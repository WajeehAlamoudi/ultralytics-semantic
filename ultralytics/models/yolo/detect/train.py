# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import math
import random
from copy import copy
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from ultralytics.data import build_dataloader, build_yolo_dataset
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.models import yolo
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK
from ultralytics.utils.patches import override_configs
from ultralytics.utils.plotting import plot_images, plot_labels
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model
from ultralytics.utils.semantic import CLIPTextEncoder, SemanticLossParams, SemanticProjectionHead
from torch.utils.checkpoint import checkpoint as _grad_ckpt


class _GradCkptWrapper(nn.Module):
    """Wraps a neck block with gradient checkpointing during training.

    Uses self.wrapped(x) (i.e. __call__) for inference so AMP dtype handling,
    model.half(), and model.eval() all propagate correctly through the registered
    child module — none of which work when forward() is replaced by a plain function.
    """

    def __init__(self, wrapped):
        super().__init__()
        self.wrapped = wrapped  # registered child → model.half() / eval() propagate

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.wrapped, name)  # forward f, i, type, np, etc.

    def forward(self, x):
        if self.training:
            return _grad_ckpt(self.wrapped, x, use_reentrant=True)
        return self.wrapped(x)  # __call__ path — AMP handles dtypes correctly


class DetectionTrainer(BaseTrainer):
    """A class extending the BaseTrainer class for training based on a detection model.

    This trainer specializes in object detection tasks, handling the specific requirements for training YOLO models for
    object detection including dataset building, data loading, preprocessing, and model configuration.

    Attributes:
        model (DetectionModel): The YOLO detection model being trained.
        data (dict): Dictionary containing dataset information including class names and number of classes.
        loss_names (tuple): Names of the loss components used in training (box_loss, cls_loss, dfl_loss).

    Methods:
        build_dataset: Build YOLO dataset for training or validation.
        get_dataloader: Construct and return dataloader for the specified mode.
        preprocess_batch: Preprocess a batch of images by scaling and converting to float.
        set_model_attributes: Set model attributes based on dataset information.
        get_model: Return a YOLO detection model.
        get_validator: Return a validator for model evaluation.
        label_loss_items: Return a loss dictionary with labeled training loss items.
        progress_string: Return a formatted string of training progress.
        plot_training_samples: Plot training samples with their annotations.
        plot_training_labels: Create a labeled training plot of the YOLO model.
        auto_batch: Calculate optimal batch size based on model memory requirements.

    Examples:
        >>> from ultralytics.models.yolo.detect import DetectionTrainer
        >>> args = dict(model="yolo26n.pt", data="coco8.yaml", epochs=3)
        >>> trainer = DetectionTrainer(overrides=args)
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict[str, Any] | None = None, _callbacks: dict | None = None):
        """Initialize a DetectionTrainer object for training YOLO object detection models.

        Args:
            cfg (dict, optional): Default configuration dictionary containing training parameters.
            overrides (dict, optional): Dictionary of parameter overrides for the default configuration.
            _callbacks (dict, optional): Dictionary of callback functions to be executed during training.
        """
        super().__init__(cfg, overrides, _callbacks)
        self.clip_encoder = None
        self._sem_disabled = False  # set True when sem_warmup=-2
        self.add_callback("on_fit_epoch_end", self._maybe_plot_semantic)
        self.add_callback("on_fit_epoch_end", self._update_semantic_weights)

    def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
        """Build YOLO Dataset for training or validation.

        Args:
            img_path (str): Path to the folder containing images.
            mode (str): 'train' mode or 'val' mode, users are able to customize different augmentations for each mode.
            batch (int, optional): Size of batches, this is for 'rect' mode.

        Returns:
            (Dataset): YOLO dataset object configured for the specified mode.
        """
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct and return dataloader for the specified mode.

        Args:
            dataset_path (str): Path to the dataset.
            batch_size (int): Number of images per batch.
            rank (int): Process rank for distributed training.
            mode (str): 'train' for training dataloader, 'val' for validation dataloader.

        Returns:
            (DataLoader): PyTorch dataloader object.
        """
        assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
        with torch_distributed_zero_first(rank):  # init dataset *.cache only once if DDP
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
        )

    def preprocess_batch(self, batch: dict) -> dict:
        """Preprocess a batch of images by scaling and converting to float.

        Args:
            batch (dict): Dictionary containing batch data with 'img' tensor.

        Returns:
            (dict): Preprocessed batch with normalized images.
        """
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=self.device.type == "cuda")
        batch["img"] = batch["img"].float() / 255
        if self.args.multi_scale > 0.0:
            imgs = batch["img"]
            sz = (
                random.randrange(
                    max(self.stride, int(self.args.imgsz * (1.0 - self.args.multi_scale))),  # min imgsz
                    int(self.args.imgsz * (1.0 + self.args.multi_scale) + self.stride),  # max imgsz
                )
                // self.stride
                * self.stride
            )  # size
            sf = sz / max(imgs.shape[2:])  # scale factor
            if sf != 1:
                ns = [
                    math.ceil(x * sf / self.stride) * self.stride for x in imgs.shape[2:]
                ]  # new shape (stretched to gs-multiple)
                imgs = nn.functional.interpolate(imgs, size=ns, mode="bilinear", align_corners=False)
            batch["img"] = imgs

        if not self._sem_disabled:
            if self.clip_encoder is None:
                self.clip_encoder = CLIPTextEncoder(device=self.device)
            if batch.get("comments"):
                batch["comment_embeds"] = self.clip_encoder.encode(batch["comments"])
            if batch.get("image_comment"):
                batch["image_comment_embeds"] = self.clip_encoder.encode(batch["image_comment"])

        return batch

    def set_model_attributes(self):
        """Set model attributes based on dataset information."""
        # Nl = de_parallel(self.model).model[-1].nl  # number of detection layers (to scale hyps)
        # self.args.box *= 3 / nl  # scale to layers
        # self.args.cls *= self.data["nc"] / 80 * 3 / nl  # scale to classes and layers
        # self.args.cls *= (self.args.imgsz / 640) ** 2 * 3 / nl  # scale to image size and layers
        self.model.nc = self.data["nc"]  # attach number of classes to model
        self.model.names = self.data["names"]  # attach class names to model
        self.model.args = self.args  # attach hyperparameters to model
        if getattr(self.model, "end2end"):
            self.model.set_head_attr(max_det=self.args.max_det)
        sem_warmup = int(getattr(self.args, "sem_warmup", 10))
        if sem_warmup == -1:
            self._sem_disabled = True
            self.args.sem_active = False
            LOGGER.info("Semantic conditioning disabled (sem_warmup=-1) — pure YOLO training")
            return

        m = unwrap_model(self.model).model[-1]  # Detect module
        # Sum of neck feature channels across all FPN scales (e.g. 128+256+512=896 for yolo11n)
        # cv2[i][0] is the first Conv in each scale's box branch — confirmed safe for all standard
        # Detect-based heads (v3–v12, v10Detect, OBB, Pose, Segment). RTDETRDecoder not supported.
        try:
            feat_dim = sum(m.cv2[i][0].conv.in_channels for i in range(m.nl))
        except (AttributeError, IndexError):
            LOGGER.warning(
                "SemanticProjectionHead: could not infer neck channel dimensions from detection head. "
                "Supported models: YOLOv3, v5, v8, v9, v10, v11, v12, v26, and their variants (OBB, Pose, Segment). "
                "Unsupported: RTDETRDecoder and custom heads without standard cv2 branch. "
                f"Falling back to feat_dim=256*nl={256 * m.nl}."
            )
            feat_dim = 256 * m.nl
        unwrap_model(self.model).proj_head = SemanticProjectionHead(feat_dim, embed_dim=512).to(self.device)

        # Gradient checkpointing on neck blocks: recompute activations during backward
        # instead of storing them — fixes OOM when semantic loss extends the gradient
        # graph through neck features without touching imgsz or detaching gradients.
        # _GradCkptWrapper is defined at module level (required for pickle/torch.save).
        _ckpt_types = {"C3k2", "C2PSA", "C2f", "C3"}
        _patched = 0
        _model_layers = unwrap_model(self.model).model
        for i, _layer in enumerate(_model_layers):
            if type(_layer).__name__ in _ckpt_types:
                _model_layers[i] = _GradCkptWrapper(_layer)
                _patched += 1
        LOGGER.info(f"Gradient checkpointing applied to {_patched} neck blocks for semantic training")

        # Validate and resolve semantic hyperparameters before training starts
        init_tau   = float(getattr(self.args, "tau", 0.07))

        if not isinstance(sem_warmup, int) or (sem_warmup < 0 and sem_warmup != -1):
            raise ValueError(f"sem_warmup must be a non-negative integer, got {sem_warmup!r}")
        if sem_warmup >= self.args.epochs:
            raise ValueError(
                f"sem_warmup={sem_warmup} must be less than epochs={self.args.epochs} — "
                "semantic losses would never activate"
            )
        if not (0.01 <= init_tau <= 1.0):
            raise ValueError(
                f"tau={init_tau} is out of range. Valid range: [0.01, 1.0]. "
                "Typical values: 0.07 (CLIP default, sharp) — 0.5 (soft, good for early training)"
            )

        def _resolve_weight(name):
            """Return (float, mode_str) — float or None if auto."""
            val = getattr(self.args, name, None)
            if val is None:
                return None, "auto"
            val = float(val)
            if val < 0:
                raise ValueError(f"{name}={val} must be >= 0")
            if val == 0:
                LOGGER.warning(f"{name}=0 will completely silence that loss component")
            return val, f"{val}"

        fixed_sem, sem_mode = _resolve_weight("sem_weight")
        fixed_neg, neg_mode = _resolve_weight("neg_weight")
        fixed_fp,  fp_mode  = _resolve_weight("fp_weight")

        LOGGER.info(
            f"Semantic loss weights — "
            f"sem_weight: {sem_mode}  |  "
            f"neg_weight: {neg_mode}  |  "
            f"fp_weight: {fp_mode}  |  "
            f"tau: {init_tau} (initial, then learned)  |  "
            f"sem_warmup: {sem_warmup} epochs"
        )

        unwrap_model(self.model).sem_params = SemanticLossParams(
            init_tau=init_tau,
            fixed_sem=fixed_sem,
            fixed_neg=fixed_neg,
            fixed_fp=fixed_fp,
        ).to(self.device)

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance.

        Class weights are computed based on inverse class frequency in the training dataset,
        raised to the power of cls_pw (0 < cls_pw <= 1 dampens, cls_pw > 1 amplifies).
        Final weights are normalized so their mean equals 1.0.
        """
        assert 0 <= self.args.cls_pw <= 1.0, "cls_pw must be in the range [0, 1]"
        if self.args.cls_pw == 0.0:
            return
        classes = np.concatenate([lb["cls"].flatten() for lb in self.train_loader.dataset.labels], 0)
        class_counts = np.bincount(classes.astype(int), minlength=self.data["nc"]).astype(np.float32)
        class_counts = np.where(class_counts == 0, 1.0, class_counts)

        weights = (1.0 / class_counts) ** self.args.cls_pw  # apply power directly
        weights = weights / weights.mean()  # normalize so mean equals 1.0
        self.model.class_weights = torch.from_numpy(weights).to(self.device)
        LOGGER.info(f"Class weights: {self.model.class_weights.cpu().numpy().round(3)}")

    def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True):
        """Return a YOLO detection model.

        Args:
            cfg (str, optional): Path to model configuration file.
            weights (str, optional): Path to model weights.
            verbose (bool): Whether to display model information.

        Returns:
            (DetectionModel): YOLO detection model.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"], verbose=verbose and RANK == -1)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a DetectionValidator for YOLO model validation."""
        if self._sem_disabled:
            self.loss_names = "box_loss", "cls_loss", "dfl_loss"
        else:
            self.loss_names = "box_loss", "cls_loss", "dfl_loss", "sem_loss", "neg_loss", "fp_loss"
        return yolo.detect.DetectionValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def label_loss_items(self, loss_items: list[float] | None = None, prefix: str = "train"):
        """Return a loss dict with labeled training loss items tensor.

        Args:
            loss_items (list[float], optional): List of loss values.
            prefix (str): Prefix for keys in the returned dictionary.

        Returns:
            (dict | list): Dictionary of labeled loss items if loss_items is provided, otherwise list of keys.
        """
        keys = [f"{prefix}/{x}" for x in self.loss_names]
        if loss_items is not None:
            loss_items = [round(float(x), 5) for x in loss_items]  # convert tensors to 5 decimal place floats
            return dict(zip(keys, loss_items))
        else:
            return keys

    def progress_string(self):
        """Return a formatted string of training progress with epoch, GPU memory, loss, instances and size."""
        return ("\n" + "%11s" * (4 + len(self.loss_names))) % (
            "Epoch",
            "GPU_mem",
            *self.loss_names,
            "Instances",
            "Size",
        )

    def plot_training_samples(self, batch: dict[str, Any], ni: int) -> None:
        """Plot training samples with their annotations.

        Args:
            batch (dict[str, Any]): Dictionary containing batch data.
            ni (int): Batch index used for naming the output file.
        """
        plot_images(
            labels=batch,
            paths=batch["im_file"],
            fname=self.save_dir / f"train_batch{ni}.jpg",
            on_plot=self.on_plot,
        )

    def plot_training_labels(self):
        """Create a labeled training plot of the YOLO model."""
        boxes = np.concatenate([lb["bboxes"] for lb in self.train_loader.dataset.labels], 0)
        cls = np.concatenate([lb["cls"] for lb in self.train_loader.dataset.labels], 0)
        plot_labels(boxes, cls.squeeze(), names=self.data["names"], save_dir=self.save_dir, on_plot=self.on_plot)

    def _update_semantic_weights(self, trainer=None):
        """Control semantic loss activation via warmup gate.

        Before sem_warmup epochs → sem_active=False  (sem + neg losses silent, model learns boxes freely)
        After  sem_warmup epochs → sem_active=True   (optimizer takes full control of all weights via
                                                       SemanticLossParams learned uncertainty weighting)

        fp_loss is always active from epoch 0 — false positive suppression on negative images
        starts immediately regardless of warmup.

        Hyperparameters:
            sem_warmup (int, default 10): epochs before semantic losses activate
            tau        (float, default 0.07): initial InfoNCE temperature (then learned by optimizer)
        """
        if self._sem_disabled:
            return
        warmup = int(getattr(self.args, "sem_warmup", 10))
        was_active = getattr(self.args, "sem_active", False)
        self.args.sem_active = self.epoch >= warmup
        if self.args.sem_active and not was_active:
            LOGGER.info(f"Epoch {self.epoch}: semantic losses activated — optimizer now controls all weights")

    def _maybe_plot_semantic(self, trainer=None):
        """Called every epoch end — log semantic metrics to CSV and plot heatmap every 10 epochs."""
        if RANK not in {-1, 0} or self._sem_disabled:
            return
        self._log_semantic_metrics()
        if self.epoch % 10 == 0 or self.epoch == self.epochs - 1:
            self.plot_semantic_alignment(self.epoch)

    def _log_semantic_metrics(self):
        """Append one row of semantic metrics to semantic_results.csv."""
        proj_head = unwrap_model(self.model).proj_head
        sem_params = unwrap_model(self.model).sem_params
        sem_align = getattr(proj_head, "_last_sem_align", float("nan"))
        sem_sep   = getattr(proj_head, "_last_sem_sep",   float("nan"))
        tau       = getattr(sem_params, "_last_tau",       float("nan")) if sem_params else float("nan")
        sigma_sem = sem_params.log_sigma_sem.exp().item() if sem_params else float("nan")
        sigma_neg = sem_params.log_sigma_neg.exp().item() if sem_params else float("nan")
        sigma_fp  = sem_params.log_sigma_fp.exp().item()  if sem_params else float("nan")

        csv_path = self.save_dir / "semantic_results.csv"
        header = "epoch,sem_align,sem_sep,tau,sigma_sem,sigma_neg,sigma_fp\n"
        row    = f"{self.epoch},{sem_align:.5f},{sem_sep:.5f},{tau:.5f},{sigma_sem:.5f},{sigma_neg:.5f},{sigma_fp:.5f}\n"
        if not csv_path.exists():
            csv_path.write_text(header)
        with open(csv_path, "a") as f:
            f.write(row)

    def plot_semantic_alignment(self, epoch: int) -> None:
        """Save a heatmap of the InfoNCE similarity matrix from the last training batch.

        The matrix rows = visual anchor projections, cols = CLIP text embeddings.
        A bright diagonal means the model correctly discriminates between descriptions.
        Saved to: runs/detect/train/semantic_epoch{N}.png
        """
        proj_head = unwrap_model(self.model).proj_head
        logits = getattr(proj_head, "_last_sem_logits", None)
        if logits is None:
            return

        n = min(logits.shape[0], 32)
        sim = logits[:n, :n].float().numpy()

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(sim, cmap="viridis", aspect="auto")
        plt.colorbar(im, ax=ax, label="Similarity score (before softmax)")
        ax.set_title(
            f"Semantic Similarity Matrix — Epoch {epoch}\n"
            "Bright diagonal = visual features aligned with their text descriptions"
        )
        ax.set_xlabel("Text embeddings (comment index)")
        ax.set_ylabel("Visual embeddings (anchor index)")
        plt.tight_layout()
        fname = self.save_dir / f"semantic_epoch{epoch}.png"
        plt.savefig(fname, dpi=150)
        plt.close()
        LOGGER.info(f"Semantic alignment plot → {fname}")

    def plot_semantic_space(self) -> None:
        """Generate t-SNE plot of semantic space at end of training.

        Runs one validation batch through the model, collects ROI-pooled visual
        features and their matched CLIP text embeddings, projects both into 512-dim
        space, then plots them together in 2D using t-SNE (falls back to PCA if
        sklearn is unavailable).

        Circles  = visual features (colored by class index)
        Diamonds = CLIP text embeddings (same color as their matched visual feature)
        Lines    = connect each visual feature to its matched text embedding

        Saved to: runs/detect/train/semantic_space.png
        """
        try:
            import numpy as np_local
            proj_head = unwrap_model(self.model).proj_head
            if proj_head is None or self.clip_encoder is None:
                return

            self.model.eval()
            vis_vecs, txt_vecs, cls_ids, comments_list = [], [], [], []

            with torch.no_grad():
                for batch in self.test_loader:
                    batch = self.preprocess_batch(batch)
                    comment_embeds = batch.get("comment_embeds")
                    if comment_embeds is None:
                        continue
                    preds  = self.model(batch["img"])
                    feats  = preds["feats"]
                    img_sz = batch["img"].shape[-1]

                    gt_boxes = batch["bboxes"].clone()
                    gt_boxes[:, 0] = (batch["bboxes"][:, 0] - batch["bboxes"][:, 2] / 2) * img_sz
                    gt_boxes[:, 1] = (batch["bboxes"][:, 1] - batch["bboxes"][:, 3] / 2) * img_sz
                    gt_boxes[:, 2] = (batch["bboxes"][:, 0] + batch["bboxes"][:, 2] / 2) * img_sz
                    gt_boxes[:, 3] = (batch["bboxes"][:, 1] + batch["bboxes"][:, 3] / 2) * img_sz
                    gt_boxes = gt_boxes.clamp(0, img_sz).to(self.device)
                    img_idx  = batch["batch_idx"].long().to(self.device)
                    bw = gt_boxes[:, 2] - gt_boxes[:, 0]
                    bh = gt_boxes[:, 3] - gt_boxes[:, 1]
                    ok = (bw >= 2) & (bh >= 2) & (comment_embeds.to(self.device).norm(dim=-1) > 0)
                    if ok.sum() < 2:
                        continue

                    from ultralytics.utils.semantic import roi_pool_neck_features
                    roi_f = roi_pool_neck_features(feats, gt_boxes[ok], img_idx[ok], img_sz)
                    vis   = proj_head(roi_f.float())
                    txt   = comment_embeds.to(self.device)[ok].float()
                    vis_vecs.append(vis.cpu().numpy())
                    txt_vecs.append(txt.cpu().numpy())
                    cls_ids.extend(batch["cls"][ok].long().cpu().tolist())
                    if len(vis_vecs) >= 5:   # enough batches for a meaningful plot
                        break

            self.model.train()
            if not vis_vecs:
                return

            vis_all = np_local.concatenate(vis_vecs)   # (N, 512)
            txt_all = np_local.concatenate(txt_vecs)   # (N, 512)
            combined = np_local.concatenate([vis_all, txt_all])  # (2N, 512)
            n = len(vis_all)

            try:
                from sklearn.manifold import TSNE
                coords = TSNE(n_components=2, perplexity=min(30, n - 1),
                              random_state=42).fit_transform(combined)
            except ImportError:
                from numpy.linalg import svd
                _, _, Vt = svd(combined - combined.mean(0), full_matrices=False)
                coords = combined @ Vt[:2].T   # PCA fallback

            vis_2d = coords[:n]
            txt_2d = coords[n:]
            colors = plt.cm.tab20(np_local.array(cls_ids) % 20)

            fig, ax = plt.subplots(figsize=(10, 8))
            for i in range(n):
                ax.plot([vis_2d[i, 0], txt_2d[i, 0]],
                        [vis_2d[i, 1], txt_2d[i, 1]],
                        color=colors[i], alpha=0.3, linewidth=0.8)
            ax.scatter(vis_2d[:, 0], vis_2d[:, 1], c=colors, marker="o",
                       s=60, zorder=3, label="Visual features")
            ax.scatter(txt_2d[:, 0], txt_2d[:, 1], c=colors, marker="D",
                       s=60, zorder=3, edgecolors="black", linewidths=0.5, label="Text embeddings")
            ax.set_title(
                "Semantic Space — End of Training\n"
                "Circles=visual  Diamonds=text  Lines=matched pairs  Colors=class"
            )
            ax.legend()
            plt.tight_layout()
            fname = self.save_dir / "semantic_space.png"
            plt.savefig(fname, dpi=150)
            plt.close()
            LOGGER.info(f"Semantic space plot → {fname}")

        except Exception as e:
            LOGGER.warning(f"Semantic space plot failed: {e}")
        finally:
            self.model.train()

    def final_eval(self):
        """Run standard final eval then save semantic plots and space visualization."""
        super().final_eval()
        if RANK in {-1, 0} and not self._sem_disabled:
            self.plot_semantic_alignment(self.epoch)
            self.plot_semantic_space()

    def auto_batch(self):
        """Get optimal batch size by calculating memory occupation of model.

        Returns:
            (int): Optimal batch size.
        """
        with override_configs(self.args, overrides={"cache": False}) as self.args:
            train_dataset = self.build_dataset(self.data["train"], mode="train", batch=16)
        max_num_obj = max(len(label["cls"]) for label in train_dataset.labels) * 4  # 4 for mosaic augmentation
        n = len(train_dataset)
        del train_dataset  # free memory
        return super().auto_batch(max_num_obj, dataset_size=n)
