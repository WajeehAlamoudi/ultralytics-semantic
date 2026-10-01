from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


class CLIPTextEncoder:
    """Frozen CLIP text encoder. Converts strings to unit-norm 512-dim embeddings."""

    def __init__(self, model_name="ViT-B/32", device="cpu"):
        try:
            import clip
        except ImportError:
            raise ImportError("pip install git+https://github.com/openai/CLIP.git")
        self._clip = clip
        self.model, _ = clip.load(model_name, device=device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.device = device
        self._cache: dict[str, torch.Tensor] = {}  # text → unit-norm embedding on CPU

    @torch.no_grad()
    def encode(self, texts: list) -> torch.Tensor:
        """Encode list of strings with in-memory cache. Returns (N, 512).

        Each unique string is encoded once and cached on CPU for the lifetime of
        the encoder. Subsequent calls for the same string are free (dict lookup).
        Empty strings always return zero vectors and are not cached.
        """
        uncached = [t for t in texts if t not in self._cache]
        if uncached:
            tokens = self._clip.tokenize(uncached, truncate=True).to(self.device)
            embeds = self.model.encode_text(tokens).float()
            mask = torch.tensor([len(t.strip()) > 0 for t in uncached], dtype=embeds.dtype, device=self.device)
            embeds = F.normalize(embeds * mask.unsqueeze(1), dim=-1)
            for t, e in zip(uncached, embeds):
                if len(t.strip()) > 0:  # only cache non-empty strings
                    self._cache[t] = e.cpu()  # store on CPU to save GPU memory
        return torch.stack(
            [
                self._cache[t].to(self.device) if t in self._cache else torch.zeros(512, device=self.device)
                for t in texts
            ]
        )


def roi_pool_neck_features(
    feats: list, boxes_xyxy: torch.Tensor, img_idx: torch.Tensor, img_size: int, output_size: int = 4
) -> torch.Tensor:
    """ROI-pool multi-scale neck features for each GT box.

    For each GT box, crops and average-pools all FPN scale feature maps within the box region, then concatenates across
    scales.

    Args:
        feats: list of (B, C_i, H_i, W_i) neck feature maps (P3, P4, P5)
        boxes_xyxy: (N, 4) GT boxes in pixel coords [x1, y1, x2, y2]
        img_idx: (N,) image index for each box within the batch
        img_size: input image size (assumed square)
        output_size: spatial size after ROI align (default 4x4 → averaged to 1x1)

    Returns:
        (N, sum(C_i)) pooled feature vector per GT box
    """
    rois = torch.cat([img_idx.float().unsqueeze(1), boxes_xyxy.float()], dim=1)  # (N, 5)
    pooled = []
    for feat in feats:
        scale = feat.shape[-1] / img_size  # spatial scale relative to input
        # Call C++ CUDA extension directly — bypasses the Python roi_align wrapper which
        # has a torch.compile hook that materializes a [K,C,PH,PW,IY,IX] bilinear tensor (~26 GiB).
        roi_feat = torch.ops.torchvision.roi_align(feat, rois, scale, output_size, output_size, -1, True)
        roi_feat = roi_feat.mean(dim=[-2, -1])  # (N, C_i) — average pool spatial dims
        pooled.append(roi_feat)
    return torch.cat(pooled, dim=1)  # (N, sum(C_i))


class SemanticLossParams(nn.Module):
    """Learnable loss-weighting parameters (Kendall et al. 2018) + InfoNCE temperature.

    All four variables are optimized automatically by the main optimizer: log_sigma_sem → uncertainty weight for
    sem_loss (box-comment contrastive) log_sigma_neg → uncertainty weight for neg_loss (scene-comment contrastive)
    log_sigma_fp → uncertainty weight for fp_loss (false positive penalty) log_tau → InfoNCE temperature τ =
    exp(log_tau), clamped to [0.01, 1.0]

    Weighting formula (per loss): L_weighted = 0.5 * L * exp(-2 * log_sigma) + log_sigma When L is large → optimizer
    increases sigma → effective weight drops (prevents domination) When L is small → optimizer decreases sigma →
    effective weight rises (pays more attention)
    """

    def __init__(
        self,
        init_tau: float = 0.07,
        fixed_sem: float | None = None,
        fixed_neg: float | None = None,
        fixed_fp: float | None = None,
    ):
        super().__init__()
        self.log_sigma_sem = nn.Parameter(torch.zeros(1))
        self.log_sigma_neg = nn.Parameter(torch.zeros(1))
        self.log_sigma_fp = nn.Parameter(torch.zeros(1))
        self.log_tau = nn.Parameter(torch.tensor(math.log(init_tau)))
        # None = auto (learned), float = fixed (user-specified)
        self.fixed_sem = fixed_sem
        self.fixed_neg = fixed_neg
        self.fixed_fp = fixed_fp

    @property
    def tau(self) -> torch.Tensor:
        return self.log_tau.exp().clamp(0.01, 1.0)

    def weight(self, loss: torch.Tensor, log_sigma: nn.Parameter, fixed: float | None = None) -> torch.Tensor:
        """Apply fixed weight if user specified one, otherwise learned uncertainty weighting."""
        if fixed is not None:
            return loss * fixed
        return 0.5 * loss * (-2.0 * log_sigma).exp() + log_sigma


class SemanticProjectionHead(nn.Module):
    """Projects YOLO anchor features into CLIP embedding space (512-dim)."""

    def __init__(self, in_dim: int, embed_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)
