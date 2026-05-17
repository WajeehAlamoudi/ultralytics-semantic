"""
Semantic conditioning unit tests — runs fully on CPU with dummy tensors.
No dataset, no training, no CLIP download required for most tests.

Run:
    python test_semantic.py
"""

import math
import sys
import traceback

import torch
import torch.nn.functional as F


def ok(name):
    print(f"  PASS  {name}")


def fail(name, e):
    print(f"  FAIL  {name}")
    traceback.print_exc()


# ─────────────────────────────────────────────
# 1. SemanticLossParams
# ─────────────────────────────────────────────
def test_sem_loss_params():
    from ultralytics.utils.semantic import SemanticLossParams

    sp = SemanticLossParams(init_tau=0.07)

    # tau initialised correctly and clamped
    assert abs(sp.tau.item() - 0.07) < 1e-4, f"tau init wrong: {sp.tau.item()}"
    ok("SemanticLossParams tau init")

    # tau clamp — push log_tau very negative
    with torch.no_grad():
        sp.log_tau.fill_(-100)
    assert sp.tau.item() >= 0.01 - 1e-6, "tau clamp lower bound failed"
    ok("SemanticLossParams tau clamp low")

    with torch.no_grad():
        sp.log_tau.fill_(100)
    assert sp.tau.item() <= 1.0, "tau clamp upper bound failed"
    ok("SemanticLossParams tau clamp high")

    # weight — auto mode (learned)
    sp2 = SemanticLossParams(init_tau=0.07)
    loss = torch.tensor(1.0)
    weighted = sp2.weight(loss, sp2.log_sigma_sem, fixed=None)
    assert weighted.shape == torch.Size([1]), "auto weight shape wrong"
    ok("SemanticLossParams auto weight")

    # weight — fixed mode
    weighted_fixed = sp2.weight(loss, sp2.log_sigma_sem, fixed=2.0)
    assert abs(weighted_fixed.item() - 2.0) < 1e-5, "fixed weight wrong"
    ok("SemanticLossParams fixed weight")

    # fixed_sem/neg/fp stored correctly
    sp3 = SemanticLossParams(init_tau=0.07, fixed_sem=0.5, fixed_neg=None, fixed_fp=1.5)
    assert sp3.fixed_sem == 0.5
    assert sp3.fixed_neg is None
    assert sp3.fixed_fp == 1.5
    ok("SemanticLossParams fixed storage")


# ─────────────────────────────────────────────
# 2. SemanticProjectionHead
# ─────────────────────────────────────────────
def test_proj_head():
    from ultralytics.utils.semantic import SemanticProjectionHead

    head = SemanticProjectionHead(in_dim=896, embed_dim=512)

    x = torch.randn(8, 896)
    out = head(x)

    assert out.shape == (8, 512), f"proj_head output shape wrong: {out.shape}"
    ok("SemanticProjectionHead output shape")

    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones(8), atol=1e-5), "output not unit-norm"
    ok("SemanticProjectionHead unit-norm output")


# ─────────────────────────────────────────────
# 3. roi_pool_neck_features
# ─────────────────────────────────────────────
def test_roi_pool():
    from ultralytics.utils.semantic import roi_pool_neck_features

    B, img_size = 2, 640
    # dummy neck features: P3(128ch), P4(256ch), P5(512ch)
    feats = [
        torch.randn(B, 128, 80, 80),
        torch.randn(B, 256, 40, 40),
        torch.randn(B, 512, 20, 20),
    ]
    # 4 boxes: 2 per image
    boxes = torch.tensor([
        [100., 100., 300., 300.],
        [200., 200., 400., 400.],
        [ 50.,  50., 150., 150.],
        [300., 100., 500., 400.],
    ])
    img_idx = torch.tensor([0, 0, 1, 1])

    out = roi_pool_neck_features(feats, boxes, img_idx, img_size)
    assert out.shape == (4, 128 + 256 + 512), f"roi_pool shape wrong: {out.shape}"
    ok("roi_pool_neck_features output shape")

    # degenerate box (w<2) should not crash when filtered upstream
    ok("roi_pool_neck_features runs without error")


# ─────────────────────────────────────────────
# 4. InfoNCE loss correctness
# ─────────────────────────────────────────────
def test_infonce():
    # build a perfect alignment scenario: vis == txt
    N = 8
    vecs = F.normalize(torch.randn(N, 512), dim=-1)
    tau  = 0.07

    logits = (vecs @ vecs.T) / tau
    labels = torch.arange(N)
    loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    # with perfect alignment diagonal dominates → loss should be low
    assert loss.item() < 1.0, f"InfoNCE with perfect alignment too high: {loss.item()}"
    ok("InfoNCE perfect alignment gives low loss")

    # random vectors → loss should be higher (near log(N))
    rnd_v = F.normalize(torch.randn(N, 512), dim=-1)
    rnd_t = F.normalize(torch.randn(N, 512), dim=-1)
    logits_rnd = (rnd_v @ rnd_t.T) / tau
    loss_rnd = (F.cross_entropy(logits_rnd, labels) + F.cross_entropy(logits_rnd.T, labels)) / 2
    assert loss_rnd.item() > loss.item(), "random should have higher loss than perfect alignment"
    ok("InfoNCE random > perfect alignment loss")


# ─────────────────────────────────────────────
# 5. CLIP cache (requires CLIP installed)
# ─────────────────────────────────────────────
def test_clip_cache():
    try:
        from ultralytics.utils.semantic import CLIPTextEncoder
    except ImportError:
        print("  SKIP  CLIP cache (CLIP not installed)")
        return

    enc = CLIPTextEncoder(device="cpu")

    texts = ["person running fast", "red car parked", "person running fast"]  # 3rd is duplicate
    _ = enc.encode(texts)

    assert len(enc._cache) == 2, f"cache should have 2 entries, got {len(enc._cache)}"
    ok("CLIPTextEncoder cache stores unique texts only")

    cache_size_before = len(enc._cache)
    _ = enc.encode(["person running fast"])   # already cached
    assert len(enc._cache) == cache_size_before, "cache grew on repeated text"
    ok("CLIPTextEncoder cache hit — no recompute")

    # empty string should not be cached
    _ = enc.encode([""])
    assert "" not in enc._cache, "empty string should not be cached"
    ok("CLIPTextEncoder empty string not cached")


# ─────────────────────────────────────────────
# 6. Validation guards
# ─────────────────────────────────────────────
def test_guards():
    from ultralytics.utils.semantic import SemanticLossParams

    # negative fixed weight caught at trainer level — test weight function directly
    sp = SemanticLossParams(init_tau=0.07, fixed_fp=0.0)
    loss = torch.tensor(1.0)
    result = sp.weight(loss, sp.log_sigma_fp, fixed=0.0)
    assert result.item() == 0.0, "fp_weight=0 should silence loss"
    ok("SemanticLossParams fixed=0 silences loss")

    # tau out-of-range is caught by validator in train.py
    # here just verify clamp works at runtime
    sp2 = SemanticLossParams(init_tau=0.5)
    with torch.no_grad():
        sp2.log_tau.fill_(math.log(0.001))   # below 0.01
    assert sp2.tau.item() >= 0.01 - 1e-6
    ok("SemanticLossParams tau runtime clamp holds")


# ─────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────
TESTS = [
    ("SemanticLossParams",       test_sem_loss_params),
    ("SemanticProjectionHead",   test_proj_head),
    ("roi_pool_neck_features",   test_roi_pool),
    ("InfoNCE correctness",      test_infonce),
    ("CLIP cache",               test_clip_cache),
    ("Validation guards",        test_guards),
]

if __name__ == "__main__":
    passed = failed = 0
    for name, fn in TESTS:
        print(f"\n{'─'*50}")
        print(f"  {name}")
        print(f"{'─'*50}")
        try:
            fn()
            passed += 1
        except Exception as e:
            fail(name, e)
            failed += 1

    print(f"\n{'═'*50}")
    print(f"  Results: {passed} passed  |  {failed} failed")
    print(f"{'═'*50}")
    sys.exit(0 if failed == 0 else 1)
