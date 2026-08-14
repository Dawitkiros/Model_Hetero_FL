"""CONTROL 1 — frozen small-random shell on ResNet: confirm acc_full ≈ acc_core.

Build the rate-1.0 model with the trained core embedded (lower-triangular, core path
preserved), set the shell-resident params to small-random (sigma), FROZEN (no training),
recalibrate BN, evaluate on the GLOBAL test set. Report acc_full, acc_core, delta for
sigma in {0.001, 0.01, 0.05}. Tests whether a frozen near-zero shell leaves acc_full≈acc_core
on ResNet+sBN (vs the MNIST/2-layer result), where residual skip-paths + BN recal can differ.
"""
import sys
import numpy as np
import torch
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from flore.hetero_models import create_hetero_model, HeteroPreResNet18, calibrate_bn_stats
from probe.skyline_common import DEV, embed_core_and_masks, test_loader, calib_loader

ck = torch.load(str(_P(__file__).resolve().parent / "core025_sbn.pt"), map_location="cpu")
core_state = ck["state"]
testld = test_loader(); calibld = calib_loader()

@torch.no_grad()
def acc(model):
    calibrate_bn_stats(model, calibld, DEV, num_batches=20); model.eval()
    c = t = 0
    for x, y in testld:
        x = x.to(DEV); c += (model(x).argmax(1).cpu() == y).sum().item(); t += y.size(0)
    return c / t

# acc_core = the rate-0.25 prefix standalone (the frozen core) — built use_scaler=False
# to MATCH how the core was trained (default create_hetero_model uses use_scaler=True,
# whose train-mode scaler corrupts BN calibration -> random acc).
core = HeteroPreResNet18(in_channels=3, num_classes=100, model_rate=0.25,
                         use_scaler=False, norm_type="sbn").to(DEV)
core.load_state_dict({k: v.to(DEV) for k, v in core_state.items()})
acc_core = acc(core)
print(f"[Control 1] acc_core (rate-0.25 frozen core) = {acc_core:.4f}")

print(f"[Control 1] sigma  acc_full  acc_core  delta")
torch.manual_seed(0)
for sigma in (0.001, 0.01, 0.05):
    full = create_hetero_model("HeteroPreResNet18", num_classes=100, model_rate=1.0, norm_type="sbn").to(DEV)
    masks = {k: v.to(DEV) for k, v in embed_core_and_masks(full, core_state).items()}
    sd = full.state_dict()
    for k, p in sd.items():
        m = masks.get(k)
        if m is not None and (m > 0).any():
            noise = torch.randn_like(p) * sigma
            sd[k] = p * (1 - m) + noise * m       # core kept, shell set to small-random
    full.load_state_dict(sd)
    af = acc(full)
    print(f"[Control 1] {sigma:<6} {af:.4f}   {acc_core:.4f}   {af-acc_core:+.4f}", flush=True)
print("CONTROL1_DONE")
