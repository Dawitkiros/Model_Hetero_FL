"""CONTROL 2 — zeroed-shell HeteroFL: toxicity vs artifact.

For a trained HeteroFL global model (rate 1.0), evaluate three ways on the GLOBAL test
set with identical BN recal:
  (i)   normal full model
  (ii)  extracted rate-0.25 core (HeteroFL static prefix extraction + ScalerV2 eval mode)
  (iii) full model with the SHELL zeroed (everything outside the core-prefix block -> 0)
If (iii) recovers to ~ (ii), the degradation is SHELL TOXICITY, not a BN/eval artifact.

Usage: python -m probe.control2_zeroed_shell <finalmodel.pt> <label>
"""
import sys
import torch
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from flore.hetero_models import create_hetero_model, ScalerV2, calibrate_bn_stats
from flore.hetero_utils import create_extractor
from probe.skyline_common import DEV, test_loader, calib_loader

CKPT = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "?"
testld = test_loader(); calibld = calib_loader()
state = torch.load(CKPT, map_location="cpu")

def set_always(m, v):
    for mm in m.modules():
        if isinstance(mm, ScalerV2):
            mm.always_scale = v

@torch.no_grad()
def acc(model):
    calibrate_bn_stats(model, calibld, DEV, num_batches=20); model.eval()
    c = t = 0
    for x, y in testld:
        x = x.to(DEV); c += (model(x).argmax(1).cpu() == y).sum().item(); t += y.size(0)
    return c / t

# (i) normal full
full = create_hetero_model("HeteroPreResNet18", num_classes=100, model_rate=1.0, norm_type="sbn").to(DEV)
full.load_state_dict({k: v.to(DEV) for k, v in state.items()})
acc_full = acc(full)

# (ii) extracted rate-0.25 core (same path the FL eval used)
core_net = create_hetero_model("HeteroPreResNet18", num_classes=100, model_rate=0.25, norm_type="sbn").to(DEV)
ext = create_extractor("static", full, num_classes=100)
core_state = ext.extract_sub_model(core_net, 0, 1)
core_net.load_state_dict(core_state, strict=True)
set_always(core_net, True)                    # force ScalerV2 in eval (as server_app does)
acc_core = acc(core_net)

# (iii) full model with shell zeroed (keep only the core-prefix block of every layer)
full_z = create_hetero_model("HeteroPreResNet18", num_classes=100, model_rate=1.0, norm_type="sbn").to(DEV)
full_z.load_state_dict({k: v.to(DEV) for k, v in state.items()})
sd = full_z.state_dict()
cref = core_net.state_dict()
for name, p in sd.items():
    if name in cref and cref[name].shape != p.shape:
        z = torch.zeros_like(p)
        sl = tuple(slice(0, s) for s in cref[name].shape)
        z[sl] = p[sl]                          # keep core block, zero everything shell
        sd[name] = z
full_z.load_state_dict(sd)
acc_zeroed = acc(full_z)

print(f"[Control 2 {LABEL}] normal_full={acc_full:.4f}  core={acc_core:.4f}  zeroed_shell={acc_zeroed:.4f}")
verdict = "SHELL TOXICITY (zeroed recovers core)" if abs(acc_zeroed - acc_core) < 0.05 else \
          ("ARTIFACT (zeroed still broken)" if abs(acc_zeroed - acc_full) < 0.05 else "ambiguous")
print(f"[Control 2 {LABEL}] verdict: {verdict}")
print(f"CONTROL2_{LABEL}_DONE")
