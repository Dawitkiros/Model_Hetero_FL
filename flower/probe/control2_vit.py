"""CONTROL 2 (ViT) — zeroed-shell HeteroFL: toxicity vs artifact.

For a trained HeteroFL global ViT (rate 1.0), evaluate three ways on the CIFAR-100
test set:
  (i)   normal full model
  (ii)  extracted rate-r core (static prefix extraction + ScalerV2 eval mode)
  (iii) full model with the SHELL zeroed (everything outside the core-prefix -> 0)
If (iii) recovers to ~ (ii), the degradation is SHELL TOXICITY, not an eval artifact.
LayerNorm has no running stats, so no BN calibration is needed.

Usage: python -m probe.control2_vit <finalmodel.pt> <label> [core_rate]
"""
import sys
import torch

from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from flore.hetero_models import create_hetero_model, ScalerV2
from flore.hetero_utils import create_extractor
from probe.skyline_common import DEV, test_loader

CKPT = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else "?"
CORE_RATE = float(sys.argv[3]) if len(sys.argv) > 3 else 0.125
NC = 100
testld = test_loader()
state = torch.load(CKPT, map_location="cpu")


def set_always(m, v):
    for mm in m.modules():
        if isinstance(mm, ScalerV2):
            mm.always_scale = v


@torch.no_grad()
def acc(model):
    model.eval()
    c = t = 0
    for x, y in testld:
        x = x.to(DEV)
        c += (model(x).argmax(1).cpu() == y).sum().item()
        t += y.size(0)
    return c / t


full = create_hetero_model("HeteroViT", num_classes=NC, model_rate=1.0).to(DEV)
full.load_state_dict({k: v.to(DEV) for k, v in state.items()})
acc_full = acc(full)

core_net = create_hetero_model("HeteroViT", num_classes=NC, model_rate=CORE_RATE).to(DEV)
ext = create_extractor("static", full, num_classes=NC)
core_state = ext.extract_sub_model(core_net, 0, 1)
core_net.load_state_dict(core_state, strict=True)
set_always(core_net, True)
acc_core = acc(core_net)

full_z = create_hetero_model("HeteroViT", num_classes=NC, model_rate=1.0).to(DEV)
full_z.load_state_dict({k: v.to(DEV) for k, v in state.items()})
sd = full_z.state_dict()
cref = core_net.state_dict()
for name, p in sd.items():
    if name in cref and cref[name].shape != p.shape:
        z = torch.zeros_like(p)
        sl = tuple(slice(0, s) for s in cref[name].shape)
        z[sl] = p[sl]
        sd[name] = z
full_z.load_state_dict(sd)
acc_zeroed = acc(full_z)

print(f"[Control 2 {LABEL}] normal_full={acc_full:.4f}  core={acc_core:.4f}  zeroed_shell={acc_zeroed:.4f}")
verdict = "SHELL TOXICITY (zeroed recovers core)" if abs(acc_zeroed - acc_core) < 0.05 else \
          ("ARTIFACT (zeroed still broken)" if abs(acc_zeroed - acc_full) < 0.05 else "ambiguous")
print(f"[Control 2 {LABEL}] verdict: {verdict}")
print(f"CONTROL2_{LABEL}_DONE")
