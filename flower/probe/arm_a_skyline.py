"""ARM A — full-client shell skyline (oracle upper bound at p_full=0.10).

Freeze a rate-0.25 core embedded in the rate-1.0 model; train ONLY shell-resident
params on the POOLED data of the 5 full-capacity clients (partitions 0-4),
centralized + unconstrained (no tanh/L2/near-zero/LR-damping). BN recal, then eval
the full model on the GLOBAL test set with a seen/unseen-class breakdown.

Usage: python -m probe.arm_a_skyline [all]   ('all' -> Arm B: pool all 50 clients)
"""
import argparse
import sys, time
import numpy as np
import torch, torch.nn as nn
from pathlib import Path as _P; sys.path.insert(0, str(_P(__file__).resolve().parents[1]))
from flore.hetero_models import create_hetero_model, calibrate_bn_stats
from probe.skyline_common import (
    DEV, full_client_indices, all_client_indices, loader_from_indices, test_loader,
    calib_loader, embed_core_and_masks, eval_seen_unseen)

ap = argparse.ArgumentParser()
ap.add_argument("n", nargs="?", default="5")
ap.add_argument("--alpha", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--core-ckpt", default=str(_P(__file__).resolve().parent / "core025_sbn.pt"))
args = ap.parse_args()
N = 50 if args.n == "all" else int(args.n)
ARM = f"N{N}"
if args.alpha != 0.1:
    ARM += f"_a{args.alpha}"
if args.seed != 42:
    ARM += f"_s{args.seed}"
if "core025" not in args.core_ckpt:
    import os
    ARM += "_" + os.path.basename(args.core_ckpt).replace("_sbn.pt", "")
CKPT = args.core_ckpt

ck = torch.load(CKPT, map_location="cpu")
core_state = ck["state"]
print(f"[Arm {ARM}] loaded frozen core: standalone acc={ck['acc']:.4f}")

pooled, counts, _ = full_client_indices(50, args.alpha, args.seed, n_full=N)
seen_mask = counts > 0
print(f"[Arm {ARM}] pooled samples={len(pooled)}  seen_classes={int(seen_mask.sum())}/100  "
      f"min/median/max class count among seen="
      f"{int(counts[seen_mask].min())}/{int(np.median(counts[seen_mask]))}/{int(counts[seen_mask].max())}")

# Held-out split of the POOLED data for honest epoch selection (no test peeking).
rng = np.random.default_rng(0)
perm = rng.permutation(len(pooled))
n_val = max(1, int(0.2 * len(pooled)))
val_idx = [pooled[i] for i in perm[:n_val]]
tr_idx = [pooled[i] for i in perm[n_val:]]
print(f"[Arm {ARM}] train={len(tr_idx)} heldout_val={len(val_idx)} (epoch selected on held-out, not test)")

trainld = loader_from_indices(tr_idx, train=True, batch=128)
valld = loader_from_indices(val_idx, train=False, batch=256)
testld = test_loader(); calibld = calib_loader()

@torch.no_grad()
def val_acc(model):
    model.eval(); c = t = 0
    for x, y in valld:
        x = x.to(DEV); c += (model(x).argmax(1).cpu() == y).sum().item(); t += y.size(0)
    return c / t

full = create_hetero_model("HeteroPreResNet18", num_classes=100, model_rate=1.0, norm_type="sbn").to(DEV)
masks = embed_core_and_masks(full, core_state)
masks = {k: v.to(DEV) for k, v in masks.items()}
n_train_param = sum(int((masks[n] > 0).sum()) for n, _ in full.named_parameters() if n in masks)
n_total = sum(p.numel() for p in full.parameters())
print(f"[Arm {ARM}] trainable shell params={n_train_param/1e6:.3f}M / {n_total/1e6:.3f}M total")

# reference: full model with embedded core + UNtrained (init) shell (epoch 0 candidate)
acc0, s0, u0, _ = eval_seen_unseen(full, testld, seen_mask, calibld)
v0 = val_acc(full)
print(f"[Arm {ARM}] pre-shell-train (random shell): val={v0:.4f} global={acc0:.4f} seen={s0:.4f} unseen={u0:.4f}")

opt = torch.optim.SGD(full.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
EPOCHS = 80
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss()

# best selected by HELD-OUT val acc; we report the global test acc AT that epoch.
best = {"val": v0, "global": acc0, "seen": s0, "unseen": u0, "epoch": 0}
print(f"[Arm {ARM}] epoch loss val global seen unseen")
for ep in range(1, EPOCHS + 1):
    full.train(); tot = nb = 0
    for x, y in trainld:
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad(); loss = crit(full(x), y); loss.backward()
        for n, p in full.named_parameters():
            if p.grad is not None and n in masks:
                p.grad.mul_(masks[n])
        opt.step(); tot += loss.item(); nb += 1
    sched.step()
    if ep % 5 == 0 or ep == EPOCHS:
        acc, s, u, _ = eval_seen_unseen(full, testld, seen_mask, calibld)
        v = val_acc(full)
        print(f"[Arm {ARM}] {ep:3d} {tot/nb:.3f} {v:.4f} {acc:.4f} {s:.4f} {u:.4f}", flush=True)
        if v > best["val"]:
            best = {"val": v, "global": acc, "seen": s, "unseen": u, "epoch": ep}
print(f"[Arm {ARM}] SKYLINE (epoch {best['epoch']} by held-out val={best['val']:.4f}): "
      f"global={best['global']:.4f} seen={best['seen']:.4f} unseen={best['unseen']:.4f}  "
      f"(frozen-core standalone {ck['acc']:.4f}; pre-train global {acc0:.4f})")
print(f"ARM_{ARM}_DONE")
