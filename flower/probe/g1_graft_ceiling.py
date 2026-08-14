"""G1 — graft-ceiling gate for the feature-anchored branch.

Frozen rate-0.25 core trunk (core025_sbn.pt, the skyline anchor, standalone 0.470).
A GroupNorm branch is grafted at one of {layer1, layer2, layer3, gap} and trained
centrally on REAL pooled CIFAR-100 features (full 50k train = the federation-wide
class-indexed channel's data). Fused logits = frozen core logits + branch logits.
Held-out epoch selection (5k split, never test). Targets: core 0.470 (floor),
full-depth-shell oracle 0.5453 (prize). Kill if best graft < ~+3pp over core.
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import torchvision as tv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from flore.hetero_models import HeteroPreResNet, PreActHeteroBasicBlock
from probe.skyline_common import ROOT, TF_TRAIN, TF_TEST, DEV, test_loader

LOCAL_CORE_CKPT = PROJECT_ROOT / "probe" / "core025_sbn.pt"
CORE_CKPT = str(LOCAL_CORE_CKPT)
CORE_CH = {"layer1": 16, "layer2": 32, "layer3": 64}
GAP_DIM = 128


def build_core(ckpt=CORE_CKPT, recal=False):
    net = HeteroPreResNet(PreActHeteroBasicBlock, [2, 2, 2, 2], in_channels=3,
                          num_classes=100, model_rate=0.25, norm_type="sbn")
    state = torch.load(ckpt, map_location="cpu")
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    net.load_state_dict(state, strict=True)
    net.to(DEV)
    if recal:
        from flore.hetero_models import calibrate_bn_stats, ScalerV2
        from probe.skyline_common import calib_loader
        for m in net.modules():
            if isinstance(m, ScalerV2):
                m.always_scale = True
        calibrate_bn_stats(net, calib_loader(), DEV, num_batches=20)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def trunk_forward(core, x, graft):
    out = core.conv1(x)
    out = core.stem_scaler(out)
    out = core.layer1(out)
    if graft == "layer1":
        feat = out
    out = core.layer2(out)
    if graft == "layer2":
        feat = out
    out = core.layer3(out)
    if graft == "layer3":
        feat = out
    out = core.layer4(out)
    out = F.relu(core.bn_final(out))
    out = core.avgpool(out)
    out = out.view(out.size(0), -1)
    if graft == "gap":
        feat = out
    logits = core.linear(out)
    return feat, logits


class GraftBranch(nn.Module):
    def __init__(self, graft, gap_arch="mlp"):
        super().__init__()
        self.graft = graft
        if graft == "gap":
            if gap_arch == "ccvr":
                self.net = nn.Linear(GAP_DIM, 100)
                nn.init.zeros_(self.net.weight)
                nn.init.zeros_(self.net.bias)
            elif gap_arch == "linear":
                self.net = nn.Linear(GAP_DIM, 100)
            else:
                self.net = nn.Sequential(
                    nn.Linear(GAP_DIM, 1024), nn.ReLU(inplace=True),
                    nn.Linear(1024, 1024), nn.ReLU(inplace=True),
                    nn.Linear(1024, 100))
            return
        donor = HeteroPreResNet(PreActHeteroBasicBlock, [2, 2, 2, 2], in_channels=3,
                                num_classes=100, model_rate=0.75, use_scaler=False,
                                norm_type="gn")
        tails = {"layer1": ["layer2", "layer3", "layer4"],
                 "layer2": ["layer3", "layer4"],
                 "layer3": ["layer4"]}[graft]
        in_planes = {"layer1": 48, "layer2": 96, "layer3": 192}[graft]
        self.adapter = nn.Conv2d(CORE_CH[graft], in_planes, kernel_size=1, bias=False)
        self.tail = nn.Sequential(*[copy.deepcopy(getattr(donor, t)) for t in tails])
        self.bn_final = nn.GroupNorm(32, 384)
        self.head = nn.Linear(384, 100)

    def forward(self, feat):
        if self.graft == "gap":
            return self.net(feat)
        out = self.adapter(feat)
        out = self.tail(out)
        out = F.relu(self.bn_final(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.head(out)


def evaluate(core, branch, loader, graft):
    branch.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            feat, core_logits = trunk_forward(core, x, graft)
            logits = core_logits + branch(feat)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graft", required=True, choices=["layer1", "layer2", "layer3", "gap"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--core-ckpt", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    core = build_core(args.core_ckpt, recal=True) if args.core_ckpt else build_core()
    branch = GraftBranch(args.graft).to(DEV)
    nparams = sum(p.numel() for p in branch.parameters())

    base = tv.datasets.CIFAR100(ROOT, train=True, download=True)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(base.targets))
    tr_idx, ho_idx = perm[:45000].tolist(), perm[45000:].tolist()
    tr_ds = tv.datasets.CIFAR100(ROOT, train=True, transform=TF_TRAIN)
    ho_ds = tv.datasets.CIFAR100(ROOT, train=True, transform=TF_TEST)
    trld = DataLoader(Subset(tr_ds, tr_idx), batch_size=128, shuffle=True,
                      num_workers=4, drop_last=True)
    hold = DataLoader(Subset(ho_ds, ho_idx), batch_size=256, shuffle=False, num_workers=2)
    testld = test_loader()

    opt = torch.optim.SGD(branch.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_ho, best_state = -1.0, None
    for ep in range(args.epochs):
        branch.train()
        for x, y in trld:
            x, y = x.to(DEV), y.to(DEV)
            with torch.no_grad():
                feat, core_logits = trunk_forward(core, x, args.graft)
            logits = core_logits + branch(feat)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        ho = evaluate(core, branch, hold, args.graft)
        if ho > best_ho:
            best_ho = ho
            best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
        if (ep + 1) % 5 == 0:
            print(f"[G1 {args.graft}] epoch {ep+1}/{args.epochs} holdout={ho:.4f} "
                  f"best={best_ho:.4f}", flush=True)

    branch.load_state_dict(best_state)
    test_acc = evaluate(core, branch, testld, args.graft)
    print(f"[G1 {args.graft}] branch_params={nparams:,} best_holdout={best_ho:.4f} "
          f"TEST={test_acc:.4f} (core ref 0.470, oracle 0.5453)", flush=True)
    print(f"G1_{args.graft}_DONE", flush=True)


if __name__ == "__main__":
    main()
