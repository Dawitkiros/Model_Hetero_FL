"""G2 — surrogate-fidelity gate for the feature-anchored branch.

Same frozen trunk + branch as G1, but the branch trains on VIRTUAL features sampled
from per-class moments of the real (no-augmentation) train-split features — the
quantity the federated class-indexed channel pools exactly. Held-out epoch selection
and final test are always on REAL features/images. Modes:
  --mode virtual   per-class Gaussian surrogate (gap: Ledoit-Wolf full cov;
                   layer3: diagonal over flattened C*H*W)
  --mode realnoaug real cached no-aug features (control: isolates the augmentation
                   share of the G1->G2 drop from the moment-fidelity share)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision as tv
from sklearn.covariance import LedoitWolf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from probe.g1_graft_ceiling import build_core, trunk_forward, GraftBranch, evaluate
from probe.skyline_common import ROOT, TF_TEST, DEV, test_loader


def core_head_forward(core, feat, graft):
    import torch.nn.functional as F
    if graft == "gap":
        return core.linear(feat)
    out = core.layer4(feat)
    out = F.relu(core.bn_final(out))
    out = core.avgpool(out).flatten(1)
    return core.linear(out)


def extract_features(core, indices, graft, batch=512):
    ds = tv.datasets.CIFAR100(ROOT, train=True, transform=TF_TEST)
    ld = DataLoader(Subset(ds, indices), batch_size=batch, shuffle=False, num_workers=4)
    feats, labels = [], []
    with torch.no_grad():
        for x, y in ld:
            f, _ = trunk_forward(core, x.to(DEV), graft)
            feats.append(f.flatten(1).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


class PerLocSampler:
    def __init__(self, feats, labels, ch=64, hw=64):
        self.ch, self.hw = ch, hw
        self.classes = sorted(set(labels.tolist()))
        self.mu, self.L = {}, {}
        for c in self.classes:
            fc = feats[labels == c].view(-1, ch, hw).permute(0, 2, 1).double().numpy()
            mus = np.stack([fc[:, l].mean(0) for l in range(hw)])
            Ls = np.stack([np.linalg.cholesky(
                LedoitWolf().fit(fc[:, l]).covariance_ + 1e-6 * np.eye(ch))
                for l in range(hw)])
            self.mu[c] = torch.tensor(mus, dtype=torch.float32)
            self.L[c] = torch.tensor(Ls, dtype=torch.float32)

    def sample(self, n_per_class, generator):
        xs, ys = [], []
        for c in self.classes:
            z = torch.randn(n_per_class, self.hw, self.ch, generator=generator)
            x = self.mu[c] + torch.einsum("nlk,ljk->nlj", z, self.L[c])
            xs.append(x.permute(0, 2, 1).reshape(n_per_class, -1))
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=generator)
        return x[perm], y[perm]


class GaussianSampler:
    def __init__(self, feats, labels, full_cov):
        self.classes = sorted(set(labels.tolist()))
        self.mu, self.scale, self.full_cov = {}, {}, full_cov
        for c in self.classes:
            fc = feats[labels == c].double().numpy()
            self.mu[c] = torch.tensor(fc.mean(0), dtype=torch.float32)
            if full_cov:
                cov = LedoitWolf().fit(fc).covariance_
                L = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
                self.scale[c] = torch.tensor(L, dtype=torch.float32)
            else:
                self.scale[c] = torch.tensor(fc.std(0) + 1e-6, dtype=torch.float32)

    def sample(self, n_per_class, generator):
        xs, ys = [], []
        for c in self.classes:
            z = torch.randn(n_per_class, self.mu[c].numel(), generator=generator)
            if self.full_cov:
                x = self.mu[c] + z @ self.scale[c].T
            else:
                x = self.mu[c] + z * self.scale[c]
            xs.append(x)
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=generator)
        return x[perm], y[perm]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graft", required=True, choices=["layer3", "gap"])
    ap.add_argument("--mode", required=True,
                    choices=["virtual", "virtual_perloc", "realnoaug"])
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--core-ckpt", default=None)
    ap.add_argument("--gap-arch", default="mlp", choices=["mlp", "linear", "ccvr"])
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    if args.core_ckpt:
        core = build_core(args.core_ckpt, recal=True)
    else:
        core = build_core()
    branch = GraftBranch(args.graft, gap_arch=args.gap_arch).to(DEV)
    shape = {"layer3": (64, 8, 8), "gap": (128,)}[args.graft]

    base = tv.datasets.CIFAR100(ROOT, train=True, download=True)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(base.targets))
    tr_idx, ho_idx = perm[:45000].tolist(), perm[45000:].tolist()
    ho_ds = tv.datasets.CIFAR100(ROOT, train=True, transform=TF_TEST)
    hold = DataLoader(Subset(ho_ds, ho_idx), batch_size=256, shuffle=False, num_workers=2)
    testld = test_loader()

    print(f"[G2 {args.graft} {args.mode}] extracting train-split features", flush=True)
    feats, labels = extract_features(core, tr_idx, args.graft)
    if args.mode == "virtual":
        sampler = GaussianSampler(feats, labels, full_cov=(args.graft == "gap"))
        print(f"[G2 {args.graft} {args.mode}] stats built "
              f"({'LW full cov' if args.graft == 'gap' else 'diag'})", flush=True)
    elif args.mode == "virtual_perloc":
        assert args.graft == "layer3"
        sampler = PerLocSampler(feats, labels)
        print(f"[G2 {args.graft} {args.mode}] stats built (per-location LW 64x64)",
              flush=True)

    opt = torch.optim.SGD(branch.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_ho, best_state = -1.0, None
    for ep in range(args.epochs):
        branch.train()
        if args.mode.startswith("virtual"):
            ep_x, ep_y = sampler.sample(450, gen)
        else:
            order = torch.randperm(len(labels), generator=gen)
            ep_x, ep_y = feats[order], labels[order]
        for i in range(0, len(ep_y) - 127, 128):
            x = ep_x[i:i + 128].view(-1, *shape).to(DEV)
            y = ep_y[i:i + 128].to(DEV)
            with torch.no_grad():
                core_logits = core_head_forward(core, x, args.graft)
            loss = crit(core_logits + branch(x), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        ho = evaluate(core, branch, hold, args.graft)
        if ho > best_ho:
            best_ho = ho
            best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
        if (ep + 1) % 10 == 0:
            print(f"[G2 {args.graft} {args.mode}] epoch {ep+1}/{args.epochs} "
                  f"holdout={ho:.4f} best={best_ho:.4f}", flush=True)

    branch.load_state_dict(best_state)
    test_acc = evaluate(core, branch, testld, args.graft)
    print(f"[G2 {args.graft} {args.mode}] best_holdout={best_ho:.4f} TEST={test_acc:.4f} "
          f"(core 0.470, G1-aug: layer3 0.5918 / gap 0.5435)", flush=True)
    print(f"G2_{args.graft}_{args.mode}_DONE", flush=True)


if __name__ == "__main__":
    main()
