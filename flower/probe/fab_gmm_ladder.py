"""FAB GMM-surrogate ladder: does a richer per-class surrogate beat single-Gaussian CCVR?

The deployable FAB number is FAB-linear = CCVR-linear = 0.563 on the FL-trained rate-0.25
CIFAR-100 core, with a single per-class Gaussian (LedoitWolf full cov) virtual-feature
channel. This script tests whether a mixture-of-Gaussians (GMM) surrogate, optionally with
a richer head, lets the recalibrated classifier climb above 0.563 -- which would make CCVR
the single-Gaussian-linear special case of a more general method rather than an equal.

Ladder (per seed): surrogate x head x arm.
  surrogate: gauss_lw (=CCVR baseline) | gmm_tied_K{1,3,5,10} | gmm_full_K3 | gmm_diag_K10
  head:      linear | mlp        (does a richer surrogate let the MLP stop overfitting?)
  arm:       fab (fuse core logits) | ccvr (head alone, isolates the surrogate effect)
Anchors: core standalone floor; realnoaug (real no-aug features = surrogate-free ceiling).
Diagnostic: per-class BIC multimodality (fraction of classes with BIC-optimal K>1).

Shared with the Table-4 reproduction: same core, 128-d GAP, 45k/5k split, 450/class
resampled each epoch, 60-epoch SGD+cosine, holdout best-epoch selection, real test eval.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from probe.g1_graft_ceiling import build_core, trunk_forward, GraftBranch, GAP_DIM
from probe.g2_surrogate import extract_features, core_head_forward, GaussianSampler
from probe.skyline_common import ROOT, TF_TEST, DEV, test_loader

import torchvision as tv
from torch.utils.data import DataLoader, Subset


class GMMSampler:
    """Per-class K-component GMM virtual-feature sampler (full | tied | diag cov).

    Sampling uses the torch generator (component pick + Gaussian draw) so seeding matches
    GaussianSampler; the single-Gaussian case is the K=1 special instance.
    """

    def __init__(self, feats, labels, n_components, cov_type, reg_covar=1e-3):
        self.classes = sorted(set(labels.tolist()))
        self.cov_type = cov_type
        self.d = feats.shape[1]
        self.w, self.mu, self.L, self.scale = {}, {}, {}, {}
        for c in self.classes:
            fc = feats[labels == c].double().numpy()
            gm = GaussianMixture(n_components=n_components, covariance_type=cov_type,
                                 reg_covar=reg_covar, random_state=0, max_iter=300,
                                 n_init=1).fit(fc)
            self.w[c] = torch.tensor(gm.weights_, dtype=torch.float32)
            self.mu[c] = torch.tensor(gm.means_, dtype=torch.float32)
            K = gm.means_.shape[0]
            if cov_type == "diag":
                self.scale[c] = torch.tensor(np.sqrt(gm.covariances_ + 1e-12),
                                             dtype=torch.float32)
            elif cov_type == "tied":
                Lk = np.linalg.cholesky(gm.covariances_ + 1e-6 * np.eye(self.d))
                self.L[c] = torch.tensor(np.repeat(Lk[None], K, axis=0), dtype=torch.float32)
            else:
                Ls = np.stack([np.linalg.cholesky(gm.covariances_[k] + 1e-6 * np.eye(self.d))
                               for k in range(K)])
                self.L[c] = torch.tensor(Ls, dtype=torch.float32)

    def sample(self, n_per_class, generator):
        xs, ys = [], []
        for c in self.classes:
            comp = torch.multinomial(self.w[c], n_per_class, replacement=True,
                                     generator=generator)
            z = torch.randn(n_per_class, self.d, generator=generator)
            mu = self.mu[c][comp]
            if self.cov_type == "diag":
                x = mu + z * self.scale[c][comp]
            else:
                x = mu + torch.einsum("nde,ne->nd", self.L[c][comp], z)
            xs.append(x)
            ys.append(torch.full((n_per_class,), c, dtype=torch.long))
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=generator)
        return x[perm], y[perm]


class RealSampler:
    """Real no-aug features, reshuffled each epoch (surrogate-free ceiling anchor)."""

    def __init__(self, feats, labels):
        self.feats, self.labels = feats, labels

    def sample(self, n_per_class, generator):
        order = torch.randperm(len(self.labels), generator=generator)
        return self.feats[order], self.labels[order]


def make_branch(head, fuse):
    if head == "linear":
        return GraftBranch("gap", gap_arch=("linear" if fuse else "ccvr")).to(DEV)
    return GraftBranch("gap", gap_arch="mlp").to(DEV)


def evaluate(core, branch, loader, fuse):
    branch.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEV), y.to(DEV)
            feat, core_logits = trunk_forward(core, x, "gap")
            logits = (core_logits + branch(feat)) if fuse else branch(feat)
            correct += (logits.argmax(1) == y).sum().item()
            total += y.size(0)
    return correct / total


def train_arm(core, sampler, hold, testld, seed, epochs, lr, fuse, head):
    branch = make_branch(head, fuse)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.SGD(branch.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.CrossEntropyLoss()
    best_ho, best_state = -1.0, None
    for ep in range(epochs):
        branch.train()
        ep_x, ep_y = sampler.sample(450, gen)
        for i in range(0, len(ep_y) - 127, 128):
            x = ep_x[i:i + 128].view(-1, GAP_DIM).to(DEV)
            y = ep_y[i:i + 128].to(DEV)
            if fuse:
                with torch.no_grad():
                    core_logits = core_head_forward(core, x, "gap")
                logits = core_logits + branch(x)
            else:
                logits = branch(x)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
        ho = evaluate(core, branch, hold, fuse)
        if ho > best_ho:
            best_ho = ho
            best_state = {k: v.detach().cpu().clone() for k, v in branch.state_dict().items()}
    branch.load_state_dict(best_state)
    return best_ho, evaluate(core, branch, testld, fuse)


def multimodality(feats, labels, kmax=8):
    opt_ks = []
    for c in sorted(set(labels.tolist())):
        fc = feats[labels == c].double().numpy()
        bics = [GaussianMixture(k, covariance_type="diag", reg_covar=1e-4, random_state=0,
                                max_iter=100).fit(fc).bic(fc) for k in range(1, kmax + 1)]
        opt_ks.append(int(np.argmin(bics) + 1))
    opt_ks = np.array(opt_ks)
    return float((opt_ks > 1).mean()), float(opt_ks.mean())


SURROGATES = [
    ("gauss_lw",     lambda f, l: GaussianSampler(f, l, full_cov=True)),
    ("gmm_tied_K1",  lambda f, l: GMMSampler(f, l, 1, "tied")),
    ("gmm_tied_K3",  lambda f, l: GMMSampler(f, l, 3, "tied")),
    ("gmm_tied_K5",  lambda f, l: GMMSampler(f, l, 5, "tied")),
    ("gmm_tied_K10", lambda f, l: GMMSampler(f, l, 10, "tied")),
    ("gmm_full_K3",  lambda f, l: GMMSampler(f, l, 3, "full")),
    ("gmm_diag_K10", lambda f, l: GMMSampler(f, l, 10, "diag")),
]
HEADS = ["linear", "mlp"]
ARMS = [("fab", True), ("ccvr", False)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-ckpt", default=str(PROJECT_ROOT / "results" / "cifar100_headroom" /
                                               "graft" / "homo_small_s42_finalmodel.pt"))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    core = build_core(args.core_ckpt, recal=True)
    testld = test_loader()

    core.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in testld:
            x, y = x.to(DEV), y.to(DEV)
            _, cl = trunk_forward(core, x, "gap")
            correct += (cl.argmax(1) == y).sum().item(); total += y.size(0)
    floor = correct / total
    print(f"ANCHOR seed={args.seed} kind=core_floor test={floor:.4f}", flush=True)

    base = tv.datasets.CIFAR100(ROOT, train=True, download=True)
    rng = np.random.default_rng(42)
    perm = rng.permutation(len(base.targets))
    tr_idx, ho_idx = perm[:45000].tolist(), perm[45000:].tolist()
    ho_ds = tv.datasets.CIFAR100(ROOT, train=True, transform=TF_TEST)
    hold = DataLoader(Subset(ho_ds, ho_idx), batch_size=256, shuffle=False, num_workers=2)

    feats, labels = extract_features(core, tr_idx, "gap")
    fk, mk = multimodality(feats, labels)
    print(f"DIAG seed={args.seed} multimodality frac_classes_optK_gt1={fk:.3f} mean_optK={mk:.2f}",
          flush=True)

    real = RealSampler(feats, labels)
    for head in HEADS:
        ho, t = train_arm(core, real, hold, testld, args.seed, args.epochs, args.lr, True, head)
        print(f"ANCHOR seed={args.seed} kind=realnoaug_{head}_fab test={t:.4f} holdout={ho:.4f}",
              flush=True)

    for sname, build in SURROGATES:
        sampler = build(feats, labels)
        print(f"[seed {args.seed}] surrogate {sname} built", flush=True)
        for head in HEADS:
            for arm, fuse in ARMS:
                ho, t = train_arm(core, sampler, hold, testld, args.seed,
                                  args.epochs, args.lr, fuse, head)
                print(f"RESULT seed={args.seed} surrogate={sname} head={head} arm={arm} "
                      f"test={t:.4f} holdout={ho:.4f}", flush=True)

    print(f"GMM_LADDER_DONE seed={args.seed}", flush=True)


if __name__ == "__main__":
    main()
