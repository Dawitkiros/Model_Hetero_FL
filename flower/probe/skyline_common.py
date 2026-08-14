"""Shared helpers for the oracle skyline arms (A/B) on CIFAR-100.

- reconstruct client partitions exactly as the FL stack does (Dirichlet, seed)
- identify the 5 full-capacity clients (partitions 0-4 at p_full=0.10)
- build a rate-1.0 HeteroPreResNet18, embed a frozen rate-0.25 core prefix,
  build per-parameter freeze masks (freeze the [:core_out,:core_in] block of
  every layer + core BN affine + classifier core columns)
- BN recalibration + global-test eval with seen/unseen-class breakdown
"""
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from flore.hetero_models import HeteroPreResNet18, create_hetero_model, calibrate_bn_stats

ROOT = os.environ.get("FLORE_DATA_ROOT", str(PROJECT_ROOT / "data"))
MEAN, STD = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
TF_TRAIN = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(),
                      T.ToTensor(), T.Normalize(MEAN, STD)])
TF_TEST = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def dirichlet_partition(targets, num_clients=50, alpha=0.1, seed=42):
    """Mirror flore.dataset.partition_dirichlet exactly."""
    targets = np.array(targets)
    num_classes = len(np.unique(targets))
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(num_clients)]
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        rng.shuffle(idx_k)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        for i, idx in enumerate(np.split(idx_k, proportions)):
            client_indices[i].extend(idx.tolist())
    return client_indices


def full_client_indices(num_clients=50, alpha=0.1, seed=42, n_full=5):
    """Pooled train indices of the n_full full-capacity clients (partitions 0..n_full-1),
    plus per-class counts over those pooled indices."""
    base = tv.datasets.CIFAR100(ROOT, train=True, download=True)
    parts = dirichlet_partition(base.targets, num_clients, alpha, seed)
    pooled = [i for p in range(n_full) for i in parts[p]]
    counts = np.bincount(np.array(base.targets)[pooled], minlength=100)
    return pooled, counts, parts


def all_client_indices(num_clients=50, alpha=0.1, seed=42):
    base = tv.datasets.CIFAR100(ROOT, train=True, download=True)
    parts = dirichlet_partition(base.targets, num_clients, alpha, seed)
    pooled = [i for p in range(num_clients) for i in parts[p]]
    counts = np.bincount(np.array(base.targets)[pooled], minlength=100)
    return pooled, counts, parts


def loader_from_indices(indices, train=True, batch=128, workers=4):
    ds = tv.datasets.CIFAR100(ROOT, train=True, download=False,
                              transform=TF_TRAIN if train else TF_TEST)
    return DataLoader(Subset(ds, indices), batch_size=batch, shuffle=train,
                      num_workers=workers, drop_last=False)


def test_loader(batch=256):
    ds = tv.datasets.CIFAR100(ROOT, train=False, download=True, transform=TF_TEST)
    return DataLoader(ds, batch_size=batch, shuffle=False, num_workers=4)


def calib_loader(n=5000, batch=256):
    ds = tv.datasets.CIFAR100(ROOT, train=True, download=False, transform=TF_TEST)
    return DataLoader(Subset(ds, list(range(n))), batch_size=batch, shuffle=False, num_workers=2)


def core_slices(full_param, core_param):
    return tuple(slice(0, s) for s in core_param.shape)


def embed_core_and_masks(full_model, core_state):
    """Lower-triangular oracle embedding. Place rate-0.25 core into the full model and
    return freeze masks (1.0=trainable shell, 0.0=frozen core). The CORE PATH is exactly
    preserved: core output rows are frozen entirely and their shell-input columns are
    zeroed (core sees only core inputs). Only shell output filters / shell BN / the
    classifier's shell columns are trainable."""
    full_state = full_model.state_dict()
    masks = {}
    for name, fp in full_state.items():
        if name not in core_state:
            masks[name] = torch.ones_like(fp, dtype=torch.float32)          # purely shell
            continue
        cs = core_state[name].to(fp.dtype)
        if cs.shape == fp.shape:
            fp.copy_(cs)
            masks[name] = torch.zeros_like(fp, dtype=torch.float32)          # frozen (e.g. linear bias)
            continue
        m = torch.ones_like(fp, dtype=torch.float32)
        if fp.dim() == 4:                 # conv [out, in, kh, kw]: out- and in-scaled
            co, ci = cs.shape[0], cs.shape[1]
            fp[:co, :ci] = cs             # embed core block
            fp[:co, ci:] = 0.0            # lower-triangular: core rows ignore shell inputs
            m[:co, :] = 0.0               # freeze ALL core output rows
        elif fp.dim() == 2:               # linear [out=classes(fixed), in]: in-scaled only
            ci = cs.shape[1]
            fp[:, :ci] = cs               # embed core feature columns
            m[:, :ci] = 0.0               # freeze core feature columns; train shell columns
        elif fp.dim() == 1:               # BN affine [C]: channel-scaled
            c = cs.shape[0]
            fp[:c] = cs
            m[:c] = 0.0
        masks[name] = m
    full_model.load_state_dict(full_state)
    return masks


@torch.no_grad()
def eval_seen_unseen(model, testld, seen_mask, recal_loader):
    calibrate_bn_stats(model, recal_loader, DEV, num_batches=20)
    model.eval()
    nc = 100
    correct = np.zeros(nc); total = np.zeros(nc)
    for x, y in testld:
        x = x.to(DEV); pred = model(x).argmax(1).cpu().numpy()
        yy = y.numpy()
        for c in range(nc):
            m = yy == c
            total[c] += m.sum(); correct[c] += (pred[m] == c).sum()
    acc = correct.sum() / total.sum()
    seen = seen_mask.astype(bool)
    acc_seen = correct[seen].sum() / max(total[seen].sum(), 1)
    acc_unseen = correct[~seen].sum() / max(total[~seen].sum(), 1)
    return acc, acc_seen, acc_unseen, int(seen.sum())
