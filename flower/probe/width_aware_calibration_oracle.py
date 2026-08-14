#!/usr/bin/env python3
"""Central oracle for width-aware feature calibration.

This is the first implementation step for the mixed-width calibration ladder. It
does not implement federated upload or missing-width EM. It unifies the existing
core/full-width CCVR and GMM feature-space probes so we can test whether richer
calibration is promising before building a protocol.

Example:
  python probe/width_aware_calibration_oracle.py \
    --checkpoint results/fullwidth_ccvr/homo_large_s42_finalmodel.pt \
    --model-rate 1.0 \
    --surrogates gauss_lw,gmm_tied_K5,gmm_full_K3 \
    --head-seeds 42 43 44
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from sklearn.covariance import LedoitWolf
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe.fab_gmm_ladder import GMMSampler  # noqa: E402
from probe.g1_graft_ceiling import trunk_forward  # noqa: E402
from probe.g2_surrogate import GaussianSampler  # noqa: E402
from probe.skyline_common import DEV, ROOT, TF_TEST  # noqa: E402
from flore.hetero_models import ScalerV2, calibrate_bn_stats, create_hetero_model  # noqa: E402
from flore.hetero_utils import create_extractor, get_model_rate_for_partition, parse_rate_config  # noqa: E402
from flore.strategies.fiarse_strategy import create_fiarse_strategy  # noqa: E402
from flore.strategies.fedrolex_mapped_strategy import create_fedrolex_mapped_strategy  # noqa: E402

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py<3.11 fallback
    import tomli as tomllib


NUM_CLASSES = 100
SPLIT_SEED = 42

# Dataset-aware eval: default CIFAR-100, switched to CIFAR-10 via configure_dataset().
_CIFAR10_TF = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
])
_DATASET_CLS = tv.datasets.CIFAR100
_DATASET_TF = TF_TEST


_FEMNIST_CACHE = "data/femnist_cache.pt"
_FEMNIST_TF = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize((0.737869, 0.737869, 0.737869), (0.431342, 0.431342, 0.431342)),
])


class FemnistDataset:
    """flwrlabs/femnist adapted to the torchvision Dataset interface used by the probe loaders.
    28x28 grayscale -> pad to 32, replicate to 3 channels (matches fastsim GpuFemnist).
    train=True exposes a fixed stratified 50k subset (for CCVR calibration); train=False the full test."""
    _cache = None

    def __init__(self, root=None, train=True, transform=None, download=False):
        if FemnistDataset._cache is None:
            FemnistDataset._cache = torch.load(_FEMNIST_CACHE, map_location="cpu", weights_only=False)
        c = FemnistDataset._cache
        self.transform = transform
        if train:
            x, y = c["train_x"].numpy(), c["train_y"].numpy()
            rng = np.random.default_rng(1234)
            sel = []
            for cl in range(62):
                idx = np.where(y == cl)[0]
                sel.append(rng.choice(idx, size=min(len(idx), 810), replace=False))
            sel = np.concatenate(sel)
            self.data, self._targets = x[sel], y[sel]
        else:
            self.data, self._targets = c["test_x"].numpy(), c["test_y"].numpy()
        self.targets = self._targets.tolist()

    def __len__(self):
        return len(self._targets)

    def __getitem__(self, i):
        img = np.pad(self.data[i], 2, mode="constant")          # 28x28 -> 32x32
        img = np.stack([img, img, img], axis=-1)                # HWC 3ch uint8
        img = self.transform(img) if self.transform is not None else torch.from_numpy(img)
        return img, int(self._targets[i])


def configure_dataset(dataset_name: str, num_classes: int) -> None:
    """Point the eval data pipeline at the checkpoint's dataset. Call before any eval."""
    global NUM_CLASSES, _DATASET_CLS, _DATASET_TF
    NUM_CLASSES = int(num_classes)
    if str(dataset_name).lower() == "cifar10":
        _DATASET_CLS = tv.datasets.CIFAR10
        _DATASET_TF = _CIFAR10_TF
    elif str(dataset_name).lower() == "femnist":
        _DATASET_CLS = FemnistDataset
        _DATASET_TF = _FEMNIST_TF
    else:
        _DATASET_CLS = tv.datasets.CIFAR100
        _DATASET_TF = TF_TEST


class RealFeatureDataset:
    def __init__(self, features: torch.Tensor, labels: torch.Tensor):
        self.features = features.float()
        self.labels = labels.long()

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.labels[index]


class ClasswiseGaussianSampler:
    """Per-class Gaussian with a class-level sampler API."""

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        full_cov: bool,
        classes: Iterable[int],
        fallback_dim: Optional[int] = None,
    ):
        self.classes = [int(c) for c in classes]
        self.d = int(features.shape[1]) if features.numel() else int(fallback_dim or 0)
        self.full_cov = bool(full_cov)
        self.mu: Dict[int, torch.Tensor] = {}
        self.scale: Dict[int, torch.Tensor] = {}
        self.has_data: Dict[int, bool] = {}

        for c in self.classes:
            fc = features[labels == c].double().numpy()
            if len(fc) == 0:
                self.mu[c] = torch.zeros(self.d, dtype=torch.float32)
                self.scale[c] = torch.eye(self.d, dtype=torch.float32) if self.full_cov else torch.zeros(self.d)
                self.has_data[c] = False
                continue

            self.mu[c] = torch.tensor(fc.mean(axis=0), dtype=torch.float32)
            self.has_data[c] = True
            if self.full_cov and len(fc) >= 2:
                cov = LedoitWolf().fit(fc).covariance_
                chol = np.linalg.cholesky(cov + 1e-6 * np.eye(cov.shape[0]))
                self.scale[c] = torch.tensor(chol, dtype=torch.float32)
            elif self.full_cov:
                self.scale[c] = torch.zeros((self.d, self.d), dtype=torch.float32)
            else:
                self.scale[c] = torch.tensor(fc.std(axis=0) + 1e-6, dtype=torch.float32)

    def sample_class(self, c: int, n: int, generator: torch.Generator) -> torch.Tensor:
        z = torch.randn(n, self.d, generator=generator)
        if not self.has_data.get(int(c), False):
            return torch.zeros(n, self.d, dtype=torch.float32)
        if self.full_cov:
            return self.mu[int(c)] + z @ self.scale[int(c)].T
        return self.mu[int(c)] + z * self.scale[int(c)]

    def sample(self, n_per_class: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for c in self.classes:
            xs.append(self.sample_class(c, n_per_class, generator))
            ys.append(torch.full((n_per_class,), int(c), dtype=torch.long))
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=generator)
        return x[perm], y[perm]


class ClasswiseDiagGMMSampler:
    """Robust per-class diagonal GMM sampler.

    Classes with too little wider-client data fall back to deterministic means, and
    classes with no wider-client data emit zeros. That is intentional for the
    mixed-width oracle: the core channel remains all-client, while missing shell
    coordinates should not hallucinate class evidence.
    """

    def __init__(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        n_components: int,
        classes: Iterable[int],
        fallback_dim: Optional[int] = None,
        reg_covar: float = 1e-3,
    ):
        self.classes = [int(c) for c in classes]
        self.d = int(features.shape[1]) if features.numel() else int(fallback_dim or 0)
        self.n_components = int(n_components)
        self.w: Dict[int, torch.Tensor] = {}
        self.mu: Dict[int, torch.Tensor] = {}
        self.scale: Dict[int, torch.Tensor] = {}
        self.has_data: Dict[int, bool] = {}

        for c in self.classes:
            fc = features[labels == c].double().numpy()
            if len(fc) == 0:
                self.w[c] = torch.ones(1, dtype=torch.float32)
                self.mu[c] = torch.zeros(1, self.d, dtype=torch.float32)
                self.scale[c] = torch.zeros(1, self.d, dtype=torch.float32)
                self.has_data[c] = False
                continue
            if len(fc) == 1:
                self.w[c] = torch.ones(1, dtype=torch.float32)
                self.mu[c] = torch.tensor(fc[None, 0], dtype=torch.float32)
                self.scale[c] = torch.zeros(1, self.d, dtype=torch.float32)
                self.has_data[c] = True
                continue

            k_eff = max(1, min(self.n_components, len(fc)))
            gm = GaussianMixture(
                n_components=k_eff,
                covariance_type="diag",
                reg_covar=reg_covar,
                random_state=0,
                max_iter=300,
                n_init=1,
            ).fit(fc)
            self.w[c] = torch.tensor(gm.weights_, dtype=torch.float32)
            self.mu[c] = torch.tensor(gm.means_, dtype=torch.float32)
            self.scale[c] = torch.tensor(np.sqrt(gm.covariances_ + 1e-12), dtype=torch.float32)
            self.has_data[c] = True

    def sample_class(self, c: int, n: int, generator: torch.Generator) -> torch.Tensor:
        c = int(c)
        if not self.has_data.get(c, False):
            return torch.zeros(n, self.d, dtype=torch.float32)
        comp = torch.multinomial(self.w[c], n, replacement=True, generator=generator)
        z = torch.randn(n, self.d, generator=generator)
        return self.mu[c][comp] + z * self.scale[c][comp]

    def sample(self, n_per_class: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for c in self.classes:
            xs.append(self.sample_class(c, n_per_class, generator))
            ys.append(torch.full((n_per_class,), int(c), dtype=torch.long))
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=generator)
        return x[perm], y[perm]


class FactorizedMixedWidthSampler:
    """Concatenate all-client core samples with eligible-only wider-coordinate samples."""

    def __init__(self, core_sampler, extra_sampler):
        self.core_sampler = core_sampler
        self.extra_sampler = extra_sampler
        self.classes = list(core_sampler.classes)

    def sample(self, n_per_class: int, generator: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for c in self.classes:
            core = self.core_sampler.sample_class(c, n_per_class, generator)
            extra = self.extra_sampler.sample_class(c, n_per_class, generator)
            xs.append(torch.cat([core, extra], dim=1))
            ys.append(torch.full((n_per_class,), int(c), dtype=torch.long))
        x, y = torch.cat(xs), torch.cat(ys)
        perm = torch.randperm(len(y), generator=generator)
        return x[perm], y[perm]


def parse_surrogates(raw: str) -> List[str]:
    surrogates = [item.strip() for item in raw.split(",") if item.strip()]
    if not surrogates:
        raise ValueError("At least one surrogate is required")
    return surrogates


def infer_fl_seed(checkpoint: Path) -> int:
    match = re.search(r"(?:^|[_-])s(\d+)(?:[_-]|\.|$)", checkpoint.name)
    return int(match.group(1)) if match else -1


def _norm_keys(obj):
    if isinstance(obj, dict):
        return {str(k).replace("-", "_"): _norm_keys(v) for k, v in obj.items()}
    return obj


def read_toml(path: Path) -> dict:
    with path.open("rb") as handle:
        return _norm_keys(tomllib.load(handle))


def cfg_get(cfg: dict, path: str, default=None):
    cur = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def make_split() -> Tuple[List[int], List[int]]:
    dataset = _DATASET_CLS(ROOT, train=True, download=True)
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(dataset.targets))
    return permutation[:45000].tolist(), permutation[45000:].tolist()


def make_holdout_loader(indices: Iterable[int], batch_size: int, workers: int) -> DataLoader:
    dataset = _DATASET_CLS(ROOT, train=True, transform=_DATASET_TF)
    return DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def make_test_loader(batch_size: int, workers: int) -> DataLoader:
    dataset = _DATASET_CLS(ROOT, train=False, download=True, transform=_DATASET_TF)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers)


def make_calib_loader(
    batch_size: int,
    workers: int,
    n_samples: int = 5000,
    indices: Optional[Iterable[int]] = None,
) -> DataLoader:
    dataset = _DATASET_CLS(ROOT, train=True, download=False, transform=_DATASET_TF)
    if indices is None:
        selected = list(range(n_samples))
    else:
        selected = list(indices)[:n_samples]
    return DataLoader(
        Subset(dataset, selected),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )


def load_frozen_model_local(
    checkpoint: Path,
    model_rate: float,
    bn_batches: int,
    batch_size: int,
    workers: int,
    calib_indices: Optional[Iterable[int]] = None,
) -> nn.Module:
    model = create_hetero_model(
        "HeteroPreResNet18",
        in_channels=3,
        num_classes=NUM_CLASSES,
        model_rate=model_rate,
        norm_type="sbn",
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    model.load_state_dict(state, strict=True)
    model.to(DEV)

    if model_rate < 1.0:
        for module in model.modules():
            if isinstance(module, ScalerV2):
                module.always_scale = True

    calibrate_bn_stats(
        model,
        make_calib_loader(batch_size=batch_size, workers=workers, indices=calib_indices),
        DEV,
        num_batches=bn_batches,
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def set_scaler_eval_mode(model: nn.Module, force: bool) -> None:
    for module in model.modules():
        if isinstance(module, ScalerV2):
            module.always_scale = force


def _find_partition_for_rate(cfg: dict, target_rate: float) -> int:
    num_clients = int(cfg_get(cfg, "num_clients", 50))
    rates, dist = parse_rate_config(
        cfg_get(cfg, "model_rates", None),
        cfg_get(cfg, "rate_distribution", None),
    )
    for partition_id in range(num_clients):
        rate = get_model_rate_for_partition(partition_id, num_clients, rates, dist)
        if abs(float(rate) - float(target_rate)) < 1e-12:
            return partition_id
    raise ValueError(f"No partition maps to model_rate={target_rate}")


def _strategy_core_state(
    full: nn.Module,
    core: nn.Module,
    cfg: dict,
    extractor: str,
) -> OrderedDict[str, torch.Tensor]:
    """Extract a method-specific small model from a trained full model.

    HeteroFL's static-prefix core can use `create_extractor("static")`.
    FIARSE needs its importance-ranked channel map; using the static prefix gives
    random-like accuracy and is not a valid FIARSE small-tier baseline.
    """
    model_name = cfg_get(cfg, "model.name", "HeteroPreResNet18")
    if model_name != "HeteroPreResNet18" and extractor != "static":
        raise ValueError(f"Method-specific extraction is only implemented for HeteroPreResNet18, got {model_name}")

    num_classes = int(cfg_get(cfg, "model.num_classes", NUM_CLASSES))
    core_rate = float(cfg_get(cfg, "core_model_rate", 0.25))
    rnd_seed = int(cfg_get(cfg, "rnd_seed", cfg_get(cfg, "dataset.seed", 42)))

    if extractor == "static":
        static_extractor = create_extractor(
            strategy="static",
            global_model=full,
            num_classes=num_classes,
            seed=rnd_seed,
        )
        state = static_extractor.extract_sub_model(
            client_model=core,
            client_idx=0,
            round_num=int(cfg_get(cfg, "num_server_rounds", 300)),
        )
        return OrderedDict((key, value.detach().cpu()) for key, value in state.items())

    if extractor == "fiarse":
        num_clients = int(cfg_get(cfg, "num_clients", 50))
        strategy = create_fiarse_strategy(
            evaluate_fn=None,
            num_clients=num_clients,
            clients_per_round=2,
            num_classes=num_classes,
            model_rates=cfg_get(cfg, "model_rates", None),
            rate_distribution=cfg_get(cfg, "rate_distribution", None),
            core_model_rate=core_rate,
            norm_type=cfg_get(cfg, "norm_type", "sbn"),
        )
        full_cpu_state = OrderedDict(
            (key, value.detach().cpu()) for key, value in full.state_dict().items()
        )
        strategy.global_model.load_state_dict(full_cpu_state, strict=True)
        strategy.global_parameters = strategy.global_model.state_dict()
        full_pid = _find_partition_for_rate(cfg, 1.0)
        core_pid = _find_partition_for_rate(cfg, core_rate)
        strategy._get_client_model_rate("c_full", full_pid)
        strategy._get_client_model_rate("c_core", core_pid)
        param_idx = strategy._split_model(["c_full", "c_core"])
        local_params = strategy._distribute(["c_full", "c_core"], param_idx)
        return OrderedDict((key, value.detach().cpu()) for key, value in local_params[1].items())

    if extractor == "fedrolex":
        num_clients = int(cfg_get(cfg, "num_clients", 50))
        overlap_cfg = cfg_get(cfg, "overlap", 0.0)
        overlap = None if (overlap_cfg is None or float(overlap_cfg) < 0) else float(overlap_cfg)
        strategy = create_fedrolex_mapped_strategy(
            evaluate_fn=None,
            num_clients=num_clients,
            clients_per_round=2,
            num_classes=num_classes,
            overlap=overlap,
            global_model_rate=float(cfg_get(cfg, "global_model_rate", 1.0)),
            model_split_mode=cfg_get(cfg, "model_split_mode", "fix"),
            reshuffle_rounds=int(cfg_get(cfg, "reshuffle_rounds", 512)),
            weighting=cfg_get(cfg, "weighting", "avg"),
            model_rates=cfg_get(cfg, "model_rates", None),
            rate_distribution=cfg_get(cfg, "rate_distribution", None),
            core_model_rate=core_rate,
            norm_type=cfg_get(cfg, "norm_type", "sbn"),
        )
        full_cpu_state = OrderedDict(
            (key, value.detach().cpu()) for key, value in full.state_dict().items()
        )
        strategy.global_model.load_state_dict(full_cpu_state, strict=True)
        strategy.global_parameters = strategy.global_model.state_dict()
        # Deterministic FedRolex core = the rate-0.25 rolled window a core client
        # receives at the final training round (overlap<0 -> canonical roll =
        # rounds % output_size). Mirrors the FIARSE branch above.
        strategy.rounds = max(int(cfg_get(cfg, "num_server_rounds", 300)) - 1, 0)
        full_pid = _find_partition_for_rate(cfg, 1.0)
        core_pid = _find_partition_for_rate(cfg, core_rate)
        strategy._get_client_model_rate("c_full", full_pid)
        strategy._get_client_model_rate("c_core", core_pid)
        param_idx = strategy._split_model(["c_full", "c_core"])
        local_params = strategy._distribute(["c_full", "c_core"], param_idx)
        return OrderedDict((key, value.detach().cpu()) for key, value in local_params[1].items())

    raise ValueError(f"Unknown core extractor {extractor!r}")


def infer_core_extractor(args: argparse.Namespace, cfg: dict) -> str:
    requested = str(args.core_extractor).lower()
    if requested != "auto":
        return requested
    algorithm = str(cfg_get(cfg, "algorithm.name", "heterofl")).lower()
    if algorithm == "fiarse":
        return "fiarse"
    if algorithm in {"fedrolex", "fedrolex_mapped"}:
        return "fedrolex"
    return "static"


def load_hetero_full_and_core(
    checkpoint: Path,
    config_path: Path,
    bn_batches: int,
    batch_size: int,
    workers: int,
    core_extractor: str = "static",
    calib_indices: Optional[Iterable[int]] = None,
) -> Tuple[nn.Module, nn.Module, dict]:
    cfg = read_toml(config_path)
    model_name = cfg_get(cfg, "model.name", "HeteroPreResNet18")
    num_classes = int(cfg_get(cfg, "model.num_classes", NUM_CLASSES))
    norm_type = cfg_get(cfg, "norm_type", "sbn")
    core_rate = float(cfg_get(cfg, "core_model_rate", 0.25))
    rnd_seed = int(cfg_get(cfg, "rnd_seed", cfg_get(cfg, "dataset.seed", 42)))

    full = create_hetero_model(
        model_name,
        in_channels=3,
        num_classes=num_classes,
        model_rate=1.0,
        norm_type=norm_type,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state" in state:
        state = state["state"]
    full.load_state_dict(state, strict=True)
    full.to(DEV)

    core = create_hetero_model(
        model_name,
        in_channels=3,
        num_classes=num_classes,
        model_rate=core_rate,
        norm_type=norm_type,
    )
    core_state = _strategy_core_state(full, core, cfg, core_extractor)
    core.load_state_dict(
        OrderedDict((key, value.detach().cpu()) for key, value in core_state.items()),
        strict=True,
    )
    core.to(DEV)

    calib_loader = make_calib_loader(batch_size=batch_size, workers=workers, indices=calib_indices)
    calibrate_bn_stats(full, calib_loader, DEV, num_batches=bn_batches)
    calibrate_bn_stats(core, calib_loader, DEV, num_batches=bn_batches)
    set_scaler_eval_mode(core, bool(cfg_get(cfg, "core_force_scaler_eval", True)))
    full.eval()
    core.eval()
    for model in (full, core):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    return full, core, cfg


def dirichlet_partition(targets: Iterable[int], num_clients: int, alpha: float, seed: int) -> List[List[int]]:
    targets = np.asarray(list(targets))
    num_classes = len(np.unique(targets))
    rng = np.random.default_rng(seed)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in range(num_classes):
        idx_cls = np.where(targets == cls)[0]
        rng.shuffle(idx_cls)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        split_points = (np.cumsum(proportions) * len(idx_cls)).astype(int)[:-1]
        for client_id, idx in enumerate(np.split(idx_cls, split_points)):
            client_indices[client_id].extend(idx.tolist())
    return client_indices


def build_train_index_rates(cfg: dict) -> np.ndarray:
    dataset = _DATASET_CLS(ROOT, train=True, download=True)
    num_clients = int(cfg_get(cfg, "num_clients", 50))
    rates, dist = parse_rate_config(
        cfg_get(cfg, "model_rates", None),
        cfg_get(cfg, "rate_distribution", None),
    )
    partitions = dirichlet_partition(
        dataset.targets,
        num_clients=num_clients,
        alpha=float(cfg_get(cfg, "dataset.alpha", 0.1)),
        seed=int(cfg_get(cfg, "dataset.seed", cfg_get(cfg, "rnd_seed", 42))),
    )
    index_rates = np.zeros(len(dataset.targets), dtype=np.float32)
    for client_id, indices in enumerate(partitions):
        rate = get_model_rate_for_partition(client_id, num_clients, rates, dist)
        index_rates[np.asarray(indices, dtype=np.int64)] = float(rate)
    return index_rates


@torch.no_grad()
def extract_eval_features(
    model: nn.Module,
    indices: Iterable[int],
    batch_size: int,
    workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = _DATASET_CLS(ROOT, train=True, transform=_DATASET_TF)
    loader = DataLoader(
        Subset(dataset, list(indices)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
    )
    features, labels = [], []
    for images, target in loader:
        feat, _ = trunk_forward(model, images.to(DEV), "gap")
        features.append(feat.flatten(1).cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def extract_test_features(
    model: nn.Module,
    batch_size: int,
    workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = make_test_loader(batch_size=batch_size, workers=workers)
    features, labels = [], []
    for images, target in loader:
        feat, _ = trunk_forward(model, images.to(DEV), "gap")
        features.append(feat.flatten(1).cpu())
        labels.append(target)
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def original_head_accuracy(model: nn.Module, batch_size: int) -> float:
    correct = total = 0
    for images, labels in make_test_loader(batch_size=batch_size, workers=0):
        images, labels = images.to(DEV), labels.to(DEV)
        _, logits = trunk_forward(model, images, "gap")
        correct += (logits.argmax(1) == labels).sum().item()
        total += labels.numel()
    return correct / total


def build_sampler(name: str, features: torch.Tensor, labels: torch.Tensor):
    if name == "gauss_lw":
        return GaussianSampler(features, labels, full_cov=True)
    if name == "gauss_diag":
        return GaussianSampler(features, labels, full_cov=False)

    match = re.fullmatch(r"gmm_(tied|full|diag)_K(\d+)", name)
    if not match:
        raise ValueError(
            f"Unknown surrogate {name!r}; expected gauss_lw, gauss_diag, "
            "or gmm_{tied,full,diag}_K<int>"
        )
    cov_type, n_components = match.group(1), int(match.group(2))
    if cov_type == "diag":
        return ClasswiseDiagGMMSampler(
            features,
            labels,
            n_components=n_components,
            classes=list(range(NUM_CLASSES)),
            fallback_dim=int(features.shape[1]),
        )
    return GMMSampler(features, labels, n_components=n_components, cov_type=cov_type)


def build_mixed_sampler(name: str, context: Dict[str, torch.Tensor | int]):
    core_features = context["core_features"]
    core_labels = context["core_labels"]
    extra_features = context["extra_features"]
    extra_labels = context["extra_labels"]
    extra_dim = int(context["extra_dim"])
    classes = list(range(NUM_CLASSES))

    if name == "gauss_lw":
        core_sampler = ClasswiseGaussianSampler(core_features, core_labels, full_cov=True, classes=classes)
        # Wider coordinates are sparse under p_full=0.10; diagonal shrinkage is the
        # conservative mixed-width version of full-covariance CCVR.
        extra_sampler = ClasswiseGaussianSampler(
            extra_features,
            extra_labels,
            full_cov=False,
            classes=classes,
            fallback_dim=extra_dim,
        )
        return FactorizedMixedWidthSampler(core_sampler, extra_sampler)
    if name == "gauss_diag":
        core_sampler = ClasswiseGaussianSampler(core_features, core_labels, full_cov=False, classes=classes)
        extra_sampler = ClasswiseGaussianSampler(
            extra_features,
            extra_labels,
            full_cov=False,
            classes=classes,
            fallback_dim=extra_dim,
        )
        return FactorizedMixedWidthSampler(core_sampler, extra_sampler)

    match = re.fullmatch(r"gmm_diag_K(\d+)", name)
    if not match:
        raise ValueError(
            f"Mixed-width views support gauss_lw, gauss_diag, and gmm_diag_K<int>; got {name!r}"
        )
    n_components = int(match.group(1))
    core_sampler = ClasswiseDiagGMMSampler(core_features, core_labels, n_components, classes=classes)
    extra_sampler = ClasswiseDiagGMMSampler(
        extra_features,
        extra_labels,
        n_components,
        classes=classes,
        fallback_dim=extra_dim,
    )
    return FactorizedMixedWidthSampler(core_sampler, extra_sampler)


def feature_dataset(features: torch.Tensor, labels: torch.Tensor, batch_size: int) -> DataLoader:
    return DataLoader(
        RealFeatureDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
    )


def class_count_summary(labels: torch.Tensor) -> Dict[str, int | float]:
    counts = torch.bincount(labels.long(), minlength=NUM_CLASSES).cpu().numpy()
    nonzero = counts[counts > 0]
    return {
        "classes_seen": int((counts > 0).sum()),
        "min_class_count": int(nonzero.min()) if len(nonzero) else 0,
        "median_class_count": float(np.median(nonzero)) if len(nonzero) else 0.0,
        "max_class_count": int(nonzero.max()) if len(nonzero) else 0,
    }


def add_baseline_deltas(rows: List[Dict[str, object]]) -> None:
    baselines: Dict[Tuple[str, int], Dict[str, object]] = {}
    for row in rows:
        if row.get("view") == "coremodel" and row.get("surrogate") == "gauss_lw":
            baselines[(str(row.get("checkpoint")), int(row.get("head_seed")))] = row
    for row in rows:
        baseline = baselines.get((str(row.get("checkpoint")), int(row.get("head_seed"))))
        if baseline is None:
            continue
        row["delta_vs_core_gauss_acc"] = float(row["test_acc"]) - float(baseline["test_acc"])
        row["delta_vs_core_gauss_nll"] = float(row["test_nll"]) - float(baseline["test_nll"])
        row["delta_vs_core_gauss_ece"] = float(row["test_ece"]) - float(baseline["test_ece"])


def parse_view_specs(raw: str, core_dim: int, full_dim: int) -> List[Dict[str, object]]:
    specs = []
    for item in [part.strip() for part in raw.split(",") if part.strip()]:
        parts = item.split(":")
        name = parts[0]
        if name == "coremodel":
            k = core_dim if len(parts) == 1 or parts[1] in {"all", ""} else int(parts[1])
            specs.append({"view": "coremodel", "source": "core", "prefix_k": k, "eligible": "all"})
        elif name == "fullcore":
            k = core_dim if len(parts) == 1 or parts[1] in {"all", ""} else int(parts[1])
            specs.append({"view": "fullcore", "source": "full", "prefix_k": k, "eligible": "all"})
        elif name == "full":
            k = full_dim if len(parts) == 1 or parts[1] in {"all", ""} else int(parts[1])
            eligible = parts[2] if len(parts) > 2 else "all"
            specs.append({"view": "full", "source": "full", "prefix_k": k, "eligible": eligible})
        elif name == "prefix":
            if len(parts) < 2:
                raise ValueError("prefix view requires prefix:<k>[:all|eligible]")
            specs.append({
                "view": f"prefix{parts[1]}",
                "source": "full",
                "prefix_k": int(parts[1]),
                "eligible": parts[2] if len(parts) > 2 else "eligible",
            })
        elif name == "mixed":
            if len(parts) < 2:
                raise ValueError("mixed view requires mixed:<k>[:eligible]")
            specs.append({
                "view": f"mixed{parts[1]}",
                "source": "mixed",
                "prefix_k": int(parts[1]),
                "eligible": parts[2] if len(parts) > 2 else "eligible",
            })
        else:
            raise ValueError(f"Unknown view spec {item!r}")
    return specs


@torch.no_grad()
def classifier_metrics(classifier: nn.Module, loader: DataLoader) -> Dict[str, float]:
    classifier.eval()
    correct = total = 0
    nll_sum = 0.0
    confidences: List[torch.Tensor] = []
    correctness: List[torch.Tensor] = []

    for features, labels in loader:
        features = features.to(DEV)
        labels = labels.to(DEV)
        logits = classifier(features)
        probs = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        correct_batch = pred.eq(labels)
        correct += correct_batch.sum().item()
        total += labels.numel()
        nll_sum += F.cross_entropy(logits, labels, reduction="sum").item()
        confidences.append(conf.cpu())
        correctness.append(correct_batch.float().cpu())

    conf_all = torch.cat(confidences)
    corr_all = torch.cat(correctness)
    ece = expected_calibration_error(conf_all, corr_all)
    return {
        "acc": correct / total,
        "nll": nll_sum / total,
        "ece": ece,
    }


def expected_calibration_error(
    confidences: torch.Tensor,
    correctness: torch.Tensor,
    bins: int = 15,
) -> float:
    ece = torch.tensor(0.0)
    edges = torch.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            weight = mask.float().mean()
            ece += weight * (confidences[mask].mean() - correctness[mask].mean()).abs()
    return float(ece)


def train_classifier(
    sampler,
    feature_dim: int,
    holdout_loader: DataLoader,
    test_loader_features: DataLoader,
    head_seed: int,
    epochs: int,
    lr: float,
    samples_per_class: int,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    torch.manual_seed(head_seed)
    classifier = nn.Linear(feature_dim, NUM_CLASSES).to(DEV)
    nn.init.zeros_(classifier.weight)
    nn.init.zeros_(classifier.bias)

    generator = torch.Generator().manual_seed(head_seed)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_holdout: Dict[str, float] | None = None
    best_state: Dict[str, torch.Tensor] | None = None
    best_epoch = -1

    for epoch in range(epochs):
        classifier.train()
        epoch_features, epoch_labels = sampler.sample(samples_per_class, generator)
        for start in range(0, len(epoch_labels) - 127, 128):
            features = epoch_features[start : start + 128].view(-1, feature_dim).to(DEV)
            labels = epoch_labels[start : start + 128].to(DEV)
            loss = criterion(classifier(features), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        holdout = classifier_metrics(classifier, holdout_loader)
        if best_holdout is None or holdout["acc"] > best_holdout["acc"]:
            best_holdout = holdout
            best_epoch = epoch + 1
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in classifier.state_dict().items()
            }

    if best_state is None or best_holdout is None:
        raise RuntimeError("Training did not produce a selected classifier")
    classifier.load_state_dict(best_state)
    return best_holdout, classifier_metrics(classifier, test_loader_features), best_epoch


def save_outputs(result_dir: Path, args: argparse.Namespace, rows: List[Dict[str, object]]) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "config.json").open("w") as handle:
        json.dump({key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, handle, indent=2)
    with (result_dir / "summary.json").open("w") as handle:
        json.dump(rows, handle, indent=2)
    if rows:
        columns: List[str] = []
        for row in rows:
            for key in row:
                if key not in columns:
                    columns.append(key)
        with (result_dir / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)


def summarize(rows: List[Dict[str, object]]) -> None:
    by_surrogate: Dict[Tuple[str, str], List[float]] = {}
    for row in rows:
        key = (str(row.get("view", "single")), str(row["surrogate"]))
        by_surrogate.setdefault(key, []).append(float(row["test_acc"]))
    for (view, surrogate), values in by_surrogate.items():
        arr = np.asarray(values, dtype=np.float64)
        std = arr.std(ddof=1) if len(arr) > 1 else 0.0
        print(
            f"SUMMARY view={view} surrogate={surrogate} test_mean={arr.mean():.4f} "
            f"test_std={std:.4f} n={len(arr)}",
            flush=True,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Central oracle for width-aware calibration")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model-rate", type=float, default=None)
    parser.add_argument("--hetero-config", default=None)
    parser.add_argument(
        "--surrogates",
        default="gauss_lw,gmm_tied_K5,gmm_full_K3",
        help="Comma-separated: gauss_lw, gauss_diag, gmm_{tied,full,diag}_K<int>",
    )
    parser.add_argument(
        "--views",
        default="coremodel,fullcore,full:512:all,prefix:192:eligible,prefix:256:eligible,prefix:512:eligible",
        help=(
            "Only with --hetero-config. Comma-separated view specs: coremodel, "
            "fullcore, full[:k[:all|eligible]], prefix:<k>[:all|eligible], "
            "mixed:<k>[:eligible]. Mixed views concatenate all-client extracted-core "
            "statistics with wider-coordinate statistics from eligible clients."
        ),
    )
    parser.add_argument(
        "--core-extractor",
        default="auto",
        choices=["auto", "static", "fiarse", "fedrolex"],
        help=(
            "How to extract the rate-core model from a hetero checkpoint. auto uses "
            "static for HeteroFL/homo, FIARSE importance extraction for FIARSE, and "
            "fails closed for FedRolex because the final checkpoint does not encode "
            "the rolling window state."
        ),
    )
    parser.add_argument("--head-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--samples-per-class", type=int, default=450)
    parser.add_argument("--bn-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--feature-batch", type=int, default=512)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for model evaluation and head training: auto, cpu, cuda, or cuda:<index>.",
    )
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "width_aware_calibration"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested CUDA device {raw!r}, but torch.cuda.is_available() is false")
    return device


def main() -> None:
    global DEV
    args = parse_args()
    DEV = resolve_device(args.device)
    print(f"DEVICE {DEV}", flush=True)
    if args.debug:
        args.epochs = min(args.epochs, 2)
        args.head_seeds = args.head_seeds[:1]
        args.samples_per_class = min(args.samples_per_class, 20)

    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    np.random.seed(SPLIT_SEED)
    torch.manual_seed(SPLIT_SEED)
    fl_seed = infer_fl_seed(checkpoint)
    train_idx, hold_idx = make_split()
    if args.debug:
        train_idx = train_idx[:2000]
        hold_idx = hold_idx[:500]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    result_dir = Path(args.output_root) / f"{timestamp}{tag}"
    rows: List[Dict[str, object]] = []

    if args.hetero_config:
        config_path = Path(args.hetero_config).resolve()
        if not config_path.is_file():
            raise FileNotFoundError(config_path)
        cfg_for_extractor = read_toml(config_path)
        configure_dataset(
            cfg_get(cfg_for_extractor, "dataset.name", "cifar100"),
            int(cfg_get(cfg_for_extractor, "model.num_classes", NUM_CLASSES)),
        )
        core_extractor = infer_core_extractor(args, cfg_for_extractor)
        full_model, core_model, cfg = load_hetero_full_and_core(
            checkpoint,
            config_path,
            args.bn_batches,
            args.batch_size,
            args.workers,
            core_extractor,
            calib_indices=train_idx,
        )
        raw_full = original_head_accuracy(full_model, args.batch_size)
        raw_core = original_head_accuracy(core_model, args.batch_size)
        print(
            f"SETUP checkpoint={checkpoint.name} fl_seed={fl_seed} hetero_config={config_path.name} "
            f"core_extractor={core_extractor} raw_full={raw_full:.4f} raw_core={raw_core:.4f}",
            flush=True,
        )
        print("Extracting full/core train/holdout/test features", flush=True)
        full_train, train_labels = extract_eval_features(full_model, train_idx, args.feature_batch, args.workers)
        full_hold, hold_labels = extract_eval_features(full_model, hold_idx, args.batch_size, args.workers)
        full_test, test_labels = extract_test_features(full_model, args.batch_size, args.workers)
        core_train, core_labels = extract_eval_features(core_model, train_idx, args.feature_batch, args.workers)
        core_hold, _ = extract_eval_features(core_model, hold_idx, args.batch_size, args.workers)
        core_test, _ = extract_test_features(core_model, args.batch_size, args.workers)
        if not torch.equal(train_labels, core_labels):
            raise RuntimeError("Full/core train label order mismatch")

        index_rates = build_train_index_rates(cfg)
        train_rates = index_rates[np.asarray(train_idx, dtype=np.int64)]
        core_dim = int(core_train.shape[1])
        full_dim = int(full_train.shape[1])
        view_specs = parse_view_specs(args.views, core_dim=core_dim, full_dim=full_dim)
        views = []
        for spec in view_specs:
            source = str(spec["source"])
            prefix_k = int(spec["prefix_k"])
            eligible = str(spec["eligible"])
            sampler_context = None
            if source == "core":
                tr, ho, te = core_train[:, :prefix_k], core_hold[:, :prefix_k], core_test[:, :prefix_k]
                mask = np.ones(len(train_labels), dtype=bool)
                view_feature_dim = int(tr.shape[1])
            else:
                if eligible == "eligible":
                    min_rate = min(1.0, prefix_k / full_dim)
                    mask = train_rates + 1e-12 >= min_rate
                elif eligible == "all":
                    mask = np.ones(len(train_labels), dtype=bool)
                else:
                    raise ValueError(f"Unknown eligibility mode {eligible!r}")
                if source == "mixed":
                    if prefix_k <= core_dim:
                        raise ValueError("mixed view requires prefix_k > core feature dimension")
                    tr = torch.empty((len(train_labels), prefix_k), dtype=torch.float32)
                    ho = torch.cat([core_hold, full_hold[:, core_dim:prefix_k]], dim=1)
                    te = torch.cat([core_test, full_test[:, core_dim:prefix_k]], dim=1)
                    sampler_context = {
                        "core_features": core_train,
                        "core_labels": train_labels,
                        "extra_features": full_train[torch.from_numpy(mask), core_dim:prefix_k],
                        "extra_labels": train_labels[torch.from_numpy(mask)],
                        "extra_dim": prefix_k - core_dim,
                    }
                    mask = np.ones(len(train_labels), dtype=bool)
                    view_feature_dim = prefix_k
                else:
                    tr, ho, te = full_train[:, :prefix_k], full_hold[:, :prefix_k], full_test[:, :prefix_k]
                    view_feature_dim = int(tr.shape[1])
            view_labels = train_labels[torch.from_numpy(mask)]
            count_stats = class_count_summary(view_labels)
            if sampler_context is not None:
                extra_stats = class_count_summary(sampler_context["extra_labels"])
                count_stats = {
                    **count_stats,
                    "extra_n_train": int(len(sampler_context["extra_labels"])),
                    "extra_classes_seen": extra_stats["classes_seen"],
                    "extra_min_class_count": extra_stats["min_class_count"],
                    "extra_median_class_count": extra_stats["median_class_count"],
                    "extra_max_class_count": extra_stats["max_class_count"],
                }
            views.append((
                spec,
                tr[torch.from_numpy(mask)],
                view_labels,
                feature_dataset(ho, hold_labels, args.batch_size),
                feature_dataset(te, test_labels, args.batch_size),
                count_stats,
                view_feature_dim,
                sampler_context,
            ))
    else:
        if args.model_rate is None:
            raise ValueError("--model-rate is required unless --hetero-config is provided")
        model = load_frozen_model_local(
            checkpoint,
            args.model_rate,
            args.bn_batches,
            args.batch_size,
            args.workers,
            calib_indices=train_idx,
        )
        raw_acc = original_head_accuracy(model, args.batch_size)
        print(
            f"SETUP checkpoint={checkpoint.name} fl_seed={fl_seed} "
            f"rate={args.model_rate:g} raw_test={raw_acc:.4f}",
            flush=True,
        )
        print("Extracting train/holdout/test features", flush=True)
        train_features, train_labels = extract_eval_features(
            model,
            train_idx,
            batch_size=args.feature_batch,
            workers=args.workers,
        )
        hold_features, hold_labels = extract_eval_features(
            model,
            hold_idx,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        test_features, test_labels = extract_test_features(
            model,
            batch_size=args.batch_size,
            workers=args.workers,
        )
        views = [(
            {"view": "single", "source": "single", "prefix_k": int(train_features.shape[1]), "eligible": "all"},
            train_features,
            train_labels,
            feature_dataset(hold_features, hold_labels, args.batch_size),
            feature_dataset(test_features, test_labels, args.batch_size),
            class_count_summary(train_labels),
            int(train_features.shape[1]),
            None,
        )]
        raw_full = raw_acc
        raw_core = raw_acc

    for spec, train_features, train_labels, hold_loader, test_feature_loader, count_stats, feature_dim, sampler_context in views:
        if len(train_labels) == 0:
            print(f"SKIP view={spec['view']} prefix_k={spec['prefix_k']} no eligible training examples")
            continue
        for surrogate_name in parse_surrogates(args.surrogates):
            print(
                f"Building view={spec['view']} prefix_k={spec['prefix_k']} eligible={spec['eligible']} "
                f"surrogate={surrogate_name} n_train={len(train_labels)}",
                flush=True,
            )
            if sampler_context is None:
                sampler = build_sampler(surrogate_name, train_features, train_labels)
            else:
                sampler = build_mixed_sampler(surrogate_name, sampler_context)
            for head_seed in args.head_seeds:
                holdout, test, best_epoch = train_classifier(
                    sampler=sampler,
                    feature_dim=feature_dim,
                    holdout_loader=hold_loader,
                    test_loader_features=test_feature_loader,
                    head_seed=head_seed,
                    epochs=args.epochs,
                    lr=args.lr,
                    samples_per_class=args.samples_per_class,
                )
                row: Dict[str, object] = {
                    "checkpoint": checkpoint.name,
                    "fl_seed": fl_seed,
                    "model_rate": args.model_rate if args.model_rate is not None else "",
                    "hetero_config": Path(args.hetero_config).name if args.hetero_config else "",
                    "core_extractor": core_extractor if args.hetero_config else "",
                    "view": spec["view"],
                    "source": spec["source"],
                    "prefix_k": spec["prefix_k"],
                    "eligible": spec["eligible"],
                    "feature_dim": feature_dim,
                    "n_train": int(len(train_labels)),
                    "raw_full_test": raw_full,
                    "raw_core_test": raw_core,
                    "surrogate": surrogate_name,
                    "head_seed": head_seed,
                    "best_epoch": best_epoch,
                    "holdout_acc": holdout["acc"],
                    "holdout_nll": holdout["nll"],
                    "holdout_ece": holdout["ece"],
                    "test_acc": test["acc"],
                    "test_nll": test["nll"],
                    "test_ece": test["ece"],
                    **count_stats,
                }
                rows.append(row)
                add_baseline_deltas(rows)
                print(
                    f"RESULT view={spec['view']} prefix_k={spec['prefix_k']} eligible={spec['eligible']} "
                    f"surrogate={surrogate_name} head_seed={head_seed} "
                    f"holdout={holdout['acc']:.4f} test={test['acc']:.4f} "
                    f"nll={test['nll']:.4f} ece={test['ece']:.4f} "
                    f"delta_vs_core_gauss={row.get('delta_vs_core_gauss_acc', '')} "
                    f"best_epoch={best_epoch}",
                    flush=True,
                )
                save_outputs(result_dir, args, rows)

    add_baseline_deltas(rows)
    summarize(rows)
    save_outputs(result_dir, args, rows)
    print(f"WROTE {result_dir}", flush=True)


if __name__ == "__main__":
    main()
