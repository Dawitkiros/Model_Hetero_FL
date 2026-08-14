"""DepthFL: depth-heterogeneous FL with self-distillation.

DepthFL (Kim et al., ICLR 2023) partitions clients by network DEPTH: a client
holds the stem plus the first ``n_blocks`` stages of a ResNet-18, each with its
own early-exit classifier. Local training applies cross-entropy to every held
exit plus all-pairs self-distillation among the exits (each exit is a student
distilled against every other exit, teacher detached, temperature 1).

This port keeps the mechanism that matters for the "toxic depth" question -
depth heterogeneity + self-distillation - and, per the chosen configuration,
uses plain FedAvg as the base optimizer rather than the paper's FedDyn (which
requires per-client control-variate state that is awkward in flwr simulation).
Cross-entropy and KD are otherwise faithful to the upstream baseline.
"""

import math
from collections import OrderedDict
from logging import INFO
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flwr.common.logger import log
from flwr.common.typing import NDArrays, Scalar
from torch.optim import SGD
from torch.utils.data import DataLoader

from .depthfl_models import KLLoss, create_depthfl_model, test_depthfl


# ---------------------------------------------------------------------------
# Depth assignment (mirrors hetero_utils.get_model_rate_for_partition)
# ---------------------------------------------------------------------------

DEFAULT_DEPTH_LEVELS = [4, 1]
DEFAULT_DEPTH_DISTRIBUTION = [0.5, 0.5]


def parse_depth_config(
    depth_levels: Optional[object] = None,
    depth_distribution: Optional[object] = None,
) -> Tuple[List[int], List[float]]:
    """Parse depth levels / distribution from TOML lists or CSV strings."""
    def _to_list(value, default, cast):
        if value is None:
            return list(default)
        if isinstance(value, str):
            parts = [p.strip() for p in value.split(",") if p.strip()]
            return [cast(p) for p in parts] if parts else list(default)
        return [cast(v) for v in value]

    levels = _to_list(depth_levels, DEFAULT_DEPTH_LEVELS, int)
    dist = _to_list(depth_distribution, DEFAULT_DEPTH_DISTRIBUTION, float)
    if len(dist) != len(levels):
        raise ValueError(
            f"depth_distribution length ({len(dist)}) must match depth_levels length ({len(levels)})"
        )
    total = float(sum(dist))
    if total <= 0:
        raise ValueError("depth_distribution must sum to a positive value")
    dist = [d / total for d in dist]
    return levels, dist


def get_depth_for_partition(
    partition_id: int,
    num_clients: int,
    depth_levels: List[int],
    depth_distribution: List[float],
) -> int:
    """Deterministically map a partition id to a depth level."""
    levels, dist = parse_depth_config(depth_levels, depth_distribution)
    if partition_id < 0:
        return levels[-1]
    cumulative, total = [], 0.0
    for ratio in dist:
        total += ratio
        cumulative.append(total)
    normalized = partition_id / num_clients
    for i, threshold in enumerate(cumulative):
        if normalized < threshold:
            return levels[i]
    return levels[-1]


def get_all_client_depths(
    num_clients: int, depth_levels: List[int], depth_distribution: List[float]
) -> Dict[int, int]:
    """Depth level for every partition id."""
    return {
        pid: get_depth_for_partition(pid, num_clients, depth_levels, depth_distribution)
        for pid in range(num_clients)
    }


def consistency_weight(training_round: int, rampup_rounds: int = 300) -> float:
    """Gaussian ramp-up for the self-distillation weight (0 -> 1 over rampup)."""
    if rampup_rounds <= 0:
        return 1.0
    current = float(min(max(training_round, 0), rampup_rounds))
    phase = 1.0 - current / rampup_rounds
    return float(math.exp(-5.0 * phase * phase))


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def _autocast(device: torch.device, dtype) -> object:
    """Autocast context for the forward/loss region; nullcontext when dtype is None."""
    import contextlib
    if dtype is None:
        return contextlib.nullcontext()
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    return torch.autocast(device_type, dtype=dtype)


def train_depthfl(
    net: nn.Module,
    trainloader: DataLoader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    kd: bool,
    cw: float,
    extended: bool = True,
    autocast_dtype=None,
) -> None:
    """Local training: per-held-exit CE + all-pairs self-distillation.

    ``extended=False`` reproduces the InclusiveFL ablation (train only the
    deepest held exit, no KD). ``cw`` is the current self-distillation weight.
    """
    criterion = nn.CrossEntropyLoss()
    criterion_kl = KLLoss().to(device)
    optimizer = SGD(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with _autocast(device, autocast_dtype):
                output_lst = net(images)
                k = len(output_lst)
                loss = torch.zeros(1, device=device)
                for i, branch_output in enumerate(output_lst):
                    if not extended and i != k - 1:
                        continue
                    loss = loss + criterion(branch_output, labels)
                    if kd and k > 1:
                        for j, other in enumerate(output_lst):
                            if j == i:
                                continue
                            loss = loss + cw * criterion_kl(branch_output, other.detach()) / (k - 1)
            loss.backward()
            optimizer.step()


# ---------------------------------------------------------------------------
# Server-side evaluation (toxic-depth metric)
# ---------------------------------------------------------------------------

def gen_depthfl_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
    global_depth: int,
    core_depth: int,
    start_time: Optional[float] = None,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Evaluate the full-depth global model: ensemble acc vs shallow-exit acc.

    ``acc_full`` = ensemble (mean-logit) accuracy of the full-depth model.
    ``acc_core`` = accuracy of the ``core_depth`` exit (what a shallow client
    deploys). ``delta_acc = acc_full - acc_core`` is the toxic-depth analog of
    the width toxic-shell metric.
    """
    num_classes = int(run_config.model.num_classes)
    norm_type = str(run_config.get("norm_type", "bn")).lower()
    if start_time is None:
        import time
        start_time = time.time()

    def evaluate(server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]):
        net = create_depthfl_model(global_depth, num_classes=num_classes, norm_type=norm_type)
        state = OrderedDict()
        for (k, ref), v in zip(net.state_dict().items(), parameters_ndarrays):
            if k.endswith("num_batches_tracked"):
                state[k] = torch.tensor(v, dtype=torch.long)
            else:
                state[k] = torch.tensor(v, dtype=torch.float32)
        net.load_state_dict(state, strict=True)
        net.to(device)

        loss_full, acc_full, acc_single = test_depthfl(net, testloader, device)
        core_idx = max(0, min(int(core_depth) - 1, len(acc_single) - 1))
        acc_core = acc_single[core_idx]
        delta_acc = acc_full - acc_core

        metrics: Dict[str, Scalar] = {
            "accuracy": acc_full,
            "acc_full": acc_full,
            "acc_core": acc_core,
            "delta_acc": delta_acc,
        }
        for i, a in enumerate(acc_single):
            metrics[f"acc_exit{i + 1}"] = a

        if run_config.get("wandb_enabled", False):
            try:
                import time as _time
                import wandb
                wall = _time.time() - start_time
                payload = {
                    "test/accuracy": acc_full,
                    "test/acc_core": acc_core,
                    "test/delta_acc": delta_acc,
                    "server_round": server_round,
                    "training_round": max(0, server_round - 1),
                    "wall_clock_sec": wall,
                }
                for i, a in enumerate(acc_single):
                    payload[f"test/acc_exit{i + 1}"] = a
                wandb.log(payload, step=max(0, server_round - 1))
            except ImportError:
                pass

        log(INFO, f"[DepthFL] round={server_round} acc_full(ensemble)={acc_full:.4f} "
                  f"acc_core(exit{core_depth})={acc_core:.4f} delta={delta_acc:.4f} "
                  f"per_exit={[round(a, 4) for a in acc_single]}")
        return loss_full, metrics

    return evaluate
