"""FjORD local training: Ordered Dropout with self-distillation.

FjORD (Horvath et al., NeurIPS 2021) differs from HeteroFL in the LOCAL
training loop, not the server. Two ingredients:

1. Ordered Dropout (OD): each round a client samples, per mini-batch, a width
   ``p`` uniformly from the nested widths it is capable of (all configured
   ``model_rates`` no larger than its own ``max_rate``) and trains the
   contiguous first-``p`` sub-network. Low-index channels are shared by every
   width, so they carry the most-reused knowledge.
2. Self-distillation (the "+KD" variant): the client's own max-width sub-model
   is the teacher for the sampled smaller sub-model within the same step. The
   teacher is trained on hard labels; the student is trained on the teacher's
   soft predictions (temperature 1, teacher detached).

This implementation keeps the FLoRE / HeteroFL backbone (``HeteroPreResNet``
with sBN + Scaler) unchanged so the server aggregation and the core-extraction
toxicity metric remain identical to HeteroFL. The only variable is the local
loop. The nested width-``p`` forward reuses the exact width-``p`` module built by
``create_hetero_model`` (so Scaler and sBN behave identically to a HeteroFL
width-``p`` client); its parameters are synced from the master's contiguous
prefix each step, and its gradients are scattered back into that prefix so a
single optimizer updates the shared weights. This deviates from the upstream
FjORD baseline, which uses per-width BatchNorm and no Scaler on a post-act
ResNet-18; see the strategy docstring.
"""

import contextlib
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from .hetero_models import create_hetero_model
from .hetero_utils import extract_state_dict_for_model


def _autocast(device: torch.device, dtype: Optional[torch.dtype]):
    """Autocast context for the forward/loss region; nullcontext when dtype is None."""
    if dtype is None:
        return contextlib.nullcontext()
    device_type = device.type if isinstance(device, torch.device) else str(device).split(":")[0]
    return torch.autocast(device_type, dtype=dtype)


def ordered_dropout_candidate_rates(
    model_rates: List[float], max_rate: float
) -> List[float]:
    """Nested widths a client can train: unique configured rates <= max_rate.

    Returned sorted ascending so the last entry is the client's own width.
    """
    tol = 1e-9
    cands = sorted({float(r) for r in model_rates if float(r) <= max_rate + tol})
    if not cands:
        cands = [float(max_rate)]
    return cands


def _soft_cross_entropy(student_logits: torch.Tensor, soft_target: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of student logits against a (detached) probability target."""
    log_prob = F.log_softmax(student_logits, dim=1)
    return -(soft_target * log_prob).sum(dim=1).mean()


def _scatter_grad_prefix(master: nn.Module, sub: nn.Module) -> None:
    """Add sub-model parameter grads into the master's contiguous prefix.

    Mirrors the first-k width-slice convention used by extraction/aggregation:
    a sub parameter of shape ``s`` maps to ``master_param[:s0, :s1, ...]``.
    """
    master_params = dict(master.named_parameters())
    for name, sub_param in sub.named_parameters():
        if sub_param.grad is None:
            continue
        master_param = master_params[name]
        if master_param.grad is None:
            master_param.grad = torch.zeros_like(master_param)
        slices = tuple(slice(0, s) for s in sub_param.shape)
        master_param.grad[slices] += sub_param.grad


def _sync_sub_from_master(master: nn.Module, sub: nn.Module) -> None:
    """Copy the master's contiguous prefix (params + buffers) into the sub-model and drop stale grads."""
    sub_state = extract_state_dict_for_model(master.state_dict(), sub.state_dict())
    sub.load_state_dict(sub_state, strict=True)
    sub.zero_grad(set_to_none=True)


def train_fjord(
    net: nn.Module,
    trainloader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    model_name: str,
    num_classes: int,
    norm_type: str,
    candidate_rates: List[float],
    max_rate: float,
    kd: bool = True,
    seed: Optional[int] = None,
    sub_model_cache: Optional[Dict[float, nn.Module]] = None,
    autocast_dtype: Optional[torch.dtype] = None,
) -> None:
    """Train ``net`` (the client's max-width model) with Ordered Dropout + KD.

    ``net`` is a ``HeteroPreResNet`` at width ``max_rate``. Per mini-batch a
    width ``p`` in ``candidate_rates`` is sampled uniformly. The width-``p``
    sub-network is executed as a shared-parameter view of ``net``; gradients are
    accumulated into ``net`` and a single SGD optimizer updates it.
    """
    rng = np.random.default_rng(seed)
    optimizer = SGD(net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)

    # Pre-build one sub-model per strictly-smaller candidate width (reused across
    # batches; synced from the master each step). The max-width path uses `net`.
    # A caller-provided cache reuses sub-model instances across fit calls; their
    # parameters and buffers are fully overwritten by _sync_sub_from_master
    # before every use, so caching changes execution only, not semantics.
    sub_models: Dict[float, nn.Module] = sub_model_cache if sub_model_cache is not None else {}
    for rate in candidate_rates:
        if rate < max_rate - 1e-9 and rate not in sub_models:
            sub = create_hetero_model(
                model_name, in_channels=3, num_classes=num_classes,
                model_rate=rate, norm_type=norm_type,
            ).to(device)
            sub_models[rate] = sub
    for sub in sub_models.values():
        sub.train()

    criterion = nn.CrossEntropyLoss()
    mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
    net.train()
    for _ in range(epochs):
        for data, target in trainloader:
            if mark_step is not None:
                mark_step()
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)

            p = float(rng.choice(candidate_rates))
            is_full = p >= max_rate - 1e-9

            if kd:
                # Teacher: client's max-width model on hard labels. The soft
                # target is materialized BEFORE backward: under compiled
                # cudagraphs the backward replay overwrites the forward's
                # output buffer, so a later read of teacher_logits is invalid.
                with _autocast(device, autocast_dtype):
                    teacher_logits = net(data)
                    teacher_loss = criterion(teacher_logits, target)
                soft_target = F.softmax(teacher_logits.detach().float(), dim=1)
                teacher_loss.backward()

                # Student: sampled width on the teacher's soft predictions.
                if is_full:
                    with _autocast(device, autocast_dtype):
                        student_logits = net(data)
                        student_loss = _soft_cross_entropy(student_logits, soft_target)
                    student_loss.backward()
                else:
                    sub = sub_models[p]
                    _sync_sub_from_master(net, sub)
                    with _autocast(device, autocast_dtype):
                        student_logits = sub(data)
                        student_loss = _soft_cross_entropy(student_logits, soft_target)
                    student_loss.backward()
                    _scatter_grad_prefix(net, sub)
            else:
                # No distillation: a single hard-label pass at the sampled width.
                if is_full:
                    with _autocast(device, autocast_dtype):
                        logits = net(data)
                        loss = criterion(logits, target)
                    loss.backward()
                else:
                    sub = sub_models[p]
                    _sync_sub_from_master(net, sub)
                    with _autocast(device, autocast_dtype):
                        logits = sub(data)
                        loss = criterion(logits, target)
                    loss.backward()
                    _scatter_grad_prefix(net, sub)

            optimizer.step()
