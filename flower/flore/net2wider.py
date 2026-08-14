"""Server-side Net2Wider-style shell repair for model-heterogeneous FL.

**v4 — supports both ResNet and PreResNet (auto-detected).**

Mappings are defined per *activation space* (the channel dimension shared by
adjacent layers and across both branches of a residual addition).

Supports:
  - Standard ResNet (BasicBlock + Bottleneck)
  - PreResNet / Pre-activated ResNet (HeteroPreResNet18 etc.)
  - Auto-detection from state dict keys
  - Both ``downsample.*`` and ``shortcut_conv.*`` naming conventions
  - Identity-skip consistency (reuses input space)

Key difference between ResNet and PreResNet:
  - ResNet:    BN follows conv  →  bn1 = space H,  bn2 = space O
  - PreResNet: BN precedes conv →  bn1 = space I,  bn2 = space H
  - PreResNet has no shortcut BN, no stem BN, but has bn_final
"""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Dict, List, MutableMapping, Optional, Set, Tuple

import numpy as np
import torch

StateDict = "OrderedDict[str, torch.Tensor]"

_BLOCK_RE = re.compile(r"^(layer\d+)\.(\d+)\.(.+)$")


# ═══════════════════════════════════════════════════════════════════════════
# Persistent state
# ═══════════════════════════════════════════════════════════════════════════


def create_shell_repair_state(seed: int = 42) -> Dict[str, object]:
    """Create persistent repair state (call once, reuse across rounds)."""
    return {
        "seed": int(seed),
        "activation_maps": {},
        "space_meta": {},
        "graph_spec": None,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Low-level helpers
# ═══════════════════════════════════════════════════════════════════════════


def _core_prefix_dim(
    full_dim: int, core_rate: float, keep_full: bool = False
) -> int:
    if keep_full:
        return full_dim
    core_dim = int(np.ceil(full_dim * float(core_rate)))
    return max(1, min(core_dim, full_dim))


def _make_fixed_map(
    full_dim: int, core_dim: int, *, seed: int, key: str
) -> torch.Tensor:
    if core_dim >= full_dim:
        return torch.arange(full_dim, dtype=torch.long)
    out = torch.arange(full_dim, dtype=torch.long)
    shell_dim = full_dim - core_dim
    local_seed = int(seed + (abs(hash(key)) % 2_000_000_000))
    rng = np.random.default_rng(local_seed)
    out[core_dim:] = torch.tensor(
        rng.integers(0, core_dim, size=shell_dim, endpoint=False),
        dtype=torch.long,
    )
    return out


def _get_or_create_space_map(
    space_id: str,
    full_dim: int,
    core_dim: int,
    repair_state: MutableMapping[str, object],
) -> torch.Tensor:
    activation_maps: dict = repair_state.setdefault("activation_maps", {})
    space_meta: dict = repair_state.setdefault("space_meta", {})

    if space_id in activation_maps:
        old_full, old_core = space_meta[space_id]
        if old_full != full_dim or old_core != core_dim:
            raise ValueError(
                f"Inconsistent space definition for {space_id!r}: "
                f"previously ({old_full}, {old_core}), now ({full_dim}, {core_dim})."
            )
        return activation_maps[space_id]

    seed = int(repair_state.get("seed", 42))
    map_ = _make_fixed_map(full_dim, core_dim, seed=seed, key=space_id)
    activation_maps[space_id] = map_
    space_meta[space_id] = (full_dim, core_dim)
    return map_


def _space_fan_counts(
    space_map: torch.Tensor, core_dim: int, dtype: torch.dtype, device
) -> torch.Tensor:
    return (
        torch.bincount(space_map, minlength=core_dim)
        .clamp_min(1)
        .to(device=device, dtype=dtype)
    )


def _maybe_add_noise(
    repaired: torch.Tensor,
    shell_mask: torch.Tensor,
    noise_std: float,
    *,
    allow_noise: bool,
) -> None:
    if noise_std <= 0 or not allow_noise or not shell_mask.any():
        return
    noise = torch.randn_like(repaired) * float(noise_std)
    repaired[shell_mask] += noise[shell_mask]


# ═══════════════════════════════════════════════════════════════════════════
# Graph spec: topology analysis
# ═══════════════════════════════════════════════════════════════════════════


def _sorted_block_ids(keys) -> List[str]:
    block_ids: Set[str] = set()
    for k in keys:
        m = _BLOCK_RE.match(k)
        if m is not None:
            block_ids.add(f"{m.group(1)}.{m.group(2)}")

    def sort_key(bid: str) -> Tuple[int, int]:
        layer, block = bid.split(".")
        return (int(layer.replace("layer", "")), int(block))

    return sorted(block_ids, key=sort_key)


def _detect_arch(keys: List[str]) -> str:
    """Detect architecture from state dict keys.

    Returns "preresnet", "resnet_bottleneck", or "resnet_basic".
    """
    has_bn_final = any(k.startswith("bn_final.") for k in keys)
    has_stem_bn = any(
        k.startswith("bn1.") and not k.startswith("layer") for k in keys
    )

    # PreResNet: has bn_final, no stem bn1
    if has_bn_final and not has_stem_bn:
        return "preresnet"

    # Bottleneck: has conv3 in blocks
    if any(_BLOCK_RE.match(k) and ".conv3." in k for k in keys):
        return "resnet_bottleneck"

    return "resnet_basic"


def _build_graph_spec(state_dict: StateDict) -> Dict[str, object]:
    """Analyse the state dict and assign activation spaces.

    Handles ResNet (BasicBlock / Bottleneck) and PreResNet.
    """
    keys = list(state_dict.keys())
    block_ids = _sorted_block_ids(keys)
    arch = _detect_arch(keys)

    # Detect projection skip per block.
    has_proj: Dict[str, bool] = {}
    for bid in block_ids:
        has_proj[bid] = any(
            k.startswith(f"{bid}.downsample.")
            or k.startswith(f"{bid}.shortcut.")
            or k.startswith(f"{bid}.shortcut_conv.")
            for k in keys
        )

    block_input_space: Dict[str, str] = {}
    block_mid_space: Dict[str, str] = {}
    block_mid2_space: Dict[str, str] = {}
    block_output_space: Dict[str, str] = {}

    visible_space = "stem_out"
    for bid in block_ids:
        block_input_space[bid] = visible_space
        block_mid_space[bid] = f"{bid}.mid"

        if arch == "resnet_bottleneck":
            block_mid2_space[bid] = f"{bid}.mid2"

        if has_proj[bid]:
            block_output_space[bid] = f"{bid}.out"
        else:
            block_output_space[bid] = visible_space

        visible_space = block_output_space[bid]

    return {
        "arch": arch,
        "block_ids": block_ids,
        "has_proj": has_proj,
        "block_input_space": block_input_space,
        "block_mid_space": block_mid_space,
        "block_mid2_space": block_mid2_space,
        "block_output_space": block_output_space,
        "final_visible_space": visible_space,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-tensor role classification
# ═══════════════════════════════════════════════════════════════════════════


def _parse_tensor_role(
    key: str,
    tensor: torch.Tensor,
    graph_spec: Dict[str, object],
    core_model_rate: float,
) -> Optional[Dict[str, object]]:
    """Classify a parameter tensor and assign its activation spaces."""
    if tensor.ndim == 0 or key.endswith("num_batches_tracked"):
        return None

    arch: str = graph_spec["arch"]
    is_preresnet = arch == "preresnet"
    is_bottleneck = arch == "resnet_bottleneck"

    # ── Stem conv ─────────────────────────────────────────────────
    if key == "conv1.weight" and tensor.ndim == 4:
        out_full, in_full = tensor.shape[0], tensor.shape[1]
        return {
            "kind": "conv",
            "in_space": "rgb",
            "out_space": "stem_out",
            "in_full": in_full,
            "in_core": _core_prefix_dim(in_full, core_model_rate, keep_full=(in_full <= 3)),
            "out_full": out_full,
            "out_core": _core_prefix_dim(out_full, core_model_rate),
        }

    # ── Stem BN (ResNet only — PreResNet has no stem BN) ──────────
    if (
        key.startswith("bn1.")
        and not key.startswith("layer")
        and tensor.ndim == 1
        and not is_preresnet
    ):
        full = tensor.shape[0]
        return {
            "kind": "norm",
            "space": "stem_out",
            "full": full,
            "core": _core_prefix_dim(full, core_model_rate),
        }

    # ── bn_final (PreResNet only — final pre-activation) ─────────
    if key.startswith("bn_final.") and tensor.ndim == 1 and is_preresnet:
        full = tensor.shape[0]
        return {
            "kind": "norm",
            "space": graph_spec["final_visible_space"],
            "full": full,
            "core": _core_prefix_dim(full, core_model_rate),
        }

    # ── Block layers ──────────────────────────────────────────────
    m = _BLOCK_RE.match(key)
    if m is not None:
        bid = f"{m.group(1)}.{m.group(2)}"
        suffix = m.group(3)

        I = graph_spec["block_input_space"][bid]
        H = graph_spec["block_mid_space"][bid]
        O = graph_spec["block_output_space"][bid]

        # ── conv1 ────────────────────────────────────────────────
        if suffix == "conv1.weight" and tensor.ndim == 4:
            out_full, in_full = tensor.shape[0], tensor.shape[1]
            return {
                "kind": "conv",
                "in_space": I,
                "out_space": H,
                "in_full": in_full,
                "in_core": _core_prefix_dim(in_full, core_model_rate),
                "out_full": out_full,
                "out_core": _core_prefix_dim(out_full, core_model_rate),
            }

        # ── bn1 ──────────────────────────────────────────────────
        if suffix.startswith("bn1.") and tensor.ndim == 1:
            full = tensor.shape[0]
            if is_preresnet:
                # PreResNet: bn1 PRECEDES conv1, normalizes block INPUT
                space = I
            else:
                # ResNet: bn1 FOLLOWS conv1, normalizes conv1 output
                space = H
            return {
                "kind": "norm",
                "space": space,
                "full": full,
                "core": _core_prefix_dim(full, core_model_rate),
            }

        # ── conv2 ────────────────────────────────────────────────
        if suffix == "conv2.weight" and tensor.ndim == 4:
            out_full, in_full = tensor.shape[0], tensor.shape[1]
            if is_bottleneck:
                H2 = graph_spec["block_mid2_space"][bid]
                out_space = H2
            elif is_preresnet:
                # PreResNet BasicBlock: conv2 is last conv → output
                out_space = O
            else:
                # ResNet BasicBlock: conv2 is last conv → output
                out_space = O
            return {
                "kind": "conv",
                "in_space": H,
                "out_space": out_space,
                "in_full": in_full,
                "in_core": _core_prefix_dim(in_full, core_model_rate),
                "out_full": out_full,
                "out_core": _core_prefix_dim(out_full, core_model_rate),
            }

        # ── bn2 ──────────────────────────────────────────────────
        if suffix.startswith("bn2.") and tensor.ndim == 1:
            full = tensor.shape[0]
            if is_preresnet:
                # PreResNet: bn2 PRECEDES conv2, normalizes conv1's output = H
                space = H
            elif is_bottleneck:
                H2 = graph_spec["block_mid2_space"][bid]
                space = H2
            else:
                # ResNet BasicBlock: bn2 FOLLOWS conv2 = O
                space = O
            return {
                "kind": "norm",
                "space": space,
                "full": full,
                "core": _core_prefix_dim(full, core_model_rate),
            }

        # ── conv3 / bn3 (Bottleneck only) ────────────────────────
        if is_bottleneck:
            if suffix == "conv3.weight" and tensor.ndim == 4:
                H2 = graph_spec["block_mid2_space"][bid]
                out_full, in_full = tensor.shape[0], tensor.shape[1]
                return {
                    "kind": "conv",
                    "in_space": H2,
                    "out_space": O,
                    "in_full": in_full,
                    "in_core": _core_prefix_dim(in_full, core_model_rate),
                    "out_full": out_full,
                    "out_core": _core_prefix_dim(out_full, core_model_rate),
                }

            if suffix.startswith("bn3.") and tensor.ndim == 1:
                full = tensor.shape[0]
                return {
                    "kind": "norm",
                    "space": O,
                    "full": full,
                    "core": _core_prefix_dim(full, core_model_rate),
                }

        # ── Projection skip ──────────────────────────────────────
        # ResNet:    downsample.0.weight / shortcut.0.weight (conv)
        #            downsample.1.*     / shortcut.1.*      (BN)
        # PreResNet: shortcut_conv.weight                    (conv, no BN)

        is_proj_conv = (
            (suffix == "shortcut_conv.weight" and tensor.ndim == 4)
            or (
                (suffix.startswith("downsample.0.") or suffix.startswith("shortcut.0."))
                and suffix.endswith("weight")
                and tensor.ndim == 4
            )
        )
        if is_proj_conv:
            out_full, in_full = tensor.shape[0], tensor.shape[1]
            return {
                "kind": "conv",
                "in_space": I,
                "out_space": O,
                "in_full": in_full,
                "in_core": _core_prefix_dim(in_full, core_model_rate),
                "out_full": out_full,
                "out_core": _core_prefix_dim(out_full, core_model_rate),
            }

        # ResNet projection BN (PreResNet has no shortcut BN)
        is_proj_bn = (
            not is_preresnet
            and (suffix.startswith("downsample.1.") or suffix.startswith("shortcut.1."))
            and tensor.ndim == 1
        )
        if is_proj_bn:
            full = tensor.shape[0]
            return {
                "kind": "norm",
                "space": O,
                "full": full,
                "core": _core_prefix_dim(full, core_model_rate),
            }

    # ── Classifier ────────────────────────────────────────────────
    if key in {"linear.weight", "fc.weight"} and tensor.ndim == 2:
        out_full, in_full = tensor.shape[0], tensor.shape[1]
        return {
            "kind": "linear",
            "in_space": graph_spec["final_visible_space"],
            "out_space": "logits",
            "in_full": in_full,
            "in_core": _core_prefix_dim(in_full, core_model_rate),
            "out_full": out_full,
            "out_core": out_full,
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════
# Tensor repair functions
# ═══════════════════════════════════════════════════════════════════════════


def _repair_weight_tensor(
    key: str,
    current: torch.Tensor,
    *,
    in_space: str,
    out_space: str,
    in_full: int,
    in_core: int,
    out_full: int,
    out_core: int,
    repair_state: MutableMapping[str, object],
    noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    repaired = current.detach().clone()
    dev = current.device

    out_map = _get_or_create_space_map(out_space, out_full, out_core, repair_state).to(dev)
    in_map = _get_or_create_space_map(in_space, in_full, in_core, repair_state).to(dev)
    in_fan = _space_fan_counts(in_map, in_core, dtype=current.dtype, device=dev)

    core_source = current[:out_core, :in_core].detach().clone()

    shell_out = slice(out_core, out_full)
    shell_in = slice(in_core, in_full)

    if out_full > out_core:
        repaired[shell_out, :in_core] = core_source[out_map[shell_out], :in_core]

    if in_full > in_core:
        scale = in_fan[in_map[shell_in]]
        if current.ndim == 4:
            scale = scale.view(1, -1, 1, 1)
            repaired[:out_core, shell_in] = core_source[:, in_map[shell_in], :, :] / scale
        else:
            scale = scale.view(1, -1)
            repaired[:out_core, shell_in] = core_source[:, in_map[shell_in]] / scale

    if out_full > out_core and in_full > in_core:
        scale = in_fan[in_map[shell_in]]
        if current.ndim == 4:
            scale = scale.view(1, -1, 1, 1)
            repaired[shell_out, shell_in] = (
                core_source[out_map[shell_out]][:, in_map[shell_in], :, :] / scale
            )
        else:
            scale = scale.view(1, -1)
            repaired[shell_out, shell_in] = (
                core_source[out_map[shell_out]][:, in_map[shell_in]] / scale
            )

    shell_mask = torch.zeros_like(current, dtype=torch.bool)
    shell_mask[out_core:, ...] = True
    if in_core < in_full:
        if current.ndim == 4:
            shell_mask[:, in_core:, :, :] = True
        else:
            shell_mask[:, in_core:] = True

    _maybe_add_noise(repaired, shell_mask, noise_std, allow_noise=True)
    return repaired, shell_mask


def _repair_norm_tensor(
    key: str,
    current: torch.Tensor,
    *,
    space: str,
    full: int,
    core: int,
    repair_state: MutableMapping[str, object],
    noise_std: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    repaired = current.detach().clone()
    if full <= core:
        return repaired, torch.zeros_like(current, dtype=torch.bool)

    space_map = _get_or_create_space_map(space, full, core, repair_state).to(current.device)
    repaired[core:] = current[space_map[core:]]

    shell_mask = torch.zeros_like(current, dtype=torch.bool)
    shell_mask[core:] = True
    _maybe_add_noise(
        repaired, shell_mask, noise_std,
        allow_noise=not key.endswith("running_var"),
    )
    return repaired, shell_mask


# ═══════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════


def build_repaired_state_dict(
    global_state_dict: StateDict,
    repair_state: MutableMapping[str, object],
    *,
    core_model_rate: float,
    noise_std: float = 0.0,
) -> Tuple[StateDict, Dict[str, float]]:
    """Build a repaired state dict with graph-consistent mappings."""
    graph_spec = repair_state.get("graph_spec")
    if graph_spec is None:
        graph_spec = _build_graph_spec(global_state_dict)
        repair_state["graph_spec"] = graph_spec

    repaired: StateDict = OrderedDict()
    shell_delta_sq = 0.0
    num_repaired_tensors = 0

    for key, tensor in global_state_dict.items():
        current = tensor.detach().clone()
        meta = _parse_tensor_role(key, current, graph_spec, core_model_rate)

        if meta is None:
            repaired[key] = current
            continue

        kind = meta["kind"]

        if kind in {"conv", "linear"}:
            repaired_tensor, shell_mask = _repair_weight_tensor(
                key, current,
                in_space=meta["in_space"], out_space=meta["out_space"],
                in_full=meta["in_full"], in_core=meta["in_core"],
                out_full=meta["out_full"], out_core=meta["out_core"],
                repair_state=repair_state, noise_std=noise_std,
            )
        elif kind == "norm":
            repaired_tensor, shell_mask = _repair_norm_tensor(
                key, current,
                space=meta["space"], full=meta["full"], core=meta["core"],
                repair_state=repair_state, noise_std=noise_std,
            )
        else:
            repaired[key] = current
            continue

        repaired[key] = repaired_tensor
        if shell_mask.any():
            delta = (repaired_tensor - current).float()
            shell_delta_sq += float(torch.sum(delta[shell_mask] ** 2).item())
            num_repaired_tensors += 1

    metrics = {
        "shell_repair_shell_l2": float(np.sqrt(shell_delta_sq)),
        "shell_repair_num_tensors": float(num_repaired_tensors),
    }
    return repaired, metrics


def repair_shell_if_stale(
    global_state_dict: StateDict,
    repair_state: MutableMapping[str, object],
    *,
    rounds_since_last_full: float,
    threshold: int,
    beta: float,
    noise_std: float,
    core_model_rate: float,
) -> Tuple[StateDict, Dict[str, float]]:
    """Repair shell tensors when the full model has been stale for too long."""
    metrics: Dict[str, float] = {
        "shell_repair_applied": 0.0,
        "shell_repair_skipped_recent_full": 1.0,
        "shell_repair_threshold": float(threshold),
        "shell_repair_beta": float(beta),
        "shell_repair_noise_std": float(noise_std),
        "shell_repair_shell_l2": 0.0,
        "shell_repair_num_tensors": 0.0,
    }

    if threshold < 0 or rounds_since_last_full < threshold:
        return (
            OrderedDict((k, v.detach().clone()) for k, v in global_state_dict.items()),
            metrics,
        )

    repaired_state, repair_metrics = build_repaired_state_dict(
        global_state_dict, repair_state,
        core_model_rate=core_model_rate, noise_std=noise_std,
    )

    beta = float(beta)
    blended: StateDict = OrderedDict()
    for key, current in global_state_dict.items():
        current_t = current.detach().clone()
        repaired_t = repaired_state[key].to(device=current_t.device, dtype=current_t.dtype)
        if current_t.dtype.is_floating_point:
            blended[key] = beta * current_t + (1.0 - beta) * repaired_t
        else:
            blended[key] = current_t

    metrics.update(repair_metrics)
    metrics["shell_repair_applied"] = 1.0
    metrics["shell_repair_skipped_recent_full"] = 0.0
    return blended, metrics
