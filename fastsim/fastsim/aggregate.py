from collections import OrderedDict

import torch

BN_BUFFER_SUFFIXES = ("num_batches_tracked", "running_mean", "running_var")


def accumulate_slice(aggregated, counts, client_param, num_samples):
    shape = client_param.shape
    if len(shape) == 4:
        aggregated[: shape[0], : shape[1], :, :] += client_param * num_samples
        counts[: shape[0], : shape[1], :, :] += num_samples
    elif len(shape) == 2:
        aggregated[: shape[0], : shape[1]] += client_param * num_samples
        counts[: shape[0], : shape[1]] += num_samples
    elif len(shape) == 1:
        aggregated[: shape[0]] += client_param * num_samples
        counts[: shape[0]] += num_samples
    else:
        aggregated += client_param * num_samples
        counts += num_samples


def aggregate_hetero(global_state, client_states, client_weights):
    aggregated = {}
    counts = {}
    for key, val in global_state.items():
        if key.endswith(BN_BUFFER_SUFFIXES):
            continue
        aggregated[key] = torch.zeros_like(val, dtype=torch.float32)
        counts[key] = torch.zeros_like(val, dtype=torch.float32)
    for state, weight in zip(client_states, client_weights):
        for key, client_param in state.items():
            if key not in aggregated:
                continue
            accumulate_slice(aggregated[key], counts[key], client_param.float(), weight)
    out = {}
    for key, val in global_state.items():
        if key.endswith(BN_BUFFER_SUFFIXES):
            out[key] = val.clone()
        else:
            mask = counts[key] > 0
            merged = val.clone().float()
            merged[mask] = aggregated[key][mask] / counts[key][mask]
            out[key] = merged.to(val.dtype)
    return out


def aggregate_depthfl(global_state, client_states):
    """Positional unweighted count aggregation (DepthFL upstream semantics).

    A depth-``d`` client's state_dict is a strict positional prefix of the
    full-depth global state_dict, so position ``i`` names the same tensor for
    every client deep enough to hold it. Each position is averaged over exactly
    the clients that held it; positions no client held keep the global value.
    """
    keys = list(global_state.keys())
    n = len(keys)
    acc = [torch.zeros_like(global_state[k], dtype=torch.float32) for k in keys]
    counts = [0.0] * n
    for state in client_states:
        for i, arr in enumerate(state.values()):
            if i >= n:
                break
            acc[i] += arr.float()
            counts[i] += 1.0
    out = OrderedDict()
    for i, k in enumerate(keys):
        if counts[i] > 0:
            val = acc[i] / counts[i]
            if k.endswith("num_batches_tracked"):
                out[k] = val.round().to(torch.long)
            else:
                out[k] = val.to(global_state[k].dtype)
        else:
            out[k] = global_state[k].clone()
    return out
