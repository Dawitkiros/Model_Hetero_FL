"""FIARSE strategy: importance-aware submodel extraction (Wu et al., NeurIPS 2024).

Structured-channel adaptation: per-layer top-k output channels ranked by
filter L2 norm of the current global model, recomputed every round, with
index-mapped distribution/aggregation reused from FedRolexMappedStrategy.
See probe/FIARSE_NOTES.md for the fidelity assessment.
"""

from collections import OrderedDict
from logging import INFO
from typing import Callable, List

import numpy as np
import torch
from flwr.common.logger import log

from .fedrolex_mapped_strategy import FedRolexMappedStrategy


class FiarseStrategy(FedRolexMappedStrategy):
    """Server strategy with per-round importance-ranked channel extraction."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        log(INFO, "FIARSE initialized (channel importance = filter L2 norm, recomputed each round)")

    @staticmethod
    def _channel_importance(weight: torch.Tensor) -> torch.Tensor:
        return weight.detach().float().reshape(weight.size(0), -1).norm(dim=1)

    def _split_model(self, client_ids: List[str]) -> List[OrderedDict]:
        idx_i = [None for _ in range(len(client_ids))]
        idx = [OrderedDict() for _ in range(len(client_ids))]

        for k, v in self.global_parameters.items():
            parameter_type = k.split(".")[-1]
            importance = None
            if parameter_type == "weight" and v.dim() > 1 and ("conv1" in k or "conv2" in k):
                importance = self._channel_importance(v)
            for m, client_id in enumerate(client_ids):
                if "weight" in parameter_type or "bias" in parameter_type:
                    if parameter_type == "weight":
                        if v.dim() > 1:
                            input_size = v.size(1)
                            output_size = v.size(0)
                            if "conv1" in k or "conv2" in k:
                                if idx_i[m] is None:
                                    idx_i[m] = torch.arange(input_size, device=v.device)
                                input_idx = idx_i[m]
                                scaler_rate = self._get_client_model_rate(client_id) / self.global_model_rate
                                local_output_size = int(np.ceil(output_size * scaler_rate))
                                local_output_size = max(1, min(local_output_size, output_size))
                                output_idx = importance.topk(local_output_size).indices.sort().values
                                idx_i[m] = output_idx
                            elif "shortcut" in k:
                                conv1_key = self._shortcut_conv1_key(k)
                                if conv1_key in idx[m]:
                                    input_idx = idx[m][conv1_key][1]
                                else:
                                    input_idx = idx_i[m]
                                output_idx = idx_i[m]
                            elif "linear" in k:
                                input_idx = idx_i[m]
                                output_idx = torch.arange(output_size, device=v.device)
                            else:
                                raise ValueError(f"Unhandled parameter name: {k}")
                            idx[m][k] = (output_idx, input_idx)
                        else:
                            idx[m][k] = idx_i[m]
                    else:
                        input_size = v.size(0)
                        if "linear" in k:
                            idx[m][k] = torch.arange(input_size, device=v.device)
                        else:
                            idx[m][k] = idx_i[m]
                else:
                    if v.dim() == 1 and ("running_mean" in k or "running_var" in k):
                        idx[m][k] = idx_i[m]
        return idx


def create_fiarse_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    **kwargs,
) -> FiarseStrategy:
    """Create FIARSE strategy."""
    return FiarseStrategy(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        num_classes=num_classes,
        num_clients=num_clients,
        **kwargs,
    )
