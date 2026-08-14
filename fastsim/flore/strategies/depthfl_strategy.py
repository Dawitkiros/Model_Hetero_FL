"""DepthFL strategy: depth-heterogeneous distribution and aggregation.

Server side, DepthFL distributes a depth-``d`` prefix of the full (depth-4)
global model to each client and aggregates heterogeneous depths by parameter
POSITION: a shallow client's parameter list is a strict prefix of the deep
model's list, so position ``i`` denotes the same tensor for every client deep
enough to hold it. Each position is averaged (unweighted, per the upstream
DepthFL baseline) over exactly the clients that hold it; positions no client
held keep the previous global value.

The round-1 capability-discovery handshake mirrors the width HeteroStrategy so
the shared server/evaluate conventions (training_round = server_round - 1, LR
schedule offset) line up.
"""

import hashlib
from collections import OrderedDict
from logging import INFO, WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..depthfl import get_depth_for_partition, parse_depth_config
from ..depthfl_models import create_depthfl_model, num_state_entries


class DepthFLStrategy(FedAvg):
    """Server strategy for depth-heterogeneous federated learning."""

    def __init__(
        self,
        *,
        fraction_fit: float = 0.0,
        fraction_evaluate: float = 0.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 0,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        num_classes: int = 100,
        num_clients: int = 100,
        norm_type: str = "bn",
        depth_levels: Optional[object] = None,
        depth_distribution: Optional[object] = None,
        global_depth: Optional[int] = None,
        core_depth: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            evaluate_fn=evaluate_fn,
            on_fit_config_fn=on_fit_config_fn,
            on_evaluate_config_fn=on_evaluate_config_fn,
            accept_failures=accept_failures,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.num_classes = num_classes
        self.num_clients = num_clients
        self.norm_type = norm_type
        self.depth_levels, self.depth_distribution = parse_depth_config(depth_levels, depth_distribution)
        self.global_depth = int(global_depth if global_depth is not None else max(self.depth_levels))
        self.core_depth = int(core_depth if core_depth is not None else min(self.depth_levels))
        self.seed = seed
        self.current_round = 0

        self.global_model = create_depthfl_model(self.global_depth, self.num_classes, self.norm_type)
        self.global_keys = list(self.global_model.state_dict().keys())
        # Precompute the number of state_dict tensors held by each depth level.
        self.depth_entries: Dict[int, int] = {
            d: num_state_entries(d, self.num_classes, self.norm_type)
            for d in sorted(set(self.depth_levels))
        }
        self.client_to_partition: Dict[str, int] = {}
        self.next_fallback_index = 0
        log(INFO, f"DepthFL initialized: depths={self.depth_levels} dist={self.depth_distribution} "
                  f"global_depth={self.global_depth} core_depth={self.core_depth} "
                  f"entries_per_depth={self.depth_entries}")

    def _get_client_depth(self, client_id: str, partition_id: Optional[int] = None) -> int:
        if partition_id is not None:
            self.client_to_partition[client_id] = int(partition_id)
            return get_depth_for_partition(int(partition_id), self.num_clients,
                                           self.depth_levels, self.depth_distribution)
        if client_id in self.client_to_partition:
            return get_depth_for_partition(self.client_to_partition[client_id], self.num_clients,
                                           self.depth_levels, self.depth_distribution)
        idx = self.next_fallback_index % self.num_clients
        self.next_fallback_index += 1
        self.client_to_partition[client_id] = idx
        log(WARNING, f"No partition_id for client {client_id[:16]}..., fallback index {idx}")
        return get_depth_for_partition(idx, self.num_clients, self.depth_levels, self.depth_distribution)

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager
                      ) -> List[Tuple[ClientProxy, FitIns]]:
        self.current_round = server_round
        global_ndarrays = parameters_to_ndarrays(parameters)

        # Keep the server-side global model in sync for aggregation/return.
        state = OrderedDict()
        for (k, _), v in zip(self.global_model.state_dict().items(), global_ndarrays):
            if k.endswith("num_batches_tracked"):
                state[k] = torch.tensor(v, dtype=torch.long)
            else:
                state[k] = torch.tensor(v, dtype=torch.float32)
        self.global_model.load_state_dict(state, strict=True)

        if server_round == 1:
            log(INFO, "Round 1: DepthFL CAPABILITY DISCOVERY - sampling ALL clients")
            num_available = client_manager.num_available()
            clients = client_manager.sample(num_clients=num_available, min_num_clients=num_available)
        else:
            sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
            clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)

        fit_configurations = []
        for client in clients:
            client_id = client.cid
            config: Dict[str, Scalar] = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            config["current_round"] = server_round

            if server_round == 1:
                # Discovery: send the full model; clients only report capabilities.
                sub_ndarrays = global_ndarrays
                config["depth"] = self.global_depth
            else:
                depth = self._get_client_depth(client_id)
                k_d = self.depth_entries[depth]
                sub_ndarrays = global_ndarrays[:k_d]
                config["depth"] = depth

            fit_ins = FitIns(ndarrays_to_parameters(sub_ndarrays), config)
            fit_configurations.append((client, fit_ins))
        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if server_round == 1:
            depth_counts: Dict[int, int] = {}
            for client_proxy, fit_res in results:
                partition_id = fit_res.metrics.get("partition_id", None)
                if partition_id is not None:
                    d = self._get_client_depth(client_proxy.cid, int(partition_id))
                    depth_counts[d] = depth_counts.get(d, 0) + 1
            log(INFO, f"DepthFL discovery: {len(self.client_to_partition)} mappings, depth dist={depth_counts}")
            global_params = [v.cpu().numpy() for v in self.global_model.state_dict().values()]
            return ndarrays_to_parameters(global_params), {"discovery": True}

        global_state = self.global_model.state_dict()
        keys = self.global_keys
        n = len(keys)
        acc = [torch.zeros_like(global_state[k], dtype=torch.float32) for k in keys]
        counts = [0.0] * n

        for _client_proxy, fit_res in results:
            arrays = parameters_to_ndarrays(fit_res.parameters)
            # A client's params are a positional prefix of the full-depth model.
            for i, arr in enumerate(arrays):
                if i >= n:
                    break
                acc[i] += torch.tensor(arr, dtype=torch.float32)
                counts[i] += 1.0

        new_state = OrderedDict()
        for i, k in enumerate(keys):
            if counts[i] > 0:
                val = acc[i] / counts[i]
                if k.endswith("num_batches_tracked"):
                    new_state[k] = val.round().to(torch.long)
                else:
                    new_state[k] = val.to(global_state[k].dtype)
            else:
                new_state[k] = global_state[k]
        self.global_model.load_state_dict(new_state, strict=True)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        result = [new_state[k].cpu().numpy() for k in keys]
        return ndarrays_to_parameters(result), metrics_aggregated

    def _resolve_client_index(self, client_id: str) -> int:
        if client_id in self.client_to_partition:
            return int(self.client_to_partition[client_id]) % self.num_clients
        try:
            return int(client_id) % self.num_clients
        except (TypeError, ValueError):
            pass
        digest = hashlib.md5(client_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.num_clients


def create_depthfl_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 100,
    **kwargs,
) -> DepthFLStrategy:
    """Create a DepthFL strategy."""
    return DepthFLStrategy(
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
