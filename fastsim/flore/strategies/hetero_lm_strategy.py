"""Strategies for Model-Heterogeneous Federated Learning on Language Models.

Implements HeteroFL for GPT-2 style language models on StackOverflow.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Callable
from collections import OrderedDict
from logging import INFO, WARNING

from flwr.common import (
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.common.typing import NDArrays
from flwr.common.logger import log
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..hetero_gpt2 import create_hetero_lm_model
from ..hetero_utils import get_model_rate_for_partition, parse_rate_config


class HeteroLMStrategy(FedAvg):
    """Strategy for model-heterogeneous federated learning on language models.

    Handles:
    - Maintaining a global GPT-2/LSTM model (largest model)
    - Extracting width-scaled sub-models for heterogeneous clients
    - Aggregating updates from clients with different model widths
    """

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[Callable] = None,
        on_fit_config_fn: Optional[Callable] = None,
        on_evaluate_config_fn: Optional[Callable] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[Callable] = None,
        evaluate_metrics_aggregation_fn: Optional[Callable] = None,
        # LM-specific parameters
        vocab_size: int = 10000,
        max_seq_len: int = 256,
        model_type: str = "gpt2",  # "gpt2" or "lstm"
        global_model_rate: float = 1.0,
        num_clients: int = 100,
        seed: int = 42,
        model_rates: Optional[List[float]] = None,
        rate_distribution: Optional[List[float]] = None,
        model_config: Optional[Dict] = None,
    ):
        """Initialize heterogeneous LM FL strategy."""
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

        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.model_type = model_type
        self.global_model_rate = global_model_rate
        self.num_clients = num_clients
        self.seed = seed
        self.current_round = 0
        self.model_rates, self.rate_distribution = parse_rate_config(
            model_rates, rate_distribution
        )
        self.model_config = model_config or {}

        # Create global model
        self.global_model = self._create_model(global_model_rate)
        log(INFO, f"Created global {model_type} model with rate={global_model_rate}, params={self.global_model.get_num_params():,}")

        # Log model rate distribution
        log(INFO, f"Model rates config: rates={self.model_rates}, distribution={self.rate_distribution}")

        # Map client_id to partition_id
        self.client_to_partition: Dict[str, int] = {}
        self.next_fallback_index = 0

    def _create_model(self, model_rate: float) -> nn.Module:
        """Create model with specified rate."""
        return create_hetero_lm_model(
            model_type=self.model_type,
            model_rate=model_rate,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            model_config=self.model_config,
            use_scaler=True,
        )

    def _get_client_model_rate(self, client_id: str, partition_id: Optional[int] = None) -> float:
        """Get model rate for a specific client."""
        if partition_id is not None:
            self.client_to_partition[client_id] = partition_id
            return get_model_rate_for_partition(
                partition_id,
                self.num_clients,
                model_rates=self.model_rates,
                rate_distribution=self.rate_distribution,
            )

        if client_id in self.client_to_partition:
            partition_id = self.client_to_partition[client_id]
            return get_model_rate_for_partition(
                partition_id,
                self.num_clients,
                model_rates=self.model_rates,
                rate_distribution=self.rate_distribution,
            )

        idx = self.next_fallback_index % self.num_clients
        self.next_fallback_index += 1
        self.client_to_partition[client_id] = idx
        log(WARNING, f"No partition_id for client {client_id[:16]}..., using fallback index {idx}")
        return get_model_rate_for_partition(
            idx,
            self.num_clients,
            model_rates=self.model_rates,
            rate_distribution=self.rate_distribution,
        )

    def _extract_sub_parameters(self, client_id: str, round_num: int) -> NDArrays:
        """Extract sub-model parameters for a client.

        For language models, we use static extraction (HeteroFL style):
        - Take the first k channels/dimensions for smaller models
        """
        if round_num == 1:
            # Discovery round - send placeholder
            model = self.global_model
            log(INFO, f"Round 1 (Discovery): Sending placeholder to client {client_id[:16]}...")
        else:
            model_rate = self._get_client_model_rate(client_id)
            model = self._create_model(model_rate)

            if round_num <= 4:
                log(INFO, f"Round {round_num}: Extracting sub-model for client {client_id[:16]}... with rate={model_rate:.2f}")

            # Extract parameters from global model to client model
            self._copy_global_to_client(model)

        return [v.cpu().numpy() for v in model.state_dict().values()]

    def _copy_global_to_client(self, client_model: nn.Module):
        """Copy parameters from global model to client model (width slicing)."""
        global_state = self.global_model.state_dict()
        client_state = client_model.state_dict()

        new_state = OrderedDict()
        for key in client_state.keys():
            if key in global_state:
                global_param = global_state[key]
                client_shape = client_state[key].shape

                # Slice to match client shape
                if len(client_shape) == 0:
                    # Scalar
                    new_state[key] = global_param.clone()
                elif len(client_shape) == 1:
                    # 1D (bias, LayerNorm)
                    new_state[key] = global_param[:client_shape[0]].clone()
                elif len(client_shape) == 2:
                    # 2D (Linear, Embedding)
                    # For Embedding: [vocab_size, embed_dim]
                    # For Linear: [out_features, in_features]
                    if 'embedding' in key.lower() or 'embed' in key.lower():
                        # Embedding: keep vocab, slice embed_dim
                        new_state[key] = global_param[:, :client_shape[1]].clone()
                    else:
                        # Linear weight
                        new_state[key] = global_param[:client_shape[0], :client_shape[1]].clone()
                elif len(client_shape) == 3:
                    # 3D (rare, but handle)
                    new_state[key] = global_param[:client_shape[0], :client_shape[1], :client_shape[2]].clone()
                else:
                    new_state[key] = global_param.clone()
            else:
                new_state[key] = client_state[key].clone()

        client_model.load_state_dict(new_state)

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        self.current_round = server_round

        # Update global model with current parameters
        global_params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.global_model.state_dict().keys(), global_params)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.tensor(v)
        self.global_model.load_state_dict(state_dict, strict=True)

        # Round 1: Sample ALL clients for discovery
        if server_round == 1:
            log(INFO, "=" * 60)
            log(INFO, "Round 1: CAPABILITY DISCOVERY - Sampling ALL clients")
            log(INFO, "=" * 60)
            num_available = client_manager.num_available()
            clients = client_manager.sample(
                num_clients=num_available, min_num_clients=num_available
            )
        else:
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

        fit_configurations = []

        for client in clients:
            client_id = client.cid
            sub_params = self._extract_sub_parameters(client_id, server_round)

            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)

            config["current_round"] = server_round

            if server_round == 1:
                model_rate = 1.0
            else:
                model_rate = self._get_client_model_rate(client_id)
            config["model_rate"] = model_rate

            fit_ins = FitIns(ndarrays_to_parameters(sub_params), config)
            fit_configurations.append((client, fit_ins))

        return fit_configurations

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from heterogeneous clients."""
        if not results:
            return None, {}

        # Round 1: Discovery
        if server_round == 1:
            log(INFO, "=" * 60)
            log(INFO, f"Round 1: CAPABILITY DISCOVERY - Received {len(results)} client responses")
            log(INFO, "=" * 60)

            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                partition_id = fit_res.metrics.get("partition_id", None)

                if partition_id is not None:
                    partition_id = int(partition_id)
                    self._get_client_model_rate(client_id, partition_id)

            log(INFO, f"Learned {len(self.client_to_partition)} client->partition mappings")

            global_params = [v.cpu().numpy() for v in self.global_model.state_dict().values()]
            return ndarrays_to_parameters(global_params), {"discovery": True}

        # Round 2+: Aggregation
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        aggregated_ndarrays = self._aggregate_hetero(weights_results, results)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated

    def _aggregate_hetero(
        self,
        weights_results: List[Tuple[NDArrays, int]],
        results: List[Tuple[ClientProxy, FitRes]],
    ) -> NDArrays:
        """Aggregate heterogeneous client updates into global model."""
        global_state = self.global_model.state_dict()
        aggregated = OrderedDict()
        weights_sum = OrderedDict()

        for key in global_state.keys():
            aggregated[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            weights_sum[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

        for (client_arrays, num_samples), (client_proxy, fit_res) in zip(weights_results, results):
            client_id = client_proxy.cid
            model_rate = self._get_client_model_rate(client_id)
            client_model = self._create_model(model_rate)

            # Map arrays to state dict
            client_state = OrderedDict()
            for (key, _), array in zip(client_model.state_dict().items(), client_arrays):
                client_state[key] = torch.tensor(array, dtype=torch.float32)

            # Add to aggregated with width slicing
            for key in client_state.keys():
                if key in aggregated:
                    client_param = client_state[key]
                    self._aggregate_width_slice(
                        aggregated[key], weights_sum[key],
                        client_param, num_samples, key
                    )

        # Normalize
        for key in global_state.keys():
            mask = weights_sum[key] > 0
            aggregated[key][mask] = aggregated[key][mask] / weights_sum[key][mask]
            aggregated[key][~mask] = global_state[key][~mask]

        return [aggregated[key].cpu().numpy() for key in global_state.keys()]

    def _aggregate_width_slice(
        self,
        aggregated: torch.Tensor,
        counts: torch.Tensor,
        client_param: torch.Tensor,
        num_samples: int,
        key: str,
    ) -> None:
        """Aggregate width slice from client into global model."""
        client_shape = client_param.shape

        if len(client_shape) == 0:
            # Scalar
            aggregated += client_param * num_samples
            counts += num_samples

        elif len(client_shape) == 1:
            n = client_shape[0]
            aggregated[:n] += client_param * num_samples
            counts[:n] += num_samples

        elif len(client_shape) == 2:
            if 'embedding' in key.lower() or 'embed' in key.lower():
                # Embedding: [vocab_size, embed_dim]
                aggregated[:, :client_shape[1]] += client_param * num_samples
                counts[:, :client_shape[1]] += num_samples
            else:
                # Linear: [out, in]
                out_f, in_f = client_shape[0], client_shape[1]
                aggregated[:out_f, :in_f] += client_param * num_samples
                counts[:out_f, :in_f] += num_samples

        elif len(client_shape) == 3:
            d0, d1, d2 = client_shape
            aggregated[:d0, :d1, :d2] += client_param * num_samples
            counts[:d0, :d1, :d2] += num_samples

        else:
            aggregated += client_param * num_samples
            counts += num_samples


def create_hetero_lm_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    vocab_size: int = 10000,
    max_seq_len: int = 256,
    model_type: str = "gpt2",
    model_rates: Optional[List[float]] = None,
    rate_distribution: Optional[List[float]] = None,
    model_config: Optional[Dict] = None,
    **kwargs
) -> HeteroLMStrategy:
    """Create HeteroFL strategy for language modeling."""
    return HeteroLMStrategy(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_type=model_type,
        num_clients=num_clients,
        model_rates=model_rates,
        rate_distribution=rate_distribution,
        model_config=model_config,
        **kwargs
    )
