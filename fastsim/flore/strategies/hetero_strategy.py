"""Strategies for Model-Heterogeneous Federated Learning.

Implements:
- HeteroFL: Static sub-model extraction
- FedRolex: Rolling window sub-model extraction
- Random: Random sub-model extraction baseline
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
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

from ..hetero_models import create_hetero_model
from ..hetero_utils import (
    create_extractor,
    SubModelExtractor,
    get_model_rate_for_partition,
    get_all_client_model_rates,
    parse_rate_config,
)
from ..net2wider import create_shell_repair_state, repair_shell_if_stale


class HeteroStrategy(FedAvg):
    """Base strategy for model-heterogeneous federated learning.

    This strategy handles:
    - Maintaining a global model (typically the largest model)
    - Extracting sub-models for heterogeneous clients
    - Aggregating updates from clients with different model sizes
    - Distributing appropriate sub-models to clients each round
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
        # Hetero-specific parameters
        extraction_mode: str = "static",
        num_classes: int = 10,
        model_name: str = "HeteroPreResNet18",
        global_model_rate: float = 1.0,
        model_split_mode: str = "fix",
        num_clients: int = 100,
        overlap: float = 0.0,
        seed: int = 42,
        model_rates: Optional[Union[str, List[float]]] = None,
        rate_distribution: Optional[Union[str, List[float]]] = None,
        norm_type: str = "sbn",
        core_model_rate: Optional[float] = None,
        enable_shell_repair: bool = False,
        shell_repair_threshold: int = 0,
        shell_repair_beta: float = 0.5,
        shell_repair_noise_std: float = 0.0,
        shell_repair_seed: Optional[int] = None,
    ):
        """Initialize heterogeneous FL strategy.

        Args:
            extraction_mode: Sub-model extraction mode ('static', 'rolling', 'random')
            num_classes: Number of output classes
            global_model_rate: Width multiplier for global model (typically 1.0)
            model_split_mode: How to distribute model rates ('fix', 'uniform', 'dirichlet')
            num_clients: Total number of clients
            overlap: Overlap ratio for rolling window (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
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

        self.extraction_mode = extraction_mode
        self.num_classes = num_classes
        self.model_name = model_name
        self.global_model_rate = global_model_rate
        self.model_split_mode = model_split_mode
        self.num_clients = num_clients
        self.overlap = overlap
        self.seed = seed
        self.norm_type = norm_type
        self.model_rates, self.rate_distribution = parse_rate_config(model_rates, rate_distribution)
        self.current_round = 0
        self.core_model_rate = float(min(self.model_rates) if core_model_rate is None else core_model_rate)
        self.enable_shell_repair = bool(enable_shell_repair)
        self.shell_repair_threshold = int(shell_repair_threshold)
        self.shell_repair_beta = float(shell_repair_beta)
        self.shell_repair_noise_std = float(shell_repair_noise_std)
        self.shell_repair_seed = int(seed if shell_repair_seed is None else shell_repair_seed)
        self.shell_repair_state = create_shell_repair_state(seed=self.shell_repair_seed)

        # Create global model
        self.global_model = create_hetero_model(
            self.model_name,
            in_channels=3,
            num_classes=num_classes,
            model_rate=global_model_rate,
            norm_type=self.norm_type,
        )
        log(INFO, f"Created global model with rate={global_model_rate}")

        # Create model rate distribution for clients (using hetero_utils for consistency)
        self.client_model_rates = get_all_client_model_rates(
            num_clients=num_clients,
            model_rates=self.model_rates,
            rate_distribution=self.rate_distribution,
        )
        rate_values = list(self.client_model_rates.values())
        unique_rates, counts = np.unique(rate_values, return_counts=True)
        log(INFO, f"Model rates config: rates={self.model_rates}, distribution={self.rate_distribution}")
        log(INFO, f"Client model rate distribution: {dict(zip(unique_rates, counts))}")

        # Create sub-model extractor
        self.extractor = create_extractor(
            strategy=extraction_mode,
            global_model=self.global_model,
            num_classes=num_classes,
            overlap=overlap,
            seed=seed,
        )
        log(INFO, f"Using {extraction_mode} extraction mode with overlap={overlap}")

        # Map client_id to partition_id for model rate assignment
        # Note: We don't cache client models because Ray reuses processes
        self.client_to_partition: Dict[str, int] = {}
        self.next_fallback_index = 0  # Sequential fallback before partition_id is known
        self.last_round_n_full_sampled: int = 0
        self.last_round_rounds_since_last_full: int = -1
        self.last_full_client_round: Optional[int] = None
        self.last_round_shell_repair_metrics: Dict[str, float] = {
            "shell_repair_applied": 0.0,
            "shell_repair_skipped_recent_full": 1.0,
            "shell_repair_threshold": float(self.shell_repair_threshold),
            "shell_repair_beta": float(self.shell_repair_beta),
            "shell_repair_noise_std": float(self.shell_repair_noise_std),
            "shell_repair_shell_l2": 0.0,
            "shell_repair_num_tensors": 0.0,
        }
        dist_sum = float(sum(self.rate_distribution)) if self.rate_distribution else 0.0
        self.p_full_target = 0.0
        if dist_sum > 0:
            for rate, prob in zip(self.model_rates, self.rate_distribution):
                if abs(float(rate) - 1.0) < 1e-12:
                    self.p_full_target = float(prob) / dist_sum
                    break

    def _get_client_model_rate(self, client_id: str, partition_id: Optional[int] = None) -> float:
        """Get model rate for a specific client.

        Uses the same deterministic mapping as the client:
        partition_id -> model_rate based on distribution [40%, 30%, 20%, 10%]

        Args:
            client_id: Client ID (hash)
            partition_id: Optional partition ID from client's fit result metrics

        Returns:
            Model rate for this client
        """
        # If partition_id is provided, store the mapping and use deterministic rate
        if partition_id is not None:
            self.client_to_partition[client_id] = partition_id
            # Use same mapping function as client to ensure consistency
            return get_model_rate_for_partition(
                partition_id,
                self.num_clients,
                self.model_rates,
                self.rate_distribution,
            )

        # If we already know this client's partition_id, use it
        if client_id in self.client_to_partition:
            partition_id = self.client_to_partition[client_id]
            return get_model_rate_for_partition(
                partition_id,
                self.num_clients,
                self.model_rates,
                self.rate_distribution,
            )

        # Fallback for unknown clients (before partition_id is received)
        # Use sequential index as temporary assignment
        idx = self.next_fallback_index % self.num_clients
        self.next_fallback_index += 1
        self.client_to_partition[client_id] = idx
        log(WARNING, f"No partition_id for client {client_id[:16]}..., using fallback index {idx}")
        return get_model_rate_for_partition(
            idx,
            self.num_clients,
            self.model_rates,
            self.rate_distribution,
        )

    def _create_client_model(self, client_id: str) -> nn.Module:
        """Create or retrieve client model.

        Note: We always create a fresh model based on current rate assignment.
        This is necessary because Ray reuses ClientAppActor processes across different
        client_ids, so the same process might handle clients with different rates.
        """
        model_rate = self._get_client_model_rate(client_id)
        # Always create fresh model - don't cache by client_id
        # because Ray process pools mean the same client_id might have different rates
        # across rounds if partition_id mapping changed
        model = create_hetero_model(
            self.model_name,
            in_channels=3,
            num_classes=self.num_classes,
            model_rate=model_rate,
            norm_type=self.norm_type,
        )
        return model

    def _extract_sub_parameters(self, client_id: str, round_num: int) -> NDArrays:
        """Extract sub-model parameters for a client.

        Round 1: Capability discovery - send minimal placeholder parameters
        Round 2+: Send correct heterogeneous model parameters
        """
        if round_num == 1:
            # Round 1: Capability discovery - parameters won't be used by client
            # Just send placeholder (full model) - client will ignore and return capabilities
            client_model = self.global_model
            log(INFO, f"Round 1 (Discovery): Sending placeholder to client {client_id[:16]}...")
        else:
            # Round 2+: Use actual heterogeneous models based on partition_id
            client_model = self._create_client_model(client_id)
            _ = self._get_client_model_rate(client_id)

        client_idx = self._resolve_client_index(client_id)

        # Extract sub-model from global model
        sub_params = self.extractor.extract_sub_model(
            client_model=client_model,
            client_idx=client_idx,
            round_num=round_num,
        )

        # Convert to numpy arrays
        return [v.cpu().numpy() for v in sub_params.values()]

    def _resolve_client_index(self, client_id: str) -> int:
        """Resolve a stable client index for extractor bookkeeping.

        Prefer the learned partition mapping when available; otherwise derive a
        deterministic fallback from the client ID string.
        """
        if client_id in self.client_to_partition:
            return int(self.client_to_partition[client_id]) % self.num_clients

        # Flower simulation client IDs are typically large integer strings.
        try:
            return int(client_id) % self.num_clients
        except (TypeError, ValueError):
            pass

        # Backward-compatible fallback for IDs like "...-7".
        if "-" in client_id:
            tail = client_id.rsplit("-", 1)[-1]
            if tail.isdigit():
                return int(tail) % self.num_clients

        # Deterministic hash fallback for arbitrary IDs.
        digest = hashlib.md5(client_id.encode("utf-8")).hexdigest()
        return int(digest[:8], 16) % self.num_clients

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training.

        Round 1: Capability discovery - sample ALL clients to learn partition_id mapping
        Round 2+: Normal training with heterogeneous models
        """
        self.current_round = server_round

        # Update global model with current parameters
        global_params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.global_model.state_dict().keys(), global_params)
        state_dict = OrderedDict()
        for k, v in params_dict:
            if k.endswith('num_batches_tracked'):
                state_dict[k] = torch.tensor(v, dtype=torch.long)
            else:
                state_dict[k] = torch.tensor(v, dtype=torch.float32)
        self.global_model.load_state_dict(state_dict, strict=True)

        # Round 1: Sample ALL clients for capability discovery
        if server_round == 1:
            log(INFO, "=" * 60)
            log(INFO, "Round 1: CAPABILITY DISCOVERY - Sampling ALL clients")
            log(INFO, "=" * 60)
            num_available = client_manager.num_available()
            clients = client_manager.sample(
                num_clients=num_available, min_num_clients=num_available
            )
        else:
            # Round 2+: Normal sampling
            sample_size, min_num_clients = self.num_fit_clients(
                client_manager.num_available()
            )
            clients = client_manager.sample(
                num_clients=sample_size, min_num_clients=min_num_clients
            )

        # Create fit instructions for each client
        fit_configurations = []
        n_full_sampled = 0

        for client in clients:
            client_id = client.cid

            # Extract sub-model parameters for this client
            sub_params = self._extract_sub_parameters(client_id, server_round)

            # Create fit configuration
            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)

            config["current_round"] = server_round

            # Add model rate to config
            if server_round == 1:
                model_rate = 1.0  # Placeholder for discovery round
            else:
                model_rate = self._get_client_model_rate(client_id)
            config["model_rate"] = model_rate
            if abs(float(model_rate) - 1.0) < 1e-12:
                n_full_sampled += 1

            fit_ins = FitIns(ndarrays_to_parameters(sub_params), config)
            fit_configurations.append((client, fit_ins))

        self.last_round_n_full_sampled = n_full_sampled
        if n_full_sampled > 0:
            self.last_full_client_round = server_round
            self.last_round_rounds_since_last_full = 0
        elif self.last_full_client_round is None:
            self.last_round_rounds_since_last_full = -1
        else:
            self.last_round_rounds_since_last_full = server_round - self.last_full_client_round

        return fit_configurations

    def get_round_sampling_metrics(self) -> Dict[str, float]:
        """Return compact per-round sampling stats for server-side logging."""
        return {
            "n_full_sampled": float(self.last_round_n_full_sampled),
            "rounds_since_last_full": float(self.last_round_rounds_since_last_full),
            "p_full_target": float(self.p_full_target),
            **self.last_round_shell_repair_metrics,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate model updates from heterogeneous clients.

        Round 1: Capability discovery - no aggregation, just learn client mappings
        Round 2+: Proper heterogeneous aggregation
        """
        if not results:
            return None, {}

        # Round 1: Capability discovery - collect partition_id from all clients
        if server_round == 1:
            log(INFO, "=" * 60)
            log(INFO, f"Round 1: CAPABILITY DISCOVERY - Received {len(results)} client responses")
            log(INFO, "=" * 60)

            # Extract partition_id from all clients
            rate_counts: Dict[float, int] = {}
            for client_proxy, fit_res in results:
                client_id = client_proxy.cid
                partition_id = fit_res.metrics.get("partition_id", None)
                model_rate = fit_res.metrics.get("model_rate", None)

                if partition_id is not None:
                    partition_id = int(partition_id)
                    self._get_client_model_rate(client_id, partition_id)

                    if model_rate is not None:
                        model_rate = float(model_rate)
                        rate_counts[model_rate] = rate_counts.get(model_rate, 0) + 1

            log(INFO, f"Learned {len(self.client_to_partition)} client->partition mappings")
            log(INFO, f"Model rate distribution: {rate_counts}")
            log(INFO, "=" * 60)

            # Return current global parameters unchanged (no training happened)
            global_params = [v.cpu().numpy() for v in self.global_model.state_dict().values()]
            return ndarrays_to_parameters(global_params), {"discovery": True}

        # Round 2+: Actual aggregation
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for client, fit_res in results
        ]

        aggregated_ndarrays = self._aggregate_hetero(weights_results, results)

        # Aggregate custom metrics if available
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
        """Aggregate heterogeneous client updates into global model.

        Implements selective averaging (Equation 10 from FedRolex paper):
        For each parameter k, we only average over clients that contributed to that parameter.

        Note: This is only called for Round 2+ (actual training rounds).
        Round 1 (capability discovery) is handled separately in aggregate_fit.
        """
        # Initialize accumulator with global model structure
        global_state = self.global_model.state_dict()
        aggregated = OrderedDict()
        weights_sum = OrderedDict()  # Per-parameter contribution counts

        for key in global_state.keys():
            aggregated[key] = torch.zeros_like(global_state[key], dtype=torch.float32)
            weights_sum[key] = torch.zeros_like(global_state[key], dtype=torch.float32)

        # Heterogeneous aggregation with width slicing
        for (client_arrays, num_samples), (client_proxy, fit_res) in zip(weights_results, results):
            client_id = client_proxy.cid

            # Get client model to know parameter shapes
            client_model = self._create_client_model(client_id)

            # Map client arrays to state dict
            client_state = OrderedDict()
            for (key, _), array in zip(client_model.state_dict().items(), client_arrays):
                if key.endswith('num_batches_tracked'):
                    client_state[key] = torch.tensor(array, dtype=torch.long)
                else:
                    client_state[key] = torch.tensor(array, dtype=torch.float32)

            # Add weighted contribution with proper width slicing
            # For width-heterogeneous models, client parameters are smaller than global
            # We aggregate into the first k elements of each dimension
            for key in client_state.keys():
                if key in aggregated:
                    client_param = client_state[key]
                    global_param = aggregated[key]
                    weight_param = weights_sum[key]

                    # Use width slicing for all parameters (handles all layer types)
                    self._aggregate_width_slice(global_param, weight_param,
                                               client_param, num_samples)

        # Normalize by per-parameter counts (Equation 10: selective averaging)
        for key in global_state.keys():
            if key.endswith(('num_batches_tracked', 'running_mean', 'running_var')):
                # sBN: BN running buffers are never FedAvg-aggregated; they are
                # recomputed post-training via calibrate_bn_stats. Keep global value.
                aggregated[key] = global_state[key]
            else:
                # Normalize by actual contribution count per parameter
                mask = weights_sum[key] > 0
                aggregated[key][mask] = aggregated[key][mask] / weights_sum[key][mask]
                # Parameters not updated by any client keep their old values
                aggregated[key][~mask] = global_state[key][~mask]

        if self.enable_shell_repair:
            aggregated, repair_metrics = repair_shell_if_stale(
                aggregated,
                self.shell_repair_state,
                rounds_since_last_full=float(self.last_round_rounds_since_last_full),
                threshold=self.shell_repair_threshold,
                beta=self.shell_repair_beta,
                noise_std=self.shell_repair_noise_std,
                core_model_rate=self.core_model_rate,
            )
            self.last_round_shell_repair_metrics = repair_metrics
        else:
            self.last_round_shell_repair_metrics = {
                "shell_repair_applied": 0.0,
                "shell_repair_skipped_recent_full": 1.0,
                "shell_repair_threshold": float(self.shell_repair_threshold),
                "shell_repair_beta": float(self.shell_repair_beta),
                "shell_repair_noise_std": float(self.shell_repair_noise_std),
                "shell_repair_shell_l2": 0.0,
                "shell_repair_num_tensors": 0.0,
            }

        self.global_model.load_state_dict(aggregated, strict=True)

        # Convert to numpy
        result = [aggregated[key].cpu().numpy() for key in global_state.keys()]

        return result

    def _get_output_size(self, model: nn.Module) -> int:
        """Get the output dimension of the model's final layer."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'linear' in name.lower():
                return module.out_features
        raise ValueError("Could not find output linear layer")

    def _aggregate_width_slice(self, aggregated: torch.Tensor, counts: torch.Tensor,
                               client_param: torch.Tensor, num_samples: int) -> None:
        """Aggregate width slice from client into global model (in-place).

        For width-heterogeneous models, client updates only affect a subset of channels.
        This adds the client's update to the appropriate channel slice.

        Args:
            aggregated: Accumulated global parameter (modified in-place)
            counts: Per-parameter counts (modified in-place)
            client_param: Client's parameter update
            num_samples: Number of samples from this client
        """
        client_shape = client_param.shape

        # Conv2d weight: [out_channels, in_channels, H, W]
        if len(client_shape) == 4:
            out_ch, in_ch = client_shape[0], client_shape[1]
            aggregated[:out_ch, :in_ch, :, :] += client_param * num_samples
            counts[:out_ch, :in_ch, :, :] += num_samples

        # Linear weight: [out_features, in_features]
        elif len(client_shape) == 2:
            out_f, in_f = client_shape[0], client_shape[1]
            aggregated[:out_f, :in_f] += client_param * num_samples
            counts[:out_f, :in_f] += num_samples

        # BatchNorm/Bias: [num_features]
        elif len(client_shape) == 1:
            n_f = client_shape[0]
            aggregated[:n_f] += client_param * num_samples
            counts[:n_f] += num_samples

        # Scalar or other
        else:
            aggregated += client_param * num_samples
            counts += num_samples

    def _get_client_indices(self, client_idx: int, local_output_size: int, round_num: int) -> np.ndarray:
        """Get the global indices that a client's sub-model corresponds to.

        This retrieves the indices that were stored by the extractor during extraction.
        All extraction modes (static, rolling, random) now store their indices during
        extract_sub_model(), so we can simply retrieve them here.

        This approach is superior to reconstruction because:
        1. Works correctly for all modes (especially random)
        2. Uses exact indices that were used for extraction (no drift)
        3. Single source of truth (extractor decides indices once)
        4. Simpler and more maintainable

        Args:
            client_idx: Client index
            local_output_size: Size of client's local model
            round_num: Current round number

        Returns:
            Array of global model indices that this client updated
        """
        # Get indices from extractor (stored during extraction phase)
        indices = self.extractor.get_indices(client_idx)

        if len(indices) == 0:
            # Fallback for edge cases (e.g., client not in current round)
            log(WARNING, f"No indices found for client {client_idx} in round {round_num}, "
                        f"using static fallback (first {local_output_size} indices)")
            # Use static extraction as safe default
            if self.extraction_mode == "static":
                return np.arange(local_output_size)
            else:
                # For rolling/random, this shouldn't happen, but use static as safe default
                log(WARNING, f"Using static fallback for {self.extraction_mode} mode - "
                            f"this may indicate a bug in the extractor")
                return np.arange(local_output_size)

        return indices


# Convenience functions to create specific strategies

def create_heterofl_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    **kwargs
) -> HeteroStrategy:
    """Create HeteroFL strategy (static extraction)."""
    return HeteroStrategy(
        fraction_fit=0.0,  # Use min_fit_clients instead
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        extraction_mode="static",
        num_classes=num_classes,
        num_clients=num_clients,
        **kwargs
    )


def create_fedrolex_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    overlap: float = 0.0,
    **kwargs
) -> HeteroStrategy:
    """Create FedRolex strategy (rolling window extraction)."""
    return HeteroStrategy(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        extraction_mode="rolling",
        num_classes=num_classes,
        num_clients=num_clients,
        overlap=overlap,
        **kwargs
    )


def create_random_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    **kwargs
) -> HeteroStrategy:
    """Create random extraction strategy."""
    return HeteroStrategy(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        extraction_mode="random",
        num_classes=num_classes,
        num_clients=num_clients,
        **kwargs
    )
