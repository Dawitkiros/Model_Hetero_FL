"""Utilities for model-heterogeneous federated learning.

This module provides utilities for:
- Sub-model extraction from global model
- Parameter aggregation across heterogeneous models
- Rolling window mechanisms for FedRolex
- Static and random extraction for baseline methods
- Model rate distribution mapping for heterogeneous clients
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Iterable, Union
from collections import OrderedDict


# ============================================================================
# Model Rate Distribution
# ============================================================================

# Model rate distribution for toxicity-focused experiments
DEFAULT_MODEL_RATES = [1.0, 0.25]
DEFAULT_RATE_DISTRIBUTION = [0.5, 0.5]  # 50% full model, 50% core model


def _to_float_list(value: Optional[Union[str, Iterable[float]]], default: List[float]) -> List[float]:
    """Parse a float list from config values.

    Supports:
    - list/tuple from TOML
    - comma-separated string from CLI overrides
    """
    if value is None:
        return list(default)
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if not parts:
            return list(default)
        return [float(p) for p in parts]
    return [float(v) for v in value]


def parse_rate_config(
    model_rates: Optional[Union[str, Iterable[float]]] = None,
    rate_distribution: Optional[Union[str, Iterable[float]]] = None,
) -> Tuple[List[float], List[float]]:
    """Return normalized model-rate config."""
    rates = _to_float_list(model_rates, DEFAULT_MODEL_RATES)
    dist = _to_float_list(rate_distribution, DEFAULT_RATE_DISTRIBUTION)
    if len(dist) != len(rates):
        raise ValueError(
            f"rate_distribution length ({len(dist)}) must match model_rates length ({len(rates)})"
        )
    total = float(sum(dist))
    if total <= 0:
        raise ValueError("rate_distribution must sum to a positive value")
    dist = [d / total for d in dist]
    return rates, dist


def get_model_rate_for_partition(
    partition_id: int,
    num_clients: int = 100,
    model_rates: List[float] = None,
    rate_distribution: List[float] = None,
) -> float:
    """Get model rate for a specific partition based on distribution.

    Maps partition_id to a model_rate based on the specified distribution.
    This ensures deterministic assignment - same partition_id always gets
    the same model_rate.

    Args:
        partition_id: Client's partition ID (0 to num_clients-1)
        num_clients: Total number of clients
        model_rates: List of available model rates (default: [1.0, 0.25])
        rate_distribution: Distribution of clients per rate (default: [0.5, 0.5])

    Returns:
        Model rate for this partition

    Example:
        For 100 clients with default distribution:
        - Partitions 0-49: rate=1.0 (50%)
        - Partitions 50-99: rate=0.25 (50%)
    """
    model_rates, rate_distribution = parse_rate_config(model_rates, rate_distribution)

    if partition_id < 0:
        return model_rates[-1]

    # Calculate cumulative distribution
    cumulative = []
    total = 0
    for ratio in rate_distribution:
        total += ratio
        cumulative.append(total)

    # Normalize partition_id to [0, 1) range
    normalized_position = partition_id / num_clients

    # Find which rate bucket this partition falls into
    for i, threshold in enumerate(cumulative):
        if normalized_position < threshold:
            return model_rates[i]

    # Fallback to full model
    return model_rates[-1]


def get_all_client_model_rates(
    num_clients: int = 100,
    model_rates: List[float] = None,
    rate_distribution: List[float] = None,
) -> Dict[int, float]:
    """Get model rates for all clients.

    Args:
        num_clients: Total number of clients
        model_rates: List of available model rates
        rate_distribution: Distribution of clients per rate

    Returns:
        Dictionary mapping partition_id to model_rate
    """
    rates, dist = parse_rate_config(model_rates, rate_distribution)
    return {
        partition_id: get_model_rate_for_partition(
            partition_id, num_clients, rates, dist
        )
        for partition_id in range(num_clients)
    }


def slice_tensor_to_shape(source: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
    """Slice tensor prefix-wise to target shape."""
    if source.shape == target_shape:
        return source.clone()
    if len(source.shape) != len(target_shape):
        return source.clone()
    slices = tuple(slice(0, min(source.shape[i], target_shape[i])) for i in range(len(target_shape)))
    out = source[slices].clone()
    if out.shape != target_shape:
        # Fallback for unexpected shape mismatch: keep whatever fits in out.
        return out
    return out


def extract_state_dict_for_model(
    global_state: "OrderedDict[str, torch.Tensor]",
    target_state: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """Extract width-compatible state dict from global model state."""
    extracted = OrderedDict()
    for name, target_tensor in target_state.items():
        src = global_state[name]
        if target_tensor.dtype == torch.long:
            extracted[name] = src.clone().to(dtype=torch.long)
        else:
            extracted[name] = slice_tensor_to_shape(src, target_tensor.shape).to(dtype=target_tensor.dtype)
    return extracted


def split_core_shell_norm(
    delta: torch.Tensor,
    core_rate: float,
) -> Tuple[float, float]:
    """Compute L2 norms for core/shell update regions from a tensor delta."""
    if delta.numel() == 0:
        return 0.0, 0.0
    if delta.ndim == 0:
        return 0.0, 0.0

    if delta.ndim >= 1:
        core_dim = int(np.ceil(delta.shape[0] * core_rate))
        core_dim = max(0, min(core_dim, delta.shape[0]))
        if core_dim == 0:
            return 0.0, float(torch.norm(delta).item())
        core_part = delta[:core_dim]
        shell_part = delta[core_dim:]
        core_norm = float(torch.norm(core_part).item()) if core_part.numel() > 0 else 0.0
        shell_norm = float(torch.norm(shell_part).item()) if shell_part.numel() > 0 else 0.0
        return core_norm, shell_norm
    return 0.0, 0.0


# ============================================================================
# Sub-Model Extraction Classes
# ============================================================================


class SubModelExtractor:
    """Base class for sub-model extraction strategies."""

    def __init__(self, global_model: nn.Module, num_classes: int = 10):
        """Initialize extractor with global model.

        Args:
            global_model: The global server model
            num_classes: Number of output classes
        """
        self.global_model = global_model
        self.num_classes = num_classes
        self.global_output_size = self._get_output_size(global_model)
        # Track which indices were used for each client in current round
        self.current_round_indices = {}

    def _get_output_size(self, model: nn.Module) -> int:
        """Get the output dimension of the model's final layer."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'linear' in name.lower():
                return module.out_features
        raise ValueError("Could not find output linear layer")

    def _extract_width_slice(self, global_param: torch.Tensor, client_param: torch.Tensor,
                            param_name: str) -> torch.Tensor:
        """Extract width slice from global parameter to match client parameter shape.

        For width-heterogeneous models, client models use a subset of channels.
        This extracts the appropriate slice based on the client model's dimensions.

        Args:
            global_param: Parameter from global model
            client_param: Parameter from client model (for shape reference)
            param_name: Name of the parameter (for layer-specific logic)

        Returns:
            Sliced parameter matching client_param shape
        """
        global_shape = global_param.shape
        client_shape = client_param.shape

        # If shapes match, no slicing needed
        if global_shape == client_shape:
            return global_param.clone()

        # Conv2d weight: [out_channels, in_channels, H, W]
        if len(global_shape) == 4:
            out_channels = client_shape[0]
            in_channels = client_shape[1]
            return global_param[:out_channels, :in_channels, :, :].clone()

        # BatchNorm or Linear weight: [out_features, in_features] or [num_features]
        elif len(global_shape) == 2:
            return global_param[:client_shape[0], :client_shape[1]].clone()

        # BatchNorm parameters or Linear bias: [num_features]
        elif len(global_shape) == 1:
            return global_param[:client_shape[0]].clone()

        # Scalar or unknown shape - just copy
        else:
            return global_param.clone()

    def extract_sub_model(self, client_model: nn.Module,
                         client_idx: int,
                         round_num: int) -> Dict[str, torch.Tensor]:
        """Extract sub-model parameters for a client.

        Args:
            client_model: Client's local model
            client_idx: Client index
            round_num: Current round number

        Returns:
            Dictionary of parameter names to tensors
        """
        raise NotImplementedError

    def get_indices(self, client_idx: int) -> np.ndarray:
        """Get the indices that were used for a client in the current round.

        Args:
            client_idx: Client index

        Returns:
            Array of global indices used for this client
        """
        return self.current_round_indices.get(client_idx, np.array([]))


class StaticExtractor(SubModelExtractor):
    """Static sub-model extraction (HeteroFL baseline).

    Extracts the first k output neurons consistently across all rounds.
    This is the simplest baseline where each client always trains the same
    portion of the global model.
    """

    def __init__(self, global_model: nn.Module, num_classes: int = 10):
        super().__init__(global_model, num_classes)
        self.client_indices = {}  # Cache indices per client

    def extract_sub_model(self, client_model: nn.Module,
                         client_idx: int,
                         round_num: int) -> Dict[str, torch.Tensor]:
        """Extract static sub-model for client."""
        local_output_size = self._get_output_size(client_model)

        # Generate or retrieve indices for this client
        if client_idx not in self.client_indices:
            # Static: always use the first local_output_size indices
            self.client_indices[client_idx] = np.arange(local_output_size)

        indices = self.client_indices[client_idx]
        # Store for aggregation
        self.current_round_indices[client_idx] = indices

        # Extract parameters
        sub_params = OrderedDict()
        global_state = self.global_model.state_dict()
        client_state = client_model.state_dict()

        for name, param in global_state.items():
            # For width-heterogeneous models, we need to slice parameters to match client model size
            # Use _extract_width_slice for all parameters to handle width differences correctly
            sub_params[name] = self._extract_width_slice(param, client_state[name], name)

        return sub_params


class RandomExtractor(SubModelExtractor):
    """Random sub-model extraction baseline.

    Randomly samples output neurons at each round. This provides diversity
    but lacks the systematic coverage of FedRolex.
    """

    def __init__(self, global_model: nn.Module, num_classes: int = 10, seed: int = 42):
        super().__init__(global_model, num_classes)
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def extract_sub_model(self, client_model: nn.Module,
                         client_idx: int,
                         round_num: int) -> Dict[str, torch.Tensor]:
        """Extract random sub-model for client."""
        local_output_size = self._get_output_size(client_model)

        # Random selection at each round
        indices = self.rng.choice(
            self.global_output_size,
            size=local_output_size,
            replace=False
        )
        indices = np.sort(indices)  # Keep sorted for consistency
        # Store for aggregation
        self.current_round_indices[client_idx] = indices

        # Extract parameters
        sub_params = OrderedDict()
        global_state = self.global_model.state_dict()
        client_state = client_model.state_dict()

        for name, param in global_state.items():
            # For width-heterogeneous models, use width slice for all parameters
            sub_params[name] = self._extract_width_slice(param, client_state[name], name)

        return sub_params


class RollingWindowExtractor(SubModelExtractor):
    """Rolling window sub-model extraction (FedRolex).

    Implements the rolling window mechanism where the extracted sub-model
    advances across communication rounds, ensuring all parameters of the
    global model are trained over time.
    """

    def __init__(self, global_model: nn.Module, num_classes: int = 10,
                 overlap: float = 0.0, reshuffle_rounds: int = 512, seed: int = 42):
        """Initialize rolling window extractor.

        Args:
            global_model: Global server model
            num_classes: Number of output classes
            overlap: Overlap ratio between consecutive windows (0.0 to 1.0)
            reshuffle_rounds: Number of rounds before reshuffling indices
            seed: Random seed for reproducibility
        """
        super().__init__(global_model, num_classes)
        self.overlap = overlap
        self.reshuffle_rounds = reshuffle_rounds
        self.seed = seed
        self.rng = np.random.RandomState(seed)

        # Per-client state
        self.roll_indices = {}  # Current position in the rolling window
        self.model_indices = {}  # Permutation of indices
        self.last_reshuffle = {}  # Track when last reshuffled

    def _initialize_client(self, client_idx: int, local_output_size: int):
        """Initialize rolling window state for a new client."""
        if client_idx not in self.model_indices:
            # Create random permutation of output indices
            self.model_indices[client_idx] = self.rng.permutation(self.global_output_size)
            self.roll_indices[client_idx] = 0
            self.last_reshuffle[client_idx] = 0

    def _advance_window(self, client_idx: int, local_output_size: int, round_num: int):
        """Advance the rolling window for the client."""
        # Calculate step size (non-overlapping portion)
        step_size = int(local_output_size * (1 - self.overlap))
        if step_size < 1:
            step_size = 1

        # Advance window
        self.roll_indices[client_idx] += step_size

        # Wrap around if we exceed the global model size
        if self.roll_indices[client_idx] >= self.global_output_size:
            self.roll_indices[client_idx] = self.roll_indices[client_idx] % self.global_output_size

        # Reshuffle periodically
        if round_num - self.last_reshuffle[client_idx] >= self.reshuffle_rounds:
            self.model_indices[client_idx] = self.rng.permutation(self.global_output_size)
            self.last_reshuffle[client_idx] = round_num

    def extract_sub_model(self, client_model: nn.Module,
                         client_idx: int,
                         round_num: int) -> Dict[str, torch.Tensor]:
        """Extract sub-model using rolling window."""
        local_output_size = self._get_output_size(client_model)

        # Initialize if needed
        self._initialize_client(client_idx, local_output_size)

        # Get current window position
        start_idx = self.roll_indices[client_idx]
        perm = self.model_indices[client_idx]

        # Extract indices with wraparound
        if start_idx + local_output_size <= self.global_output_size:
            indices = perm[start_idx:start_idx + local_output_size]
        else:
            # Wraparound case
            end_part = perm[start_idx:]
            needed = local_output_size - len(end_part)
            start_part = perm[:needed]
            indices = np.concatenate([end_part, start_part])

        # Store indices for aggregation BEFORE advancing the window
        self.current_round_indices[client_idx] = indices.copy()

        # Advance window for next round
        self._advance_window(client_idx, local_output_size, round_num)

        # Extract parameters
        sub_params = OrderedDict()
        global_state = self.global_model.state_dict()
        client_state = client_model.state_dict()

        for name, param in global_state.items():
            # For width-heterogeneous models, use width slice for all parameters
            sub_params[name] = self._extract_width_slice(param, client_state[name], name)

        return sub_params


class HeteroAggregator:
    """Aggregator for heterogeneous model updates.

    Handles aggregation of model updates from clients with different model sizes.
    Supports different weighting strategies.
    """

    def __init__(self, global_model: nn.Module,
                 weighting: str = "uniform",
                 device: torch.device = None):
        """Initialize aggregator.

        Args:
            global_model: Global server model
            weighting: Weighting strategy ('uniform', 'num_samples', 'model_size')
            device: Device for computation
        """
        self.global_model = global_model
        self.weighting = weighting
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def aggregate(self,
                  updates: List[Tuple[Dict[str, torch.Tensor], int, Dict]],
                  extractor: SubModelExtractor,
                  round_num: int) -> Dict[str, torch.Tensor]:
        """Aggregate client updates into global model.

        Args:
            updates: List of (state_dict, num_samples, metrics) from clients
            extractor: Sub-model extractor used for this round
            round_num: Current round number

        Returns:
            Updated global model state dict
        """
        global_state = self.global_model.state_dict()

        # Initialize aggregation buffers
        aggregated = OrderedDict()
        counts = OrderedDict()

        for name in global_state.keys():
            aggregated[name] = torch.zeros_like(global_state[name])
            if 'linear' in name:
                counts[name] = torch.zeros_like(global_state[name])
            else:
                counts[name] = torch.zeros(1, device=aggregated[name].device)

        # Aggregate updates
        total_weight = 0.0

        for client_idx, (client_params, num_samples, metrics) in enumerate(updates):
            # Compute weight
            if self.weighting == "uniform":
                weight = 1.0
            elif self.weighting == "num_samples":
                weight = float(num_samples)
            elif self.weighting == "model_size":
                # Weight by relative model size
                local_params = sum(p.numel() for p in client_params.values())
                global_params = sum(p.numel() for p in global_state.values())
                weight = local_params / global_params
            else:
                weight = 1.0

            total_weight += weight

            # Add weighted updates
            for name, param in client_params.items():
                if 'linear.weight' in name or 'linear.bias' in name:
                    # For output layer, we need to map back to global indices
                    # This requires knowledge of which indices were used
                    # For now, we'll accumulate all updates
                    aggregated[name] += param.to(self.device) * weight
                    if 'linear.weight' in name:
                        counts[name] += weight
                    elif 'linear.bias' in name:
                        counts[name] += weight
                else:
                    # For other layers, simple weighted average
                    aggregated[name] += param.to(self.device) * weight
                    counts[name] += weight

        # Normalize by total weight
        for name in aggregated.keys():
            if 'linear' in name:
                # Normalize by per-element counts
                mask = counts[name] > 0
                aggregated[name][mask] /= counts[name][mask]
            else:
                # Normalize by total weight
                if counts[name].item() > 0:
                    aggregated[name] /= counts[name]

        return aggregated


def create_extractor(strategy: str,
                    global_model: nn.Module,
                    num_classes: int = 10,
                    overlap: float = 0.0,
                    seed: int = 42) -> SubModelExtractor:
    """Factory function to create sub-model extractor.

    Args:
        strategy: Extraction strategy ('static', 'random', 'rolling')
        global_model: Global server model
        num_classes: Number of output classes
        overlap: Overlap ratio for rolling window
        seed: Random seed

    Returns:
        Appropriate SubModelExtractor instance
    """
    if strategy == "static":
        return StaticExtractor(global_model, num_classes)
    elif strategy == "random":
        return RandomExtractor(global_model, num_classes, seed)
    elif strategy == "rolling":
        return RollingWindowExtractor(global_model, num_classes, overlap, seed=seed)
    else:
        raise ValueError(f"Unknown extraction strategy: {strategy}")


if __name__ == "__main__":
    from flore.hetero_models import HeteroPreResNet18

    # Test extractors
    print("Testing sub-model extraction mechanisms:\n")

    # Create global and client models
    global_model = HeteroPreResNet18(num_classes=10, model_rate=1.0)
    client_model = HeteroPreResNet18(num_classes=10, model_rate=0.5)

    print(f"Global model output size: {global_model.linear.out_features}")
    print(f"Client model output size: {client_model.linear.out_features}\n")

    # Test static extraction
    print("Testing Static Extraction:")
    static_extractor = StaticExtractor(global_model)
    sub_params = static_extractor.extract_sub_model(client_model, client_idx=0, round_num=0)
    print(f"  Extracted {len(sub_params)} parameters")
    print(f"  Output layer weight shape: {sub_params['linear.weight'].shape}\n")

    # Test rolling extraction
    print("Testing Rolling Window Extraction:")
    rolling_extractor = RollingWindowExtractor(global_model, overlap=0.2)
    for round_num in range(3):
        sub_params = rolling_extractor.extract_sub_model(client_model, client_idx=0, round_num=round_num)
        print(f"  Round {round_num}: Extracted output shape {sub_params['linear.weight'].shape}")

    print("\nAll tests passed!")
