"""Flower Client for Model-Heterogeneous Federated Learning.

Supports:
- HeteroFL: Static sub-model extraction
- FedRolex: Rolling window sub-model extraction
- Homogeneous baselines
"""

import os
import torch
import torch.nn as nn
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, NDArrays, Scalar
from flwr.common.logger import log
from logging import INFO

from .hetero_models import create_hetero_model, get_model_info
from .models import test, train_fedavg, train_fedavg_adam
from .dataset import load_partition_data
from .utils import context_to_easydict
from .hetero_utils import get_model_rate_for_partition, parse_rate_config


def _build_shell_lr_mask(net, model_name, num_classes, norm_type, core_rate, shell_scale, device):
    """Per-parameter grad mask: 1.0 on the rate-`core_rate` prefix (core) block,
    `shell_scale` elsewhere (shell). Width-scaled dims are detected by comparing each
    tensor's shape to the rate-`core_rate` reference model (same prefix convention as
    HeteroFL extraction). Non-width tensors (e.g. classifier rows) keep full LR."""
    ref = create_hetero_model(model_name, in_channels=3, num_classes=num_classes,
                              model_rate=core_rate, norm_type=norm_type)
    ref_shapes = {k: tuple(v.shape) for k, v in ref.state_dict().items()}
    mask = {}
    for name, p in net.named_parameters():
        full = tuple(p.shape)
        cs = ref_shapes.get(name)
        if cs is None or cs == full:
            continue
        m = torch.full(full, float(shell_scale), device=device)
        m[tuple(slice(0, c) for c in cs)] = 1.0
        mask[name] = m
    return mask


class HeteroClient(NumPyClient):
    """Client for model-heterogeneous federated learning.

    Each client trains a model with potentially different width (number of channels)
    based on their model_rate parameter.
    """

    def __init__(
        self,
        net: nn.Module,
        trainloader,
        valloader,
        device: torch.device,
        context: Context,
        config: Dict,
        model_rate: float,
    ):
        """Initialize heterogeneous client.

        Args:
            net: Local model (size determined by model_rate)
            trainloader: Training data loader
            valloader: Validation data loader
            device: Device to train on
            context: Flower context
            config: Configuration dictionary
            model_rate: Model width multiplier (0.25, 0.5, 0.75, 1.0)
        """
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.device = device
        self.context = context
        self.config = config
        self.model_rate = model_rate
        self.net.to(self.device)

        # Log model info
        info = get_model_info(self.net)
        log(INFO, f"Client {context.node_id}: Model rate={model_rate:.2f}, "
                  f"Params={info['total_params']:,}, Size={info['model_size_mb']:.2f}MB")

    def get_properties(self, config: Dict[str, Scalar]) -> Dict[str, Scalar]:
        """Get client properties including partition_id and model_rate.

        This allows the server to learn client capabilities before training.
        """
        partition_id = int(self.context.node_config.get("partition-id", -1))
        return {
            "partition_id": partition_id,
            "model_rate": self.model_rate,
        }

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Get model parameters as numpy arrays."""
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def set_parameters(self, parameters: NDArrays) -> None:
        """Set model parameters from numpy arrays."""
        params_dict = zip(self.net.state_dict().keys(), parameters)
        state_dict = OrderedDict()

        for k, v in params_dict:
            if k.endswith('num_batches_tracked'):
                tensor = torch.tensor(v, dtype=torch.long)
            else:
                tensor = torch.tensor(v, dtype=torch.float32)
            state_dict[k] = tensor

        self.net.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Train the model on local data.

        Round 1: Capability discovery - no training, just report partition_id and model_rate
        Round 2+: Actual training with heterogeneous models

        Args:
            parameters: Global model parameters (extracted for this client's model size)
            config: Training configuration

        Returns:
            Tuple of (updated_parameters, num_samples, metrics)
        """
        current_round = int(config.get("current_round", 0))
        partition_id = int(self.context.node_config.get("partition-id", -1))

        # Round 1: Capability discovery - no training
        if current_round == 1:
            log(INFO, f"Round 1 (Discovery): partition_id={partition_id}, model_rate={self.model_rate:.2f}")
            # Return current parameters (not used) and capability info
            return (
                self.get_parameters(config={}),
                len(self.trainloader.dataset),
                {
                    "partition_id": partition_id,
                    "model_rate": self.model_rate,
                    "discovery": True,
                }
            )

        # Round 2+: Actual training
        assigned_rate = float(config.get("model_rate", self.model_rate))
        if abs(assigned_rate - self.model_rate) > 1e-12:
            # Keep client model consistent with server-issued rate.
            self.model_rate = assigned_rate
            self.net = create_hetero_model(
                self.config.model.name,
                in_channels=3,
                num_classes=self.config.model.num_classes,
                model_rate=self.model_rate,
                norm_type=self.config.get("norm_type", "sbn"),
            )
            self.net.to(self.device)

        # Set parameters from server (should match our model size)
        self.set_parameters(parameters)

        learning_rate = float(config.get("learning_rate", self.config["learning_rate"]))

        # Net2Net pilot: damp the shell learning rate on full-width clients only.
        shell_mask = None
        shell_lr_scale = float(os.environ.get("SHELL_LR_SCALE", "1.0"))
        if shell_lr_scale < 1.0 and self.model_rate > 0.99:
            if getattr(self, "_shell_mask", None) is None or self._shell_mask_rate != self.model_rate:
                self._shell_mask = _build_shell_lr_mask(
                    self.net, self.config.model.name, self.config.model.num_classes,
                    self.config.get("norm_type", "sbn"),
                    float(self.config.get("core_model_rate", 0.25)),
                    shell_lr_scale, self.device)
                self._shell_mask_rate = self.model_rate
            shell_mask = self._shell_mask

        # Train
        client_optimizer = str(self.config.get("client_optimizer", "sgd")).lower()
        if client_optimizer == "adamw":
            train_fedavg_adam(
                self.net,
                self.trainloader,
                self.device,
                epochs=self.config["local_epochs"],
                global_round=current_round,
                lr=learning_rate,
                weight_decay=self.config["weight_decay"],
                total_rounds=int(self.config.get("num_server_rounds", 500)),
            )
        else:
            train_fedavg(
                self.net,
                self.trainloader,
                self.device,
                epochs=self.config["local_epochs"],
                learning_rate=learning_rate,
                momentum=self.config["momentum"],
                weight_decay=self.config["weight_decay"],
                shell_mask=shell_mask,
            )

        # Return updated parameters
        updated_params = self.get_parameters(config={})
        num_samples = len(self.trainloader.dataset)

        metrics = {
            "model_rate": self.model_rate,
            "round": current_round,
            "partition_id": partition_id,
        }

        return updated_params, num_samples, metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """Evaluate the model on local validation data.

        Args:
            parameters: Model parameters
            config: Evaluation configuration

        Returns:
            Tuple of (loss, num_samples, metrics)
        """
        self.set_parameters(parameters)
        loss, accuracy = test(self.net, self.valloader, self.device)

        return float(loss), len(self.valloader.dataset), {"accuracy": float(accuracy)}


def client_fn(context: Context):
    """Create a Flower client instance.

    Args:
        context: Flower context with configuration

    Returns:
        Flower client
    """
    # Parse configuration
    configs = context_to_easydict(context)
    run_config = configs.run_config

    # Dispatch to LM client for language modeling algorithms
    algorithm = run_config.algorithm.name.lower()
    if algorithm in ["heterofl_lm", "stackoverflow", "lm"]:
        from .client_app_lm import client_fn as lm_client_fn
        return lm_client_fn(context)

    # Get client-specific configuration
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(configs.node_config.num_partitions)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_rates, rate_distribution = parse_rate_config(
        run_config.get("model_rates", None),
        run_config.get("rate_distribution", None),
    )

    # Get model rate for this client based on partition_id.
    model_rate = get_model_rate_for_partition(
        partition_id=partition_id,
        num_clients=num_partitions,
        model_rates=model_rates,
        rate_distribution=rate_distribution,
    )

    # Create model with appropriate width based on assigned rate
    net = create_hetero_model(
        run_config.model.name,
        in_channels=3,
        num_classes=run_config.model.num_classes,
        model_rate=model_rate,
        norm_type=run_config.get("norm_type", "sbn"),
    )

    # Log client configuration
    log(INFO, f"Client partition_id={partition_id}: model_rate={model_rate:.2f}")

    # Load data partition
    trainloader, valloader = load_partition_data(partition_id, run_config)

    # Create and return client
    return HeteroClient(
        net=net,
        trainloader=trainloader,
        valloader=valloader,
        device=device,
        context=context,
        config=run_config,
        model_rate=model_rate,
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn=client_fn)
