"""Flower Client for Model-Heterogeneous Federated Learning on Language Models."""

import torch
import torch.nn as nn
from collections import OrderedDict
from logging import INFO

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.logger import log

from .hetero_gpt2 import create_hetero_lm_model
from .dataset_lm import load_lm_partition_data
from .hetero_utils import get_model_rate_for_partition, parse_rate_config
from .utils import context_to_easydict


class HeteroLMClient(NumPyClient):
    """Client for heterogeneous federated language modeling."""

    def __init__(
        self,
        partition_id: int,
        config: dict,
    ):
        self.partition_id = partition_id
        self.config = config

        # Get model rate for this client
        num_clients = config.get("num_clients", 100)
        model_rates, rate_distribution = parse_rate_config(
            config.get("model_rates", None),
            config.get("rate_distribution", None),
        )
        self.model_rates = model_rates
        self.rate_distribution = rate_distribution
        self.model_rate = get_model_rate_for_partition(
            partition_id,
            num_clients,
            model_rates=model_rates,
            rate_distribution=rate_distribution,
        )

        # Model configuration
        self.model_type = config.get("model_type", "gpt2")
        self.vocab_size = config.get("dataset", {}).get("vocab_size", 10000)
        self.max_seq_len = config.get("dataset", {}).get("max_seq_len", 256)

        # Create model
        self.model = self._create_model()

        log(INFO, f"Client partition_id={partition_id}: model_rate={self.model_rate:.2f}, "
            f"params={self.model.get_num_params():,}")

        # Load data
        self.trainloader, self.valloader = load_lm_partition_data(partition_id, config)

        # Training config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.local_epochs = config.get("local_epochs", 1)

    def _create_model(self) -> nn.Module:
        """Create model with appropriate rate."""
        return create_hetero_lm_model(
            model_type=self.model_type,
            model_rate=self.model_rate,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            model_config=self.config.get("model", {}),
            use_scaler=True,
        )

    def get_parameters(self, config):
        """Get model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters from server."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.tensor(v)
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Train model on local data."""
        current_round = config.get("current_round", 0)

        # Round 1: Discovery - just return capabilities
        if current_round == 1:
            log(INFO, f"Round 1 (Discovery): partition_id={self.partition_id}, "
                f"model_rate={self.model_rate:.2f}")

            return self.get_parameters(config), len(self.trainloader.dataset), {
                "partition_id": self.partition_id,
                "model_rate": self.model_rate,
                "discovery": True,
            }

        # Round 2+: Actual training
        self.set_parameters(parameters)

        learning_rate = float(config.get("learning_rate", self.config.get("learning_rate", 0.0001)))
        weight_decay = float(self.config.get("weight_decay", 0.01))

        self.model.to(self.device)
        self.model.train()

        # Use AdamW for Transformer models
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        total_loss = 0.0
        num_batches = 0

        for epoch in range(self.local_epochs):
            for batch in self.trainloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                optimizer.zero_grad()
                logits, loss = self.model(input_ids, labels=labels)

                loss.backward()
                # Gradient clipping for Transformers
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)

        return self.get_parameters(config), len(self.trainloader.dataset), {
            "partition_id": self.partition_id,
            "model_rate": self.model_rate,
            "train_loss": avg_loss,
        }

    def evaluate(self, parameters, config):
        """Evaluate model on validation data."""
        self.set_parameters(parameters)
        self.model.to(self.device)
        self.model.eval()

        total_loss = 0.0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.valloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits, loss = self.model(input_ids, labels=labels)

                non_pad_mask = labels != -100
                num_tokens = non_pad_mask.sum().item()

                total_loss += loss.item() * num_tokens
                total_tokens += num_tokens

        avg_loss = total_loss / max(total_tokens, 1)

        return avg_loss, len(self.valloader.dataset), {
            "partition_id": self.partition_id,
            "model_rate": self.model_rate,
        }


def client_fn(context: Context):
    """Create a Flower client for language modeling."""
    configs = context_to_easydict(context)
    run_config = configs.run_config
    partition_id = int(context.node_config["partition-id"])

    return HeteroLMClient(
        partition_id=partition_id,
        config=run_config,
    ).to_client()


app = ClientApp(client_fn=client_fn)
