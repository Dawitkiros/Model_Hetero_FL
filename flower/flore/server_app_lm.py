"""Flower Server for Model-Heterogeneous Federated Learning on Language Models."""

import json
import time
import math
import torch
from typing import Callable, Dict, Optional, Tuple, List
from collections import OrderedDict

from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from .strategies.hetero_lm_strategy import create_hetero_lm_strategy
from .hetero_gpt2 import create_hetero_lm_model
from .dataset_lm import load_lm_test_data
from .server import ResultsSaverServer, history_saver
from .utils import context_to_easydict


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from cross-entropy loss."""
    return math.exp(min(loss, 20))  # Cap to avoid overflow


def test_lm(model, dataloader, device):
    """Evaluate language model on test data.

    Returns:
        Tuple of (loss, perplexity)
    """
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            logits, loss = model(input_ids, labels=labels)

            # Count non-padding tokens
            non_pad_mask = labels != -100
            num_tokens = non_pad_mask.sum().item()

            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = compute_perplexity(avg_loss)

    return avg_loss, perplexity


def gen_evaluate_fn_lm(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
    start_time: Optional[float] = None,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate evaluation function for language modeling."""

    if start_time is None:
        start_time = time.time()

    def evaluate(
        server_round: int,
        parameters_ndarrays: NDArrays,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test data."""
        training_round = max(0, server_round - 1)

        # Create global model
        model_type = run_config.get("model_type", "gpt2")
        vocab_size = run_config.get("dataset", {}).get("vocab_size", 10000)
        max_seq_len = run_config.get("dataset", {}).get("max_seq_len", 256)

        net = create_hetero_lm_model(
            model_type=model_type,
            model_rate=run_config.get("global_model_rate", 1.0),
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            model_config=run_config.get("model", {}),
            use_scaler=True,
        )

        # Load parameters
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict()
        for k, v in params_dict:
            state_dict[k] = torch.tensor(v)
        net.load_state_dict(state_dict, strict=True)

        # Evaluate
        loss, perplexity = test_lm(net, testloader, device=device)

        # Log to WandB if enabled
        if run_config.get("wandb_enabled", False):
            try:
                import wandb

                wall_clock_sec = time.time() - start_time
                wall_clock_hours = wall_clock_sec / 3600.0

                if server_round == 1:
                    wandb.log({
                        "discovery/test_loss": loss,
                        "discovery/test_perplexity": perplexity,
                        "server_round": server_round,
                        "training_round": 0,
                        "phase": "discovery",
                        "wall_clock_sec": wall_clock_sec,
                        "wall_clock_hours": wall_clock_hours,
                        "test_vs_time/perplexity": perplexity,
                        "test_vs_time/loss": loss,
                    }, step=0)
                else:
                    base_lr = run_config.get("learning_rate", 0.0001)
                    # LR decay for LM (decay at 60% and 80% of training)
                    total_rounds = run_config.get("num_server_rounds", 500)
                    decay_milestones = [int(0.6 * total_rounds), int(0.8 * total_rounds)]

                    lr = base_lr
                    for milestone in decay_milestones:
                        if server_round >= milestone:
                            lr *= 0.1

                    wandb.log({
                        "test/loss": loss,
                        "test/perplexity": perplexity,
                        "train/learning_rate": lr,
                        "server_round": server_round,
                        "training_round": training_round,
                        "phase": "training",
                        "wall_clock_sec": wall_clock_sec,
                        "wall_clock_hours": wall_clock_hours,
                        "test_vs_time/perplexity": perplexity,
                        "test_vs_time/loss": loss,
                    }, step=training_round)
            except ImportError:
                pass

        return loss, {"perplexity": perplexity}

    return evaluate


def fit_metrics_aggregation_lm(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics from clients."""
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    num_clients = len(metrics)

    # Average loss across clients
    total_loss = sum(
        num_examples * m.get("train_loss", 0)
        for num_examples, m in metrics
    )
    avg_loss = total_loss / total_examples if total_examples > 0 else 0

    return {
        "total_examples": total_examples,
        "num_clients": num_clients,
        "avg_train_loss": avg_loss,
        "avg_train_perplexity": compute_perplexity(avg_loss),
    }


def create_lr_schedule_fn_lm(base_lr: float, total_rounds: int):
    """Create learning rate schedule for language modeling."""
    # Decay at 60% and 80% of training
    decay_milestones = [int(0.6 * total_rounds) + 1, int(0.8 * total_rounds) + 1]

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        if server_round == 1:
            return {"current_round": server_round, "learning_rate": base_lr}

        lr = base_lr
        for milestone in decay_milestones:
            if server_round >= milestone:
                lr *= 0.1

        return {"current_round": server_round, "learning_rate": lr}

    return fit_config


def server_fn(context: Context):
    """Create Flower server for language modeling FL."""
    experiment_start_time = time.time()

    configs = context_to_easydict(context)
    run_config = configs.run_config

    algorithm = run_config.algorithm.name.lower()
    num_rounds = int(run_config.num_server_rounds)
    num_clients = int(run_config.num_clients)
    clients_per_round = int(run_config.clients_per_round)

    # Set random seed
    rnd_seed = int(run_config.get("rnd_seed", 42))
    torch.manual_seed(rnd_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(rnd_seed)
        torch.cuda.manual_seed_all(rnd_seed)
    import numpy as np
    import random
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)

    # Print configuration
    print("=" * 70)
    print("HeteroFL Language Modeling Experiment")
    print("=" * 70)
    print(json.dumps(run_config, indent=2))
    print("=" * 70)
    print(f"Algorithm: {algorithm}")
    print(f"Random Seed: {rnd_seed}")
    print(f"Total Rounds: {num_rounds}")
    print(f"Clients: {num_clients} total, {clients_per_round} per round")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model configuration
    model_type = run_config.get("model_type", "gpt2")
    vocab_size = run_config.get("dataset", {}).get("vocab_size", 10000)
    max_seq_len = run_config.get("dataset", {}).get("max_seq_len", 256)

    # Create global model
    global_model = create_hetero_lm_model(
        model_type=model_type,
        model_rate=run_config.get("global_model_rate", 1.0),
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_config=run_config.get("model", {}),
        use_scaler=True,
    )

    print(f"Global model: {model_type}, params={global_model.get_num_params():,}")

    ndarrays = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # Load test data
    testloader = load_lm_test_data(run_config)
    evaluate_fn = gen_evaluate_fn_lm(testloader, device, run_config, experiment_start_time)

    # Initialize WandB
    if run_config.get("wandb_enabled", False):
        try:
            import wandb
            wandb.init(
                project=run_config.get("wandb_project", "hetero-fl-lm"),
                name=run_config.get("wandb_name", f"{algorithm}_lm_experiment"),
                config=run_config
            )
            wandb.define_metric("wall_clock_sec")
            wandb.define_metric("wall_clock_hours")
            wandb.define_metric("test_vs_time/*", step_metric="wall_clock_sec")
            print("[Info] WandB initialized successfully")
        except ImportError:
            print("[Warning] WandB not installed")

    # Create LR schedule
    lr_schedule_fn = create_lr_schedule_fn_lm(
        base_lr=run_config.get("learning_rate", 0.0001),
        total_rounds=num_rounds,
    )

    # Create strategy
    strategy = create_hetero_lm_strategy(
        evaluate_fn=evaluate_fn,
        num_clients=num_clients,
        clients_per_round=clients_per_round,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        model_type=model_type,
        global_model_rate=run_config.get("global_model_rate", 1.0),
        model_rates=run_config.get("model_rates", None),
        rate_distribution=run_config.get("rate_distribution", None),
        model_config=run_config.get("model", {}),
        initial_parameters=parameters,
        on_fit_config_fn=lr_schedule_fn,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_lm,
    )

    print(f"Using HeteroFL strategy for {model_type} language modeling")
    print("=" * 70)

    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=history_saver,
        run_config=run_config,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(server=server, config=config)


app = ServerApp(server_fn=server_fn)
