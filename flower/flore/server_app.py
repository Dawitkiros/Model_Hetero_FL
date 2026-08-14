"""Flower Server for Model-Heterogeneous Federated Learning."""

import json
import os
import time
import torch
from typing import Callable, Dict, Optional, Tuple, List
from collections import OrderedDict

from flwr.server import ServerApp, ServerAppComponents, ServerConfig, SimpleClientManager
from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.common.typing import NDArrays, Scalar
from torch.utils.data import DataLoader

from .strategies.hetero_strategy import (
    create_heterofl_strategy,
    create_random_strategy,
)
from .strategies.fedrolex_mapped_strategy import create_fedrolex_mapped_strategy
from .strategies.fiarse_strategy import create_fiarse_strategy
from .strategies.flore_strategy import create_flore_strategy
from .hetero_models import create_hetero_model, ScalerV2, calibrate_bn_stats
from .hetero_utils import create_extractor
from .models import test
from .dataset import load_test_data, load_calibration_data
from .server import ResultsSaverServer, history_saver
from .utils import context_to_easydict


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    run_config: dict,
    calibration_loader: Optional[DataLoader] = None,
    start_time: Optional[float] = None,
    round_stats_provider: Optional[Callable[[], Dict[str, Scalar]]] = None,
) -> Callable[[int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]]:
    """Generate evaluation function for heterogeneous FL.

    Args:
        testloader: Test data loader
        device: Device for evaluation
        run_config: Run configuration
        calibration_loader: Optional DataLoader for BN calibration (HeteroFL paper Section 3.2)
        start_time: Experiment start time for wall clock logging
        round_stats_provider: Optional callback returning current round sampling stats

    Returns:
        Evaluation function
    """
    # Initialize start time if not provided
    if start_time is None:
        start_time = time.time()

    norm_type = str(run_config.get("norm_type", "sbn")).lower()
    core_model_rate = float(run_config.get("core_model_rate", 0.25))
    bn_recalibrate_eval = bool(run_config.get("bn_recalibrate_eval", True))
    diag_core_only_full_eval = bool(run_config.get("diag_core_only_full_eval", False))

    def _load_state_dict_from_ndarrays(net: torch.nn.Module, parameters_ndarrays: NDArrays) -> OrderedDict:
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict: OrderedDict = OrderedDict()
        for k, v in params_dict:
            if k.endswith("num_batches_tracked"):
                state_dict[k] = torch.tensor(v, dtype=torch.long)
            else:
                state_dict[k] = torch.tensor(v, dtype=torch.float32)
        net.load_state_dict(state_dict, strict=True)
        return state_dict

    def _set_scaler_eval_mode(net: torch.nn.Module, force_scale_in_eval: bool) -> None:
        """Force ScalerV2 modules to apply scaling during eval if requested."""
        for module in net.modules():
            if isinstance(module, ScalerV2):
                module.always_scale = force_scale_in_eval

    def _extract_core_state(
        full_model: torch.nn.Module,
        core_model: torch.nn.Module,
        server_round: int,
    ) -> "OrderedDict[str, torch.Tensor]":
        """Extract core state via the same static extractor path used for clients."""
        extractor = create_extractor(
            strategy="static",
            global_model=full_model,
            num_classes=run_config.model.num_classes,
            seed=int(run_config.get("rnd_seed", 42)),
        )
        state_via_strategy = extractor.extract_sub_model(
            client_model=core_model,
            client_idx=0,
            round_num=server_round,
        )
        return OrderedDict((k, v.detach().cpu()) for k, v in state_via_strategy.items())

    def _build_full_core_only_state(
        full_state: "OrderedDict[str, torch.Tensor]",
        core_state: "OrderedDict[str, torch.Tensor]",
    ) -> "OrderedDict[str, torch.Tensor]":
        """Embed core parameters into full-sized tensors and zero shell channels."""
        gated_state: OrderedDict = OrderedDict()
        for key, full_tensor in full_state.items():
            base = full_tensor.detach().clone()
            if key not in core_state:
                gated_state[key] = base
                continue
            core_tensor = core_state[key]
            if base.shape == core_tensor.shape:
                gated_state[key] = core_tensor.detach().clone().to(base.device, dtype=base.dtype)
                continue

            # For shell channels, zero everything by default.
            # running_var shell is set to 1 for numeric stability.
            if key.endswith("running_var"):
                base.fill_(1.0)
            else:
                base.zero_()
            slices = tuple(slice(0, d) for d in core_tensor.shape)
            base[slices] = core_tensor.detach().to(base.device, dtype=base.dtype)
            gated_state[key] = base
        return gated_state

    def evaluate(
        server_round: int,
        parameters_ndarrays: NDArrays,
        config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Evaluate global model on test data.

        Note: Round 1 is capability discovery (no training), so we adjust
        the "training_round" for logging purposes:
        - server_round 1 = discovery (training_round 0)
        - server_round 2 = first training round (training_round 1)
        """
        # Adjust round number for logging (Round 1 is discovery, not training)
        training_round = max(0, server_round - 1)

        # Create global model with full width
        net = create_hetero_model(
            run_config.model.name,
            in_channels=3,
            num_classes=run_config.model.num_classes,
            model_rate=run_config.get("global_model_rate", 1.0),
            norm_type=norm_type,
        )

        # Load full-width parameters
        full_state = _load_state_dict_from_ndarrays(net, parameters_ndarrays)

        net.to(device)

        # Evaluate core sub-model extracted from full global parameters.
        # Use the same static extractor path as server->client dispatch.
        core_net = create_hetero_model(
            run_config.model.name,
            in_channels=3,
            num_classes=run_config.model.num_classes,
            model_rate=core_model_rate,
            norm_type=norm_type,
        )
        core_force_scaler_eval = bool(run_config.get("core_force_scaler_eval", True))
        core_state = _extract_core_state(net, core_net, server_round)
        core_net.load_state_dict(core_state, strict=True)
        core_net.to(device)

        # Calibrate BN statistics before evaluation (HeteroFL paper Section 3.2)
        # This is critical for Static Batch Normalization to work correctly
        if (
            norm_type == "sbn"
            and bn_recalibrate_eval
            and calibration_loader is not None
            and server_round > 1
        ):
            calibrate_bn_stats(net, calibration_loader, device, num_batches=50)

        # Evaluate full model.
        # When global_model_rate < 1.0 (e.g. homo_small), the global model
        # has active ScalerV2 modules.  Without forcing them in eval mode the
        # activations are 4× too small and accuracy collapses to random.
        global_model_rate = float(run_config.get("global_model_rate", 1.0))
        _set_scaler_eval_mode(net, core_force_scaler_eval and global_model_rate < 1.0)
        loss_full, acc_full = test(net, testloader, device=device)

        if (
            norm_type == "sbn"
            and bn_recalibrate_eval
            and calibration_loader is not None
            and server_round > 1
        ):
            calibrate_bn_stats(core_net, calibration_loader, device, num_batches=50)

        _set_scaler_eval_mode(core_net, core_force_scaler_eval)
        loss_core, acc_core = test(core_net, testloader, device=device)
        delta_acc = acc_full - acc_core

        acc_core_only_full = float("nan")
        loss_core_only_full = float("nan")
        if diag_core_only_full_eval:
            full_core_only_net = create_hetero_model(
                run_config.model.name,
                in_channels=3,
                num_classes=run_config.model.num_classes,
                model_rate=run_config.get("global_model_rate", 1.0),
                norm_type=norm_type,
            )
            full_core_only_state = _build_full_core_only_state(full_state, core_state)
            full_core_only_net.load_state_dict(full_core_only_state, strict=True)
            full_core_only_net.to(device)
            if (
                norm_type == "sbn"
                and bn_recalibrate_eval
                and calibration_loader is not None
                and server_round > 1
            ):
                calibrate_bn_stats(full_core_only_net, calibration_loader, device, num_batches=50)
            _set_scaler_eval_mode(full_core_only_net, core_force_scaler_eval)
            loss_core_only_full, acc_core_only_full = test(full_core_only_net, testloader, device=device)

        sampling_stats: Dict[str, Scalar] = {
            "n_full_sampled": float("nan"),
            "rounds_since_last_full": float("nan"),
            "p_full_target": float("nan"),
            "shell_repair_applied": 0.0,
            "shell_repair_skipped_recent_full": 1.0,
            "shell_repair_threshold": float(run_config.get("shell_repair_threshold", 0)),
            "shell_repair_beta": float(run_config.get("shell_repair_beta", 0.5)),
            "shell_repair_noise_std": float(run_config.get("shell_repair_noise_std", 0.0)),
            "shell_repair_shell_l2": 0.0,
            "shell_repair_num_tensors": 0.0,
        }
        if round_stats_provider is not None:
            try:
                provided = round_stats_provider() or {}
                sampling_stats.update({
                    "n_full_sampled": float(provided.get("n_full_sampled", float("nan"))),
                    "rounds_since_last_full": float(provided.get("rounds_since_last_full", float("nan"))),
                    "p_full_target": float(provided.get("p_full_target", float("nan"))),
                    "shell_repair_applied": float(provided.get("shell_repair_applied", 0.0)),
                    "shell_repair_skipped_recent_full": float(provided.get("shell_repair_skipped_recent_full", 1.0)),
                    "shell_repair_threshold": float(provided.get("shell_repair_threshold", sampling_stats["shell_repair_threshold"])),
                    "shell_repair_beta": float(provided.get("shell_repair_beta", sampling_stats["shell_repair_beta"])),
                    "shell_repair_noise_std": float(provided.get("shell_repair_noise_std", sampling_stats["shell_repair_noise_std"])),
                    "shell_repair_shell_l2": float(provided.get("shell_repair_shell_l2", 0.0)),
                    "shell_repair_num_tensors": float(provided.get("shell_repair_num_tensors", 0.0)),
                })
            except Exception:
                pass

        # Log to WandB if enabled
        if run_config.get("wandb_enabled", False):
            try:
                import wandb

                # Calculate wall clock time (in seconds and hours)
                wall_clock_sec = time.time() - start_time
                wall_clock_hours = wall_clock_sec / 3600.0

                # Round 1 is discovery - log as round 0 with special marker
                if server_round == 1:
                    discovery_payload = {
                        "discovery/acc_full": acc_full,
                        "discovery/acc_core": acc_core,
                        "discovery/delta_acc": delta_acc,
                        "discovery/n_full_sampled": sampling_stats["n_full_sampled"],
                        "discovery/rounds_since_last_full": sampling_stats["rounds_since_last_full"],
                        "discovery/p_full_target": sampling_stats["p_full_target"],
                        "repair/applied": sampling_stats["shell_repair_applied"],
                        "repair/skipped_recent_full": sampling_stats["shell_repair_skipped_recent_full"],
                        "repair/threshold": sampling_stats["shell_repair_threshold"],
                        "repair/beta": sampling_stats["shell_repair_beta"],
                        "repair/noise_std": sampling_stats["shell_repair_noise_std"],
                        "repair/shell_l2": sampling_stats["shell_repair_shell_l2"],
                        "repair/num_tensors": sampling_stats["shell_repair_num_tensors"],
                        "server_round": server_round,
                        "training_round": 0,
                        "phase": "discovery",
                        "wall_clock_sec": wall_clock_sec,
                        "wall_clock_hours": wall_clock_hours,
                        # Metrics with wall_clock_sec as x-axis
                        "test_vs_time/accuracy": acc_full,
                        "test_vs_time/loss": loss_full,
                        "test_vs_time/accuracy_core": acc_core,
                        "test_vs_time/delta_acc": delta_acc,
                    }
                    if diag_core_only_full_eval:
                        discovery_payload["discovery/acc_core_only_full"] = acc_core_only_full
                        discovery_payload["discovery/loss_core_only_full"] = loss_core_only_full
                        discovery_payload["discovery/core_only_full_minus_core"] = acc_core_only_full - acc_core
                    wandb.log(discovery_payload, step=0)
                else:
                    # Calculate learning rate for this round
                    # Decay schedule depends on partitioning type
                    dataset_config = run_config.get("dataset", {})
                    partitioning = dataset_config.get("partitioning", "dirichlet").lower() if isinstance(dataset_config, dict) else getattr(dataset_config, "partitioning", "dirichlet").lower()
                    if partitioning == "iid":
                        decay_milestones = [151, 251]  # Adjusted for discovery round
                    else:
                        decay_milestones = [801, 1501]  # Adjusted for discovery round

                    base_lr = run_config.get("learning_rate", 0.0002)
                    lr = base_lr
                    for milestone in decay_milestones:
                        if server_round >= milestone:
                            lr *= 0.1

                    train_payload = {
                        "test/accuracy": acc_full,
                        "test/accuracy_core": acc_core,
                        "test/delta_acc": delta_acc,
                        "sampling/n_full_sampled": sampling_stats["n_full_sampled"],
                        "sampling/rounds_since_last_full": sampling_stats["rounds_since_last_full"],
                        "sampling/p_full_target": sampling_stats["p_full_target"],
                        "repair/applied": sampling_stats["shell_repair_applied"],
                        "repair/skipped_recent_full": sampling_stats["shell_repair_skipped_recent_full"],
                        "repair/threshold": sampling_stats["shell_repair_threshold"],
                        "repair/beta": sampling_stats["shell_repair_beta"],
                        "repair/noise_std": sampling_stats["shell_repair_noise_std"],
                        "repair/shell_l2": sampling_stats["shell_repair_shell_l2"],
                        "repair/num_tensors": sampling_stats["shell_repair_num_tensors"],
                        "train/learning_rate": lr,
                        "server_round": server_round,
                        "training_round": training_round,
                        "phase": "training",
                        "wall_clock_sec": wall_clock_sec,
                        "wall_clock_hours": wall_clock_hours,
                        # Metrics with wall_clock_sec as x-axis
                        "test_vs_time/accuracy": acc_full,
                        "test_vs_time/loss": loss_full,
                        "test_vs_time/accuracy_core": acc_core,
                        "test_vs_time/delta_acc": delta_acc,
                    }
                    if diag_core_only_full_eval:
                        train_payload["test/accuracy_core_only_full"] = acc_core_only_full
                        train_payload["test/loss_core_only_full"] = loss_core_only_full
                        train_payload["test/core_only_full_minus_core"] = acc_core_only_full - acc_core
                    wandb.log(train_payload, step=training_round)
            except ImportError:
                pass

        metrics_out: Dict[str, Scalar] = {
            "accuracy": acc_full,
            "acc_full": acc_full,
            "acc_core": acc_core,
            "delta_acc": delta_acc,
            "n_full_sampled": sampling_stats["n_full_sampled"],
            "rounds_since_last_full": sampling_stats["rounds_since_last_full"],
            "p_full_target": sampling_stats["p_full_target"],
            "shell_repair_applied": sampling_stats["shell_repair_applied"],
            "shell_repair_skipped_recent_full": sampling_stats["shell_repair_skipped_recent_full"],
            "shell_repair_threshold": sampling_stats["shell_repair_threshold"],
            "shell_repair_beta": sampling_stats["shell_repair_beta"],
            "shell_repair_noise_std": sampling_stats["shell_repair_noise_std"],
            "shell_repair_shell_l2": sampling_stats["shell_repair_shell_l2"],
            "shell_repair_num_tensors": sampling_stats["shell_repair_num_tensors"],
        }
        if diag_core_only_full_eval:
            metrics_out["acc_core_only_full"] = acc_core_only_full
            metrics_out["loss_core_only_full"] = loss_core_only_full
            metrics_out["core_only_full_minus_core"] = acc_core_only_full - acc_core
        return loss_full, metrics_out

    return evaluate


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Weighted average of accuracy metric for evaluation."""
    accuracies = [num_examples * float(m["accuracy"]) for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)} if examples else {"accuracy": 0.0}


def fit_metrics_aggregation(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Aggregate fit metrics from clients.

    Collects useful training statistics including:
    - Total examples trained
    - Model rate distribution (for heterogeneous FL)
    - Discovery phase indicator
    """
    if not metrics:
        return {}

    total_examples = sum(num_examples for num_examples, _ in metrics)
    num_clients = len(metrics)

    # Check if this is a discovery round
    is_discovery = any(m.get("discovery", False) for _, m in metrics)

    if is_discovery:
        # Discovery round - collect model rate distribution
        rate_counts: Dict[float, int] = {}
        for _, m in metrics:
            rate = float(m.get("model_rate", 1.0))
            rate_counts[rate] = rate_counts.get(rate, 0) + 1

        out = {
            "discovery": True,
            "num_clients_discovered": num_clients,
        }
        for rate, count in sorted(rate_counts.items()):
            out[f"rate_{rate:.2f}"] = count
        return out

    # Training round - aggregate training stats
    # Collect model rates used in this round
    rate_counts: Dict[float, int] = {}
    for _, m in metrics:
        rate = float(m.get("model_rate", 1.0))
        rate_counts[rate] = rate_counts.get(rate, 0) + 1

    out = {
        "total_examples": total_examples,
        "num_clients": num_clients,
        "avg_examples_per_client": total_examples / num_clients if num_clients > 0 else 0,
        "n_full_sampled": float(rate_counts.get(1.0, 0)),
    }
    if rate_counts:
        min_rate = min(rate_counts)
        out["n_small_sampled"] = float(rate_counts.get(min_rate, 0))
    for rate, count in sorted(rate_counts.items()):
        out[f"rate_{rate:.2f}"] = count
    return out


def create_lr_schedule_fn(base_lr: float, decay_schedule: List[int], decay_factor: float = 0.1):
    """Create learning rate schedule function based on FedRolex paper.

    Note: Round 1 is capability discovery (no training), so learning rate
    schedule is based on training_round = server_round - 1.
    Decay milestones are adjusted accordingly (e.g., [801, 1501] for server rounds).

    Args:
        base_lr: Initial learning rate (e.g., 0.0002)
        decay_schedule: List of training rounds to decay LR (e.g., [800, 1500])
        decay_factor: Factor to multiply LR by at each decay step (default: 0.1)

    Returns:
        Function that returns config dict with current learning rate
    """
    # Adjust decay schedule to account for discovery round
    # training_round 800 = server_round 801
    adjusted_schedule = [milestone + 1 for milestone in decay_schedule]

    def fit_config(server_round: int) -> Dict[str, Scalar]:
        """Return training configuration with learning rate schedule."""
        # Round 1 is discovery - return base LR (not used anyway)
        if server_round == 1:
            return {
                "current_round": server_round,
                "learning_rate": base_lr,
            }

        lr = base_lr

        # Apply decay based on which milestones have been passed
        for milestone in adjusted_schedule:
            if server_round >= milestone:
                lr *= decay_factor

        return {
            "current_round": server_round,
            "learning_rate": lr,
        }

    return fit_config


def server_fn(context: Context):
    """Create Flower server for model-heterogeneous FL.

    Args:
        context: Flower context

    Returns:
        Server app components
    """
    # Parse configuration to check algorithm type
    configs = context_to_easydict(context)
    run_config = configs.run_config
    algorithm = run_config.algorithm.name.lower()

    # Dispatch to LM server for language modeling algorithms
    if algorithm in ["heterofl_lm", "stackoverflow", "lm"]:
        from .server_app_lm import server_fn as lm_server_fn
        return lm_server_fn(context)

    # Record experiment start time for wall clock logging
    experiment_start_time = time.time()

    # Extract algorithm and parameters
    num_rounds = int(run_config.num_server_rounds)
    num_clients = int(run_config.num_clients)
    clients_per_round = int(run_config.clients_per_round)

    # Set random seed for reproducibility
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
    print("Model-Heterogeneous Federated Learning Experiment")
    print("=" * 70)
    print(json.dumps(run_config, indent=2))
    print("=" * 70)
    print(f"Algorithm: {algorithm}")
    print(f"Random Seed: {rnd_seed}")
    print(f"Total Rounds: {num_rounds}")
    print(f"  - Round 1: Capability Discovery (no training)")
    print(f"  - Rounds 2-{num_rounds}: Actual Training ({num_rounds - 1} training rounds)")
    print(f"Clients: {num_clients} total")
    print(f"  - Round 1: ALL {num_clients} clients for discovery")
    print(f"  - Round 2+: {clients_per_round} clients per round")
    print("=" * 70)

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create global model and initial parameters
    global_model = create_hetero_model(
        run_config.model.name,
        in_channels=3,
        num_classes=run_config.model.num_classes,
        model_rate=run_config.get("global_model_rate", 1.0),
        norm_type=run_config.get("norm_type", "sbn"),
    )

    if os.environ.get("NET2NET_INIT") == "1":
        from .net2wider import create_shell_repair_state, build_repaired_state_dict
        _cr = float(run_config.get("core_model_rate", 0.25))
        _ns = float(os.environ.get("NET2NET_NOISE", "0.001"))
        _rs = create_shell_repair_state(seed=int(run_config.get("rnd_seed", 42)))
        _repaired, _ = build_repaired_state_dict(
            global_model.state_dict(), _rs, core_model_rate=_cr, noise_std=_ns)
        global_model.load_state_dict(_repaired)
        print(f"[net2net] function-preserving shell init applied (core_rate={_cr}, noise_std={_ns})")

    ndarrays = [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    parameters = ndarrays_to_parameters(ndarrays)

    # Load test data and calibration data for BN calibration
    testloader = load_test_data(run_config)
    norm_type = str(run_config.get("norm_type", "sbn")).lower()
    bn_recalibrate_eval = bool(run_config.get("bn_recalibrate_eval", True))
    calibration_loader = None
    if norm_type == "sbn" and bn_recalibrate_eval:
        calibration_loader = load_calibration_data(run_config, num_samples=5000)
    strategy = None

    def round_sampling_stats_provider() -> Dict[str, Scalar]:
        if strategy is None:
            return {}
        get_stats = getattr(strategy, "get_round_sampling_metrics", None)
        if callable(get_stats):
            return get_stats()
        return {}

    evaluate_fn = gen_evaluate_fn(
        testloader,
        device,
        run_config,
        calibration_loader,
        experiment_start_time,
        round_sampling_stats_provider,
    )

    # Initialize WandB if enabled
    if run_config.get("wandb_enabled", False):
        try:
            import wandb
            wandb.init(
                project=run_config.get("wandb_project", "model-hetero-fl"),
                name=run_config.get("wandb_name", f"{algorithm}_experiment"),
                config=run_config
            )
            # Define custom x-axis metrics for wall clock charts
            # This creates automatic charts: accuracy vs wall_clock_sec
            wandb.define_metric("wall_clock_sec")
            wandb.define_metric("wall_clock_hours")
            wandb.define_metric("test_vs_time/*", step_metric="wall_clock_sec")
            print("[Info] WandB initialized successfully")
        except ImportError:
            print("[Warning] WandB not installed, skipping WandB logging")

    # Create learning rate schedule function
    # Decay schedule depends on partitioning type:
    # - IID: decay at [150, 250] (shorter training)
    # - Dirichlet (non-IID): decay at [800, 1500] (FedRolex paper)
    dataset_cfg = run_config.get("dataset", {})
    partitioning = (dataset_cfg.get("partitioning", "dirichlet") if isinstance(dataset_cfg, dict)
                    else getattr(dataset_cfg, "partitioning", "dirichlet")).lower()
    if partitioning == "iid":
        decay_schedule = [150, 250]
    else:
        # Dirichlet or other non-IID partitioning
        decay_schedule = [800, 1500]

    lr_schedule_fn = create_lr_schedule_fn(
        base_lr=run_config.get("learning_rate", 0.0002),
        decay_schedule=decay_schedule,
        decay_factor=0.1
    )
    adjusted_schedule = [m + 1 for m in decay_schedule]  # Account for discovery round
    print(f"Learning rate schedule:")
    print(f"  - Base LR: {run_config.get('learning_rate', 0.0002)}")
    print(f"  - Partitioning: {partitioning}")
    print(f"  - Decay at training rounds {decay_schedule} (server rounds {adjusted_schedule})")
    print(f"  - Decay factor: 0.1")

    # Create strategy based on algorithm
    if algorithm == "heterofl":
        strategy = create_heterofl_strategy(
            evaluate_fn=evaluate_fn,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_classes=run_config.model.num_classes,
            model_name=run_config.model.name,
            global_model_rate=run_config.get("global_model_rate", 1.0),
            model_split_mode=run_config.get("model_split_mode", "fix"),
            model_rates=run_config.get("model_rates", None),
            rate_distribution=run_config.get("rate_distribution", None),
            norm_type=run_config.get("norm_type", "sbn"),
            core_model_rate=run_config.get("core_model_rate", 0.25),
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            enable_shell_repair=run_config.get("enable_shell_repair", False),
            shell_repair_threshold=run_config.get("shell_repair_threshold", 0),
            shell_repair_beta=run_config.get("shell_repair_beta", 0.5),
            shell_repair_noise_std=run_config.get("shell_repair_noise_std", 0.0),
            shell_repair_seed=run_config.get("rnd_seed", 42),
        )
        print("Using HeteroFL strategy (static extraction)")

    elif algorithm == "flore":
        strategy = create_flore_strategy(
            evaluate_fn=evaluate_fn,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_classes=run_config.model.num_classes,
            model_name=run_config.model.name,
            global_model_rate=run_config.get("global_model_rate", 1.0),
            model_split_mode=run_config.get("model_split_mode", "fix"),
            model_rates=run_config.get("model_rates", None),
            rate_distribution=run_config.get("rate_distribution", None),
            norm_type=run_config.get("norm_type", "sbn"),
            core_model_rate=run_config.get("core_model_rate", 0.25),
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            lift_scale=run_config.get("lift_scale", 1.0),
            enable_shell_repair=run_config.get("enable_shell_repair", False),
            shell_repair_threshold=run_config.get("shell_repair_threshold", 0),
            shell_repair_beta=run_config.get("shell_repair_beta", 0.5),
            shell_repair_noise_std=run_config.get("shell_repair_noise_std", 0.0),
            shell_repair_seed=run_config.get("rnd_seed", 42),
        )
        print(
            "Using FLoRE strategy "
            f"(pre-aggregation lifted deltas, lift_scale={run_config.get('lift_scale', 1.0)})"
        )

    elif algorithm == "fedrolex":
        # Canonical FedRolex path: index-mapped rolling extraction/aggregation.
        strategy = create_fedrolex_mapped_strategy(
            evaluate_fn=evaluate_fn,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_classes=run_config.model.num_classes,
            global_model_rate=run_config.get("global_model_rate", 1.0),
            model_split_mode=run_config.get("model_split_mode", "fix"),
            overlap=(None if run_config.get("overlap", 0.0) < 0 else run_config.get("overlap", 0.0)),
            weighting=run_config.get("weighting", "avg"),
            reshuffle_rounds=run_config.get("reshuffle_rounds", 512),
            model_rates=run_config.get("model_rates", None),
            rate_distribution=run_config.get("rate_distribution", None),
            core_model_rate=run_config.get("core_model_rate", 0.25),
            norm_type=run_config.get("norm_type", "sbn"),
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            enable_shell_repair=run_config.get("enable_shell_repair", False),
            shell_repair_threshold=run_config.get("shell_repair_threshold", 0),
            shell_repair_beta=run_config.get("shell_repair_beta", 0.5),
            shell_repair_noise_std=run_config.get("shell_repair_noise_std", 0.0),
            shell_repair_seed=run_config.get("rnd_seed", 42),
        )
        print(
            "Using FedRolex strategy "
            f"(rolling window, overlap={run_config.get('overlap', 0.0)}, "
            f"weighting={run_config.get('weighting', 'avg')})"
        )

    elif algorithm == "fedrolex_mapped":
        print("WARNING: 'fedrolex_mapped' is deprecated; use 'fedrolex'.")
        strategy = create_fedrolex_mapped_strategy(
            evaluate_fn=evaluate_fn,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_classes=run_config.model.num_classes,
            global_model_rate=run_config.get("global_model_rate", 1.0),
            model_split_mode=run_config.get("model_split_mode", "fix"),
            overlap=(None if run_config.get("overlap", 0.0) < 0 else run_config.get("overlap", 0.0)),
            weighting=run_config.get("weighting", "avg"),
            reshuffle_rounds=run_config.get("reshuffle_rounds", 512),
            model_rates=run_config.get("model_rates", None),
            rate_distribution=run_config.get("rate_distribution", None),
            core_model_rate=run_config.get("core_model_rate", 0.25),
            norm_type=run_config.get("norm_type", "sbn"),
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            enable_shell_repair=run_config.get("enable_shell_repair", False),
            shell_repair_threshold=run_config.get("shell_repair_threshold", 0),
            shell_repair_beta=run_config.get("shell_repair_beta", 0.5),
            shell_repair_noise_std=run_config.get("shell_repair_noise_std", 0.0),
            shell_repair_seed=run_config.get("rnd_seed", 42),
        )
        print(
            "Using FedRolex-mapped strategy "
            f"(rolling window, overlap={run_config.get('overlap', 0.0)}, "
            f"weighting={run_config.get('weighting', 'avg')})"
        )

    elif algorithm == "fiarse":
        strategy = create_fiarse_strategy(
            evaluate_fn=evaluate_fn,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_classes=run_config.model.num_classes,
            global_model_rate=run_config.get("global_model_rate", 1.0),
            model_split_mode=run_config.get("model_split_mode", "fix"),
            weighting=run_config.get("weighting", "avg"),
            model_rates=run_config.get("model_rates", None),
            rate_distribution=run_config.get("rate_distribution", None),
            core_model_rate=run_config.get("core_model_rate", 0.25),
            norm_type=run_config.get("norm_type", "sbn"),
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            enable_shell_repair=run_config.get("enable_shell_repair", False),
            shell_repair_threshold=run_config.get("shell_repair_threshold", 0),
            shell_repair_beta=run_config.get("shell_repair_beta", 0.5),
            shell_repair_noise_std=run_config.get("shell_repair_noise_std", 0.0),
            shell_repair_seed=run_config.get("rnd_seed", 42),
        )
        print(
            "Using FIARSE strategy "
            f"(importance-ranked channels, weighting={run_config.get('weighting', 'avg')})"
        )

    elif algorithm == "random":
        strategy = create_random_strategy(
            evaluate_fn=evaluate_fn,
            num_clients=num_clients,
            clients_per_round=clients_per_round,
            num_classes=run_config.model.num_classes,
            global_model_rate=run_config.get("global_model_rate", 1.0),
            model_split_mode=run_config.get("model_split_mode", "fix"),
            model_rates=run_config.get("model_rates", None),
            rate_distribution=run_config.get("rate_distribution", None),
            norm_type=run_config.get("norm_type", "sbn"),
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
        )
        print("Using Random extraction strategy")

    elif algorithm in ["homo_small", "homo_large"]:
        # Homogeneous baselines - use standard FedAvg
        from flwr.server.strategy import FedAvg

        model_rate = 0.25 if algorithm == "homo_small" else 1.0
        print(f"Using Homogeneous baseline with model_rate={model_rate}")

        # For homogeneous, all clients use the same model size
        strategy = FedAvg(
            fraction_fit=0.0,
            fraction_evaluate=0.0,
            min_fit_clients=clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
        )

    elif algorithm == "fst_fedadam":
        # Frozen-shell core + FedAdam server optimizer (debug/ fst_fedadam port).
        # Run the homogeneous core (set global-model-rate = core rate); under a
        # frozen shell acc_full == acc_core, so this measures the method's ceiling.
        from .strategies.fst_fedadam_strategy import FstFedAdamStrategy

        print(f"Using fst_fedadam (FedAdam server-opt on core, "
              f"model_rate={run_config.get('global_model_rate', 0.25)})")
        strategy = FstFedAdamStrategy(
            fraction_fit=0.0,
            fraction_evaluate=0.0,
            min_fit_clients=clients_per_round,
            min_evaluate_clients=0,
            min_available_clients=num_clients,
            evaluate_fn=evaluate_fn,
            initial_parameters=parameters,
            on_fit_config_fn=lr_schedule_fn,
            fit_metrics_aggregation_fn=fit_metrics_aggregation,
            eta=float(os.environ.get("FST_SERVER_LR", run_config.get("server_lr", 0.03))),
            tau=float(os.environ.get("FST_ADAM_TAU", run_config.get("adam_tau", 1e-3))),
            beta_1=float(run_config.get("adam_b1", 0.9)),
            beta_2=float(run_config.get("adam_b2", 0.99)),
            param_keys=list(global_model.state_dict().keys()),
        )

    else:
        raise ValueError(
            f"Unknown algorithm: '{algorithm}'. "
            f"Available: ['heterofl', 'flore', 'fedrolex', 'fedrolex_mapped' (deprecated), 'fiarse', 'random', 'homo_small', 'homo_large']"
        )

    print("=" * 70)

    # Create server components
    client_manager = SimpleClientManager()
    server = ResultsSaverServer(
        client_manager=client_manager,
        strategy=strategy,
        results_saver_fn=history_saver,
        run_config=run_config,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(server=server, config=config)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)
