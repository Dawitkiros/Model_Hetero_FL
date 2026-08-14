"""FedRolex-mapped strategy based on the original implementation."""

import copy
from collections import OrderedDict
from logging import INFO
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
from flwr.common.typing import NDArrays
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from ..hetero_models import HeteroPreResNet18
from ..hetero_utils import (
    get_model_rate_for_partition,
    parse_rate_config,
    split_core_shell_norm,
)
from ..net2wider import create_shell_repair_state, repair_shell_if_stale


class FedRolexMappedStrategy(FedAvg):
    """Server strategy mirroring FedRolex index-mapped aggregation."""

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
        num_classes: int = 10,
        global_model_rate: float = 1.0,
        model_split_mode: str = "fix",
        num_clients: int = 100,
        overlap: Optional[float] = 0.0,
        weighting: str = "avg",
        reshuffle_rounds: int = 512,
        model_rates: Optional[Union[str, List[float]]] = None,
        rate_distribution: Optional[Union[str, List[float]]] = None,
        core_model_rate: float = 0.25,
        norm_type: str = "sbn",
        enable_shell_repair: bool = False,
        shell_repair_threshold: int = 0,
        shell_repair_beta: float = 0.5,
        shell_repair_noise_std: float = 0.0,
        shell_repair_seed: Optional[int] = None,
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
        self.global_model_rate = float(global_model_rate)
        # Kept for API compatibility with other strategies/configs.
        self.model_split_mode = str(model_split_mode)
        self.num_clients = int(num_clients)
        self.overlap = overlap
        self.weighting = weighting
        self.reshuffle_rounds = int(reshuffle_rounds)
        self.core_model_rate = float(core_model_rate)
        self.enable_shell_repair = bool(enable_shell_repair)
        self.shell_repair_threshold = int(shell_repair_threshold)
        self.shell_repair_beta = float(shell_repair_beta)
        self.shell_repair_noise_std = float(shell_repair_noise_std)
        self.shell_repair_seed = int(42 if shell_repair_seed is None else shell_repair_seed)

        self.model_rates, self.rate_distribution = parse_rate_config(model_rates, rate_distribution)
        self.p_full_target = 0.0
        for rate, prob in zip(self.model_rates, self.rate_distribution):
            if abs(float(rate) - 1.0) < 1e-12:
                self.p_full_target = float(prob)
                break

        self.rounds = 0
        self.time_since_last_full = 0
        self.last_round_metrics: Dict[str, Scalar] = {}

        self.global_model = HeteroPreResNet18(
            in_channels=3,
            num_classes=num_classes,
            model_rate=self.global_model_rate,
            norm_type=norm_type,
        )
        self.global_parameters = self.global_model.state_dict()
        self.param_keys = list(self.global_parameters.keys())

        self.client_partition_map: Dict[str, int] = {}
        self.current_param_idx: Dict[str, OrderedDict] = {}

        self.model_idxs: Dict[str, torch.Tensor] = {}
        self.roll_idx: Dict[str, int] = {}
        self.tmp_counts: Dict[str, torch.Tensor] = {
            k: torch.ones_like(v, dtype=torch.float32) for k, v in self.global_parameters.items()
        }
        self.shell_repair_state = create_shell_repair_state(seed=self.shell_repair_seed)
        self._reshuffle_params()

        log(INFO, "FedRolex-mapped initialized")
        log(INFO, f"  model_rates={self.model_rates}")
        log(INFO, f"  rate_distribution={self.rate_distribution}")
        log(INFO, f"  weighting={self.weighting}, overlap={self.overlap}")
        if self.model_split_mode != "fix":
            log(INFO, f"  model_split_mode={self.model_split_mode} (treated as fix in mapped strategy)")

    @staticmethod
    def _mesh(index_tuple: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.meshgrid(index_tuple[0], index_tuple[1], indexing="ij")

    @staticmethod
    def _shortcut_conv1_key(key: str) -> str:
        prefix = key.split("shortcut")[0]
        return f"{prefix}conv1.weight"

    def _reshuffle_params(self) -> None:
        for k, v in self.global_parameters.items():
            if k.endswith("conv1.weight") or k.endswith("conv2.weight"):
                output_size = v.size(0)
                self.model_idxs[k] = torch.randperm(output_size, device=v.device)
                self.roll_idx[k] = 0

    def _get_client_model_rate(self, client_id: str, partition_id: Optional[int] = None) -> float:
        if partition_id is not None:
            self.client_partition_map[client_id] = int(partition_id)
        if client_id not in self.client_partition_map:
            raise RuntimeError(f"Missing partition_id mapping for client {client_id}")
        pid = self.client_partition_map[client_id]
        return get_model_rate_for_partition(
            partition_id=pid,
            num_clients=self.num_clients,
            model_rates=self.model_rates,
            rate_distribution=self.rate_distribution,
        )

    def _split_model(self, client_ids: List[str]) -> List[OrderedDict]:
        idx_i = [None for _ in range(len(client_ids))]
        idx = [OrderedDict() for _ in range(len(client_ids))]

        for k, v in self.global_parameters.items():
            parameter_type = k.split(".")[-1]
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

                                if self.overlap is None:
                                    roll = self.rounds % output_size
                                    model_idx = torch.arange(output_size, device=v.device)
                                else:
                                    overlap = float(self.overlap)
                                    step = int(local_output_size * (1 - overlap)) + 1
                                    self.roll_idx[k] = (self.roll_idx[k] + step) % output_size
                                    roll = self.roll_idx[k]
                                    model_idx = self.model_idxs[k]
                                model_idx = torch.roll(model_idx, roll, -1)
                                output_idx = model_idx[:local_output_size]
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

    def _distribute(self, client_ids: List[str], param_idx: List[OrderedDict]) -> List[OrderedDict]:
        local_parameters = [OrderedDict() for _ in range(len(client_ids))]
        for k, v in self.global_parameters.items():
            parameter_type = k.split(".")[-1]
            for m in range(len(client_ids)):
                if "weight" in parameter_type or "bias" in parameter_type:
                    if "weight" in parameter_type:
                        if v.dim() > 1:
                            local_parameters[m][k] = copy.deepcopy(v[self._mesh(param_idx[m][k])])
                        else:
                            local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                else:
                    if k in param_idx[m]:
                        local_parameters[m][k] = copy.deepcopy(v[param_idx[m][k]])
                    else:
                        local_parameters[m][k] = copy.deepcopy(v)
        return local_parameters

    def configure_fit(self, server_round: int, parameters: Parameters, client_manager) -> List[Tuple[ClientProxy, FitIns]]:
        self.rounds = max(server_round - 1, 0)

        global_params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.param_keys, global_params)
        state_dict = OrderedDict()
        for k, v in params_dict:
            if k.endswith("num_batches_tracked"):
                state_dict[k] = torch.tensor(v, dtype=torch.long)
            else:
                state_dict[k] = torch.tensor(v, dtype=torch.float32)
        self.global_model.load_state_dict(state_dict, strict=True)
        self.global_parameters = self.global_model.state_dict()

        if server_round == 1:
            num_available = client_manager.num_available()
            clients = client_manager.sample(num_clients=num_available, min_num_clients=num_available)
            full_params = [v.cpu().numpy() for v in self.global_parameters.values()]

            fit_cfg = []
            for client in clients:
                config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn is not None else {}
                config["current_round"] = server_round
                config["model_rate"] = 1.0
                fit_cfg.append((client, FitIns(ndarrays_to_parameters(full_params), config)))

            self.current_param_idx = {}
            return fit_cfg

        sample_size, min_num_clients = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(num_clients=sample_size, min_num_clients=min_num_clients)
        client_ids = [client.cid for client in clients]

        for cid in client_ids:
            if cid not in self.client_partition_map:
                raise RuntimeError(f"Client {cid} not discovered in round 1")

        param_idx = self._split_model(client_ids)
        local_parameters = self._distribute(client_ids, param_idx)
        self.current_param_idx = {cid: param_idx[m] for m, cid in enumerate(client_ids)}

        fit_cfg = []
        for m, client in enumerate(clients):
            config = self.on_fit_config_fn(server_round) if self.on_fit_config_fn is not None else {}
            config["current_round"] = server_round
            config["model_rate"] = self._get_client_model_rate(client.cid)
            sub_params = [val.cpu().numpy() for _, val in local_parameters[m].items()]
            fit_cfg.append((client, FitIns(ndarrays_to_parameters(sub_params), config)))

        return fit_cfg

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        if server_round == 1:
            for client_proxy, fit_res in results:
                pid = int(fit_res.metrics.get("partition_id", -1))
                if pid >= 0:
                    self._get_client_model_rate(client_proxy.cid, pid)
            metrics = {
                "discovery": True,
                "num_clients_discovered": len(self.client_partition_map),
            }
            global_params = [v.cpu().numpy() for v in self.global_model.state_dict().values()]
            return ndarrays_to_parameters(global_params), metrics

        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]

        local_parameters: List[OrderedDict] = []
        param_idx_list: List[OrderedDict] = []
        client_ids: List[str] = []

        for (client_arrays, _), (client_proxy, _) in zip(weights_results, results):
            cid = client_proxy.cid
            if cid not in self.current_param_idx:
                raise RuntimeError(f"Missing param_idx for client {cid} in round {server_round}")

            client_state = OrderedDict()
            for key, array in zip(self.param_keys, client_arrays):
                if key.endswith("num_batches_tracked"):
                    client_state[key] = torch.tensor(array, dtype=torch.long)
                else:
                    client_state[key] = torch.tensor(array, dtype=torch.float32)

            local_parameters.append(client_state)
            param_idx_list.append(self.current_param_idx[cid])
            client_ids.append(cid)

        aggregated_ndarrays = self._combine(local_parameters, param_idx_list, client_ids)

        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        metrics_aggregated.update(self.last_round_metrics)

        return ndarrays_to_parameters(aggregated_ndarrays), metrics_aggregated

    def _combine(
        self,
        local_parameters: List[OrderedDict],
        param_idx: List[OrderedDict],
        client_ids: List[str],
    ) -> NDArrays:
        self.global_parameters = self.global_model.state_dict()
        prev_parameters = copy.deepcopy(self.global_parameters)
        updated_parameters = copy.deepcopy(self.global_parameters)
        tmp_counts_cpy = copy.deepcopy(self.tmp_counts)

        client_rates = [self._get_client_model_rate(cid) for cid in client_ids]
        n_full = sum(1 for r in client_rates if abs(r - 1.0) < 1e-12)
        n_small = sum(1 for r in client_rates if abs(r - self.core_model_rate) < 1e-12)

        for k, v in updated_parameters.items():
            parameter_type = k.split(".")[-1]
            if parameter_type in ("running_mean", "running_var", "num_batches_tracked"):
                # sBN: BN running buffers are never aggregated (recomputed via
                # calibrate_bn_stats); keep the current global value.
                continue
            count = v.new_zeros(v.size(), dtype=torch.float32)
            tmp_v = v.new_zeros(v.size(), dtype=torch.float32)

            for m in range(len(local_parameters)):
                local_v = local_parameters[m][k]
                if "weight" in parameter_type or "bias" in parameter_type:
                    if parameter_type == "weight" and v.dim() > 1:
                        mesh = self._mesh(param_idx[m][k])
                        if "linear" in k:
                            K = self.tmp_counts[k][mesh]
                        else:
                            output_size = v.size(0)
                            scaler_rate = client_rates[m] / self.global_model_rate
                            local_output_size = int(np.ceil(output_size * scaler_rate))
                            if self.weighting == "avg":
                                K = 1
                            elif self.weighting == "width":
                                K = local_output_size
                            elif self.weighting == "updates":
                                K = self.tmp_counts[k][mesh]
                            elif self.weighting == "updates_width":
                                K = local_output_size * self.tmp_counts[k][mesh]
                            else:
                                K = 1
                        tmp_v[mesh] += K * local_v
                        count[mesh] += K
                        tmp_counts_cpy[k][mesh] += 1
                    else:
                        idx = param_idx[m][k]
                        K = self.tmp_counts[k][idx]
                        tmp_v[idx] += K * local_v
                        count[idx] += K
                        tmp_counts_cpy[k][idx] += 1
                else:
                    if k in param_idx[m]:
                        idx = param_idx[m][k]
                        K = self.tmp_counts[k][idx]
                        tmp_v[idx] += K * local_v
                        count[idx] += K
                        tmp_counts_cpy[k][idx] += 1
                    else:
                        K = self.tmp_counts[k]
                        tmp_v += K * local_v
                        count += K
                        tmp_counts_cpy[k] += 1

            mask = count > 0
            if v.ndim == 0:
                if bool(mask.item()):
                    v.copy_((tmp_v / count).to(v.dtype))
            else:
                tmp_v[mask] = tmp_v[mask].div_(count[mask])
                v[mask] = tmp_v[mask].to(v.dtype)

        self.tmp_counts = tmp_counts_cpy
        self.global_parameters = updated_parameters
        self.global_model.load_state_dict(self.global_parameters)

        # Guardrails and update norms
        core_sq = 0.0
        shell_sq = 0.0
        max_abs_update = 0.0
        for k in self.param_keys:
            delta = (self.global_parameters[k] - prev_parameters[k]).float()
            if not torch.isfinite(delta).all():
                raise RuntimeError(f"Non-finite update detected in {k}")
            max_abs_update = max(max_abs_update, float(delta.abs().max().item()))
            if delta.ndim == 0:
                continue
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                continue
            core_norm, shell_norm = split_core_shell_norm(delta, self.core_model_rate)
            core_sq += core_norm ** 2
            shell_sq += shell_norm ** 2

        if n_full > 0:
            self.time_since_last_full = 0
        else:
            self.time_since_last_full += 1

        if self.enable_shell_repair:
            self.global_parameters, repair_metrics = repair_shell_if_stale(
                self.global_parameters,
                self.shell_repair_state,
                rounds_since_last_full=float(self.time_since_last_full),
                threshold=self.shell_repair_threshold,
                beta=self.shell_repair_beta,
                noise_std=self.shell_repair_noise_std,
                core_model_rate=self.core_model_rate,
            )
            self.global_model.load_state_dict(self.global_parameters, strict=True)
        else:
            repair_metrics = {
                "shell_repair_applied": 0.0,
                "shell_repair_skipped_recent_full": 1.0,
                "shell_repair_threshold": float(self.shell_repair_threshold),
                "shell_repair_beta": float(self.shell_repair_beta),
                "shell_repair_noise_std": float(self.shell_repair_noise_std),
                "shell_repair_shell_l2": 0.0,
                "shell_repair_num_tensors": 0.0,
            }

        self.last_round_metrics = {
            "n_full_sampled": float(n_full),
            "n_small_sampled": float(n_small),
            "time_since_last_full": float(self.time_since_last_full),
            "rounds_since_last_full": float(self.time_since_last_full),
            "p_full_target": float(self.p_full_target),
            "update_norm_core": float(np.sqrt(core_sq)),
            "update_norm_shell": float(np.sqrt(shell_sq)),
            "update_max_abs": float(max_abs_update),
            **repair_metrics,
        }

        self.rounds += 1
        if self.rounds > 0 and self.reshuffle_rounds > 0 and self.rounds % self.reshuffle_rounds == 0:
            self._reshuffle_params()

        return [self.global_parameters[k].cpu().numpy() for k in self.param_keys]

    def get_round_sampling_metrics(self) -> Dict[str, float]:
        """Expose compact per-round sampling stats for server/eval logging."""
        return {
            "n_full_sampled": float(self.last_round_metrics.get("n_full_sampled", 0.0)),
            "rounds_since_last_full": float(
                self.last_round_metrics.get("rounds_since_last_full", -1.0)
            ),
            "p_full_target": float(self.p_full_target),
            "shell_repair_applied": float(self.last_round_metrics.get("shell_repair_applied", 0.0)),
            "shell_repair_skipped_recent_full": float(
                self.last_round_metrics.get("shell_repair_skipped_recent_full", 1.0)
            ),
            "shell_repair_threshold": float(
                self.last_round_metrics.get("shell_repair_threshold", self.shell_repair_threshold)
            ),
            "shell_repair_beta": float(
                self.last_round_metrics.get("shell_repair_beta", self.shell_repair_beta)
            ),
            "shell_repair_noise_std": float(
                self.last_round_metrics.get("shell_repair_noise_std", self.shell_repair_noise_std)
            ),
            "shell_repair_shell_l2": float(
                self.last_round_metrics.get("shell_repair_shell_l2", 0.0)
            ),
            "shell_repair_num_tensors": float(
                self.last_round_metrics.get("shell_repair_num_tensors", 0.0)
            ),
        }


def create_fedrolex_mapped_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    overlap: Optional[float] = 0.0,
    **kwargs,
) -> FedRolexMappedStrategy:
    """Create FedRolex-mapped strategy."""
    return FedRolexMappedStrategy(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        num_classes=num_classes,
        num_clients=num_clients,
        overlap=overlap,
        **kwargs,
    )
