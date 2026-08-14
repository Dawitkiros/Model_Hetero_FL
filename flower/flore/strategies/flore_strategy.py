"""FLoRE strategy: pre-aggregation lifted-delta HeteroFL.

This strategy keeps the existing HeteroFL client-side extraction flow, but
changes the server aggregation step:

1. The server sends the appropriate sub-model to each client.
2. The client trains locally and returns its updated parameters.
3. The server computes the client delta against what it originally sent.
4. If the client is a small-model client, the server lifts that delta to the
   full model using the same graph-consistent activation-space mapping used by
   the shell-repair utilities.
5. The server aggregates lifted small-client deltas together with true
   full-client deltas before applying the update.

The implementation intentionally lives in this file only.  No server or client
code is modified here.
"""

from __future__ import annotations

from collections import OrderedDict
from logging import INFO
from typing import Callable, Dict, List, MutableMapping, Optional, Tuple, Union

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

from ..hetero_utils import split_core_shell_norm
from ..net2wider import (
    _build_graph_spec,
    _get_or_create_space_map,
    _parse_tensor_role,
    _space_fan_counts,
    create_shell_repair_state,
)
from .hetero_strategy import HeteroStrategy


def _state_dict_from_ndarrays(
    template_state: "OrderedDict[str, torch.Tensor]",
    ndarrays: NDArrays,
) -> "OrderedDict[str, torch.Tensor]":
    """Convert an array list into a state dict using a template for dtypes."""
    state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for (key, template_tensor), array in zip(template_state.items(), ndarrays):
        if key.endswith("num_batches_tracked") or template_tensor.dtype == torch.long:
            state_dict[key] = torch.tensor(array, dtype=torch.long)
        else:
            state_dict[key] = torch.tensor(array, dtype=template_tensor.dtype)
    return state_dict


def _state_dict_to_ndarrays(state_dict: "OrderedDict[str, torch.Tensor]") -> NDArrays:
    return [tensor.detach().cpu().numpy() for tensor in state_dict.values()]


def _zero_delta_like(
    reference_state: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, tensor in reference_state.items():
        if tensor.is_floating_point():
            delta[key] = torch.zeros_like(tensor, dtype=torch.float32)
        else:
            delta[key] = torch.zeros_like(tensor)
    return delta


def _clone_state_dict(
    state_dict: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    return OrderedDict((k, v.clone()) for k, v in state_dict.items())


def lift_core_delta_to_full(
    core_delta: "OrderedDict[str, torch.Tensor]",
    global_state_dict: "OrderedDict[str, torch.Tensor]",
    repair_state: MutableMapping[str, object],
    *,
    core_model_rate: float,
    lift_scale: float = 1.0,
) -> "OrderedDict[str, torch.Tensor]":
    """Lift a core-only delta to a full-model delta.

    The core prefix is copied exactly.  Shell entries are filled by the same
    activation-space mapping used by Net2Wider-style shell repair.
    """
    graph_spec = repair_state.get("graph_spec")
    if graph_spec is None:
        graph_spec = _build_graph_spec(global_state_dict)
        repair_state["graph_spec"] = graph_spec

    full_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()

    for key, full_tensor in global_state_dict.items():
        if key not in core_delta:
            full_delta[key] = torch.zeros_like(full_tensor)
            continue

        d_core = core_delta[key]
        if not full_tensor.is_floating_point() or d_core.ndim == 0:
            # Batch counters and other scalar state are kept unchanged by the
            # lifted update.
            full_delta[key] = torch.zeros_like(full_tensor)
            continue

        meta = _parse_tensor_role(key, full_tensor, graph_spec, core_model_rate)

        if meta is None:
            if d_core.shape == full_tensor.shape:
                full_delta[key] = d_core.clone().to(dtype=full_tensor.dtype)
            else:
                d = torch.zeros_like(full_tensor)
                slices = tuple(
                    slice(0, min(s1, s2)) for s1, s2 in zip(d_core.shape, full_tensor.shape)
                )
                d[slices] = d_core[slices]
                full_delta[key] = d
            continue

        kind = meta["kind"]
        if kind in {"conv", "linear"}:
            lifted = _lift_weight_delta(d_core, full_tensor, meta, repair_state)
        elif kind == "norm":
            lifted = _lift_norm_delta(d_core, full_tensor, meta, repair_state)
        else:
            lifted = torch.zeros_like(full_tensor)

        if lift_scale != 1.0:
            lifted = _scale_shell_region(lifted, meta, lift_scale)
        full_delta[key] = lifted

    return full_delta


def _lift_weight_delta(
    d_core: torch.Tensor,
    full_tensor: torch.Tensor,
    meta: Dict[str, object],
    repair_state: MutableMapping[str, object],
) -> torch.Tensor:
    """Lift a core weight delta [k_out, k_in, ...] to full shape."""
    out_core = int(meta["out_core"])
    in_core = int(meta["in_core"])
    out_full = int(meta["out_full"])
    in_full = int(meta["in_full"])

    device = d_core.device
    d_full = torch.zeros_like(full_tensor)

    # Exact copy of the core-core block.
    d_full[:out_core, :in_core] = d_core[:out_core, :in_core]

    out_map = _get_or_create_space_map(
        meta["out_space"], out_full, out_core, repair_state
    ).to(device)
    in_map = _get_or_create_space_map(
        meta["in_space"], in_full, in_core, repair_state
    ).to(device)
    in_fan = _space_fan_counts(in_map, in_core, dtype=d_core.dtype, device=device)

    core_block = d_core[:out_core, :in_core]
    shell_out = slice(out_core, out_full)
    shell_in = slice(in_core, in_full)

    if out_full > out_core:
        d_full[shell_out, :in_core] = core_block[out_map[shell_out], :in_core]

    if in_full > in_core:
        scale = in_fan[in_map[shell_in]]
        if full_tensor.ndim == 4:
            scale = scale.view(1, -1, 1, 1)
            d_full[:out_core, shell_in] = core_block[:, in_map[shell_in], :, :] / scale
        else:
            scale = scale.view(1, -1)
            d_full[:out_core, shell_in] = core_block[:, in_map[shell_in]] / scale

    if out_full > out_core and in_full > in_core:
        scale = in_fan[in_map[shell_in]]
        if full_tensor.ndim == 4:
            scale = scale.view(1, -1, 1, 1)
            d_full[shell_out, shell_in] = (
                core_block[out_map[shell_out]][:, in_map[shell_in], :, :] / scale
            )
        else:
            scale = scale.view(1, -1)
            d_full[shell_out, shell_in] = (
                core_block[out_map[shell_out]][:, in_map[shell_in]] / scale
            )

    return d_full


def _lift_norm_delta(
    d_core: torch.Tensor,
    full_tensor: torch.Tensor,
    meta: Dict[str, object],
    repair_state: MutableMapping[str, object],
) -> torch.Tensor:
    """Lift a core 1-D delta [k] to full shape [C]."""
    core_dim = int(meta["core"])
    full_dim = int(meta["full"])

    d_full = torch.zeros_like(full_tensor)
    d_full[:core_dim] = d_core[:core_dim]

    if full_dim > core_dim:
        space_map = _get_or_create_space_map(
            meta["space"], full_dim, core_dim, repair_state
        ).to(d_core.device)
        d_full[core_dim:] = d_core[space_map[core_dim:]]

    return d_full


def _scale_shell_region(
    delta: torch.Tensor,
    meta: Dict[str, object],
    scale: float,
) -> torch.Tensor:
    """Scale only the shell portion of a lifted delta tensor."""
    d = delta.clone()
    kind = meta["kind"]

    if kind in {"conv", "linear"}:
        oc = int(meta["out_core"])
        ic = int(meta["in_core"])
        of = int(meta["out_full"])
        inf = int(meta["in_full"])

        if of > oc:
            d[oc:] *= scale
        if inf > ic:
            if d.ndim == 4:
                d[:oc, ic:, :, :] *= scale
            elif d.ndim == 2:
                d[:oc, ic:] *= scale
    elif kind == "norm":
        cd = int(meta["core"])
        d[cd:] *= scale

    return d


class FloreStrategy(HeteroStrategy):
    """HeteroFL with pre-aggregation lifted-delta aggregation."""

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
        lift_scale: float = 1.0,
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
            extraction_mode=extraction_mode,
            num_classes=num_classes,
            model_name=model_name,
            global_model_rate=global_model_rate,
            model_split_mode=model_split_mode,
            num_clients=num_clients,
            overlap=overlap,
            seed=seed,
            model_rates=model_rates,
            rate_distribution=rate_distribution,
            norm_type=norm_type,
            core_model_rate=core_model_rate,
            enable_shell_repair=enable_shell_repair,
            shell_repair_threshold=shell_repair_threshold,
            shell_repair_beta=shell_repair_beta,
            shell_repair_noise_std=shell_repair_noise_std,
            shell_repair_seed=shell_repair_seed,
        )

        self.lift_scale = float(lift_scale)
        self.current_sent_parameters: Dict[str, "OrderedDict[str, torch.Tensor]"] = {}
        self.lift_states: Dict[float, Dict[str, object]] = {}
        self.last_round_metrics: Dict[str, Scalar] = {}

        # Keep one activation-space map cache per non-full model rate so that
        # different core widths do not collide.
        for rate in self.model_rates:
            rate_f = float(rate)
            if rate_f < 1.0 - 1e-12:
                self.lift_states[rate_f] = create_shell_repair_state(
                    seed=self.shell_repair_seed + int(round(rate_f * 1000.0))
                )

        log(INFO, "Created FLoRE strategy with pre-aggregation delta lifting")
        log(INFO, f"  lift_scale={self.lift_scale}")
        log(INFO, f"  model_rates={self.model_rates}")
        log(INFO, f"  rate_distribution={self.rate_distribution}")

    def _lift_state_for_rate(self, model_rate: float) -> Dict[str, object]:
        rate = float(model_rate)
        if rate >= 1.0 - 1e-12:
            return self.shell_repair_state
        if rate not in self.lift_states:
            self.lift_states[rate] = create_shell_repair_state(
                seed=self.shell_repair_seed + int(round(rate * 1000.0))
            )
        return self.lift_states[rate]

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training and cache sent sub-models."""
        self.current_round = server_round
        self.current_sent_parameters = {}

        global_params = parameters_to_ndarrays(parameters)
        params_dict = zip(self.global_model.state_dict().keys(), global_params)
        state_dict = OrderedDict()
        for k, v in params_dict:
            if k.endswith("num_batches_tracked"):
                state_dict[k] = torch.tensor(v, dtype=torch.long)
            else:
                state_dict[k] = torch.tensor(v, dtype=torch.float32)
        self.global_model.load_state_dict(state_dict, strict=True)

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
        n_full_sampled = 0

        for client in clients:
            client_id = client.cid
            sub_arrays = self._extract_sub_parameters(client_id, server_round)
            if server_round == 1:
                template_model = self.global_model
            else:
                template_model = self._create_client_model(client_id)
            sub_params = _state_dict_from_ndarrays(template_model.state_dict(), sub_arrays)
            self.current_sent_parameters[client_id] = _clone_state_dict(sub_params)

            config = {}
            if self.on_fit_config_fn is not None:
                config = self.on_fit_config_fn(server_round)
            config["current_round"] = server_round

            if server_round == 1:
                model_rate = 1.0
            else:
                model_rate = self._get_client_model_rate(client_id)
            config["model_rate"] = model_rate
            if abs(float(model_rate) - 1.0) < 1e-12:
                n_full_sampled += 1

            fit_ins = FitIns(ndarrays_to_parameters(sub_arrays), config)
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

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client updates by lifting small-model deltas first."""
        if not results:
            return None, {}

        if server_round == 1:
            log(INFO, "=" * 60)
            log(INFO, f"Round 1: CAPABILITY DISCOVERY - Received {len(results)} client responses")
            log(INFO, "=" * 60)

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

            global_params = [v.cpu().numpy() for v in self.global_model.state_dict().values()]
            return ndarrays_to_parameters(global_params), {"discovery": True}

        global_state = self.global_model.state_dict()
        total_examples = sum(fit_res.num_examples for _, fit_res in results)
        if total_examples <= 0:
            return ndarrays_to_parameters(_state_dict_to_ndarrays(global_state)), {}

        aggregated_delta = _zero_delta_like(global_state)

        n_full = 0
        n_small = 0
        n_lifted = 0
        core_sq = 0.0
        shell_sq = 0.0
        lifted_core_sq = 0.0
        lifted_shell_sq = 0.0
        full_core_sq = 0.0
        full_shell_sq = 0.0
        max_abs_update = 0.0

        for client_proxy, fit_res in results:
            client_id = client_proxy.cid
            client_rate = float(self._get_client_model_rate(client_id))
            is_full_model = abs(client_rate - 1.0) < 1e-12

            sent_state = self.current_sent_parameters.get(client_id)
            if sent_state is None:
                raise RuntimeError(
                    f"Missing sent parameters for client {client_id} in round {server_round}"
                )

            received_state = _state_dict_from_ndarrays(
                sent_state, parameters_to_ndarrays(fit_res.parameters)
            )

            client_delta: "OrderedDict[str, torch.Tensor]" = OrderedDict()
            for key, sent_tensor in sent_state.items():
                if key.endswith(("num_batches_tracked", "running_mean", "running_var")) or not sent_tensor.is_floating_point():
                    # sBN: BN running buffers are never aggregated (recomputed via
                    # calibrate_bn_stats); zero their delta so the global value is kept.
                    client_delta[key] = torch.zeros_like(global_state[key])
                    continue
                client_delta[key] = (
                    received_state[key].to(dtype=torch.float32)
                    - sent_tensor.to(dtype=torch.float32)
                )

            if is_full_model:
                n_full += 1
                lifted_delta = client_delta
            else:
                n_small += 1
                n_lifted += 1
                lift_state = self._lift_state_for_rate(client_rate)
                lifted_delta = lift_core_delta_to_full(
                    client_delta,
                    global_state,
                    lift_state,
                    core_model_rate=client_rate,
                    lift_scale=self.lift_scale,
                )

            weight = float(fit_res.num_examples) / float(total_examples)
            for key, delta_tensor in lifted_delta.items():
                if not global_state[key].is_floating_point():
                    continue
                delta_f = delta_tensor.to(dtype=torch.float32)
                aggregated_delta[key] += weight * delta_f

                if delta_f.ndim > 0:
                    c_norm, s_norm = split_core_shell_norm(delta_f, client_rate)
                    core_sq += c_norm ** 2 * weight
                    shell_sq += s_norm ** 2 * weight
                    max_abs_update = max(max_abs_update, float(delta_f.abs().max().item()))

                    if is_full_model:
                        full_core_sq += c_norm ** 2 * weight
                        full_shell_sq += s_norm ** 2 * weight
                    else:
                        lifted_core_sq += c_norm ** 2 * weight
                        lifted_shell_sq += s_norm ** 2 * weight

        new_state = OrderedDict()
        for key, param in global_state.items():
            if param.is_floating_point():
                new_state[key] = param + aggregated_delta[key].to(dtype=param.dtype)
            else:
                new_state[key] = param.clone()

        self.global_model.load_state_dict(new_state, strict=True)

        metrics_aggregated: Dict[str, Scalar] = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)

        self.last_round_metrics = {
            "agg_n_clients": float(len(results)),
            "agg_n_full": float(n_full),
            "agg_n_small": float(n_small),
            "agg_n_lifted": float(n_lifted),
            "agg_lift_scale": float(self.lift_scale),
            "agg_update_norm_core": float(np.sqrt(core_sq)),
            "agg_update_norm_shell": float(np.sqrt(shell_sq)),
            "agg_lifted_norm_core": float(np.sqrt(lifted_core_sq)),
            "agg_lifted_norm_shell": float(np.sqrt(lifted_shell_sq)),
            "agg_full_norm_core": float(np.sqrt(full_core_sq)),
            "agg_full_norm_shell": float(np.sqrt(full_shell_sq)),
            "agg_update_max_abs": float(max_abs_update),
            "n_full_sampled": float(self.last_round_n_full_sampled),
            "rounds_since_last_full": float(self.last_round_rounds_since_last_full),
            "p_full_target": float(self.p_full_target),
        }
        metrics_aggregated.update(self.last_round_metrics)

        return ndarrays_to_parameters(_state_dict_to_ndarrays(new_state)), metrics_aggregated

    def get_round_sampling_metrics(self) -> Dict[str, float]:
        """Return compact round-level metrics for logging."""
        return {
            "n_full_sampled": float(self.last_round_metrics.get("n_full_sampled", 0.0)),
            "rounds_since_last_full": float(
                self.last_round_metrics.get("rounds_since_last_full", -1.0)
            ),
            "p_full_target": float(self.p_full_target),
            "agg_n_full": float(self.last_round_metrics.get("agg_n_full", 0.0)),
            "agg_n_small": float(self.last_round_metrics.get("agg_n_small", 0.0)),
            "agg_n_lifted": float(self.last_round_metrics.get("agg_n_lifted", 0.0)),
            "agg_lift_scale": float(self.last_round_metrics.get("agg_lift_scale", self.lift_scale)),
            "agg_update_norm_core": float(self.last_round_metrics.get("agg_update_norm_core", 0.0)),
            "agg_update_norm_shell": float(self.last_round_metrics.get("agg_update_norm_shell", 0.0)),
            "agg_update_max_abs": float(self.last_round_metrics.get("agg_update_max_abs", 0.0)),
        }


def create_flore_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    overlap: float = 0.0,
    **kwargs,
) -> FloreStrategy:
    """Factory for the FLoRE strategy."""
    return FloreStrategy(
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


__all__ = ["FloreStrategy", "create_flore_strategy", "lift_core_delta_to_full"]
