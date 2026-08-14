"""FedAdam server optimizer for the frozen-shell core (fst_fedadam port).

Realizes the debug/ `fst_fedadam` winner in the Flower stack: under a frozen
shell acc_full == acc_core, so the method reduces to optimizing the shared core
with an adaptive server step. We run the homogeneous core (global model_rate =
core_model_rate) and replace FedAvg's server update with FedAdam:

    Delta  = FedAvg(client params) - current   (pseudo-gradient / aggregated update)
    m      = b1*m + (1-b1)*Delta
    v      = b2*v + (1-b2)*Delta^2
    new    = current + eta * m / (sqrt(v) + tau)

BN running buffers (running_mean/running_var/num_batches_tracked) and any
non-float arrays are passed through unchanged (kept at the FedAvg value); under
sBN they are recomputed via calibrate_bn_stats before any accuracy is read.
"""
from typing import List, Optional

import numpy as np
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg


class FstFedAdamStrategy(FedAvg):
    def __init__(self, *args, eta: float = 0.03, tau: float = 1e-3,
                 beta_1: float = 0.9, beta_2: float = 0.99,
                 param_keys: Optional[List[str]] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eta = eta
        self.tau = tau
        self.b1 = beta_1
        self.b2 = beta_2
        self.param_keys = param_keys
        self.current = (parameters_to_ndarrays(self.initial_parameters)
                        if self.initial_parameters is not None else None)
        self.m: Optional[List[np.ndarray]] = None
        self.v: Optional[List[np.ndarray]] = None

    @staticmethod
    def _is_bn_buffer(key: str) -> bool:
        return key.endswith(("running_mean", "running_var", "num_batches_tracked"))

    def aggregate_fit(self, server_round, results, failures):
        agg_params, metrics = super().aggregate_fit(server_round, results, failures)
        if agg_params is None:
            return agg_params, metrics
        agg = parameters_to_ndarrays(agg_params)
        if self.current is None:
            self.current = [a.copy() for a in agg]
        if self.m is None:
            self.m = [np.zeros_like(a, dtype=np.float64) for a in agg]
            self.v = [np.zeros_like(a, dtype=np.float64) for a in agg]

        new: List[np.ndarray] = []
        for i, (cur, ag) in enumerate(zip(self.current, agg)):
            key = self.param_keys[i] if self.param_keys else ""
            if (not np.issubdtype(ag.dtype, np.floating)) or self._is_bn_buffer(key):
                new.append(ag)  # BN buffers / ints: keep FedAvg value (recomputed at eval)
                continue
            g = ag.astype(np.float64) - cur.astype(np.float64)
            self.m[i] = self.b1 * self.m[i] + (1.0 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1.0 - self.b2) * g * g
            upd = cur.astype(np.float64) + self.eta * self.m[i] / (np.sqrt(self.v[i]) + self.tau)
            new.append(upd.astype(ag.dtype))

        self.current = [a.copy() for a in new]
        return ndarrays_to_parameters(new), metrics
