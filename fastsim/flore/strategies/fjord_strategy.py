"""FjORD strategy (Horvath et al., NeurIPS 2021).

Server side, FjORD is identical to HeteroFL: static contiguous first-k
sub-model extraction and position-wise selective averaging. The FjORD-specific
mechanism (Ordered Dropout + self-distillation) lives entirely in the client
local-training loop (see ``flore/fjord.py``), which is dispatched by algorithm
name. This class therefore subclasses ``HeteroStrategy`` in static mode and
exists mainly for explicit dispatch and logging.

Deviation from the upstream Flower FjORD baseline (documented deliberately):
this variant runs on the FLoRE / HeteroFL backbone (PreAct ResNet with sBN +
Scaler) so the toxicity core-extraction metric stays directly comparable to
HeteroFL, rather than porting FjORD's per-width BatchNorm / no-Scaler post-act
ResNet-18. The distinguishing FjORD mechanics (ordered-dropout width sampling
and wide->narrow self-distillation) are reproduced faithfully.
"""

from logging import INFO
from typing import Callable

from flwr.common.logger import log

from .hetero_strategy import HeteroStrategy


class FjordStrategy(HeteroStrategy):
    """HeteroFL-style server with FjORD ordered-dropout clients."""

    def __init__(self, **kwargs):
        kwargs["extraction_mode"] = "static"
        super().__init__(**kwargs)
        log(INFO, "FjORD initialized (ordered dropout + self-distillation on FLoRE backbone)")


def create_fjord_strategy(
    evaluate_fn: Callable,
    num_clients: int = 100,
    clients_per_round: int = 10,
    num_classes: int = 10,
    **kwargs,
) -> FjordStrategy:
    """Create FjORD strategy (static extraction; OD+KD handled client-side)."""
    return FjordStrategy(
        fraction_fit=0.0,
        fraction_evaluate=0.0,
        min_fit_clients=clients_per_round,
        min_evaluate_clients=0,
        min_available_clients=num_clients,
        evaluate_fn=evaluate_fn,
        num_classes=num_classes,
        num_clients=num_clients,
        **kwargs,
    )
