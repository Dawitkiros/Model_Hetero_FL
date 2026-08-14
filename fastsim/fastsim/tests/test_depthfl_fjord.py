import sys
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastsim.aggregate import aggregate_depthfl
from fastsim.config import SimConfig
from fastsim.depth import DepthFLSim
from fastsim.loop import FastSim
from fastsim.tests.test_loop import TinyData


# (a) MultiResNet depth-prefix state_dict nesting -------------------------------

def test_depthfl_state_prefix_nesting():
    from flore.depthfl_models import create_depthfl_model
    keys = {d: list(create_depthfl_model(d, num_classes=100, norm_type="bn").state_dict().keys())
            for d in (1, 2, 3, 4)}
    assert [len(keys[d]) for d in (1, 2, 3, 4)] == [93, 172, 237, 269]
    for d in (1, 2, 3):
        assert keys[d] == keys[4][: len(keys[d])], f"depth {d} not a strict prefix of depth 4"
        assert keys[d] == keys[d + 1][: len(keys[d])], f"depth {d} not a strict prefix of depth {d + 1}"


# (b) fastsim positional-count aggregation vs flore DepthFLStrategy --------------

def _synth_client_state(depth, seed):
    from flore.depthfl_models import create_depthfl_model
    torch.manual_seed(seed)
    sd = create_depthfl_model(depth, num_classes=100, norm_type="bn").state_dict()
    out = OrderedDict()
    for k, v in sd.items():
        if k.endswith("num_batches_tracked"):
            out[k] = torch.tensor((seed % 5) + 1, dtype=torch.long)
        else:
            out[k] = torch.randn_like(v.float())
    return out


def test_aggregate_depthfl_matches_flore_strategy():
    from flore.strategies.depthfl_strategy import DepthFLStrategy
    from flwr.common import FitRes, Status, Code, ndarrays_to_parameters, parameters_to_ndarrays

    strat = DepthFLStrategy(
        min_fit_clients=5, min_available_clients=50,
        num_classes=100, num_clients=50, norm_type="bn",
        depth_levels=[4, 1], depth_distribution=[0.10, 0.90],
    )
    client_states = [
        _synth_client_state(4, 11), _synth_client_state(4, 12),
        _synth_client_state(1, 21), _synth_client_state(1, 22), _synth_client_state(1, 23),
    ]

    class _Proxy:
        def __init__(self, cid):
            self.cid = cid

    results = []
    for i, st in enumerate(client_states):
        arrays = [v.numpy() for v in st.values()]
        fit_res = FitRes(
            status=Status(Code.OK, ""),
            parameters=ndarrays_to_parameters(arrays),
            num_examples=100,
            metrics={},
        )
        results.append((_Proxy(str(i)), fit_res))

    global_state = OrderedDict((k, v.detach().clone()) for k, v in strat.global_model.state_dict().items())
    fast_out = aggregate_depthfl(global_state, client_states)

    params, _ = strat.aggregate_fit(2, results, [])
    flore_arrays = parameters_to_ndarrays(params)

    keys = list(global_state.keys())
    assert len(flore_arrays) == len(keys)
    for i, k in enumerate(keys):
        a = fast_out[k]
        b = torch.tensor(flore_arrays[i])
        assert a.shape == b.shape, k
        assert torch.equal(a, b.to(a.dtype)), f"mismatch at {k}"


# (c) FjORD ordered-dropout: scatter == sub autograd, grads cleared on sync ------

def _fjord_pair(rate_sub=0.25, seed=0):
    from flore.hetero_models import create_hetero_model
    torch.manual_seed(seed)
    master = create_hetero_model("HeteroPreResNet18", in_channels=3, num_classes=10,
                                 model_rate=1.0, norm_type="sbn")
    sub = create_hetero_model("HeteroPreResNet18", in_channels=3, num_classes=10,
                              model_rate=rate_sub, norm_type="sbn")
    return master, sub


def test_fjord_sync_clears_stale_sub_grads():
    from flore.fjord import _sync_sub_from_master
    master, sub = _fjord_pair(seed=1)
    for p in sub.parameters():
        p.grad = torch.ones_like(p)
    _sync_sub_from_master(master, sub)
    assert all(p.grad is None for p in sub.parameters()), "3ecc786 regression: sub grads not cleared on sync"


def test_fjord_scatter_matches_sub_autograd_and_leaves_shell_untouched():
    from flore.fjord import _scatter_grad_prefix, _sync_sub_from_master
    master, sub = _fjord_pair(seed=2)
    master.train(); sub.train()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    _sync_sub_from_master(master, sub)
    F.cross_entropy(sub(x), y).backward()
    ref_sub_grad = {n: p.grad.detach().clone() for n, p in sub.named_parameters()}

    master.zero_grad(set_to_none=True)
    _scatter_grad_prefix(master, sub)

    master_params = dict(master.named_parameters())
    sub_params = dict(sub.named_parameters())
    touched = 0
    for name, mp in master_params.items():
        if name not in sub_params:
            continue
        sp = sub_params[name]
        slices = tuple(slice(0, s) for s in sp.shape)
        assert mp.grad is not None, name
        assert torch.equal(mp.grad[slices], ref_sub_grad[name]), f"scatter != sub autograd at {name}"
        if mp.shape != sp.shape:
            full = mp.grad.clone()
            full[slices] = 0.0
            assert torch.count_nonzero(full) == 0, f"shell region not left zero at {name}"
            touched += 1
    assert touched > 0, "expected at least one width-reduced tensor with a shell region"


def test_fjord_scatter_accumulates_into_existing_master_grad():
    from flore.fjord import _scatter_grad_prefix, _sync_sub_from_master
    master, sub = _fjord_pair(seed=3)
    master.train(); sub.train()
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    F.cross_entropy(master(x), y).backward()
    teacher_grad = {n: p.grad.detach().clone() for n, p in master.named_parameters()}

    _sync_sub_from_master(master, sub)
    F.cross_entropy(sub(x), y).backward()
    _scatter_grad_prefix(master, sub)

    sub_params = dict(sub.named_parameters())
    for name, mp in master.named_parameters():
        if name in sub_params:
            slices = tuple(slice(0, s) for s in sub_params[name].shape)
            expected = teacher_grad[name].clone()
            expected[slices] = expected[slices] + sub_params[name].grad
            assert torch.allclose(mp.grad, expected, atol=1e-6), name


# (d) FjORD candidate-rate logic ------------------------------------------------

def test_fjord_candidate_rates():
    from flore.fjord import ordered_dropout_candidate_rates
    assert ordered_dropout_candidate_rates([1.0, 0.25], 1.0) == [0.25, 1.0]
    assert ordered_dropout_candidate_rates([1.0, 0.25], 0.25) == [0.25]
    assert ordered_dropout_candidate_rates([1.0, 0.5, 0.25], 0.5) == [0.25, 0.5]
    assert ordered_dropout_candidate_rates([1.0], 0.25) == [0.25]


# smoke: both new paths run end to end on CPU ----------------------------------

def _write_cfg(tmp_path, algorithm, model_name, extra=""):
    toml = tmp_path / f"{algorithm}.toml"
    toml.write_text(
        'num-clients = 8\nnum-server-rounds = 3\nlocal-epochs = 1\n'
        'clients-per-round = 2\nrnd-seed = 42\nlearning-rate = 0.05\n'
        'momentum = 0.9\nweight-decay = 0.0005\nglobal-model-rate = 1.0\n'
        'model-rates = "1.0,0.25"\nrate-distribution = "0.5,0.5"\n'
        f'{extra}'
        f'[algorithm]\nname = "{algorithm}"\n'
        '[dataset]\nname = "cifar100"\nalpha = 0.1\nbatch-size = 16\nseed = 42\n'
        f'[model]\nname = "{model_name}"\nnum-classes = 10\n'
    )
    return SimConfig.from_toml(toml)


def test_fjord_smoke_two_rounds_cpu(tmp_path):
    cfg = _write_cfg(tmp_path, "fjord", "HeteroPreResNet18", extra='norm-type = "sbn"\nfjord-kd = true\n')
    sim = FastSim(cfg, device="cpu", data=TinyData())
    before = {k: v.clone() for k, v in sim.global_model.state_dict().items()}
    sim.run(progress_every=0, eval_every=0)
    after = sim.global_model.state_dict()
    assert sum(1 for k in before if not torch.equal(before[k], after[k])) > 0


def test_depthfl_smoke_two_rounds_cpu(tmp_path):
    cfg = _write_cfg(tmp_path, "depthfl", "MultiResNet18",
                     extra='norm-type = "bn"\ndepth-levels = "4,1"\ndepth-distribution = "0.5,0.5"\n'
                           'depthfl-kd = true\ndepthfl-extended = true\nkd-rampup-rounds = 300\n')
    sim = DepthFLSim(cfg, device="cpu", data=TinyData())
    before = {k: v.clone() for k, v in sim.global_model.state_dict().items()}
    sim.run(progress_every=0, eval_every=0)
    after = sim.global_model.state_dict()
    assert sum(1 for k in before if not torch.equal(before[k], after[k])) > 0
    depths = set(sim.depths)
    assert depths == {4, 1}
