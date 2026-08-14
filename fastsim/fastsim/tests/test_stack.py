import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastsim.config import SimConfig
from fastsim.loop import FastSim
from fastsim import data as fsdata


class RaggedData:
    def __init__(self, sizes, num_classes=10, device="cpu"):
        self.sizes = list(sizes)
        n = sum(self.sizes)
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(0)
        self.train_x = torch.rand(n, 3, 32, 32, generator=g).to(self.device)
        self.train_y = torch.randint(0, num_classes, (n,), generator=g).to(self.device)
        self.mean = torch.zeros(1, 3, 1, 1, device=self.device)
        self.std = torch.ones(1, 3, 1, 1, device=self.device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def partitions(self, num_clients, alpha, seed, partitioning="dirichlet", data_fraction=1.0):
        parts, start = [], 0
        for s in self.sizes[:num_clients]:
            parts.append(list(range(start, start + s)))
            start += s
        return parts

    def client_split(self, indices, seed):
        return fsdata.GpuCifar.client_split(self, indices, seed)


def _fjord_cfg(tmp_path, stack, kd=True, lr=0.05, num_clients=6, cpr=4, epochs=2):
    toml = tmp_path / f"fjord_{stack}_{kd}_{lr}.toml"
    toml.write_text(
        f'num-clients = {num_clients}\nnum-server-rounds = 3\nlocal-epochs = {epochs}\n'
        f'clients-per-round = {cpr}\nrnd-seed = 42\nlearning-rate = {lr}\n'
        'momentum = 0.9\nweight-decay = 0.0005\nglobal-model-rate = 1.0\n'
        'model-rates = "1.0,0.25"\nrate-distribution = "0.34,0.66"\n'
        'norm-type = "sbn"\n'
        f'fjord-kd = {"true" if kd else "false"}\n'
        f'fjord-stack = {"true" if stack else "false"}\n'
        '[algorithm]\nname = "fjord"\n'
        '[dataset]\nname = "cifar100"\nalpha = 0.1\nbatch-size = 16\nseed = 42\n'
        '[model]\nname = "HeteroPreResNet18"\nnum-classes = 10\n'
    )
    return SimConfig.from_toml(toml)


def _round_states(cfg, data):
    sim = FastSim(cfg, device="cpu", data=data)
    picks = sim.run_round(2)
    return sim, picks, {k: v.detach().clone() for k, v in sim.global_model.state_dict().items()}


def _compare(seq, stk, atol):
    worst = 0.0
    for k in seq:
        d = (seq[k].float() - stk[k].float()).abs().max().item()
        worst = max(worst, d)
        assert d <= atol, (k, d)
    return worst


def test_fjord_stacked_single_step_round_exact(tmp_path):
    sizes = (15, 15, 12, 18)
    torch.manual_seed(0)
    _, picks_a, seq = _round_states(
        _fjord_cfg(tmp_path, stack=False, num_clients=4, cpr=4, epochs=1), RaggedData(sizes))
    torch.manual_seed(999)
    sim_b, picks_b, stk = _round_states(
        _fjord_cfg(tmp_path, stack=True, num_clients=4, cpr=4, epochs=1), RaggedData(sizes))
    assert picks_a == picks_b
    assert {sim_b.rates[p] for p in picks_b} == {1.0, 0.25}
    _compare(seq, stk, atol=1e-6)


def test_fjord_stacked_multi_epoch_ragged_small_lr(tmp_path):
    sizes = (45, 18, 40, 33, 26, 50)
    torch.manual_seed(0)
    _, _, seq = _round_states(_fjord_cfg(tmp_path, stack=False, lr=1e-4), RaggedData(sizes))
    torch.manual_seed(999)
    _, _, stk = _round_states(_fjord_cfg(tmp_path, stack=True, lr=1e-4), RaggedData(sizes))
    _compare(seq, stk, atol=1e-6)


def test_fjord_stacked_multi_epoch_ragged_small_lr_nokd(tmp_path):
    sizes = (45, 18, 40, 33, 26, 50)
    torch.manual_seed(0)
    _, _, seq = _round_states(_fjord_cfg(tmp_path, stack=False, kd=False, lr=1e-4), RaggedData(sizes))
    torch.manual_seed(999)
    _, _, stk = _round_states(_fjord_cfg(tmp_path, stack=True, kd=False, lr=1e-4), RaggedData(sizes))
    _compare(seq, stk, atol=1e-6)


def test_fjord_stacked_rng_stream_identical(tmp_path):
    sizes = (45, 18, 40, 33, 26, 50)
    sim_a = FastSim(_fjord_cfg(tmp_path, stack=False), data=RaggedData(sizes), device="cpu")
    sim_b = FastSim(_fjord_cfg(tmp_path, stack=True), data=RaggedData(sizes), device="cpu")
    sim_a.run_round(2)
    sim_b.run_round(2)
    assert torch.equal(sim_a.train_gen.get_state(), sim_b.train_gen.get_state())
    assert torch.equal(sim_a.sample_gen.get_state(), sim_b.sample_gen.get_state())
