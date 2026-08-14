import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastsim.aggregate import aggregate_hetero
from fastsim.config import SimConfig
from fastsim.loop import FastSim, lr_at_round
from fastsim import data as fsdata


def test_aggregate_prefix_and_untouched():
    g = {"conv.weight": torch.ones(4, 4, 3, 3) * 10.0, "bn.running_mean": torch.ones(4) * 7.0}
    c1 = {"conv.weight": torch.ones(2, 2, 3, 3) * 1.0}
    c2 = {"conv.weight": torch.ones(4, 4, 3, 3) * 3.0}
    out = aggregate_hetero(g, [c1, c2], [100, 300])
    w = out["conv.weight"]
    assert torch.allclose(w[:2, :2], torch.full((2, 2, 3, 3), (100 * 1.0 + 300 * 3.0) / 400))
    assert torch.allclose(w[2:, 2:], torch.full((2, 2, 3, 3), 3.0))
    assert torch.allclose(out["bn.running_mean"], torch.ones(4) * 7.0)


def test_aggregate_no_clients_keeps_global():
    g = {"lin.weight": torch.randn(5, 5)}
    out = aggregate_hetero(g, [], [])
    assert torch.equal(out["lin.weight"], g["lin.weight"])


def test_accumulate_matches_flore_width_slice():
    from flore.strategies.hetero_strategy import HeteroStrategy
    from fastsim.aggregate import accumulate_slice
    torch.manual_seed(3)
    for shape, cshape in [((8, 8, 3, 3), (3, 5, 3, 3)), ((10, 6), (4, 6)), ((12,), (5,)), ((), ())]:
        agg_a = torch.zeros(shape)
        cnt_a = torch.zeros(shape)
        agg_b = agg_a.clone()
        cnt_b = cnt_a.clone()
        cp = torch.randn(cshape)
        accumulate_slice(agg_a, cnt_a, cp, 37)
        HeteroStrategy._aggregate_width_slice(None, agg_b, cnt_b, cp, 37)
        assert torch.equal(agg_a, agg_b), shape
        assert torch.equal(cnt_a, cnt_b), shape


def test_lr_schedule_flat_before_milestones():
    from fastsim.loop import decay_schedule_for
    assert lr_at_round(0.05, 2) == 0.05
    assert lr_at_round(0.05, 300) == 0.05
    assert abs(lr_at_round(0.05, 802) - 0.005) < 1e-12
    assert decay_schedule_for("dirichlet") == (800, 1500)
    assert decay_schedule_for("iid") == (150, 250)
    assert abs(lr_at_round(0.1, 152, decay_schedule_for("iid")) - 0.01) < 1e-12


class TinyData:
    def __init__(self, n=256, num_classes=10, device="cpu"):
        self.device = torch.device(device)
        g = torch.Generator().manual_seed(0)
        self.train_x = torch.rand(n, 3, 32, 32, generator=g).to(self.device)
        self.train_y = torch.randint(0, num_classes, (n,), generator=g).to(self.device)
        self.mean = torch.zeros(1, 3, 1, 1, device=self.device)
        self.std = torch.ones(1, 3, 1, 1, device=self.device)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def partitions(self, num_clients, alpha, seed, partitioning="dirichlet", data_fraction=1.0):
        idx = list(range(len(self.train_y)))
        per = len(idx) // num_clients
        return [idx[i * per:(i + 1) * per] for i in range(num_clients)]

    def client_split(self, indices, seed):
        return fsdata.GpuCifar.client_split(self, indices, seed)


def make_cfg(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text(
        'num-clients = 8\nnum-server-rounds = 3\nlocal-epochs = 1\n'
        'clients-per-round = 2\nrnd-seed = 42\nlearning-rate = 0.05\n'
        'momentum = 0.9\nweight-decay = 0.0005\nglobal-model-rate = 1.0\n'
        'model-rates = "1.0,0.25"\nrate-distribution = "0.25,0.75"\n'
        f'save-path = "{tmp_path}/smoke.csv"\n'
        '[algorithm]\nname = "heterofl"\n'
        '[dataset]\nname = "cifar100"\nalpha = 0.1\nbatch-size = 16\nseed = 42\n'
        '[model]\nname = "HeteroPreResNet18"\nnum-classes = 10\n'
    )
    return SimConfig.from_toml(toml)


def test_smoke_two_rounds_cpu(tmp_path):
    cfg = make_cfg(tmp_path)
    sim = FastSim(cfg, device="cpu", data=TinyData())
    before = {k: v.clone() for k, v in sim.global_model.state_dict().items()}
    sim.run(progress_every=0)
    after = sim.global_model.state_dict()
    changed = sum(1 for k in before if not torch.equal(before[k], after[k]))
    assert changed > 0
    core_rates = {sim.rates[pid] for pid in range(8)}
    assert core_rates == {1.0, 0.25}
    path = sim.save()
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    assert set(loaded.keys()) == set(after.keys())


def test_partitioning_honored():
    ds = object.__new__(fsdata.GpuCifar)
    ds.train_y = torch.zeros(103)
    iid = fsdata.GpuCifar.partitions(ds, 10, 0.1, 42, "iid")
    assert iid[0] == list(range(10))
    assert iid[-1] == list(range(90, 103))
    try:
        fsdata.GpuCifar.partitions(ds, 10, 0.1, 42, "shards")
        assert False
    except ValueError:
        pass


def test_save_empty_path_raises(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.save_path = ""
    sim = FastSim(cfg, device="cpu", data=TinyData())
    try:
        sim.save()
        assert False
    except ValueError as e:
        assert "save path" in str(e)


def test_config_toml_array_rates(tmp_path):
    toml = tmp_path / "arr.toml"
    toml.write_text(
        'num-clients = 4\nnum-server-rounds = 2\nlocal-epochs = 1\n'
        'clients-per-round = 2\nlearning-rate = 0.05\n'
        'model-rates = [1.0, 0.25]\nrate-distribution = [0.3, 0.7]\n'
        '[dataset]\nname = "cifar100"\n[model]\nnum-classes = 10\n'
    )
    cfg = SimConfig.from_toml(toml)
    assert cfg.model_rates == [1.0, 0.25]
    assert abs(sum(cfg.rate_distribution) - 1.0) < 1e-12


def test_deterministic_across_instances(tmp_path):
    cfg = make_cfg(tmp_path)
    s1 = FastSim(cfg, device="cpu", data=TinyData())
    s2 = FastSim(cfg, device="cpu", data=TinyData())
    torch.manual_seed(123)
    s1.run(progress_every=0)
    torch.manual_seed(456)
    s2.run(progress_every=0)
    a, b = s1.global_model.state_dict(), s2.global_model.state_dict()
    for k in a:
        assert torch.equal(a[k], b[k]), k


def test_adamw_lr_schedule_shape():
    from fastsim.loop import adamw_lr_at_round
    total, base = 600, 5e-4
    warm = [adamw_lr_at_round(base, r, total) for r in range(0, 30)]
    assert all(b >= a for a, b in zip(warm, warm[1:]))
    assert abs(adamw_lr_at_round(base, 29, total) - base) < 1e-9
    mid = adamw_lr_at_round(base, 300, total)
    end = adamw_lr_at_round(base, 599, total)
    assert base > mid > end
    assert end >= 0.05 * base - 1e-9


def test_adamw_param_groups_exclusions():
    from fastsim.loop import adamw_param_groups
    import torch.nn as nn

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 4)
            self.norm = nn.LayerNorm(4)
            self.pos_embed = nn.Parameter(torch.zeros(1, 4))
            self.cls_token = nn.Parameter(torch.zeros(1, 4))

    groups = adamw_param_groups(Toy(), 0.05)
    decay, no_decay = groups[0]["params"], groups[1]["params"]
    assert len(decay) == 1
    assert len(no_decay) == 5
    assert groups[0]["weight_decay"] == 0.05 and groups[1]["weight_decay"] == 0.0
