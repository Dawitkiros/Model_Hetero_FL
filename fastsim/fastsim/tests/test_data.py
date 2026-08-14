import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastsim.config import SimConfig
from fastsim import data as fsdata


def crop_reference(x, oy, ox, flip):
    n = x.shape[0]
    padded = F.pad(x, (4, 4, 4, 4))
    out = torch.empty_like(x)
    for i in range(n):
        c = padded[i, :, oy[i]:oy[i] + 32, ox[i]:ox[i] + 32]
        out[i] = c.flip(-1) if flip[i] else c
    return out


def test_augment_matches_reference():
    torch.manual_seed(0)
    x = torch.rand(16, 3, 32, 32)
    gen = torch.Generator().manual_seed(7)
    got = fsdata.augment(x, gen)
    gen2 = torch.Generator().manual_seed(7)
    oy = torch.randint(0, 9, (16,), generator=gen2)
    ox = torch.randint(0, 9, (16,), generator=gen2)
    flip = torch.rand(16, generator=gen2) < 0.5
    want = crop_reference(x, oy, ox, flip)
    assert torch.equal(got, want)


def test_config_reads_flore_toml(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text(
        'num-clients = 50\nnum-server-rounds = 300\nlocal-epochs = 5\n'
        'clients-per-round = 5\nrnd-seed = 43\nlearning-rate = 0.05\n'
        'momentum = 0.9\nweight-decay = 0.0005\nglobal-model-rate = 1.0\n'
        'model-rates = "0.75,0.25"\nrate-distribution = "0.30,0.70"\n'
        'save-path = "results/x.csv"\n'
        '[algorithm]\nname = "heterofl"\n'
        '[dataset]\nname = "cifar100"\npartitioning = "dirichlet"\n'
        'alpha = 0.1\nbatch-size = 32\nseed = 43\n'
        '[model]\nname = "HeteroPreResNet18"\nnum-classes = 100\n'
    )
    cfg = SimConfig.from_toml(toml)
    assert cfg.num_clients == 50
    assert cfg.model_rates == [0.75, 0.25]
    assert cfg.rate_distribution == [0.30, 0.70]
    assert cfg.alpha == 0.1
    assert cfg.data_seed == 43
    assert cfg.num_classes == 100


def test_client_split_deterministic_and_disjoint():
    ds = object.__new__(fsdata.GpuCifar)
    ds.device = torch.device("cpu")
    idx = list(range(100, 200))
    tr1, va1 = fsdata.GpuCifar.client_split(ds, idx, seed=5)
    tr2, va2 = fsdata.GpuCifar.client_split(ds, idx, seed=5)
    assert torch.equal(tr1, tr2) and torch.equal(va1, va2)
    assert len(va1) == 20 and len(tr1) == 80
    assert set(tr1.tolist()).isdisjoint(set(va1.tolist()))


def test_fraction_keep_set_stratified_deterministic():
    ds = object.__new__(fsdata.GpuCifar)
    ds.train_y = torch.tensor([0] * 100 + [1] * 60 + [2] * 3)
    k1 = fsdata.GpuCifar._fraction_keep_set(ds, 0.5, seed=42)
    k2 = fsdata.GpuCifar._fraction_keep_set(ds, 0.5, seed=42)
    assert k1 == k2
    y = ds.train_y
    counts = {c: sum(1 for i in k1 if y[i].item() == c) for c in (0, 1, 2)}
    assert counts == {0: 50, 1: 30, 2: 2}
    k3 = fsdata.GpuCifar._fraction_keep_set(ds, 0.5, seed=43)
    assert k1 != k3


def test_partitions_data_fraction_filters_iid_parts():
    ds = object.__new__(fsdata.GpuCifar)
    ds.train_y = torch.tensor(list(range(10)) * 100)
    parts = fsdata.GpuCifar.partitions(ds, 10, alpha=0.1, seed=42,
                                       partitioning="iid", data_fraction=0.5)
    total = sum(len(p) for p in parts)
    assert total == 500
    full = fsdata.GpuCifar.partitions(ds, 10, alpha=0.1, seed=42, partitioning="iid")
    for p_half, p_full in zip(parts, full):
        assert set(p_half).issubset(set(p_full))
    try:
        fsdata.GpuCifar.partitions(ds, 10, alpha=0.1, seed=42,
                                   partitioning="iid", data_fraction=0.0)
        raised = False
    except ValueError:
        raised = True
    assert raised
