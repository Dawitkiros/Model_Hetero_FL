import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastsim.config import SimConfig
from fastsim.loaders import EvalLoader


class _SeqLoader:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = type("S", (), {"__len__": lambda s: sum(len(b[1]) for b in batches)})()

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def _make_batches(n=3, bs=8, num_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [(torch.rand(bs, 3, 32, 32, generator=g),
             torch.randint(0, num_classes, (bs,), generator=g)) for _ in range(n)]


def test_config_perf_defaults(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text(
        'num-clients = 4\nnum-server-rounds = 2\nlocal-epochs = 1\n'
        'clients-per-round = 2\nlearning-rate = 0.05\n'
        '[dataset]\nname = "cifar100"\n[model]\nnum-classes = 10\n'
    )
    cfg = SimConfig.from_toml(toml)
    assert cfg.bf16 is False
    assert cfg.compile_mode == "off"
    assert cfg.cudnn_benchmark is True
    assert cfg.eval_batch == 1024


def test_fjord_sub_cache_exact_parity():
    from flore.fjord import train_fjord
    from flore.hetero_models import create_hetero_model

    def run(cache):
        torch.manual_seed(7)
        net = create_hetero_model("HeteroPreResNet18", in_channels=3, num_classes=10,
                                  model_rate=1.0, norm_type="sbn")
        loader = _SeqLoader(_make_batches())
        train_fjord(net, loader, torch.device("cpu"), epochs=1, learning_rate=0.05,
                    momentum=0.9, weight_decay=5e-4, model_name="HeteroPreResNet18",
                    num_classes=10, norm_type="sbn", candidate_rates=[0.25, 1.0],
                    max_rate=1.0, kd=True, seed=3, sub_model_cache=cache)
        return net.state_dict()

    a = run(None)
    cache = {}
    b = run(cache)
    c = run(cache)
    assert 0.25 in cache
    for k in a:
        assert torch.equal(a[k], b[k]), k
        assert torch.equal(a[k], c[k]), k


def test_depthfl_bf16_flag_runs_and_default_off():
    from flore.depthfl import train_depthfl
    from flore.depthfl_models import create_depthfl_model

    def run(dtype):
        torch.manual_seed(5)
        net = create_depthfl_model(2, num_classes=10, norm_type="bn")
        loader = _SeqLoader(_make_batches(n=2))
        train_depthfl(net, loader, torch.device("cpu"), epochs=1, learning_rate=0.1,
                      weight_decay=1e-3, kd=True, cw=0.5, extended=True,
                      autocast_dtype=dtype)
        return net.state_dict()

    fp32_a = run(None)
    fp32_b = run(None)
    for k in fp32_a:
        assert torch.equal(fp32_a[k], fp32_b[k]), k
    bf16 = run(torch.bfloat16)
    for k in bf16:
        assert torch.isfinite(bf16[k].float()).all(), k
    changed = sum(1 for k in fp32_a if not torch.equal(fp32_a[k], bf16[k]))
    assert changed > 0


def test_eval_batch_size_invariance():
    from flore.depthfl_models import create_depthfl_model, test_depthfl
    g = torch.Generator().manual_seed(1)

    class _Data:
        test_x = torch.rand(300, 3, 32, 32, generator=g)
        test_y = torch.randint(0, 10, (300,), generator=g)
        train_x = test_x
        train_y = test_y

        @staticmethod
        def normalize(x):
            return x

    torch.manual_seed(2)
    net = create_depthfl_model(4, num_classes=10, norm_type="bn")
    net.eval()
    outs = []
    for bs in (32, 256):
        loader = EvalLoader(_Data, "test", None, batch_size=bs)
        loss, acc, singles = test_depthfl(net, loader, torch.device("cpu"))
        outs.append((loss / len(loader), acc, singles))
    (l1, a1, s1), (l2, a2, s2) = outs
    assert a1 == a2
    assert s1 == s2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cudagraph kd regression needs CUDA")
def test_fjord_kd_learns_under_reduce_overhead():
    from flore.fjord import train_fjord
    from flore.hetero_models import create_hetero_model

    torch.manual_seed(9)
    net = create_hetero_model("HeteroPreResNet18", in_channels=3, num_classes=10,
                              model_rate=0.25, norm_type="sbn").cuda()
    net.compile(mode="reduce-overhead")
    g = torch.Generator().manual_seed(1)
    x = torch.rand(64, 3, 32, 32, generator=g).cuda()
    y = torch.randint(0, 10, (64,), generator=g).cuda()
    batches = [(x[i * 16:(i + 1) * 16], y[i * 16:(i + 1) * 16]) for i in range(4)]
    loader = _SeqLoader(batches)

    def ce():
        net.train()
        with torch.no_grad():
            return torch.nn.functional.cross_entropy(net(x[:32]), y[:32]).item()

    before = ce()
    train_fjord(net, loader, torch.device("cuda"), epochs=15, learning_rate=0.05,
                momentum=0.9, weight_decay=0.0, model_name="HeteroPreResNet18",
                num_classes=10, norm_type="sbn", candidate_rates=[0.25],
                max_rate=0.25, kd=True, seed=3)
    after = ce()
    assert after < before * 0.5, (before, after)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="compile parity needs CUDA")
def test_compiled_train_step_parity_fp32():
    from flore.depthfl import train_depthfl
    from flore.depthfl_models import create_depthfl_model

    def run(compiled):
        torch.manual_seed(11)
        net = create_depthfl_model(1, num_classes=10, norm_type="bn").cuda()
        if compiled:
            net.compile(mode="default")
        batches = [(x.cuda(), y.cuda()) for x, y in _make_batches(n=1, seed=4)]
        train_depthfl(net, _SeqLoader(batches), torch.device("cuda"), epochs=1,
                      learning_rate=0.01, weight_decay=1e-3, kd=True, cw=0.5, extended=True)
        return {k: v.detach().cpu() for k, v in net.state_dict().items()}

    old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        eager = run(False)
        comp = run(True)
    finally:
        torch.backends.cudnn.allow_tf32 = old_cudnn_tf32
        torch.backends.cuda.matmul.allow_tf32 = old_matmul_tf32
    assert set(eager) == set(comp)
    for k in eager:
        assert torch.allclose(eager[k].float(), comp[k].float(), atol=1e-4, rtol=1e-4), k
