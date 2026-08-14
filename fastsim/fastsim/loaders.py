import math

import torch

from .data import augment


class _Sized:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n


class ClientLoader:
    """Re-iterable loader over a client's GPU index tensor.

    Each pass (one local epoch) reshuffles and augments with the simulator's
    training generator, matching the FastSim._local_train data pipeline exactly,
    and yields ``(x, y)`` GPU batches. Re-iterable so the source training loops
    can run several epochs over it.
    """

    def __init__(self, sim, idx):
        self.sim = sim
        self.idx = idx
        self.dataset = _Sized(len(idx))

    def __iter__(self):
        sim, idx, cfg = self.sim, self.idx, self.sim.cfg
        data = sim.data
        n = len(idx)
        perm = torch.randperm(n, generator=sim.train_gen, device=sim.device)
        for start in range(0, n, cfg.batch_size):
            batch = idx[perm[start:start + cfg.batch_size]]
            x = data.train_x[batch]
            x = augment(x, sim.train_gen, hflip=cfg.hflip)
            x = data.normalize(x)
            yield x, data.train_y[batch]

    def __len__(self):
        return math.ceil(len(self.idx) / self.sim.cfg.batch_size)


class EvalLoader:
    """Re-iterable loader over GPU tensors in fixed order (normalize, no augment).

    Used for centralized test evaluation and sBN calibration.
    """

    def __init__(self, data, split, indices=None, batch_size=256):
        self.data = data
        self.src_x = data.test_x if split == "test" else data.train_x
        self.src_y = data.test_y if split == "test" else data.train_y
        if indices is None:
            indices = torch.arange(self.src_x.shape[0], device=self.src_x.device)
        self.indices = indices
        self.batch_size = batch_size
        self.dataset = _Sized(len(indices))

    def __iter__(self):
        for start in range(0, len(self.indices), self.batch_size):
            batch = self.indices[start:start + self.batch_size]
            x = self.data.normalize(self.src_x[batch])
            yield x, self.src_y[batch]

    def __len__(self):
        return math.ceil(len(self.indices) / self.batch_size)
