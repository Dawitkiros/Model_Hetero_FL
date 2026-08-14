import numpy as np
import torch
import torch.nn.functional as F
import torchvision as tv

from flore.dataset import partition_dirichlet

STATS = {
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
}

DATASETS = {"cifar100": tv.datasets.CIFAR100, "cifar10": tv.datasets.CIFAR10}


class GpuCifar:
    def __init__(self, name, device, root="./data"):
        cls = DATASETS[name]
        train = cls(root=root, train=True, download=False)
        test = cls(root=root, train=False, download=False)
        mean, std = STATS[name]
        self.device = torch.device(device)
        self.mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)
        self.train_x = self._to_device(train.data)
        self.train_y = torch.tensor(train.targets, device=self.device, dtype=torch.long)
        self.test_x = self._to_device(test.data)
        self.test_y = torch.tensor(test.targets, device=self.device, dtype=torch.long)
        self._train_raw = train

    def _to_device(self, arr):
        x = torch.from_numpy(np.ascontiguousarray(arr)).to(self.device)
        return x.permute(0, 3, 1, 2).contiguous().float().div_(255.0)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def partitions(self, num_clients, alpha, seed, partitioning="dirichlet", data_fraction=1.0):
        mode = str(partitioning).lower()
        if mode == "iid":
            total = len(self.train_y)
            per = total // num_clients
            parts = [
                list(range(pid * per, pid * per + per if pid < num_clients - 1 else total))
                for pid in range(num_clients)
            ]
        elif mode == "dirichlet":
            parts = partition_dirichlet(self._train_raw, num_clients, alpha, seed)
        else:
            raise ValueError(f"unsupported partitioning: {partitioning}")
        if not 0.0 < float(data_fraction) <= 1.0:
            raise ValueError(f"data-fraction must be in (0, 1]: {data_fraction}")
        if float(data_fraction) < 1.0:
            keep = self._fraction_keep_set(float(data_fraction), seed)
            parts = [[i for i in p if i in keep] for p in parts]
        return parts

    def _fraction_keep_set(self, fraction, seed):
        g = torch.Generator().manual_seed(seed * 7919 + 13)
        y = self.train_y.cpu()
        keep = []
        for c in torch.unique(y).tolist():
            idx = torch.nonzero(y == c, as_tuple=False).flatten()
            perm = torch.randperm(len(idx), generator=g)
            keep.append(idx[perm[: round(len(idx) * fraction)]])
        return set(torch.cat(keep).tolist())

    def client_split(self, indices, seed):
        g = torch.Generator().manual_seed(seed)
        idx = torch.tensor(indices, dtype=torch.long)
        perm = torch.randperm(len(idx), generator=g)
        n_val = int(len(idx) * 0.2)
        return idx[perm[n_val:]].to(self.device), idx[perm[:n_val]].to(self.device)


class GpuFemnist(GpuCifar):
    """flwrlabs/femnist (3597 natural writers, 62 classes, 28x28 grayscale) adapted to the
    CIFAR pipeline: pad 28->32, replicate gray->3ch, reuse HeteroPreResNet18. Natural
    writer-based partitioning. Built from data/femnist_cache.pt (scripts/build_femnist_cache.py)."""
    def __init__(self, name, device, root="./data"):
        cache = torch.load(f"{root}/femnist_cache.pt", map_location="cpu")
        self.device = torch.device(device)
        tx = cache["train_x"].to(self.device)                       # uint8 [N,28,28]
        tx = tx.unsqueeze(1).to(torch.float16).div_(255.0)          # [N,1,28,28]
        tx = F.pad(tx, (2, 2, 2, 2))                                # -> [N,1,32,32]
        self.train_x = tx.expand(-1, 3, -1, -1).contiguous()       # [N,3,32,32] fp16
        self.train_y = cache["train_y"].to(self.device).long()
        self.writer = cache["writer"].to(self.device).long()
        self.n_writers = int(cache["n_writers"])
        m, sd = 0.737869, 0.431342   # precomputed padded FEMNIST stats (avoids a full fp32 copy)
        self.mean = torch.tensor([m, m, m], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([sd, sd, sd], device=self.device).view(1, 3, 1, 1)
        self.test_x = cache["test_x"]        # uint8 [Nt,28,28] on CPU (probe handles eval)
        self.test_y = cache["test_y"]
        self._train_raw = None

    def partitions(self, num_clients, alpha, seed, partitioning="natural", data_fraction=1.0):
        mode = str(partitioning).lower()
        if mode in ("natural", "writer"):
            order = torch.argsort(self.writer)               # group examples by writer id
            w_sorted = self.writer[order]
            boundaries = torch.nonzero(torch.diff(w_sorted), as_tuple=False).flatten() + 1
            chunks = torch.tensor_split(order, boundaries.cpu())
            parts = [c.cpu().tolist() for c in chunks]       # n_writers partitions
        elif mode == "iid":
            total = len(self.train_y)
            per = total // num_clients
            parts = [list(range(pid * per, pid * per + per if pid < num_clients - 1 else total))
                     for pid in range(num_clients)]
        else:
            raise ValueError(f"femnist supports partitioning natural|iid, got {partitioning}")
        if float(data_fraction) < 1.0:
            keep = self._fraction_keep_set(float(data_fraction), seed)
            parts = [[i for i in p if i in keep] for p in parts]
        return parts


def make_dataset(name, device, root="./data"):
    if str(name).lower() == "femnist":
        return GpuFemnist(name, device, root=root)
    return GpuCifar(name, device, root=root)


def draw_augment(n, gen, device, hflip=True):
    oy = torch.randint(0, 9, (n,), generator=gen, device=device)
    ox = torch.randint(0, 9, (n,), generator=gen, device=device)
    if hflip:
        flip = torch.rand(n, generator=gen, device=device) < 0.5
    else:
        flip = None
    return oy, ox, flip


def apply_augment(x, oy, ox, flip):
    n = x.shape[0]
    device = x.device
    padded = F.pad(x, (4, 4, 4, 4))
    n_idx = torch.arange(n, device=device).view(n, 1, 1)
    y_idx = oy.view(n, 1, 1) + torch.arange(32, device=device).view(1, 32, 1)
    x_idx = ox.view(n, 1, 1) + torch.arange(32, device=device).view(1, 1, 32)
    out = padded[n_idx, :, y_idx, x_idx].permute(0, 3, 1, 2)
    if flip is not None:
        out = torch.where(flip.view(n, 1, 1, 1), out.flip(-1), out)
    return out.contiguous()


def augment(x, gen, hflip=True):
    oy, ox, flip = draw_augment(x.shape[0], gen, x.device, hflip=hflip)
    return apply_augment(x, oy, ox, flip)
