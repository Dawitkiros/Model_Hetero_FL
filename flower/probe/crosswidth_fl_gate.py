"""Sparse-participation FL cross-width test: does the central +1.1pp inplace-distillation lift to the
deployable core SURVIVE when the full-width teacher is under-trained by sparse participation?

Setup: HeteroFL on CIFAR-100, N clients, Dirichlet non-IID, clients_per_round sampled, a fraction p_full
are full-capable (rate 1.0), the rest core-only (rate 0.25). Shared-prefix BN (sBN, recalibrated at eval).
Correct HeteroFL prefix aggregation: each coordinate averaged (data-weighted) only over clients that COVER it
(core coords over all sampled clients, shell coords over full clients only; untouched if no full sampled).

Two arms (only the full clients' local objective differs):
  heterofl   : full clients train width 1.0 only              (standard HeteroFL baseline)
  crosswidth : full clients train sandwich+KD (1.0 hard-label + 0.25 inplace-distillation from 1.0's soft out)
Core-only clients train width 0.25 in both arms.
Metric: deployable core (rate 0.25) accuracy after eval-time BN recalibration; also full (1.0) accuracy.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv

WIDTHS = [0.25, 1.0]
_S = {"w": 1.0}
_RECAL = {"on": False}
def set_width(w): _S["w"] = w
def _c(cmax): return max(1, round(cmax * _S["w"]))

class SConv(nn.Module):
    def __init__(self, cin_max, cout_max, k, stride=1, pad=0, first=False):
        super().__init__()
        self.conv = nn.Conv2d(cin_max, cout_max, k, stride, pad, bias=False)
        self.cin_max, self.cout_max, self.first = cin_max, cout_max, first
    def forward(self, x):
        cin = self.cin_max if self.first else _c(self.cin_max)
        cout = _c(self.cout_max)
        return F.conv2d(x, self.conv.weight[:cout, :cin], None, self.conv.stride, self.conv.padding)

class SBN(nn.Module):
    """Shared-prefix BN. Train: batch stats, no running-stat tracking (sBN). Recalib: track running stats
    on the deployable slice. Eval: use recalibrated running stats. All at the current-width prefix."""
    def __init__(self, cmax):
        super().__init__()
        self.bn = nn.BatchNorm2d(cmax)
    def forward(self, x):
        c = x.shape[1]
        if _RECAL["on"]:
            return F.batch_norm(x, self.bn.running_mean[:c], self.bn.running_var[:c],
                                self.bn.weight[:c], self.bn.bias[:c], True, 0.1, self.bn.eps)
        if self.training:
            return F.batch_norm(x, None, None, self.bn.weight[:c], self.bn.bias[:c], True, 0.0, self.bn.eps)
        return F.batch_norm(x, self.bn.running_mean[:c], self.bn.running_var[:c],
                            self.bn.weight[:c], self.bn.bias[:c], False, 0.0, self.bn.eps)

class SLinear(nn.Module):
    def __init__(self, cin_max, out):
        super().__init__()
        self.fc = nn.Linear(cin_max, out); self.cin_max = cin_max
    def forward(self, x):
        return F.linear(x, self.fc.weight[:, :_c(self.cin_max)], self.fc.bias)

class Block(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1, self.b1 = SConv(cin, cout, 3, stride, 1), SBN(cout)
        self.c2, self.b2 = SConv(cout, cout, 3, 1, 1), SBN(cout)
        self.short = None
        if stride != 1 or cin != cout:
            self.short = nn.ModuleList([SConv(cin, cout, 1, stride, 0), SBN(cout)])
    def forward(self, x):
        out = F.relu(self.b1(self.c1(x)))
        out = self.b2(self.c2(out))
        sc = x if self.short is None else self.short[1](self.short[0](x))
        return F.relu(out + sc)

class SlimResNet18(nn.Module):
    def __init__(self, num_classes=100):
        super().__init__()
        self.stem, self.sb = SConv(3, 64, 3, 1, 1, first=True), SBN(64)
        self.l1 = self._stage(64, 64, 2, 1)
        self.l2 = self._stage(64, 128, 2, 2)
        self.l3 = self._stage(128, 256, 2, 2)
        self.l4 = self._stage(256, 512, 2, 2)
        self.fc = SLinear(512, num_classes)
    def _stage(self, cin, cout, n, stride):
        layers = [Block(cin, cout, stride)] + [Block(cout, cout, 1) for _ in range(n - 1)]
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.relu(self.sb(self.stem(x)))
        x = self.l4(self.l3(self.l2(self.l1(x))))
        return self.fc(F.adaptive_avg_pool2d(x, 1).flatten(1))

def covered(model, rate):
    """Return {full_param_name: index_tuple} covered by a client of this rate (HeteroFL prefix)."""
    set_width(rate); out = {}
    for name, mod in model.named_modules():
        if isinstance(mod, SConv):
            co = _c(mod.cout_max); ci = mod.cin_max if mod.first else _c(mod.cin_max)
            out[f"{name}.conv.weight"] = (slice(0, co), slice(0, ci))
        elif isinstance(mod, SBN):
            c = _c(mod.bn.num_features)
            for k in ("weight", "bias", "running_mean", "running_var"):
                out[f"{name}.bn.{k}"] = (slice(0, c),)
        elif isinstance(mod, SLinear):
            out[f"{name}.fc.weight"] = (slice(0, mod.fc.out_features), slice(0, _c(mod.cin_max)))
            out[f"{name}.fc.bias"] = (slice(0, mod.fc.out_features),)
    return out

MEAN = torch.tensor([0.5071, 0.4867, 0.4408]).view(1, 3, 1, 1)
STD = torch.tensor([0.2675, 0.2565, 0.2761]).view(1, 3, 1, 1)

def load_cifar100(root, device):
    tr = tv.datasets.CIFAR100(root=root, train=True, download=False)
    te = tv.datasets.CIFAR100(root=root, train=False, download=False)
    def to_t(d): return torch.from_numpy(np.ascontiguousarray(d.data)).permute(0, 3, 1, 2).float().div_(255.0)
    return (to_t(tr).to(device), torch.tensor(tr.targets, device=device),
            to_t(te).to(device), torch.tensor(te.targets, device=device))

def dirichlet_partition(y, n_clients, alpha, seed):
    rng = np.random.RandomState(seed); y = y.cpu().numpy(); K = int(y.max() + 1)
    idx_by_c = [np.where(y == c)[0] for c in range(K)]
    parts = [[] for _ in range(n_clients)]
    for c in range(K):
        rng.shuffle(idx_by_c[c])
        props = rng.dirichlet(np.repeat(alpha, n_clients))
        cuts = (np.cumsum(props) * len(idx_by_c[c])).astype(int)[:-1]
        for pid, chunk in enumerate(np.split(idx_by_c[c], cuts)):
            parts[pid].extend(chunk.tolist())
    return [np.array(p) for p in parts]

def augment(x, gen):
    n = x.shape[0]; p = F.pad(x, (4, 4, 4, 4))
    oy = torch.randint(0, 9, (n,), generator=gen, device=x.device)
    ox = torch.randint(0, 9, (n,), generator=gen, device=x.device)
    ni = torch.arange(n, device=x.device).view(n, 1, 1)
    yi = oy.view(n, 1, 1) + torch.arange(32, device=x.device).view(1, 32, 1)
    xi = ox.view(n, 1, 1) + torch.arange(32, device=x.device).view(1, 1, 32)
    out = p[ni, :, yi, xi].permute(0, 3, 1, 2)
    flip = torch.rand(n, generator=gen, device=x.device) < 0.5
    return torch.where(flip.view(n, 1, 1, 1), out.flip(-1), out).contiguous()

def local_train(model, xtr, ytr, idx, rate, arm, epochs, lr, bs, mean, std, gen):
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    for _ in range(epochs):
        perm = idx[torch.randperm(len(idx), generator=gen, device=idx.device)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            xb = (augment(xtr[b], gen) - mean) / std; yb = ytr[b]
            opt.zero_grad()
            if rate == 1.0 and arm == "crosswidth":
                set_width(1.0); of = model(xb); lf = F.cross_entropy(of, yb)
                set_width(0.25); oc = model(xb)
                lc = F.kl_div(F.log_softmax(oc, 1), F.softmax(of.detach(), 1), reduction="batchmean")
                loss = lf + lc
            else:
                set_width(rate); loss = F.cross_entropy(model(xb), yb)
            loss.backward(); opt.step()

@torch.no_grad()
def recalibrate(model, xtr, width, mean, std, gen, nb=100, bs=256):
    set_width(width)
    for m in model.modules():
        if isinstance(m, SBN): m.bn.reset_running_stats()
    _RECAL["on"] = True; model.eval()
    for _ in range(nb):
        b = torch.randint(0, len(xtr), (bs,), generator=gen, device=xtr.device)
        model((xtr[b] - mean) / std)
    _RECAL["on"] = False

@torch.no_grad()
def evaluate(model, xte, yte, width, mean, std, bs=1000):
    model.eval(); set_width(width); correct = 0
    for i in range(0, len(xte), bs):
        correct += (model((xte[i:i + bs] - mean) / std).argmax(1) == yte[i:i + bs]).sum().item()
    return correct / len(xte)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["heterofl", "crosswidth"])
    ap.add_argument("--p_full", type=float, default=0.1)
    ap.add_argument("--n_clients", type=int, default=100)
    ap.add_argument("--cpr", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = torch.device("cuda:0")
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    xtr, ytr, xte, yte = load_cifar100(a.root, dev)
    mean, std = MEAN.to(dev), STD.to(dev)
    parts = dirichlet_partition(ytr, a.n_clients, a.alpha, a.seed)
    parts = [torch.tensor(p, device=dev) for p in parts]
    sizes = np.array([len(p) for p in parts])
    n_full = int(a.p_full * a.n_clients)
    rate_of = {i: (1.0 if i < n_full else 0.25) for i in range(a.n_clients)}  # first n_full are full-capable
    glob = SlimResNet18(100).to(dev)
    rng = np.random.RandomState(a.seed)
    t0 = time.time()
    for rnd in range(a.rounds):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * rnd / a.rounds))
        sampled = rng.choice(a.n_clients, a.cpr, replace=False)
        states, rates, weights = [], [], []
        for cid in sampled:
            if len(parts[cid]) < 2: continue
            cm = SlimResNet18(100).to(dev); cm.load_state_dict(glob.state_dict())
            local_train(cm, xtr, ytr, parts[cid], rate_of[cid], a.arm, a.local_epochs, lr, a.bs, mean, std, gen)
            states.append(cm.state_dict()); rates.append(rate_of[cid]); weights.append(float(sizes[cid]))
        gsd = glob.state_dict()
        num = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gsd.items()}
        den = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in gsd.items()}
        for st, r, w in zip(states, rates, weights):
            for k, idx in covered(glob, r).items():
                num[k][idx] += w * st[k][idx].float()
                den[k][idx] += w
        for k in gsd:
            if k in num:
                m = den[k] > 0
                gsd[k][m] = (num[k][m] / den[k][m]).to(gsd[k].dtype)
        glob.load_state_dict(gsd)
        if (rnd + 1) % 100 == 0 or rnd == a.rounds - 1:
            recalibrate(glob, xtr, 0.25, mean, std, gen); ca = evaluate(glob, xte, yte, 0.25, mean, std)
            recalibrate(glob, xtr, 1.0, mean, std, gen); fa = evaluate(glob, xte, yte, 1.0, mean, std)
            print(f"[{a.arm} p{a.p_full} s{a.seed}] rnd{rnd+1} core={ca:.4f} full={fa:.4f} ({(time.time()-t0)/60:.1f}m)", flush=True)
    recalibrate(glob, xtr, 0.25, mean, std, gen); core = evaluate(glob, xte, yte, 0.25, mean, std)
    recalibrate(glob, xtr, 1.0, mean, std, gen); full = evaluate(glob, xte, yte, 1.0, mean, std)
    res = {"arm": a.arm, "p_full": a.p_full, "seed": a.seed, "n_full": n_full,
           "core_acc": core, "full_acc": full, "minutes": (time.time() - t0) / 60}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=2)
    print("RESULT", json.dumps(res), flush=True)

if __name__ == "__main__":
    main()
