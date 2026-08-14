"""Decisive gradient-conflict gate.

Both the core-protection idea (#12) and the ghost/phantom-substrate ideas (#23/#24) rest on ONE
shared premise: that on the SHARED core-prefix params, the gradient a core-only client applies
(width-0.25 forward) and the gradient a full-width client applies (width-1.0 forward) point in
CONFLICTING directions. This script measures that directly.

Isolate the width effect: for the SAME held-out minibatch, compute
  g_core^core = d/d(core-prefix params) [ CE( f_{w=0.25}(x), y ) ]
  g_core^full = d/d(core-prefix params) [ CE( f_{w=1.0}(x),  y ) ]
concatenate over all trainable core-prefix param slices (covered(model, 0.25)), and report
  cos(g_core^core, g_core^full)   -- near 1 => no conflict (both ideas solve a non-problem)
  ||g_core^core|| / ||g_core^full||  -- >>1 => core clients push the shared core "too hard"
averaged over several minibatches, at an EARLY and a LATE partially-trained HeteroFL checkpoint.

Substrate: an actual HeteroFL(p_full=0.3) slimmable model trained here for `--rounds` rounds
(the deployment-relevant, FedAvg-aggregated shared core), snapshotted early and late.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from crosswidth_fl_gate import (
    SlimResNet18, SBN, set_width, covered, load_cifar100, dirichlet_partition,
    local_train, recalibrate, evaluate, MEAN, STD,
)


def core_prefix_slices(model):
    """Ordered list of (trainable_param_name, index_tuple) covered by a rate-0.25 client."""
    cov = covered(model, 0.25)
    pnames = set(n for n, _ in model.named_parameters())
    items = [(n, idx) for n, idx in cov.items() if n in pnames]
    items.sort(key=lambda t: t[0])  # deterministic, identical order for both width forwards
    return items


def grad_on_core(model, xb, yb, width, slices):
    """Flattened grad of the width-`width` CE loss w.r.t. the core-prefix param slices."""
    set_width(width)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(xb), yb)
    loss.backward()
    pn = dict(model.named_parameters())
    parts, head_len = [], None
    for name, idx in slices:
        g = pn[name].grad
        gs = (g[idx].flatten() if g is not None
              else torch.zeros_like(pn[name][idx]).flatten())
        parts.append(gs)
    return torch.cat(parts)


def fc_mask(model, slices):
    """Boolean mask (len = concat vector) marking the classifier-head (fc) coords."""
    segs = []
    for name, idx in slices:
        n = model.get_parameter(name)[idx].numel()
        segs.append(torch.full((n,), name.startswith("fc."), dtype=torch.bool))
    return torch.cat(segs)


def measure_conflict(model, xte, yte, mean, std, slices, gen, n_batches=16, bs=128):
    model.train()  # sBN batch statistics = the training-time gradient signal; no running-stat update
    cos_all, ratio_all, cos_body = [], [], []
    g_core_sum = g_full_sum = None
    hmask = fc_mask(model, slices).to(xte.device)
    body = ~hmask
    for _ in range(n_batches):
        b = torch.randint(0, len(xte), (bs,), generator=gen, device=xte.device)
        xb = (xte[b] - mean) / std
        yb = yte[b]
        gcc = grad_on_core(model, xb, yb, 0.25, slices)
        gcf = grad_on_core(model, xb, yb, 1.0, slices)
        cos = F.cosine_similarity(gcc, gcf, dim=0).item()
        ratio = (gcc.norm() / (gcf.norm() + 1e-12)).item()
        cb = F.cosine_similarity(gcc[body], gcf[body], dim=0).item()
        cos_all.append(cos); ratio_all.append(ratio); cos_body.append(cb)
        g_core_sum = gcc.clone() if g_core_sum is None else g_core_sum + gcc
        g_full_sum = gcf.clone() if g_full_sum is None else g_full_sum + gcf
    model.zero_grad(set_to_none=True)
    agg_cos = F.cosine_similarity(g_core_sum, g_full_sum, dim=0).item()
    agg_cos_body = F.cosine_similarity(g_core_sum[body], g_full_sum[body], dim=0).item()
    agg_ratio = (g_core_sum.norm() / (g_full_sum.norm() + 1e-12)).item()
    return {
        "cos_mean": float(np.mean(cos_all)), "cos_std": float(np.std(cos_all)),
        "cos_min": float(np.min(cos_all)), "cos_max": float(np.max(cos_all)),
        "cos_body_mean": float(np.mean(cos_body)),
        "ratio_mean": float(np.mean(ratio_all)), "ratio_std": float(np.std(ratio_all)),
        "agg_cos": agg_cos, "agg_cos_body": agg_cos_body, "agg_ratio": agg_ratio,
        "n_batches": n_batches, "bs": bs,
        "n_core_coords": int(sum(model.get_parameter(n)[i].numel() for n, i in slices)),
    }


def train_heterofl(a, dev):
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    gen = torch.Generator(device=dev).manual_seed(a.seed)
    xtr, ytr, xte, yte = load_cifar100(a.root, dev)
    mean, std = MEAN.to(dev), STD.to(dev)
    parts = dirichlet_partition(ytr, a.n_clients, a.alpha, a.seed)
    parts = [torch.tensor(p, device=dev) for p in parts]
    sizes = np.array([len(p) for p in parts])
    n_full = int(a.p_full * a.n_clients)
    rate_of = {i: (1.0 if i < n_full else 0.25) for i in range(a.n_clients)}
    glob = SlimResNet18(100).to(dev)
    rng = np.random.RandomState(a.seed)
    ckpts = {}
    t0 = time.time()
    for rnd in range(a.rounds):
        lr = a.lr * 0.5 * (1 + np.cos(np.pi * rnd / a.rounds))
        sampled = rng.choice(a.n_clients, a.cpr, replace=False)
        states, rates, weights = [], [], []
        for cid in sampled:
            if len(parts[cid]) < 2: continue
            cm = SlimResNet18(100).to(dev); cm.load_state_dict(glob.state_dict())
            local_train(cm, xtr, ytr, parts[cid], rate_of[cid], "heterofl",
                        a.local_epochs, lr, a.bs, mean, std, gen)
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
        if (rnd + 1) in a.snaps:
            ckpts[rnd + 1] = {k: v.detach().clone() for k, v in glob.state_dict().items()}
            recalibrate(glob, xtr, 0.25, mean, std, gen); ca = evaluate(glob, xte, yte, 0.25, mean, std)
            recalibrate(glob, xtr, 1.0, mean, std, gen); fa = evaluate(glob, xte, yte, 1.0, mean, std)
            ckpts[rnd + 1]["_acc"] = (ca, fa)
            print(f"[heterofl p{a.p_full}] rnd{rnd+1} core={ca:.4f} full={fa:.4f} "
                  f"({(time.time()-t0)/60:.1f}m)", flush=True)
    return glob, ckpts, (xtr, ytr, xte, yte, mean, std, gen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p_full", type=float, default=0.3)
    ap.add_argument("--n_clients", type=int, default=100)
    ap.add_argument("--cpr", type=int, default=10)
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--snaps", type=str, default="25,80,300")
    ap.add_argument("--local_epochs", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--root", default="data")
    ap.add_argument("--out", default="results/grad_conflict.json")
    a = ap.parse_args()
    a.snaps = [int(s) for s in a.snaps.split(",")]
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={dev}", flush=True)

    glob, ckpts, (xtr, ytr, xte, yte, mean, std, gen) = train_heterofl(a, dev)
    slices = core_prefix_slices(glob)

    results = {}
    for rnd in a.snaps:
        tag = f"r{rnd}"
        ca, fa = ckpts[rnd].pop("_acc")
        m = SlimResNet18(100).to(dev)
        m.load_state_dict(ckpts[rnd])
        gg = torch.Generator(device=dev).manual_seed(1234)  # fixed probe minibatches, same across ckpts
        results[tag] = {"round": rnd, "core_acc": ca, "full_acc": fa,
                        **measure_conflict(m, xte, yte, mean, std, slices, gg)}
        r = results[tag]
        print(f"\n== {tag} (round {rnd}, core={ca:.4f} full={fa:.4f}) ==", flush=True)
        print(f"  cos(g_core^core, g_core^full): mean={r['cos_mean']:.4f} +- {r['cos_std']:.4f} "
              f"[min {r['cos_min']:.4f}, max {r['cos_max']:.4f}]  agg={r['agg_cos']:.4f}", flush=True)
        print(f"  body-only (no fc head) cos:    mean={r['cos_body_mean']:.4f}  agg={r['agg_cos_body']:.4f}", flush=True)
        print(f"  ||g^core||/||g^full|| ratio:   mean={r['ratio_mean']:.4f} +- {r['ratio_std']:.4f}  "
              f"agg={r['agg_ratio']:.4f}", flush=True)
        print(f"  core-prefix coords={r['n_core_coords']}", flush=True)

    meta = {"p_full": a.p_full, "n_clients": a.n_clients, "cpr": a.cpr, "rounds": a.rounds,
            "alpha": a.alpha, "seed": a.seed}
    out = {"meta": meta, "results": results}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nRESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
