#!/usr/bin/env python3
"""Direct usefulness gate for the shell-information direction (fork A, strict core-only).

Question: can shell information reach a CORE-DEPLOYABLE readout beyond core+CCVR?
A core-only client computes core features at inference, so shell info can only shape the
readout's PARAMETERS, never add shell features to the input. Distillation from the
full-width model is the ORACLE UPPER BOUND on any shell-info-to-core transport (it hands the
core student the teacher's full output directly, more generous than any gradient channel).
If even oracle distillation cannot beat core+CCVR, the strict core-only target is capped by
the core feature ceiling (W1), and no shell-gradient channel can rescue it.

Readouts, all on the SAME frozen core GAP features:
  R0 core-standalone   : the extracted core's own logits (anchor).
  R1 core+CCVR         : Ledoit-Wolf Gaussian virtual-feature linear head (deployable baseline).
  R2 core+supervised   : head trained on REAL core features + true labels (is the readout the limit?).
  R3 core+KD(full->core): head trained on REAL core features, distilled from full-width teacher
                          logits (+ true labels). This is the shell-knowledge-into-core test.
Reference: full-width teacher test acc and full-width CCVR (the +9pp ceiling).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe.g1_graft_ceiling import trunk_forward  # noqa: E402
from probe.skyline_common import DEV, ROOT, TF_TEST  # noqa: E402
import probe.width_aware_calibration_oracle as wac  # noqa: E402
from probe.width_aware_calibration_oracle import (  # noqa: E402
    RealFeatureDataset,
    build_sampler,
    classifier_metrics,
    load_hetero_full_and_core,
    make_holdout_loader,
    make_split,
    make_test_loader,
    train_classifier,
)

NUM_CLASSES = 100


@torch.no_grad()
def extract(model: nn.Module, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats, logits, labels = [], [], []
    vit_like = hasattr(model, "forward_features")
    for images, target in loader:
        if vit_like:
            gap = model.forward_features(images.to(DEV))
            out = model.linear(gap)
        else:
            gap, out = trunk_forward(model, images.to(DEV), "gap")
        feats.append(gap.flatten(1).cpu())
        logits.append(out.cpu())
        labels.append(target)
    return torch.cat(feats), torch.cat(logits), torch.cat(labels)


def index_loader(indices: List[int], batch_size: int, workers: int) -> DataLoader:
    # Dataset-aware: mirrors the oracle's train-split loader so CIFAR-10 checkpoints
    # read CIFAR-10 images with CIFAR-10 normalization once configure_dataset() has run.
    return make_holdout_loader(indices, batch_size, workers)


def feat_loader(feats: torch.Tensor, labels: torch.Tensor, batch_size: int = 512) -> DataLoader:
    return DataLoader(RealFeatureDataset(feats, labels), batch_size=batch_size, shuffle=False)


def acc_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(1) == labels).float().mean().item())


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int = NUM_CLASSES, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_head(head_type: str, in_dim: int, hidden: int) -> nn.Module:
    num_classes = wac.NUM_CLASSES
    if head_type == "linear":
        head = nn.Linear(in_dim, num_classes)
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        return head
    if head_type == "mlp":
        return MLPHead(in_dim, hidden, num_classes=num_classes)
    raise ValueError(head_type)


def train_real_head(
    train_feat: torch.Tensor,
    train_y: torch.Tensor,
    teacher_logits: torch.Tensor | None,
    hold_loader: DataLoader,
    test_loader: DataLoader,
    head_type: str,
    hidden: int,
    epochs: int,
    lr: float,
    kd_alpha: float,
    kd_temp: float,
    seed: int,
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    torch.manual_seed(seed)
    head = make_head(head_type, train_feat.shape[1], hidden).to(DEV)
    optimizer = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    use_kd = teacher_logits is not None and kd_alpha > 0.0
    tensors = [train_feat, train_y] + ([teacher_logits] if use_kd else [])
    dataset = TensorDataset(*tensors)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, generator=torch.Generator().manual_seed(seed))

    best_acc, best_state, best_epoch = -1.0, None, -1
    for epoch in range(epochs):
        head.train()
        for batch in loader:
            feats = batch[0].to(DEV)
            labels = batch[1].to(DEV)
            logits = head(feats)
            ce = F.cross_entropy(logits, labels)
            if use_kd:
                t_logits = batch[2].to(DEV)
                kd = F.kl_div(
                    F.log_softmax(logits / kd_temp, dim=1),
                    F.softmax(t_logits / kd_temp, dim=1),
                    reduction="batchmean",
                ) * (kd_temp * kd_temp)
                loss = kd_alpha * kd + (1.0 - kd_alpha) * ce
            else:
                loss = ce
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
        hold = classifier_metrics(head, hold_loader)
        if hold["acc"] > best_acc:
            best_acc = hold["acc"]
            best_epoch = epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)
    head.eval()
    return classifier_metrics(head, hold_loader), classifier_metrics(head, test_loader), best_epoch, head


@torch.no_grad()
def head_logits(head: nn.Module, feats: torch.Tensor, batch_size: int = 1024) -> torch.Tensor:
    out = []
    for start in range(0, feats.shape[0], batch_size):
        out.append(head(feats[start : start + batch_size].to(DEV)).cpu())
    return torch.cat(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--head", default="mlp", choices=["linear", "mlp"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--kd-alpha", type=float, default=0.7)
    parser.add_argument("--kd-temp", type=float, default=4.0)
    parser.add_argument("--ccvr-epochs", type=int, default=60)
    parser.add_argument("--ccvr-lr", type=float, default=0.05)
    parser.add_argument("--ccvr-samples-per-class", type=int, default=450)
    parser.add_argument("--head-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bn-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "shell_distill_gate"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    train_idx, hold_idx = make_split()
    if args.debug:
        train_idx, hold_idx = train_idx[:4000], hold_idx[:2000]

    print(f"[gate] loading full+core from {args.checkpoint}", flush=True)
    full, core, cfg = load_hetero_full_and_core(
        checkpoint=args.checkpoint,
        config_path=args.config,
        bn_batches=args.bn_batches,
        batch_size=args.batch_size,
        workers=args.workers,
        calib_indices=train_idx,
    )

    tr_loader = index_loader(train_idx, args.batch_size, args.workers)
    ho_loader = index_loader(hold_idx, args.batch_size, args.workers)
    te_loader = make_test_loader(batch_size=args.batch_size, workers=args.workers)

    print("[gate] extracting core features + full teacher logits", flush=True)
    core_tr, core_tr_logits, y_tr = extract(core, tr_loader)
    core_ho, _, y_ho = extract(core, ho_loader)
    core_te, core_te_logits, y_te = extract(core, te_loader)
    full_tr, full_tr_logits, _ = extract(full, tr_loader)
    full_ho, _, _ = extract(full, ho_loader)
    full_te, full_te_logits, _ = extract(full, te_loader)

    core_dim = core_tr.shape[1]
    full_dim = full_tr.shape[1]
    rows: List[Dict[str, object]] = []

    # references
    r0 = acc_from_logits(core_te_logits, y_te)
    teacher_acc = acc_from_logits(full_te_logits, y_te)
    rows.append({"method": "R0_core_standalone", "head": "-", "head_seed": -1, "test_acc": r0})
    rows.append({"method": "ref_fullwidth_teacher", "head": "-", "head_seed": -1, "test_acc": teacher_acc})
    print(f"[gate] R0 core-standalone test={r0:.4f}   full-width teacher test={teacher_acc:.4f}", flush=True)

    core_ho_loader = feat_loader(core_ho, y_ho)
    core_te_loader = feat_loader(core_te, y_te)
    full_ho_loader = feat_loader(full_ho, y_ho)
    full_te_loader = feat_loader(full_te, y_te)

    # Strong oracle teacher: supervised MLP head on full-width REAL features (true labels).
    # This is the best full-width predictor, the most generous KD source for R3.
    hot, tet, ept, teacher_head = train_real_head(
        full_tr, y_tr, None, full_ho_loader, full_te_loader,
        head_type="mlp", hidden=args.hidden, epochs=args.epochs, lr=args.lr,
        kd_alpha=0.0, kd_temp=args.kd_temp, seed=42,
    )
    teacher_train_logits = head_logits(teacher_head, full_tr)
    rows.append({"method": "ref_fullwidth_supervised", "head": "mlp", "head_seed": 42,
                 "best_epoch": ept, "holdout_acc": hot["acc"], "test_acc": tet["acc"]})
    print(f"[gate] full-width supervised teacher test={tet['acc']:.4f}", flush=True)

    for seed in args.head_seeds:
        # R1 core+CCVR (gauss_lw virtual-feature linear head)
        sampler = build_sampler("gauss_lw", core_tr, y_tr)
        ho1, te1, ep1 = train_classifier(
            sampler, core_dim, core_ho_loader, core_te_loader,
            head_seed=seed, epochs=args.ccvr_epochs, lr=args.ccvr_lr,
            samples_per_class=args.ccvr_samples_per_class,
        )
        rows.append({"method": "R1_core_ccvr", "head": "linear", "head_seed": seed,
                     "best_epoch": ep1, "holdout_acc": ho1["acc"], "test_acc": te1["acc"]})

        # R2 core + supervised real-feature head (true labels, no shell)
        ho2, te2, ep2, _ = train_real_head(
            core_tr, y_tr, None, core_ho_loader, core_te_loader,
            head_type=args.head, hidden=args.hidden, epochs=args.epochs, lr=args.lr,
            kd_alpha=0.0, kd_temp=args.kd_temp, seed=seed,
        )
        rows.append({"method": "R2_core_supervised", "head": args.head, "head_seed": seed,
                     "best_epoch": ep2, "holdout_acc": ho2["acc"], "test_acc": te2["acc"]})

        # R3 core + KD from full-width supervised teacher (shell knowledge into core readout)
        ho3, te3, ep3, _ = train_real_head(
            core_tr, y_tr, teacher_train_logits, core_ho_loader, core_te_loader,
            head_type=args.head, hidden=args.hidden, epochs=args.epochs, lr=args.lr,
            kd_alpha=args.kd_alpha, kd_temp=args.kd_temp, seed=seed,
        )
        rows.append({"method": "R3_core_kd_fullwidth", "head": args.head, "head_seed": seed,
                     "best_epoch": ep3, "holdout_acc": ho3["acc"], "test_acc": te3["acc"]})

        # reference: full-width CCVR ceiling (uses shell features at inference, NOT core-deployable)
        fsampler = build_sampler("gauss_lw", full_tr, y_tr)
        hof, tef, epf = train_classifier(
            fsampler, full_dim, full_ho_loader, full_te_loader,
            head_seed=seed, epochs=args.ccvr_epochs, lr=args.ccvr_lr,
            samples_per_class=args.ccvr_samples_per_class,
        )
        rows.append({"method": "ref_fullwidth_ccvr", "head": "linear", "head_seed": seed,
                     "best_epoch": epf, "holdout_acc": hof["acc"], "test_acc": tef["acc"]})

        print(f"[gate] seed={seed}  R1_ccvr={te1['acc']:.4f}  R2_sup={te2['acc']:.4f}  "
              f"R3_kd={te3['acc']:.4f}  (R3-R1={100*(te3['acc']-te1['acc']):+.2f}pp)  "
              f"fw_ccvr={tef['acc']:.4f}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.output_root) / f"{stamp}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, handle, indent=2)
    print(f"[gate] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
