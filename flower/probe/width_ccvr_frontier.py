#!/usr/bin/env python3
"""Width-accuracy frontier: what is the smallest deployable sub-model within 2-3pp of
full-width, and is the core-to-full gap a capacity (architecture) limit?

The shell-distill gate showed an oracle readout on the FL rate-0.25 core caps ~5-6pp below
full-width CCVR, i.e. the gap is core CAPACITY, not the readout. This probe maps the gap as a
function of model rate: for each rate it extracts the sub-model from the full checkpoint,
sBN-recalibrates, and reports core+CCVR (R1) and a supervised MLP on real features (R2, the
readout ceiling). The "ballpark of full-width" target then reads off directly as the smallest
rate whose R1/R2 lands within the chosen margin of full-width.
"""
import argparse
import json
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from probe.shell_distill_gate import extract, feat_loader, index_loader, train_real_head  # noqa: E402
from probe.skyline_common import DEV  # noqa: E402
from probe.width_aware_calibration_oracle import (  # noqa: E402
    build_sampler,
    cfg_get,
    configure_dataset,
    load_hetero_full_and_core,
    make_calib_loader,
    make_split,
    make_test_loader,
    read_toml,
    set_scaler_eval_mode,
    train_classifier,
)
from flore.hetero_models import calibrate_bn_stats, create_hetero_model  # noqa: E402
from flore.hetero_utils import create_extractor  # noqa: E402


def load_submodel(full, cfg, rate, calib_loader, bn_batches):
    model_name = cfg_get(cfg, "model.name", "HeteroPreResNet18")
    num_classes = int(cfg_get(cfg, "model.num_classes", 100))
    norm_type = cfg_get(cfg, "norm_type", "sbn")
    rnd_seed = int(cfg_get(cfg, "rnd_seed", cfg_get(cfg, "dataset.seed", 42)))
    sub = create_hetero_model(model_name, in_channels=3, num_classes=num_classes,
                              model_rate=float(rate), norm_type=norm_type)
    extractor = create_extractor(strategy="static", global_model=full, num_classes=num_classes, seed=rnd_seed)
    sub_state = extractor.extract_sub_model(
        client_model=sub, client_idx=0, round_num=int(cfg_get(cfg, "num_server_rounds", 300)))
    sub.load_state_dict(OrderedDict((k, v.detach().cpu()) for k, v in sub_state.items()), strict=True)
    sub.to(DEV)
    calibrate_bn_stats(sub, calib_loader, DEV, num_batches=bn_batches)
    set_scaler_eval_mode(sub, bool(cfg_get(cfg, "core_force_scaler_eval", True)))
    sub.eval()
    for p in sub.parameters():
        p.requires_grad_(False)
    return sub


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--rates", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--ccvr-epochs", type=int, default=60)
    parser.add_argument("--ccvr-lr", type=float, default=0.05)
    parser.add_argument("--ccvr-samples-per-class", type=int, default=450)
    parser.add_argument("--head-seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--bn-batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--output-root", default=str(PROJECT_ROOT / "results" / "width_ccvr_frontier"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--skip-supervised", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    cfg_probe = read_toml(args.config)
    configure_dataset(cfg_get(cfg_probe, "dataset.name", "cifar100"),
                      int(cfg_get(cfg_probe, "model.num_classes", 100)))

    rates = [float(r) for r in args.rates.split(",") if r.strip()]
    train_idx, hold_idx = make_split()
    if args.debug:
        train_idx, hold_idx = train_idx[:4000], hold_idx[:2000]

    full, _, cfg = load_hetero_full_and_core(
        checkpoint=args.checkpoint, config_path=args.config, bn_batches=args.bn_batches,
        batch_size=args.batch_size, workers=args.workers, calib_indices=train_idx)
    calib_loader = make_calib_loader(batch_size=args.batch_size, workers=args.workers, indices=train_idx)
    tr_loader = index_loader(train_idx, args.batch_size, args.workers)
    ho_loader = index_loader(hold_idx, args.batch_size, args.workers)
    te_loader = make_test_loader(batch_size=args.batch_size, workers=args.workers)

    rows: List[Dict[str, object]] = []
    for rate in rates:
        sub = load_submodel(full, cfg, rate, calib_loader, args.bn_batches)
        tr_f, tr_logits, y_tr = extract(sub, tr_loader)
        ho_f, _, y_ho = extract(sub, ho_loader)
        te_f, te_logits, y_te = extract(sub, te_loader)
        dim = tr_f.shape[1]
        r0 = float((te_logits.argmax(1) == y_te).float().mean().item())
        ho_fl = feat_loader(ho_f, y_ho)
        te_fl = feat_loader(te_f, y_te)
        for seed in args.head_seeds:
            sampler = build_sampler("gauss_lw", tr_f, y_tr)
            _, te1, ep1 = train_classifier(sampler, dim, ho_fl, te_fl, head_seed=seed,
                                           epochs=args.ccvr_epochs, lr=args.ccvr_lr,
                                           samples_per_class=args.ccvr_samples_per_class)
            if args.skip_supervised:
                r2_acc = None
            else:
                _, te2, ep2, _ = train_real_head(tr_f, y_tr, None, ho_fl, te_fl, head_type="mlp",
                                                 hidden=args.hidden, epochs=args.epochs, lr=args.lr,
                                                 kd_alpha=0.0, kd_temp=4.0, seed=seed)
                r2_acc = te2["acc"]
            rows.append({"rate": rate, "feat_dim": dim, "head_seed": seed,
                         "R0_standalone": r0, "R1_ccvr": te1["acc"], "R2_supervised": r2_acc})
            print(f"[frontier] rate={rate:.3f} dim={dim} seed={seed}  R0={r0:.4f}  "
                  f"R1_ccvr={te1['acc']:.4f}  R2_sup={r2_acc if r2_acc is None else f'{r2_acc:.4f}'}", flush=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.tag}" if args.tag else ""
    out_dir = Path(args.output_root) / f"{stamp}{tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "summary.json", "w") as handle:
        json.dump({"args": {k: str(v) for k, v in vars(args).items()}, "rows": rows}, handle, indent=2)
    print(f"[frontier] wrote {out_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
