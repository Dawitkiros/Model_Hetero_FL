import csv
from collections import OrderedDict
from pathlib import Path

import torch

from flore.depthfl import (
    consistency_weight,
    get_depth_for_partition,
    train_depthfl,
)
from flore.depthfl_models import create_depthfl_model, num_state_entries, test_depthfl

from .aggregate import aggregate_depthfl
from .data import make_dataset
from .loaders import ClientLoader, EvalLoader
from .loop import decay_schedule_for, lr_at_round


class DepthFLSim:
    """Single-process DepthFL simulator: depth-prefix distribution, positional
    unweighted count aggregation, per-exit CE + all-pairs self-distillation."""

    def __init__(self, cfg, device="cuda:0", data_root="./data", data=None):
        self.cfg = cfg
        self.device = torch.device(device)
        self.data = data if data is not None else make_dataset(cfg.dataset_name, self.device, root=data_root)
        parts = self.data.partitions(cfg.num_clients, cfg.alpha, cfg.data_seed, cfg.partitioning, cfg.data_fraction)
        self.client_train = []
        self.client_sizes = []
        for pid in range(cfg.num_clients):
            tr, _ = self.data.client_split(parts[pid], seed=cfg.data_seed * 100003 + pid)
            self.client_train.append(tr)
            self.client_sizes.append(len(tr))
        self.depths = [
            get_depth_for_partition(pid, cfg.num_clients, cfg.depth_levels, cfg.depth_distribution)
            for pid in range(cfg.num_clients)
        ]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.rnd_seed)
            self.global_model = create_depthfl_model(
                cfg.global_depth, num_classes=cfg.num_classes, norm_type=cfg.norm_type,
            ).to(self.device)
        self.depth_entries = {
            d: num_state_entries(d, cfg.num_classes, cfg.norm_type) for d in sorted(set(self.depths))
        }
        self.sample_gen = torch.Generator().manual_seed(cfg.rnd_seed)
        self.train_gen = torch.Generator(device=self.device).manual_seed(cfg.rnd_seed + 1)
        self._depth_models = {}
        self._test_loader = None
        if self.device.type == "cuda" and cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
        self._maybe_compile(self.global_model)

    def _maybe_compile(self, model):
        if self.cfg.compile_mode != "off" and self.device.type == "cuda":
            model.compile(mode=self.cfg.compile_mode)
        return model

    def _depth_model(self, depth):
        if depth not in self._depth_models:
            self._depth_models[depth] = self._maybe_compile(create_depthfl_model(
                depth, num_classes=self.cfg.num_classes, norm_type=self.cfg.norm_type,
            ).to(self.device))
        return self._depth_models[depth]

    def _load_prefix(self, model):
        gvals = list(self.global_model.state_dict().values())
        new = OrderedDict()
        for i, key in enumerate(model.state_dict().keys()):
            v = gvals[i].detach().clone()
            new[key] = v.to(torch.long) if key.endswith("num_batches_tracked") else v.to(torch.float32)
        model.load_state_dict(new, strict=True)

    def run_round(self, server_round):
        cfg = self.cfg
        picks = torch.randperm(cfg.num_clients, generator=self.sample_gen)[: cfg.clients_per_round].tolist()
        lr = lr_at_round(cfg.learning_rate, server_round, decay_schedule_for(cfg.partitioning))
        cw = consistency_weight(max(0, server_round - 1), cfg.kd_rampup_rounds)
        client_states = []
        for pid in picks:
            model = self._depth_model(self.depths[pid])
            self._load_prefix(model)
            loader = ClientLoader(self, self.client_train[pid])
            train_depthfl(
                model, loader, self.device,
                epochs=cfg.local_epochs, learning_rate=lr, weight_decay=cfg.weight_decay,
                kd=cfg.depthfl_kd, cw=cw, extended=cfg.depthfl_extended,
                autocast_dtype=torch.bfloat16 if cfg.bf16 else None,
            )
            client_states.append(OrderedDict((k, v.detach().clone()) for k, v in model.state_dict().items()))
        new_state = aggregate_depthfl(self.global_model.state_dict(), client_states)
        self.global_model.load_state_dict(new_state, strict=True)
        return picks

    def evaluate(self, server_round):
        if self._test_loader is None:
            self._test_loader = EvalLoader(self.data, "test", None, batch_size=self.cfg.eval_batch)
        with torch.inference_mode():
            loss, acc_full, acc_single = test_depthfl(self.global_model, self._test_loader, self.device)
        core_idx = max(0, min(int(self.cfg.core_depth) - 1, len(acc_single) - 1))
        acc_core = acc_single[core_idx]
        metrics = OrderedDict()
        metrics["acc_full"] = float(acc_full)
        metrics["acc_core"] = float(acc_core)
        metrics["delta_acc"] = float(acc_full - acc_core)
        for i, a in enumerate(acc_single):
            metrics[f"acc_exit{i + 1}"] = float(a)
        return metrics

    def run(self, progress_every=50, log_fn=print, eval_every=1, csv_path=None):
        writer, fh = None, None
        if csv_path:
            Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
            fh = open(csv_path, "w", newline="")
        for server_round in range(2, self.cfg.num_rounds + 1):
            self.run_round(server_round)
            if eval_every and server_round % eval_every == 0:
                metrics = self.evaluate(server_round)
                if fh is not None:
                    if writer is None:
                        writer = csv.writer(fh)
                        writer.writerow(["round"] + list(metrics.keys()))
                    writer.writerow([server_round] + [metrics[k] for k in metrics])
                    fh.flush()
                if progress_every and server_round % progress_every == 0:
                    log_fn(f"[depthfl] round {server_round}/{self.cfg.num_rounds} "
                           f"full={metrics['acc_full']:.4f} core={metrics['acc_core']:.4f} "
                           f"delta={metrics['delta_acc']:.4f}")
            elif progress_every and server_round % progress_every == 0:
                log_fn(f"[depthfl] round {server_round}/{self.cfg.num_rounds}")
        if fh is not None:
            fh.close()
        return self.global_model

    def save(self, path=None):
        if path is None:
            if not self.cfg.save_path:
                raise ValueError("no save path: config has empty save-path and no explicit path given")
            path = str(Path(self.cfg.save_path).with_suffix("")) + "_finalmodel.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.global_model.state_dict(), path)
        return path
