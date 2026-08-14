import csv
from pathlib import Path

import torch
import torch.nn.functional as F

from flore.hetero_models import create_hetero_model
from flore.hetero_utils import StaticExtractor, get_model_rate_for_partition

from .aggregate import aggregate_hetero
from .data import make_dataset, augment
from .loaders import ClientLoader, EvalLoader


def decay_schedule_for(partitioning):
    return (150, 250) if str(partitioning).lower() == "iid" else (800, 1500)


def adamw_lr_at_round(base_lr, global_round, total_rounds):
    import math
    warmup = max(1, int(0.05 * total_rounds))
    if global_round < warmup:
        return base_lr * float(global_round + 1) / warmup
    prog = (global_round - warmup) / max(1, total_rounds - warmup)
    return 0.05 * base_lr + 0.5 * (base_lr - 0.05 * base_lr) * (1 + math.cos(math.pi * prog))


def adamw_param_groups(model, weight_decay):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or n.endswith(".bias") or n.endswith(".weight_g") or "pos_embed" in n or "cls_token" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def lr_at_round(base_lr, server_round, decay_schedule=(800, 1500), decay_factor=0.1):
    lr = base_lr
    for milestone in decay_schedule:
        if server_round >= milestone + 1:
            lr *= decay_factor
    return lr


class FastSim:
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
        self.rates = [
            get_model_rate_for_partition(pid, cfg.num_clients, cfg.model_rates, cfg.rate_distribution)
            for pid in range(cfg.num_clients)
        ]
        if getattr(cfg, "rate_assignment", "sorted") == "random":
            import numpy as np
            perm = np.random.RandomState(cfg.data_seed * 100003 + 7).permutation(cfg.num_clients)
            shuffled = [None] * cfg.num_clients
            for slot, pid in enumerate(perm):
                shuffled[pid] = self.rates[slot]
            self.rates = shuffled
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg.rnd_seed)
            self.global_model = create_hetero_model(
                cfg.model_name, in_channels=3, num_classes=cfg.num_classes,
                model_rate=cfg.global_model_rate, norm_type=cfg.norm_type,
            ).to(self.device)
        self.extractor = StaticExtractor(self.global_model, num_classes=cfg.num_classes)
        self.sample_gen = torch.Generator().manual_seed(cfg.rnd_seed)
        self.train_gen = torch.Generator(device=self.device).manual_seed(cfg.rnd_seed + 1)
        self._client_models = {}
        self._fjord_subs = {}
        self._amp = self.device.type == "cuda"
        if self.device.type == "cuda" and cfg.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

    def _maybe_compile(self, model):
        if self.cfg.compile_mode != "off" and self.device.type == "cuda":
            model.compile(mode=self.cfg.compile_mode)
        return model

    def _client_model(self, rate):
        if rate not in self._client_models:
            self._client_models[rate] = self._maybe_compile(create_hetero_model(
                self.cfg.model_name, in_channels=3, num_classes=self.cfg.num_classes,
                model_rate=rate, norm_type=self.cfg.norm_type,
            ).to(self.device))
        return self._client_models[rate]

    def _local_train(self, model, idx, lr):
        cfg = self.cfg
        adamw = cfg.client_optimizer == "adamw"
        if adamw:
            opt = torch.optim.AdamW(adamw_param_groups(model, cfg.weight_decay),
                                    lr=lr, betas=(0.9, 0.999))
        else:
            opt = torch.optim.SGD(model.parameters(), lr=lr,
                                  momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        scaler = torch.amp.GradScaler(self.device.type, enabled=self._amp)
        model.train()
        n = len(idx)
        for _ in range(cfg.local_epochs):
            perm = torch.randperm(n, generator=self.train_gen, device=self.device)
            for start in range(0, n, cfg.batch_size):
                batch = idx[perm[start:start + cfg.batch_size]]
                x = self.data.train_x[batch]
                x = augment(x, self.train_gen, hflip=self.cfg.hflip)
                x = self.data.normalize(x)
                y = self.data.train_y[batch]
                opt.zero_grad(set_to_none=True)
                with torch.autocast(self.device.type, enabled=self._amp):
                    loss = F.cross_entropy(model(x), y)
                scaler.scale(loss).backward()
                if adamw:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(opt)
                scaler.update()

    def run_round(self, server_round):
        cfg = self.cfg
        picks = torch.randperm(cfg.num_clients, generator=self.sample_gen)[: cfg.clients_per_round].tolist()
        if cfg.client_optimizer == "adamw":
            lr = adamw_lr_at_round(cfg.learning_rate, server_round, cfg.num_rounds)
        else:
            lr = lr_at_round(cfg.learning_rate, server_round, decay_schedule_for(cfg.partitioning))
        states, weights = [], []
        if cfg.algorithm == "fjord" and cfg.fjord_stack:
            states, weights = self._run_fjord_round_stacked(picks, server_round, lr)
        else:
            for pid in picks:
                model = self._client_model(self.rates[pid])
                sub = self.extractor.extract_sub_model(model, pid, server_round)
                model.load_state_dict(sub, strict=True)
                if cfg.algorithm == "fjord":
                    self._train_fjord_client(model, pid, server_round, lr)
                else:
                    self._local_train(model, self.client_train[pid], lr)
                states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
                weights.append(self.client_sizes[pid])
        new_state = aggregate_hetero(self.global_model.state_dict(), states, weights)
        self.global_model.load_state_dict(new_state, strict=True)
        return picks

    def _fjord_stack_template(self, rate):
        if not hasattr(self, "_stack_templates"):
            self._stack_templates = {}
        if rate not in self._stack_templates:
            template = create_hetero_model(
                self.cfg.model_name, in_channels=3, num_classes=self.cfg.num_classes,
                model_rate=rate, norm_type=self.cfg.norm_type,
            ).to(self.device)
            template.train()
            self._stack_templates[rate] = template
        return self._stack_templates[rate]

    def _run_fjord_round_stacked(self, picks, server_round, lr):
        """FjORD round with same-width lockstep stacking. RNG consumption per
        client happens at its position in pick order, so the seeded stream is
        identical to the sequential path; only kernel-level numerics differ."""
        from flore.fjord import ordered_dropout_candidate_rates
        from .stack import FjordNarrowStack, predraw_client_steps
        cfg = self.cfg
        states, weights = [], []
        narrow = []
        for pid in picks:
            rate = self.rates[pid]
            cand = ordered_dropout_candidate_rates(cfg.model_rates, rate)
            if len(cand) == 1:
                steps = predraw_client_steps(self, self.client_train[pid], cfg.local_epochs)
                narrow.append((pid, steps, len(states)))
                states.append(None)
                weights.append(self.client_sizes[pid])
            else:
                model = self._client_model(rate)
                sub = self.extractor.extract_sub_model(model, pid, server_round)
                model.load_state_dict(sub, strict=True)
                self._train_fjord_client(model, pid, server_round, lr)
                states.append({k: v.detach().clone() for k, v in model.state_dict().items()})
                weights.append(self.client_sizes[pid])
        if narrow:
            groups = {}
            for pid, steps, slot in narrow:
                groups.setdefault(self.rates[pid], []).append((pid, steps, slot))
            if not hasattr(self, "_stack_runners"):
                self._stack_runners = {}
            for rate, members in groups.items():
                template = self._fjord_stack_template(rate)
                init_state = self.extractor.extract_sub_model(template, members[0][0], server_round)
                if rate not in self._stack_runners:
                    self._stack_runners[rate] = FjordNarrowStack(
                        template, self.device, compile_mode=cfg.compile_mode)
                runner = self._stack_runners[rate]
                outs = runner.run(init_state, [m[1] for m in members], self,
                                  lr, cfg.momentum, cfg.weight_decay, cfg.fjord_kd)
                for (pid, _, slot), state in zip(members, outs):
                    states[slot] = state
        return states, weights

    def _train_fjord_client(self, model, pid, server_round, lr):
        from flore.fjord import ordered_dropout_candidate_rates, train_fjord
        cfg = self.cfg
        max_rate = self.rates[pid]
        cand = ordered_dropout_candidate_rates(cfg.model_rates, max_rate)
        for rate in cand:
            if rate < max_rate - 1e-9 and rate not in self._fjord_subs:
                self._fjord_subs[rate] = self._maybe_compile(create_hetero_model(
                    cfg.model_name, in_channels=3, num_classes=cfg.num_classes,
                    model_rate=rate, norm_type=cfg.norm_type,
                ).to(self.device))
        loader = ClientLoader(self, self.client_train[pid])
        seed = cfg.rnd_seed * 1000003 + server_round * 131 + pid
        train_fjord(
            model, loader, self.device,
            epochs=cfg.local_epochs, learning_rate=lr,
            momentum=cfg.momentum, weight_decay=cfg.weight_decay,
            model_name=cfg.model_name, num_classes=cfg.num_classes,
            norm_type=cfg.norm_type, candidate_rates=cand, max_rate=max_rate,
            kd=cfg.fjord_kd, seed=seed,
            sub_model_cache=self._fjord_subs,
            autocast_dtype=torch.bfloat16 if cfg.bf16 else None,
        )

    def _ensure_eval_assets(self):
        if getattr(self, "_eval_ready", False):
            return
        cfg = self.cfg
        cal_gen = torch.Generator().manual_seed(cfg.rnd_seed)
        n_train = self.data.train_x.shape[0]
        calib_idx = torch.randperm(n_train, generator=cal_gen)[:5000].to(self.device)
        self._calib_loader = EvalLoader(self.data, "train", calib_idx, batch_size=cfg.batch_size)
        self._test_loader = EvalLoader(self.data, "test", None, batch_size=cfg.eval_batch)
        self._eval_full = self._maybe_compile(create_hetero_model(
            cfg.model_name, in_channels=3, num_classes=cfg.num_classes,
            model_rate=cfg.global_model_rate, norm_type=cfg.norm_type,
        ).to(self.device))
        self._eval_core = self._maybe_compile(create_hetero_model(
            cfg.model_name, in_channels=3, num_classes=cfg.num_classes,
            model_rate=cfg.core_model_rate, norm_type=cfg.norm_type,
        ).to(self.device))
        self._core_extractor = StaticExtractor(self.global_model, num_classes=cfg.num_classes)
        self._eval_ready = True

    @staticmethod
    def _set_scaler_eval(net, force):
        from flore.hetero_models import ScalerV2
        for module in net.modules():
            if isinstance(module, ScalerV2):
                module.always_scale = force

    def evaluate(self, server_round):
        """Width toxicity metric mirroring flore gen_evaluate_fn: full-width
        ensemble accuracy vs statically-extracted core accuracy, both with sBN
        recalibration and ScalerV2 eval conventions."""
        from flore.hetero_models import calibrate_bn_stats
        from flore.models import test
        self._ensure_eval_assets()
        cfg = self.cfg
        norm_type = str(cfg.norm_type).lower()
        sbn = norm_type == "sbn"
        bn_recal = bool(cfg.raw.get("bn-recalibrate-eval", True))
        core_force = bool(cfg.raw.get("core-force-scaler-eval", True))
        gmr = float(cfg.global_model_rate)
        recal = sbn and bn_recal and server_round > 1

        self._eval_full.load_state_dict(self.global_model.state_dict(), strict=True)
        if recal:
            calibrate_bn_stats(self._eval_full, self._calib_loader, self.device, num_batches=50)
        self._set_scaler_eval(self._eval_full, core_force and gmr < 1.0)
        with torch.inference_mode():
            loss_full, acc_full = test(self._eval_full, self._test_loader, self.device)

        core_state = self._core_extractor.extract_sub_model(self._eval_core, 0, server_round)
        self._eval_core.load_state_dict(core_state, strict=True)
        if recal:
            calibrate_bn_stats(self._eval_core, self._calib_loader, self.device, num_batches=50)
        self._set_scaler_eval(self._eval_core, core_force)
        with torch.inference_mode():
            loss_core, acc_core = test(self._eval_core, self._test_loader, self.device)

        return {
            "acc_full": float(acc_full),
            "acc_core": float(acc_core),
            "delta_acc": float(acc_full - acc_core),
        }

    def run(self, progress_every=50, log_fn=print, eval_every=0, csv_path=None):
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
                    log_fn(f"[fastsim] round {server_round}/{self.cfg.num_rounds} "
                           + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
            elif progress_every and server_round % progress_every == 0:
                log_fn(f"[fastsim] round {server_round}/{self.cfg.num_rounds}")
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
