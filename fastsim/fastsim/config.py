from dataclasses import dataclass

from flore.hetero_utils import parse_rate_config

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def read_toml(path):
    with open(path, "rb") as fh:
        return tomllib.load(fh)


@dataclass
class SimConfig:
    raw: dict
    num_clients: int
    num_rounds: int
    local_epochs: int
    clients_per_round: int
    rnd_seed: int
    learning_rate: float
    momentum: float
    weight_decay: float
    global_lr: float
    global_model_rate: float
    model_rates: list
    rate_distribution: list
    core_model_rate: float
    norm_type: str
    client_optimizer: str
    depth_levels: list
    depth_distribution: list
    global_depth: int
    core_depth: int
    depthfl_kd: bool
    depthfl_extended: bool
    kd_rampup_rounds: int
    fjord_kd: bool
    fjord_stack: bool
    cudnn_benchmark: bool
    compile_mode: str
    eval_batch: int
    bf16: bool
    algorithm: str
    dataset_name: str
    partitioning: str
    alpha: float
    batch_size: int
    data_seed: int
    data_fraction: float
    hflip: bool
    rate_assignment: str
    model_name: str
    num_classes: int
    save_path: str

    @classmethod
    def from_toml(cls, path):
        cfg = read_toml(path)
        ds = cfg.get("dataset", {})
        model = cfg.get("model", {})
        algo = cfg.get("algorithm", {})
        rates, dist = parse_rate_config(cfg.get("model-rates"), cfg.get("rate-distribution"))
        from flore.depthfl import parse_depth_config
        depth_levels, depth_dist = parse_depth_config(cfg.get("depth-levels"), cfg.get("depth-distribution"))
        return cls(
            raw=cfg,
            num_clients=int(cfg["num-clients"]),
            num_rounds=int(cfg["num-server-rounds"]),
            local_epochs=int(cfg["local-epochs"]),
            clients_per_round=int(cfg["clients-per-round"]),
            rnd_seed=int(cfg.get("rnd-seed", 42)),
            learning_rate=float(cfg["learning-rate"]),
            momentum=float(cfg.get("momentum", 0.9)),
            weight_decay=float(cfg.get("weight-decay", 0.0)),
            global_lr=float(cfg.get("global-lr", 1.0)),
            global_model_rate=float(cfg.get("global-model-rate", 1.0)),
            model_rates=rates,
            rate_distribution=dist,
            core_model_rate=float(cfg.get("core-model-rate", 0.25)),
            norm_type=str(cfg.get("norm-type", "sbn")),
            client_optimizer=str(cfg.get("client-optimizer", "sgd")).lower(),
            depth_levels=depth_levels,
            depth_distribution=depth_dist,
            global_depth=int(cfg.get("global-depth", max(depth_levels))),
            core_depth=int(cfg.get("core-depth", min(depth_levels))),
            depthfl_kd=bool(cfg.get("depthfl-kd", True)),
            depthfl_extended=bool(cfg.get("depthfl-extended", True)),
            kd_rampup_rounds=int(cfg.get("kd-rampup-rounds", 300)),
            fjord_kd=bool(cfg.get("fjord-kd", True)),
            fjord_stack=bool(cfg.get("fjord-stack", False)),
            cudnn_benchmark=bool(cfg.get("cudnn-benchmark", True)),
            compile_mode=str(cfg.get("compile-mode", "off")).lower(),
            eval_batch=int(cfg.get("eval-batch", 1024)),
            bf16=bool(cfg.get("bf16", False)),
            algorithm=str(algo.get("name", "heterofl")),
            dataset_name=str(ds.get("name", "cifar100")).lower(),
            partitioning=str(ds.get("partitioning", "dirichlet")),
            alpha=float(ds.get("alpha", 0.5)),
            batch_size=int(ds.get("batch-size", 32)),
            data_seed=int(ds.get("seed", 42)),
            data_fraction=float(ds.get("data-fraction", 1.0)),
            hflip=bool(cfg.get("hflip", True)),
            rate_assignment=str(cfg.get("rate-assignment", "sorted")).lower(),
            model_name=str(model.get("name", "HeteroPreResNet18")),
            num_classes=int(model.get("num-classes", 100)),
            save_path=str(cfg.get("save-path", "")),
        )
