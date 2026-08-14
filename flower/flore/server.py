"""fedprox: A Flower Baseline."""

import json
import os
import pickle
from logging import INFO
from pathlib import Path
from secrets import token_hex

from easydict import EasyDict

from flwr.common import log
from flwr.server import Server
from flwr.server.history import History

SAVE_PATH = Path(os.path.abspath(__file__)).parent.parent / "results"


class ResultsSaverServer(Server):
    """Server to save history to disk."""

    def __init__(
        self,
        *,
        client_manager,
        strategy=None,
        results_saver_fn=None,
        run_config=None,
    ):
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.results_saver_fn = results_saver_fn
        self.run_config = run_config

    def fit(self, num_rounds, timeout):
        """Run federated averaging for a number of rounds."""
        log(INFO, "Starting federated learning...")
        history, elapsed = super().fit(num_rounds, timeout)
        if self.results_saver_fn:
            log(INFO, "Results saver function provided. Executing")
            self.results_saver_fn(history, self.run_config)
        # Save final global model weights for post-hoc analysis (zeroed-shell control).
        try:
            gm = getattr(self.strategy, "global_model", None)
            sp = self.run_config.get("save_path", self.run_config.get("save-path")) if self.run_config is not None else None
            if sp and gm is None and self.run_config is not None:
                from collections import OrderedDict
                import torch
                from flwr.common import parameters_to_ndarrays
                from .hetero_models import create_hetero_model
                net = create_hetero_model(
                    self.run_config.model.name,
                    in_channels=3,
                    num_classes=self.run_config.model.num_classes,
                    model_rate=float(self.run_config.get("global_model_rate", 1.0)),
                    norm_type=self.run_config.get("norm_type", "sbn"),
                )
                nds = parameters_to_ndarrays(self.parameters)
                sd = OrderedDict()
                for k, v in zip(net.state_dict().keys(), nds):
                    dtype = torch.long if k.endswith("num_batches_tracked") else torch.float32
                    sd[k] = torch.tensor(v, dtype=dtype)
                net.load_state_dict(sd, strict=True)
                gm = net
            if gm is not None and sp:
                import os, torch
                ckpt = os.path.splitext(str(sp))[0] + "_finalmodel.pt"
                torch.save(gm.state_dict(), ckpt)
                log(INFO, f"Saved final global model -> {ckpt}")
        except Exception as e:  # never fail a run over a checkpoint
            log(INFO, f"final-model save skipped: {e}")
        return history, elapsed


def history_saver(history: History, run_config: EasyDict): # pyright: ignore[reportAttributeAccessIssue]
    """Save the history from the run to the results directory.

    Args:
        history (History): The run's history
        run_config (dict): The experiments configuration.
    """
    log(INFO, "................")
    from datetime import datetime
    import uuid

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # First 8 chars of UUID
    file_suffix: str = (
        f"{'dataset' if run_config['dataset']['name'] else ''}" # pyright: ignore[reportAttributeAccessIssue]
        f"{'dir' if run_config['dataset']['partitioning'] == 'dirichlet' else 'iid'}" # pyright: ignore[reportAttributeAccessIssue]
        f"_C={run_config['num_clients']}" # pyright: ignore[reportAttributeAccessIssue]
        f"_B={run_config['dataset']['batch_size']}" # pyright: ignore[reportAttributeAccessIssue]
        f"_E={run_config['local_epochs']}" # pyright: ignore[reportAttributeAccessIssue]
        f"_R={run_config['num_server_rounds']}" # pyright: ignore[reportAttributeAccessIssue]
        f"{timestamp}_{unique_id}"
    )

    dataset_path = Path(run_config['model']['name']) # pyright: ignore[reportAttributeAccessIssue]
    save_results_as_pickle(
        history, file_path=SAVE_PATH / dataset_path.name / file_suffix
    )
    save_config_file(
        run_config, save_path=SAVE_PATH / dataset_path.name / file_suffix
    )


def save_results_as_pickle(
    history: History,
    file_path: str | Path,
    extra_results: dict | None = None,
    default_filename: str = "results.pkl",
) -> None:
    """Save results from simulation to pickle.

    Parameters
    ----------
    history: History
        History returned by start_simulation.
    file_path: Union[str, Path]
        Path to file to create and store both history and extra_results.
        If path is a directory, the default_filename will be used.
        path doesn't exist, it will be created. If file exists, a
        randomly generated suffix will be added to the file name. This
        is done to avoid overwritting results.
    extra_results : Optional[Dict]
        A dictionary containing additional results you would like
        to be saved to disk. Default: {} (an empty dictionary)
    default_filename: Optional[str]
        File used by default if file_path points to a directory instead
        to a file. Default: "results.pkl"
    """
    path = Path(file_path)

    # ensure path exists
    path.mkdir(exist_ok=True, parents=True)

    def _add_random_suffix(path_: Path):
        """Add a randomly generated suffix to the file name (so it doesn't.

        overwrite the file).
        """
        print(f"File `{path_}` exists! ")
        suffix = token_hex(4)
        print(f"New results to be saved with suffix: {suffix}")
        return path_.parent / (path_.stem + "_" + suffix + ".pkl")

    def _complete_path_with_default_name(path_: Path):
        """Append the default file name to the path."""
        print("Using default filename")
        return path_ / default_filename

    if path.is_dir():
        path = _complete_path_with_default_name(path)

    if path.is_file():
        # file exists already
        path = _add_random_suffix(path)

    print(f"Results will be saved into: {path}")

    data = {"history": history}
    if extra_results is not None:
        data = {**data, **extra_results}

    # save results to pickle
    with open(str(path), "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def save_config_file(config: dict, save_path: Path):
    """Save the experiment's config file to the relevant directory.

    Args:
        config (dict): Experiment config
        file_path (Path):
    """
    save_path = save_path / "config.json"
    if os.path.exists(save_path):
        log(INFO, "Config for this run has already been saved before")
    else:
        with open(save_path, "w", encoding="utf8") as fp:
            json.dump(config, fp)
