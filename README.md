# When More Capacity Hurts: Unequal Parameter Exposure in Model-Heterogeneous Federated Learning

Code to reproduce the results of the paper *When More Capacity Hurts: Unequal Parameter Exposure in Model-Heterogeneous Federated Learning* (under review).

In capacity-heterogeneous federated learning with nested sub-models, parameters outside the shared core are trained only by the capable clients.
When capable participation is sparse, these coverage-starved parameters make the deployed full model worse than its own extracted core.
The paper characterizes this reversal on the width and depth axes and traces it to the volume of data reaching the exclusive parameters.

## Repository layout

The paper reports results from multiple implementations and identifies the stack wherever results appear; the release mirrors that structure.

```
flower/     Flower/Ray stack: flore package, flwr app config, probes, calibration comparison and alpha-sweep configs
fastsim/    Vectorized single-process stack: fastsim package plus its pinned flore copy, surface/exposure/depth/ViT configs and runners
```

The two `flore` copies are intentionally separate: each subtree ships exactly the code state that produced its results.
The `fastsim` stack passed model-state and evaluation equivalence gates against the Flower implementation (paper, Appendix B).

## Installation

```bash
mamba env create -f environment.yml    # or: conda env create -f environment.yml
conda activate mhfl
```

CIFAR-10 and CIFAR-100 are downloaded through `torchvision` into `flower/data/` and `fastsim/data/` on first use.

## Reproducing the paper

All fleet scripts accept `GPUS` as a comma-separated device list (default `0`) and skip already-completed cells, so they can be re-run to resume.

### Figure 1 and the deployment surface (crossover)

Trains the five-seed HeteroFL surface on CIFAR-100 and CIFAR-10 across capable fractions, then probes every checkpoint at six extraction rates with sBN recalibration and CCVR calibration.

```bash
GPUS=0,1 bash fastsim/scripts/run_surface5.sh
python flower/scripts/plot_crossover.py --out-dir figs
```

### Table 2: exposure interventions

The K-scaling arm (fixed fraction, growing fleet) and the fixed-count arm (five writers, shrinking data share):

```bash
GPUS=0,1 bash fastsim/scripts/run_kscale.sh
```

The K=500 and IID arms use the same runner with a different config directory:

```bash
GPUS=0,1 CONF=conf/acfleet OUT=results/acfleet LOG=logs/acfleet bash fastsim/scripts/run_kscale.sh
```

The K=500 fixed-count cell (1% share) and the IID ladder cells (K=100 and K=500 at five writers) have dedicated runners:

```bash
GPUS=0,1 bash fastsim/scripts/run_k500_p001.sh
GPUS=0,1 bash fastsim/scripts/run_iid_ladder.sh
```

### Matched-optimization intervention

Each eligible client's unique pool is restricted (`shell-unique-frac`) and/or its per-epoch draws rescaled (`shell-draw-frac`) while SGD steps, minibatch layout, and aggregation weights stay at baseline values; the suite includes the u=f=1 with-replacement sampler control and the K=200 draws-x4 mirror:

```bash
GPUS=0,1 bash fastsim/scripts/run_expmatch.sh
```

### Table 3: constrained shell-training oracle

Train the frozen rate-0.25 core once, then run the oracle arms on a trained full checkpoint:

```bash
cd flower
python probe/train_core025.py
python probe/arm_a_skyline.py --checkpoint <heterofl_finalmodel.pt> --config <run_config.toml>
```

### Controls (Table 1 rows)

```bash
cd flower
python probe/control2_zeroed_shell.py --checkpoint <heterofl_finalmodel.pt> --config <run_config.toml>
python probe/grad_conflict_gate.py
```

`grad_conflict_gate.py` is the self-contained gradient-alignment simulator; it trains its own width-sliced substrate.

### Figure 2: label-heterogeneity sweep (Flower/Ray)

Each cell is one `flwr` simulation over a run config in `flower/conf/alpha_sweep/`.

```bash
cd flower
script -qefc "flower-simulation --app . --run-config conf/alpha_sweep/heterofl_a30_p030.toml --num-supernodes 50 --stream" run.log
python scripts/plot_alpha_sweep.py --out-dir figs
```

The `script -qefc` wrapper gives the simulation a pty; recent `flwr` versions silently no-op without one in non-interactive shells.

### Table 4: depth axis and distillation

Six DepthFL/FjORD cells per seed, 500 rounds, with per-round CSV metrics and health guards, then last-50-round aggregation:

```bash
bash fastsim/scripts/run_depth_axis_seeds.sh
python fastsim/scripts/agg_depth_axis_seeds.py
```

`fastsim/scripts/run_depth_axis_equiv.sh` reruns the six-cell equivalence gate against the Flower originals, and `compare_depth_axis.py` prints the comparison.

### Table 5: calibrated deployment comparison

Train the uniform-harness baselines (HeteroFL, FedRolex, FIARSE, homogeneous small and large) from `flower/conf/aaai_missing_baseline/` with the `flower-simulation` command above, then probe each final model with:

```bash
cd flower
python probe/width_ccvr_frontier.py --checkpoint <finalmodel.pt> --config <run_config.toml> --rates 0.25,1.0 --skip-supervised
```

### Vision Transformer transfer

The width-sliced ViT fleet lives in `fastsim/conf/vitfleet/` (three seeds, rate-0.125 core):

```bash
cd fastsim
python scripts/fastsim_cell.py --config conf/vitfleet/fsvit_p010_s42.toml --gpu 0 --csv results/vitfleet/fsvit_p010_s42.csv
```

Evaluate slicing behavior with `flower/probe/control2_vit.py`.

## Notes

Trained checkpoints, per-round logs, and probe outputs are not included; every result is regenerated by the commands above.
Exact seeds, rounds, and protocol constants are pinned in the config files and match the paper's appendix.

## License

MIT.
