# When More Capacity Hurts: Unequal Parameter Exposure in Model-Heterogeneous Federated Learning

Code accompanying the paper at the NeurIPS 2026 workshop **Collaborative, Open, and DECentralized training of Foundation Models**.

In model-heterogeneous federated learning, every client trains a shared core, while only capable clients train the additional parameters, called the shell.
The paper asks when those additional parameters improve the model that clients deploy, and connects deployment quality to both the data reaching the shell and the optimization it receives.

## Main findings

`p_full` is the fraction of clients capable of training the full model; `K` is the total number of clients.

- **The larger model can underperform its own core before classifier calibration.** At the sparse CIFAR-100 operating point, the full model achieves 25.1 ± 2.6% raw accuracy and its extracted core achieves 38.2 ± 1.4%, across three FL seeds.
- **The crossover persists after CCVR.** On CIFAR-100, the calibrated core-minus-full gap is +2.69 ± 1.33 percentage points at `p_full=0.10` and -6.06 ± 1.40 at `p_full=0.60`, over five FL seeds. CIFAR-10 and a width-sliced Vision Transformer show the same reversal in ordering as eligibility increases.
- **Data reach and optimization dose interact.** Restricting unique data alone produces comparatively small changes in the matched experiment. Jointly reducing unique data and per-epoch draws increases the gap to +6.87 ± 1.54 points. Increasing draws fourfold at `K=200`, with five eligible clients, reduces the gap from +12.54 ± 2.71 to +4.94 ± 0.60 points.
- **Depth heterogeneity exhibits a related deployment gap.** At `p_full=0.10`, the DepthFL depth-1 exit outperforms the four-exit ensemble by 26.8 ± 1.5 points. This comparison uses the method's distinct shallow and ensemble readouts.
- **Calibration matters for small-model comparisons.** In Table 4, the homogeneous rate-0.25 model reaches 52.58 ± 0.66% after CCVR and the HeteroFL core reaches 52.34 ± 0.92%. Their paired difference is +0.24 points, with a 95% confidence interval of [-1.02, +1.49].

A positive deployment gap means **core accuracy minus full-model accuracy**, in percentage points.
The FL seed is the replication unit: classifier repetitions are averaged within each FL seed before computing means, sample standard deviations, or paired contrasts.

## Repository layout

```text
flower/     Flower/Ray training, flore models, calibration probes, baseline and alpha-sweep configs
fastsim/    Single-process simulator, its pinned flore copy, and surface/exposure/depth/ViT experiments
```

See Appendices E, F, and G of the paper for implementation details and experimental protocols.

## Installation and reproduction

Run commands from the repository root unless a different directory is shown.

```bash
mamba env create -f environment.yml
conda activate mhfl
```

Download CIFAR-10 and CIFAR-100 with `torchvision` into `flower/data/` and `fastsim/data/` before running experiments.

### Figures 1 and 2, Table 1, and matched optimization

Set `GPUS` to your available devices, such as `GPUS=0` for one GPU.

```bash
GPUS=0,1,2,3 bash fastsim/scripts/run_surface5.sh
GPUS=0,1,2,3 bash fastsim/scripts/run_kscale.sh
GPUS=0,1,2,3 CONF=conf/acfleet OUT=results/acfleet LOG=logs/acfleet bash fastsim/scripts/run_kscale.sh
GPUS=0,1,2,3 bash fastsim/scripts/run_expmatch.sh
```

These runners train checkpoints and execute the width/CCVR probes.
For narrower reruns of the exposure ladder, use [run_k500_p001.sh](fastsim/scripts/run_k500_p001.sh) or [run_iid_ladder.sh](fastsim/scripts/run_iid_ladder.sh).
Resume checks rely on existing checkpoint files or probe-directory names; verify that each expected `summary.json` is complete before aggregation.

Regenerate Figure 3 with:

```bash
python fastsim/scripts/plot_matched.py fastsim/results figs/fig_matched_readable.pdf figs/matched_results
```

### Table 2 and diagnostic controls

Use [arm_a_skyline.py](flower/probe/arm_a_skyline.py) with an extracted core checkpoint, pool `5` or `all`, and seeds 42 through 44.
Appendix B controls: [zeroed shell](flower/probe/control2_zeroed_shell.py) and [gradient alignment](flower/probe/grad_conflict_gate.py).

### Tables 3 and 5: depth and distillation

Run the [depth configs](fastsim/conf/depth_axis) with [fastsim_cell.py](fastsim/scripts/fastsim_cell.py), then summarize the CSVs using [agg_depth_axis_seeds.py](fastsim/scripts/agg_depth_axis_seeds.py).

### Table 4 and Figure 4: Flower experiments

Run the [baseline](flower/conf/aaai_missing_baseline) and [alpha-sweep](flower/conf/alpha_sweep) configs with Flower.
For Table 4 calibration, use [width_aware_calibration_oracle.py](flower/probe/width_aware_calibration_oracle.py) with the matching checkpoint and method-specific extractor (`static`, `fiarse`, or `fedrolex`).

### ViT transfer

Train the [ViT configs](fastsim/conf/vitfleet) with `FastSim`, save the checkpoint, and evaluate rates 0.125 and 1.0 with [width_ccvr_frontier.py](flower/probe/width_ccvr_frontier.py).
Slicing check: [control2_vit.py](flower/probe/control2_vit.py).

## License

MIT.
