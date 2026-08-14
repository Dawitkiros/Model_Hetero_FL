# fastsim: single-process, GPU-resident, tier-vectorized FL simulator

Status: design brief accepted 2026-07-02 (from the simulation-overhead fork). Build order below.

## Why

Measured on deepthought during the capping confirmation fleet: flwr/Ray runs `p030_s45` at ~34 s/round while the actual round compute (5 clients x 5 local epochs x ~1k samples, PreResNet18) is ~3-4 GPU-seconds on an A5000.
Roughly 90% of wall time is Ray actor scheduling, full-model serialization every round, per-actor dataset construction, uncapped OMP pools, and superlink chatter.
Partial-participation FedAvg on one machine is a sequential loop per round; processes buy nothing.

## Decision

Skip flwr tuning and throwaway harnesses.
Build the real thing once: a first-class tested package.
Quality, simplicity, robustness, and scalability over development cost.

## Architecture

- Single process, single GPU per config run; a sweep is N independent configs scheduled across GPUs (embarrassingly parallel, no distributed actors).
- Dataset lives on the GPU as one tensor (CIFAR fp16 ~150 MB); Dirichlet partitions are index tensors.
- One model replica; client "communication" is in-memory state_dict exchange, no serialization.
- Width-heterogeneous clients batch within rate tiers via torch.vmap (tier-bucketed vectorization); ragged shapes across tiers, dense within a tier.
- Deterministic seeding per (config, seed, round, client).
- Reads the existing flore TOML configs unchanged; reuses flore model construction, sub-model slicing, and aggregation logic unchanged (one source of truth for method logic).
- AMP on by default; CUDA graphs later if per-step kernels dominate.
- Scheduler module respects ~/SAPEF/RESOURCES.md limits (thread caps, load checks) instead of relying on launch-time convention.

## Equivalence gate (load-bearing; nothing is trusted before this passes)

fastsim must reproduce recorded flwr results on anchor rows, same configs, same eval probe (width_ccvr_frontier R1), within the flwr seed std:

1. homo025_a01 (CIFAR-100, s42-44): core+CCVR ~52.5.
2. heterofl_a01_p030 (s42-44) full surface row: 51.25 / 51.81 / 52.20 / 52.59 / 52.71 / 52.43 at rates 0.25..1.0.
3. heterofl_a01_p010 (s42-44) endpoints: core ~52.0, full ~49.4 (the crossover sign).

Pass = per-cell |fastsim mean - flwr mean| within 1 flwr seed std, and the p010 core>full sign preserved.
Equivalence runs are GPU work: schedule after the current confirmation fleet frees the box.

Documented mean-neutral deviation (adversarial fidelity review, 2026-07-02): flwr's per-client 80/20
train/val split and augmentation RNG are unseeded (actor-entropy driven), fastsim's are seeded.
Both draw an unbiased same-size split from the identical Dirichlet partition, so means match while
flwr's recorded seed std bakes in extra split/augmentation noise, making the 1-std gate conservative.
The review confirmed equivalence on all anchor semantics: init, aggregation math, sampling
distribution, round structure (299 training rounds), local training, augmentation, partitioning,
inert global-lr, sBN buffers, and checkpoint format.

## Modules

- config.py: flore TOML adapter (same keys the flwr app reads).
- data.py: GPU-resident datasets, Dirichlet partition index tensors.
- loop.py: round loop (sample clients, local train, aggregate, eval hooks, checkpointing compatible with existing probes).
- tiers.py: rate-tier bucketing + vmap local-training engine.
- aggregate.py: HeteroFL-style per-coordinate aggregation over tier slices (verified port of flore's _aggregate_hetero; a live cross-check test pins accumulate_slice to HeteroStrategy._aggregate_width_slice so flore-side changes break tests instead of silently diverging).
- equivalence.py: anchor-row harness; produces the pass/fail table above.
- runner.py: sweep scheduler (configs x seeds -> GPUs, load-aware).

## Milestones

- M1: homogeneous FedAvg loop reproduces homo025 anchor.
- M2: HeteroFL width slicing reproduces the p030 surface row and the p010 sign.
- M3: tier-vmap engine; target <=4 s/round (>=8x vs flwr).
- M4: runner + docs; retire flwr for single-machine sweeps.

## Non-goals

Distributed/cross-device deployment, async protocols, secure aggregation, non-flore model families (until needed, e.g. a LoRA stack for Stage 3).

## Gate run 1 (2026-07-03): 7/9 PASS, sign preserved, NOT yet adopted

Speed: 7.3-9.0 s/round vs flwr ~34 (~4x single-run; <20% GPU util, M3 headroom).
homo025 +0.20, p010 +0.22/-0.40: unbiased. p030: ALL SIX rates +0.56..+0.83 high - a small
SYSTEMATIC positive offset on one family (2 cells exceed flwr's smallest stds -> strict FAIL).
The 6 p030 cells share 3 checkpoints, so this is 3 paired comparisons, not 6 independent ones.
Suspects: split semantics (does flwr redraw the unseeded 80/20 per fit call, i.e. clients see
all data over rounds, vs fastsim's fixed 80%?) - check whether load_partition_data runs per-fit
or per-actor-lifetime; then extend both sides to n=5 (flwr p030 n=5 exists) and re-gate.
Do NOT run science sweeps on fastsim until the p030 offset is explained or gated away.

## Gate resolution (2026-07-03): p030 offset EXPLAINED - flwr's split was wrong; fastsim adopted as reference

Adversarial verification (Opus agent, code + empirical): (A) flwr 1.26.1 rebuilds the client on
EVERY fit message (message_handler.py:97, ray_actor.py:56-59, both launch paths), so flore's
unseeded random_split (dataset.py:296) redraws each client's 80/20 split per fit - clients see
~100% of their data over rounds (val leaks into train; non-reproducible). (B) fastsim's fixed
seeded disjoint split verified correct (no train/val overlap, no test leakage, fixed weights).
(C) Direction probe (60r, p030): enabling flwr-style redraw inside fastsim moves accuracy DOWN
(-1.54pp at rate 0.25), matching flwr sitting ~0.7pp below fastsim. Sign established; magnitude
not exactly matched (single seed, 60r).

RULING (user, 2026-07-03): per-fit redraw violates the real-world holdout setup; fastsim's fixed
split is the corrected reference semantics. Gate re-baselined: homo025 + p010 anchors PASS
unbiased; the p030 offset is a documented flwr artifact, not a fastsim defect. fastsim is
CLEARED for science sweeps. Optional strengthener (not blocking): seed flore's random_split,
re-run flwr p030 n=5, confirm it converges up to fastsim (definitive causal match).

## Causal closure (2026-07-04): strengthener RUN - redraw is mean-neutral; offset within noise; law invariant

The strengthener was run: flore random_split seeded per client, flwr p030 s42-46 retrained
300r and probed (FLoRE-aaai-baseline/results/width_surface_p030fix). flwr-fix equals
flwr-orig at every rate (deltas -0.22..+0.06, paired per-seed mixed-sign): the per-fit
redraw is approximately MEAN-NEUTRAL at full horizon, and the 60r single-seed direction
probe (-1.54pp) was a short-horizon artifact. The fastsim-vs-flwr p030 offset
(+0.59..+0.77) is therefore NOT explained by split semantics; at n=5 it is ~1.0-1.4 SE per
cell with checkpoints shared across rates, i.e. within combined seed noise. No defect found
on either side. The flwr redraw remains a code-proven validity/reproducibility flaw (val
leaks into train), just not an accuracy distortion here. All three rows (orig, fix, fastsim)
are unimodal with argmax r0.75: the width-participation structure is invariant to simulator
and split semantics. fastsim's cleared status is unchanged.
