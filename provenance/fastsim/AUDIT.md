# Independent numerical audit of AAAI headline claims

Date: 2026-07-05.
Auditor: adversarial recompute from raw probe JSON and training logs.
Source of claims: `/home/dawitkr/SAPEF/FLoRE-aaai-baseline/AAAI_RESULTS.md`.
Recompute basis: `results/{surface5,kscale,acfleet,vitfleet}/probes/*/summary.json`, `logs/bracket/*.log`, `logs/ghost_gate/armb_se1_*.log`.
Deployable accuracy convention: per checkpoint, mean of `R1_ccvr` over head_seeds; FL-seed aggregate = mean over the s42..s46 checkpoints.

## Verdict

All headline numbers the task named reproduce from raw files.
Zero numerical mismatches at the reported precision.
One LOW-severity labeling nuance on a single qualitative annotation (K=50 p010 "interior"), described in section 2.

## 1. n=5 surface (Table 1) - surface5 probes

Recompute reproduces `results/surface5/surface_tables.txt` cell-for-cell to two decimals (all 54 CIFAR-100 cells + 18 CIFAR-10 cells), so only the argmax drift and endpoints are tabulated here.

### CIFAR-100 (n=5, R1 core+CCVR %)

| p_full | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 1.0 | argmax | core-full |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 52.15 | 52.21 | 51.94 | 51.68 | 51.08 | 49.46 | 0.375 | +2.69 |
| 0.20 | 51.80 | 52.24 | 52.55 | 52.50 | 52.48 | 51.84 | 0.5 | -0.04 |
| 0.30 | 51.81 | 52.45 | 52.89 | 53.11 | 53.36 | 53.15 | 0.75 | -1.33 |
| 0.40 | 50.33 | 51.12 | 51.89 | 52.53 | 53.26 | 53.92 | 1.0 | -3.59 |
| 0.60 | 49.09 | 49.83 | 51.19 | 52.27 | 53.57 | 55.15 | 1.0 | -6.06 |
| 0.80 | 44.30 | 46.68 | 49.57 | 52.25 | 54.90 | 58.76 | 1.0 | -14.46 |

Argmax drift 0.375 / 0.5 / 0.75 / 1.0 / 1.0 / 1.0. Claim: 0.375/0.5/0.75/1/1/1. STATUS OK.

### CIFAR-10 (n=5, R1 core+CCVR %)

| p_full | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 1.0 | argmax | core-full |
|---|---|---|---|---|---|---|---|---|
| 0.10 | 73.00 | 73.40 | 73.16 | 72.37 | 70.93 | 67.14 | 0.375 | +5.87 |
| 0.30 | 71.41 | 72.12 | 72.38 | 72.63 | 72.31 | 71.43 | 0.625 | -0.02 |
| 0.60 | 69.31 | 70.30 | 71.05 | 71.69 | 72.01 | 72.52 | 1.0 | -3.22 |

Argmax drift 0.375 / 0.625 / 1.0. Claim: 0.375/0.625/1. STATUS OK.

Interior RIDGE confirmed for the BN-family ResNet at every non-saturated row (interior argmax strictly exceeds BOTH endpoints): p010 52.21 > core 52.15 and > full 49.46; p020 52.55 > 51.80/51.84; p030 53.36 > 51.81/53.15.
This is the contrast partner for the ViT interior VALLEY in section 4.

### Raw surface (tab:rawsurface, R0 standalone, n=5) - bonus check

Raw argmax drift, CIFAR-100: 0.25 / 0.375 / 0.5 / 0.5 / 0.75 / 1.0. Claim identical. STATUS OK.
Raw argmax drift, CIFAR-10: 0.375 / 0.625 / 1.0. Claim identical. STATUS OK.
The noted "0.05pp inversion at c10 p030" reproduces (r0.5 42.57 sits 0.05 below r0.375 42.62 before rising to r0.625 42.75); argmax still 0.625.

## 2. Fraction-vs-count K-sweep + IID controls

n=3 paired (s42-44). K=50 rows are the surface5 s42-44 subset. Core = r0.25, full = r1.0. Deltas are per-seed core-full, mean +/- sample std.

| label | core | full | core-full (claim) | core-full (recomputed) | argmax | status |
|---|---|---|---|---|---|---|
| K50  p010 | 52.24 | 48.96 | +3.28+/-1.47 | +3.28+/-1.47 | 0.25* | OK (see note) |
| K100 p010 | 52.04 | 48.89 | +3.15+/-0.63 | +3.15+/-0.63 | 0.375 | OK |
| K200 p010 | 50.52 | 47.61 | +2.91+/-0.72 | +2.91+/-0.72 | 0.375 | OK |
| K500 p010 | 45.15 | 40.18 | +4.97+/-0.32 | +4.97+/-0.32 | 0.25 | OK |
| K100 p005 | 52.24 | 46.42 | +5.82+/-0.95 | +5.82+/-0.95 | 0.25 | OK |
| K200 p0025 | 51.25 | 39.13 | +12.12+/-2.79 | +12.12+/-2.79 | 0.25 | OK |
| K50  p040 | 50.14 | 54.29 | -4.15+/-0.67 | -4.15+/-0.67 | 1.0 | OK |
| K100 p040 | 50.34 | 55.07 | -4.73+/-0.55 | -4.73+/-0.55 | 1.0 | OK |
| K200 p040 | 47.45 | 55.12 | -7.66+/-1.04 | -7.66+/-1.04 | 1.0 | OK |
| K500 p040 | 40.44 | 49.34 | -8.90+/-1.55 | -8.90+/-1.55 | 1.0 | OK |
| IID p010 | 56.39 | 50.63 | +5.76+/-0.27 | +5.76+/-0.27 | 0.25 | OK |
| IID K200 p0025 | 51.27 | 33.82 | +17.44+/-0.45 | +17.44+/-0.45 | 0.25 | OK |

All twelve core-minus-full values (including the two the task named explicitly, +5.76 and +17.44) match to the last reported digit.

LOW note (*): the K=50 p010 row is labeled "interior" in the AAAI K-sweep table, but at the seed-matched n=3 subset the argmax lands on the CORE (r0.25 = 52.24) by a 0.03pp margin over the best interior width (r0.375 = 52.21).
It becomes genuinely interior at n=5 (Table 1, argmax r0.375 = 52.21 > core 52.15) and is interior at n=3 for K=100 and K=200.
This is a qualitative-label edge case inside noise (row std ~1.5pp), not a numerical error; the audited core-minus-full delta for that row is exact.

## 3. Oracle bracket ladder + SE1 same-basis quadrant

### Bracket ladder (logs/bracket, raw sBN-recal acc, n=3)

| rung | claim | recomputed | seeds | status |
|---|---|---|---|---|
| FL-full (raw R0 r1.0 het p010) | 25.1 +/- 2.6 | 25.08 +/- 2.63 | 25.85/27.24/22.15 | OK |
| Arm A (oracle shell, 5-client pool) | 32.7 | 32.74 +/- 2.37 | 34.03/34.19/30.00 | OK |
| frozen FL core standalone | 38.2 | 38.15 +/- 1.40 | 39.63/36.84/37.98 | OK |
| Arm B (oracle shell, all data, FL core) | 55.2 | 55.23 +/- 1.57 | 53.59/55.38/56.72 | OK |
| Arm B standalone homo025 core | 56.1 | 56.06 +/- 0.25 | 55.77/56.18/56.23 | OK |
| Arm B homolarge clean core | 67.6 | 67.57 +/- 0.40 | 67.30/68.03/67.37 | OK |

Ladder order FL-full 25.08 < ArmA 32.74 < core 38.15 < ArmB 55.23 holds strictly. STATUS OK.
Derived reads reproduce: ArmA is 5.41pp below core (claim 5.4); coverage component ArmB-ArmA = 22.49 (claim +22.5); substrate component ArmB_homolarge-ArmB_FL = 12.34 (claim ~12.3).

### SE1 same-basis quadrant (logs/ghost_gate/armb_se1_*.log, global SKYLINE acc %)

| quantity | log | claim | recomputed | status |
|---|---|---|---|---|
| S_fwd (fwd on, bwd off) | armb_se1_fon_boff | 67.83 | 67.83 | OK |
| S_bwd (fwd off, bwd on) | armb_se1_foff_bon | 69.44 | 69.44 | OK |
| S_onon (both on) | armb_se1_fon_bon | 67.26 | 67.26 | OK |
| S_offoff (both off) | armb_se1_foff_boff | 68.10 | 68.10 | OK |

All four exact.
Pre-registered ladder-stop: STOP iff S_fwd - S_offoff < 2.0pp.
S_fwd - S_offoff = 67.83 - 68.10 = -0.27pp < 2.0 => STOP fires. STATUS OK.
The forward pathway carries no capturable substrate share (it is slightly negative), and D = S_onon - S_offoff = -0.84pp also trips the D<4pp sanity gate (centralized SE1 does not even reproduce the in-context gap), which only reinforces the stop.

## 4. ViT second-backbone surface (vitfleet probes)

n=3 (s42-44), rates {0.125..1.0}, core = r0.125, full = r1.0.

| p_full | 0.125 | 0.25 | 0.375 | 0.5 | 0.625 | 0.75 | 1.0 | argmax | core-full (claim) | recomputed | status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.10 | 31.65 | 12.19 | 15.05 | 16.79 | 18.82 | 20.02 | 22.22 | 0.125 | +9.43+/-1.25 | +9.43+/-1.25 | OK |
| 0.30 | 29.09 | 12.83 | 16.32 | 19.27 | 22.13 | 24.13 | 27.52 | 0.125 | +1.56+/-0.54 | +1.56+/-0.54 | OK |
| 0.60 | 21.93 | 14.03 | 17.21 | 20.49 | 23.68 | 25.38 | 33.37 | 1.0 | -11.43+/-0.57 | -11.43+/-0.57 | OK |

Core/full endpoints, deltas, and per-row argmax all match exactly.
Interior VALLEY confirmed at p010 and p030: every interior width (r0.25..r0.75) sits BELOW both endpoints.
p010 interior spans 12.19 (r0.25) up to 20.02 (r0.75), both under core 31.65 and full 22.22 (claim example "12.19 at r0.25 rising to 20.02 at r0.75" reproduces exactly).
p030 interior spans 12.83 up to 24.13, both under core 29.09 and full 27.52.
At p060 the full model is the argmax so the row is not a valley, consistent with the claim scoping the valley to p010/p030.
The ViT interior VALLEY vs the ResNet interior RIDGE (section 1) is confirmed: the interior optimum is a per-channel-normalization (BN-family) phenomenon; under LayerNorm the interior collapses below both endpoints.

## Mismatch register

No numerical mismatch at reported precision across Tasks 1-4.

LOW-1 (labeling, not a number): AAAI K-sweep table labels the K=50 p010 argmax "interior"; at the seed-matched n=3 subset the argmax is the core (r0.25) by 0.03pp. Interior at n=5 and at n=3 for K=100/200. Within-noise qualitative edge case. Recommendation: either footnote the K=50 row as an n=3 near-tie or report the n=5 argmax to keep the "interior" label unambiguous.
