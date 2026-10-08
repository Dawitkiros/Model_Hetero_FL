# AAAI missing-baseline: CIFAR-100 results (2026-06-30 overnight)

Setting: HeteroPreResNet18, CIFAR-100, Dirichlet alpha=0.1, 50 clients, 5/round, 300 rounds,
3 FL seeds. Deployable object = extracted rate-0.25 core + CCVR (gauss Ledoit-Wolf, 3 head
seeds). homo_small = all clients train the rate-0.25 model (rate-distribution 0,1). The
width-het extractors get 10/30/60% full-width clients (p_full).

All competitor implementations were audited for fidelity before trusting any number; the audit
caught and we fixed a HIGH issue (FedRolex was hard-blocked from receiving a +CCVR row). See
AUDIT_FIX_NOTES.md.

## Main table (+CCVR core, test acc, mean +/- std over FL seeds)

| method      | p_full | raw_core    | raw_full    | +CCVR core   |
|-------------|-------:|------------:|------------:|-------------:|
| homo_small  | -      | 38.04+/-2.50 | 3.74+/-1.07 | **52.48+/-0.79** |
| heterofl    | 0.10   | 35.57+/-2.01 | 27.63+/-1.31 | 51.99+/-0.94 |
| fedrolex    | 0.10   | 26.19+/-2.05 | 21.10+/-5.54 | 41.17+/-0.22 |
| fedrolex@600r | 0.10 | ~26         | -           | 41.88        |
| fiarse      | 0.10   | 35.31+/-3.26 | 25.22+/-4.40 | 48.77+/-2.63 |

## Participation curve (denser sampling REVISES the earlier "monotone scissors" claim)

homo_small+CCVR is participation-free (52.48). A 3-point sample (p010/p030/p060) looked like a
clean monotone scissors, but denser sampling (p020, p040 added; p080 pending) shows it is NOT
monotone: homo-small+CCVR is statistically TIED with HeteroFL+CCVR across low-to-moderate
participation and separates clearly only at high participation (p060).

HeteroFL+CCVR vs homo_small (52.48), test acc:

| p_full | heterofl+CCVR | homo - heterofl |
|-------:|--------------:|----------------:|
| 0.10   | 51.99+/-0.94  | +0.49  (tie)    |
| 0.20   | 52.36+/-0.96  | +0.12  (tie)    |
| 0.30   | 51.25+/-0.80  | +1.23  (~tie)   |
| 0.40   | 51.72+/-1.15  | +0.76  (tie)    |
| 0.60   | 48.26+/-1.55  | +4.22  (clear)  |
| 0.80   | 44.06         | +8.42  (large)  |

Shape (6 points): FLAT tie from p010-p040 (homo ~= HeteroFL, +0.1..+1.2pp within noise), then
ACCELERATING separation at high participation (p060 +4.22, p080 +8.42) as HeteroFL degrades
(51.7 -> 48.3 -> 44.1). Not a clean monotone scissors from the toxic point; a
participation-THRESHOLD effect: width-het extraction is neutral at low full-client fraction and
becomes actively harmful vs a calibrated homogeneous-small core at high participation, because
shell co-training damages the deployable core.

FedRolex and FIARSE (3-point) trend clearer: homo - fedrolex +11.3/+13.0/+16.4 across
p010/p030/p060 (FedRolex is a weak deployable core throughout); homo - fiarse +3.7/+2.6/+4.5.
So homo-small+CCVR beats FedRolex and FIARSE at all sampled p_full, ties HeteroFL (the strongest
extractor) at low-moderate participation, and beats it increasingly at high participation.

## Reads (honest)

## CORE vs FULL +CCVR: a participation CROSSOVER (2026-07-01, complete)

HeteroFL, +CCVR on the extracted core vs the full model, across participation (homo-small+CCVR =
52.48, flat):

| p_full | core+CCVR | full+CCVR | core - full |
|-------:|---------:|---------:|------------:|
| 0.10   | 51.99    | 49.36    | **+2.63**   |
| 0.20   | 52.36    | 52.41    | -0.05       |
| 0.30   | 51.25    | 52.43    | -1.18       |
| 0.40   | 51.72    | 55.38    | -3.66       |
| 0.60   | 48.26    | 57.64    | -9.38       |
| 0.80   | 44.06    | 60.67    | -16.61      |

There is a CROSSOVER at ~p_full=0.15-0.20. At SPARSE full-client participation (p010, the
realistic width-het regime where most clients cannot afford full width), the full model is
coverage-starved and undertrained, so core+CCVR beats it by +2.63pp and homo-small+CCVR beats it
by +3.12pp. As participation rises the full model recovers and overtakes (core degrades 52->44,
full recovers 49->61).

## Reads (honest, crossover framing)

1. At the realistic sparse operating point (p_full ~= 0.10), the CALIBRATED CORE (or a homogeneous
small model + CCVR) is the best deployable object for EVERYONE - it beats the full width-het model,
not just the extracted core. The earlier "never-wins envelope" read was wrong: at sparse
participation the full model is not a good deployment.

2. Defensible contribution: "In width-heterogeneous FL under sparse full-client participation
(the common regime), the deployable object should be a CALIBRATED core (core+CCVR), which beats
the full model; the field omits both post-hoc calibration of the deployable core AND the
homogeneous-small+CCVR baseline, and does not report that the full model is undertrained/worse
than the calibrated core in this regime." Plus: CCVR applied to a width-het extracted core is a
new setup (CCVR was built for homogeneous non-IID FL).

3. Honest scope: the winning regime for core+CCVR-over-full is NARROW (p_full <= ~0.15), the p010
margin is modest (+2.63pp core-full, ~2x the seed std), and above p020 the full model wins. So the
claim must be scoped to sparse participation, not stated as "core beats full" universally. The
core-degradation + full-recovery crossover, plus the sparse-regime deployable recommendation, is
the genuine finding.

3. FedRolex is far behind (-11 to -16pp), and the charitable 600-round run (>512-channel roll
period, a full roll) only moved it +0.67pp (41.21 -> 41.88). So FedRolex's poor deployable core
is inherent to its even-spreading design (no window is continuously trained), not a round-budget
artifact. Defensible.

4. FIARSE is the closest competitor but has high seed variance (48.77 +/- 2.63 at p010; s42
alone was 51.69). Worth a look at the importance-selection instability.

5. CCVR adds ~+16pp on top of every method's raw core, and the raw columns alone would mislead:
the deployable picture only appears with calibration. This supports "the calibrated deployable
baseline is the missing comparison."

## CIFAR-10 transfer: ATTEMPTED, INVALID, needs two fixes (2026-07-01)

A CIFAR-10 transfer run (20 jobs) was trained + evaluated but the results are garbage
(raw ~1%, +CCVR ~14%, at/below the 10-class chance of 10%). Two independent bugs, both
found by scrutinizing the below-chance numbers:

1. Training impaired: `flore/dataset.py get_transforms("cifar10")` has NO normalization
(cifar100 does). And homo025 (should be best) got stuck at chance (final acc 0.10) while
HeteroFL reached 0.35 - an inconsistency that points to a real CIFAR-10 training bug beyond
normalization. Fix + verify before trusting any CIFAR-10 training.

2. Eval harness is CIFAR-100-hardcoded: `probe/width_aware_calibration_oracle.py` loads
`tv.datasets.CIFAR100`, `NUM_CLASSES = 100`, and CIFAR-100 normalization (TF_TEST) in 7+
places. It therefore evaluates a CIFAR-10 model against CIFAR-100 data. Must be made
dataset-aware (read dataset name from config; use CIFAR-10 data/transforms/num_classes=10)
before any CIFAR-10 +CCVR number is meaningful.

The CIFAR-10 checkpoints/rows under `c10_*` and `ccvr_c10/` are NOT usable. The second-dataset
transfer check is the key REMAINING gate for AAAI and must be redone properly (daytime work).

## Pending
- Redo the CIFAR-10 (or a second-backbone) transfer with the two fixes above.
- Denser CIFAR-100 HeteroFL scissors at p020/p040/p080 (running 2026-07-01 05:11) to firm the
headline curve on the validated harness.
- Two MEDIUM audit items for the RAW columns only: FedRolex raw-full disclosure, FIARSE raw-core
via importance extractor for s43/s44 (AUDIT_FIX_NOTES.md).

## Adversarial stress-test verdict (2026-07-01, workflow w61vlhxug)

Strength: WEAK. Robust: NO. Novel: PARTLY. Recommendation: REFRAME + needs-more-experiments; do NOT
write the "deploy the calibrated core, it beats the full model" headline.

Why the headline is dead:
- It holds at EXACTLY ONE sampled point (p010, +2.63pp core-full, 3/3 sign-consistent, paired t
  p=0.037). It is a tie by p020 (-0.05, p=0.85) and reverses monotonically to -16.6pp by p080. One
  winning point is a corner, not a regime.
- Rests on the unestablished assumption "p_full=0.10 is THE realistic regime." A referee who accepts
  p_full>=0.2 reverses the finding.
- WRONG deployable object: at p010, homo_small+CCVR (52.48) ties/dominates core+CCVR (51.99) and needs
  NO width-het training; homo>full at p010 (+3.12pp) does not clear alpha=0.05 (p=0.055). So the honest
  conclusion is a NEGATIVE result for width-het ("just train a homogeneous small model + CCVR"), not
  "deploy the extracted core."
- Underpowered (n=3 seeds); incremental novelty (CCVR is off-the-shelf, Luo 2021 / FedPFT 2024; the
  "full worse than core at sparse participation" is the empirical face of minimum-coverage-index theory,
  "Every Parameter Matters" NeurIPS 2023). Unaccounted CCVR privacy/comm cost. Generalization gate unmet
  (single dataset/backbone).

What SURVIVES (the strongest honest, evidence-backed claim) = a benchmark + diagnostic, NOT a method:
1. Width-het papers evaluate a RAW uncalibrated core and omit a homogeneous-small baseline; post-hoc
   CCVR adds a robust participation-independent ~+16pp and re-ranks deployables, so the raw columns
   misrank.
2. A calibrated homogeneous-small model matches/beats every width-het extracted core at low-to-moderate
   participation -> width-het extraction buys nothing over just training a small model there.
3. The full-vs-core deployment choice crosses over near p_full~0.15-0.20 (full undertrained only at
   sparse participation).

Path to lift it: >=5 seeds + denser p_full grid; fix+run second dataset/backbone; account for CCVR
cost; and to beat "width-het buys nothing, just use homo-small+CCVR" it needs an actionable method that
beats homo-small+CCVR OR a principled participation-aware deploy rule. Venue: FL workshop / benchmark
track; AAAI main is a stretch without a method. Positioned honestly: "first to characterize the
participation crossover and add the calibrated-core + homogeneous-small baselines to the width-het
benchmark" - useful but incremental.

## CIFAR-10 transfer: the crossover GENERALIZES (2026-07-01, harness fixed)

Fixed the two bugs (cifar10 normalization in get_transforms; dataset-aware oracle; CIFAR-100
regression 51.40 unchanged; homo025 CIFAR-10 above chance). HeteroFL core+CCVR vs full+CCVR,
CIFAR-10, 3 seeds:

| dataset  | p010 core / full | core-full | p060 core / full | core-full |
|----------|-----------------:|----------:|-----------------:|----------:|
| CIFAR-100| 51.99 / 49.36    | +2.63     | 48.26 / 57.64    | -9.38     |
| CIFAR-10 | 73.42 / 65.97    | +7.45     | 71.14 / 75.17    | -4.03     |

Same SIGN pattern on both datasets: core+CCVR beats the full model at sparse participation
(coverage-starved shell -> net-harmful full model), full beats core at high participation. So the
participation crossover (core-vs-full net-benefit flips with p_full) is NOT a CIFAR-100/PreResNet
artifact - it transfers, and on CIFAR-10 the sparse-regime core advantage is larger (+7.45pp).

homo-small+CCVR = 72.11 (CIFAR-10). The "homo ~= width-het core" leg is a tie with sign flipping:
CIFAR-100 homo +0.49 over HeteroFL core; CIFAR-10 HeteroFL core +1.31 over homo. Both within ~1pp
=> "width-het extraction buys ~nothing over homo-small" survives as a TIE, but homo does not
cleanly win on CIFAR-10; the >=5 seeds will decide robust-tie vs noise.

Gate status: CROSSOVER transfer -> PASS. Remaining: >=5 seeds (re-ranking + homo-vs-core tie).

## n=5 seed verdict (2026-07-01): re-ranking DID NOT survive; crossover + tie DID

Extended to 5 FL seeds (s42-46) at p010 for all methods + the HeteroFL crossover points.

n=5 p010 +CCVR core: homo 52.58+/-0.66, heterofl 52.34+/-0.92, fiarse 49.24+/-2.26, fedrolex 40.38+/-1.74.
n=5 p010 raw_core:    homo 38.16+/-1.80, heterofl 36.24+/-1.71, fiarse 36.31+/-2.90, fedrolex 26.14+/-1.68.

1. RE-RANKING (fiarse<->heterofl): NOT robust. At n=3 raw looked like fiarse (38.6) > heterofl (33.3)
   by +5.3pp, flipping to heterofl>fiarse under CCVR - a clean "raw misranks" story. At n=5 the raw
   gap collapses to +0.07pp (fiarse 36.31 vs heterofl 36.24) = a TIE. So the raw fiarse-lead was
   fiarse's n=3 seed-variance (+/-2.9), not a real misranking. Calibration still separates them
   (heterofl pulls ahead by +3.1pp), but that is "calibration resolves a raw tie," NOT "the field's
   raw leaderboard is wrong." The strong protocol/re-ranking claim - the main AAAI-main lever - does
   not survive the seeds.

2. homo ~= width-het core TIE: HOLDS. homo 52.58 vs heterofl core 52.34 at p010 = +0.24pp (within
   std), n=5. "Width-het extraction buys ~nothing over a calibrated homogeneous small model" survives
   as a robust tie (both datasets).

3. CORE-vs-FULL CROSSOVER: ROBUST at n=5. core-full = +2.41 (p010), -8.15 (p060), -16.20 (p080);
   crossover unchanged from n=3, and it also generalized to CIFAR-10. This is the strongest, most
   defensible leg (and the delta vs Lan et al.'s coverage theorem).

4. CCVR lift: robust ~+13-16pp on every raw core, omitted by the field.

NET: the robust, evidence-backed contribution is a BENCHMARK+DIAGNOSTIC - (a) core-vs-full participation
crossover confirmed on 2 datasets at n=5 (the delta), (b) homo-small+CCVR ties the best width-het core
(width-het buys ~nothing), (c) calibration matters and is omitted. The re-ranking-misranks-the-leaderboard
claim was n=3 noise. Honest venue: FL workshop / benchmark track. AAAI-main is NOT supported, because its
lever (the re-ranking) did not survive >=5 seeds.

## WIDTH x PARTICIPATION SURFACE (2026-07-02): w*(p_full) is graded and monotone

Frontier probe (probe/width_ccvr_frontier.py, sBN recal + CCVR gauss_lw head, head seeds 42-44)
on every heterofl CIFAR-100 checkpoint, rates {0.25,0.375,0.5,0.625,0.75,1.0}, FL seeds s42-44.
R1-only rerun with thread caps (--skip-supervised; R1 seeded before R2 so values are protocol-identical).

| p_full | r0.25 | r0.375 | r0.5  | r0.625 | r0.75 | r1.0  | w* (acc)      |
|-------:|------:|-------:|------:|-------:|------:|------:|:--------------|
| 0.10   | 52.02 | 52.16  | 52.09 | 51.80  | 51.17 | 49.36 | 0.375 (52.16) |
| 0.20   | 52.36 | 52.62  | 52.99 | 53.37  | 52.96 | 52.41 | 0.625 (53.37) |
| 0.30   | 51.25 | 51.81  | 52.20 | 52.59  | 52.71 | 52.43 | 0.75  (52.71) |
| 0.40   | 51.72 | 52.57  | 53.48 | 54.20  | 54.62 | 55.38 | 1.0   (55.38) |
| 0.60   | 48.26 | 49.58  | 51.34 | 53.15  | 54.63 | 56.99 | 1.0   (56.99) |
| 0.80   | 44.06 | 46.39  | 49.84 | 53.23  | 55.88 | 60.67 | 1.0   (60.67) |

Anchoring: r0.25 and r1.0 columns reproduce the oracle crossover table (p080 44.06/60.67 exact;
p010 full 49.36 exact; p060 full 56.99 vs oracle 57.64, small protocol gap from sBN handling).

Reads:
1. The optimal deployable width w* is MONOTONE INCREASING in p_full through INTERIOR values:
   0.375 -> 0.625 -> 0.75 -> 1.0. The binary core-vs-full crossover is the shadow of a smooth law.
2. Every row is unimodal: capacity gain vs coverage-starvation harm trade off at an interior point.
3. Both default policies are dominated in complementary regimes: deploy-biggest loses up to -2.80pp
   (p010) at sparse participation; deploy-core loses up to -16.61pp (p080) at high participation.
   Interior w* beats BOTH endpoints at p020 (+1.01 vs core, +0.96 vs full) and p030 (+1.46/+0.28).
4. Caveats: n=3 FL seeds; adjacent-rate gaps (0.1-0.5pp) not individually significant; the claims
   are the argmax drift across 6 rows and the large endpoint losses at the extremes.

Next: (a) Stage-2 capping intervention - train {0.25,0.625} fleet at p020 and {0.25,0.75} at p030
vs the {0.25,1.0} fleet, >=3 seeds: does capping the ARCHITECTURE at w* beat slicing after the fact?
(b) label-free w* predictor from participation/coverage logs vs this table (8 held-out rows incl c10).
(c) CIFAR-10 surface BLOCKED on dataset-aware fix: width_ccvr_frontier never calls configure_dataset
and shell_distill_gate.index_loader hardcodes CIFAR100; fix both + rerun C100 regression row first.

## Stage-2 capping verdict (2026-07-02, n=3): WINS at p030, FAILS at p020

Capped fleets ({0.25,cap} rates, identical configs otherwise) vs the uncapped {0.25,1.0} fleet,
frontier probe R1 core+CCVR, FL seeds s42-44.

p020 cap0.5:   worse at every rate (-0.07/-0.63/-1.60); best deployable 52.29 vs 53.37 (-1.08).
p020 cap0.625: worse at every rate (-0.48/-0.24/-0.99/-1.59); best 52.38 vs 53.37 (-0.98).
p030 cap0.75:  BETTER AT EVERY RATE (+0.38/+0.73/+0.70/+0.96/+0.71); best deployable 53.55@r0.625
               vs uncapped 52.71@r0.75 (+0.84), at 0.75^2 = 56% of wide-client MACs.

Reads:
1. Capping is NOT a universal win: the naive "cap at w*" rule is dead (p020 capped at its own
   w*=0.625 loses ~1pp uniformly; even the 0.25 core degrades slightly). The wide clients'
   high-band training apparently benefits the shared prefix (context/co-adaptation effect),
   so amputating too much capacity hurts all slices.
2. At p030, cap0.75 UNIFORMLY dominates (all 5 deployable widths, sign-consistent) - a genuine
   method-shaped result: better deployables at ~44% less wide-client compute.
3. The capped p030 fleet's own argmax is again INTERIOR (0.625 > 0.75): the unimodal law
   reproduces inside capped fleets.
4. CAUTION: +0.7..1.0pp margins at n=3 are exactly the magnitude that died at n=5 before
   (re-ranking lesson). Do not claim until s45/s46 confirm.

Next fleet (queued): p030cap075 s45/s46 + uncapped p030 s45/s46 (n=5 on the win);
p020cap075 s42-44 (rule-shape test: is "cap at 0.75" the rule rather than "cap at w*"?).

## Capping CONFIRMATION verdict (2026-07-02 evening, n=5 + rule-shape)

p030 cap0.75 vs uncapped, n=5 (s42-46): capped better at EVERY rate, deltas
+0.47/+0.75/+0.91/+1.11/+0.90; best deployable capped 53.63@r0.625 vs uncapped 52.60@r0.75
= +1.03pp, at 56% of wide-client MACs. UNLIKE the re-ranking claim, the margins GREW from
n=3 to n=5 (+0.84 -> +1.03). This crosses the +1pp bar the method search never cleared,
while saving 44% wide-client compute. First genuine method-shaped positive of the program.

p020 cap0.75 (rule-shape test, n=3): FAILS (-0.19..-1.12 per rate, best deployable -0.89).
Same cap level flips sign between p020 and p030 => the gate is PARTICIPATION, not cap depth.
All three cap levels fail at p020; cap0.75 wins uniformly at p030.

Rule so far: capping pays only above a participation threshold in (0.20, 0.30].
Next boundary point queued: p040cap075 s42-44 (prediction from the w* surface: at p040 the
full model is the argmax, so capping should stop winning; if it still wins, the mechanism is
concentration not argmax-tracking - either answer sharpens the rule).

## p040 boundary verdict (2026-07-02 night, n=3): capping OBSOLETE at p040 - rule complete

p040cap075 vs uncapped p040 per rate: -0.33/-0.02/+0.04/+0.09/+0.14 (all within noise, a WASH),
but best deployable capped 54.76@r0.75 vs uncapped 55.38@r1.0 = -0.62 (capping removes the
argmax slice, which at p040 is the full model).

The capping rule across four participation points:
- p020: top band is NET-POSITIVE for lower slices (capping hurts everywhere, all cap levels).
- p030: top band is NET-PARASITIC (capping helps every slice, +1.03pp best deployable, n=5).
- p040: top band is NEUTRAL to lower slices but is itself the best deployable (capping loses by omission).

So the marginal value of the widest band to the rest of the model is NON-MONOTONE in
participation (+ -> - -> 0), and width capping is a REGIME method with a narrow win band
around p030. No existing theory predicts the p020 sign. Honest claim: (i) deployment width
should track w*(p_full) (selection, robust across the whole axis); (ii) training-width capping
adds +1pp on top ONLY inside the band where the top band is parasitic; a practitioner needs a
cheap band predictor (coverage index / probe) - open question, fastsim makes the dense
band-mapping experiment (p025/p035, more caps, 5 seeds) affordable.

## fastsim adoption + dense band map (2026-07-03): capping DEMOTED under corrected semantics

fastsim (FLoRE-fastsim worktree) passed its equivalence gate after adversarial review:
homo025 and p010 anchors unbiased, p030 family showed a small systematic offset later
traced to flwr's per-fit split redraw (flwr 1.26.1 rebuilds the client per fit message, so
flore's unseeded random_split redraws each client's 80/20 split every round; val leaks into
train). User ruling: fixed seeded split = corrected reference semantics; fastsim cleared for
science sweeps at ~4x single-run / ~12x fleet speedup.

Dense band map on fastsim (p_full in {0.20,0.25,0.30,0.35,0.40} x {uncapped, cap075},
n=5, R1 core+CCVR): best-deployable capped-vs-uncapped deltas
p020 -0.24 / p025 -0.56 / p030 -0.29 / p035 +0.16 / p040 -0.63. The flwr p030 capping win
(+1.03pp, n=5) does NOT reproduce under fastsim. Capping is simulator/implementation
sensitive => DEMOTED from method to cautionary finding. (See causal closure below: the
split-semantics edit alone does not explain the flwr-fastsim level offset, so the capping
discrepancy is an unexplained implementation sensitivity, which if anything strengthens
the demotion.)

## WIDTH x PARTICIPATION SURFACES at n=5 on fastsim (2026-07-03): the law holds

CIFAR-100 (uncapped heterofl, n=5, R1 core+CCVR %): argmax drifts monotonically
p010 r0.375 -> p020 r0.5 -> p030 r0.75 -> p040 r1 -> p060 r1 -> p080 r1; every row unimodal.
CIFAR-10 interior transfer (n=5): p010 r0.375 -> p030 r0.625 -> p060 r1. The interior
optimum at matched participation confirms the drift is graded, not a core/full binary.
Full tables: FLoRE-fastsim/results/surface5/surface_tables.txt.

## Statistic-guided width selection (2026-07-03): the label-free selector VALIDATED

Method: class-stratified 50/50 split of client-side train GAP features; gauss_lw stats per
half; CCVR head trained on A-virtual samples, epoch-selected and scored on disjoint
B-virtual draws; pick argmax width. Label-free at the server, privacy profile identical to
CCVR itself (same class-conditional feature statistics).

Validation across every checkpoint family we have (flwr + fastsim, capped + uncapped,
C100 + C10), regret vs per-checkpoint oracle:
- C100 grid (87 ckpts): selector 0.114pp mean / 0.65 max / 0 over 1pp; always-core 2.537pp;
  always-full 0.426pp; beats always-full 38 win / 43 tie / 6 loss.
- C10 flwr grid (9 ckpts): selector 0.39pp; always-full 14.73pp (crossover side).
- fastsim surface5 (45 ckpts): selector 0.130pp / 1 over 1pp; always-core 3.53pp;
  always-full 1.29pp.
- COMBINED N=141: selector 0.137pp vs always-core 2.78pp vs always-full 1.61pp; only 3/141
  checkpoints exceed 1pp regret.
Interior test (c10 p030, GT argmax 0.625): selector picks 0.625 on 4/5 seeds (s42 picks
0.5, regret small), family-mean argmax matches GT. Adversarial audit clean: no label
leakage, no test access, ~12pp virtual-vs-test gap shows scores are not disguised test
accuracy, and the dimension-bias diagnostic shows it is not disguised always-full
(picks max rate 49% on C100 grid, 11% on C10).

## CAUSAL CLOSURE (2026-07-04, flwr p030fix n=5): split redraw is MEAN-NEUTRAL; offset unexplained but within noise; the LAW is invariant

Intervention: seeded flore's random_split per client (generator seeded from dataset seed +
partition id), reran flwr heterofl_a01_p030 s42-46 for 300r, probed identically
(results/width_surface_p030fix). Three rows at rates 0.25..1.0 (R1 core+CCVR %, n=5):

  flwr-orig 51.31 51.86 52.26 52.52 52.60 52.49
  flwr-fix  51.09 51.79 52.29 52.46 52.59 52.56
  fastsim   51.81 52.45 52.89 53.11 53.36 53.15

fix-minus-orig: -0.22/-0.07/+0.03/-0.06/-0.01/+0.06, paired per-seed deltas mixed-sign.
VERDICT: the per-fit split redraw is approximately mean-neutral at full horizon. The
60-round single-seed direction probe (-1.54pp) was a short-horizon artifact. The
fastsim-vs-flwr offset (+0.59..+0.77) is therefore NOT explained by split semantics; it
persists unchanged, is ~1.0-1.4 SE per cell at n=5 with shared checkpoints across rates,
i.e. within combined seed noise, and no further defect was found on either side.

What survives, cleanly: (i) the flwr flaw is real and code-proven (val leaks into train,
splits non-reproducible) - a validity/reproducibility defect, NOT an accuracy distortion
in this setting; the paper must not claim it explains level offsets. (ii) All three rows
are unimodal with argmax r0.75 at p030 (fix row 0.75 over 1.0 by a hair): the
participation-coverage law is invariant to simulator and split semantics. (iii) dataset.py
reseed edit reverted after the runs; only the pre-existing CIFAR-10 normalize fix remains
uncommitted.

## Selector count CORRECTION + paper basis (2026-07-04)

Dedup discovery: the surface5 c100 uncapped p020/p030/p040 checkpoints are bit-identical
retrains of the bandmap uncapped runs (fastsim is deterministic given config+seed; verified
tensor-equal). The "141 checkpoints" headline therefore over-counts by 15; unique total is
126. Corrected fastsim-only selector numbers (the paper's basis, one stack, dedup applied):
- CIFAR-100, 65 unique ckpts (50 bandmap capped+uncapped p020-p040 + 15 surface p010/p060/p080):
  selector 0.112pp mean / 0.65 max / 0 over 1pp; always-core 3.122pp (max 17.09);
  always-full 0.484pp (max 4.93); diag-real 0.101pp; agreement 0.585; vs always-full
  23W/37T/5L (non-trivial 61: sel 0.106 vs full 0.458, 20W/36T/5L); picks max rate 57%.
- CIFAR-10, 15 ckpts (surface5): selector 0.262pp mean / 1.29 max / 1 over 1pp; always-core
  1.662 (max 3.79); always-full 2.539 (max 10.52); diag 0.028; p030 interior GT argmax 0.625
  picked 4/5 seeds; dim-corr -0.02; picks max rate 7%.
- Combined 80: selector 0.140pp vs core 2.848 vs full 0.869; agreement 0.562.
Grader: scratchpad grade_fastsim_selector.py (session 1266da45). The flwr-trained subsets
(mixed-grid rows) agree in magnitude and remain in the full record above; the paper reports
the fastsim-only basis per the one-stack decision (user ruling 2026-07-04: no simulator
comparison in the paper; the law is simulator-invariant).

## RAW surface (2026-07-04): the law's shape exists WITHOUT calibration

Tabulated R0_standalone (sBN-recalibrated, no CCVR) from the existing surface5 probes,
n=5, both datasets. Every row rises-then-falls and the raw argmax drifts monotonically:
C100 0.25/0.375/0.5/0.5/0.75/1.0 (p010-p080), C10 0.375/0.625/1.0 (one 0.05pp inversion
at c10 p030 inside noise; raw stds 1.0-5.0 c100, 3.7-9.4 c10). Raw optimum sits ~one width
step NARROWER than calibrated at sparse p (calibration partially repairs the undertrained
wide features). Kills the "law is a property of CCVR" reviewer attack at zero GPU cost;
added to the paper as tab:rawsurface + amended the "visible only after calibration" claim
(shape is raw, levels need calibration).

## K-scaling fleet LAUNCHED (2026-07-04): fraction vs count

18 runs on fastsim (conf/kscale, results/kscale in FLoRE-fastsim): K in {100,200} x
{fixed-fraction p010/p040, fixed-count N=5 (p005@K100, p0025@K200)} x s42-44, 300r,
clients-per-round = K/10, vs existing K=50 surface5 rows. Deterministic prefix assignment
verified (N_full = 5/10/20/40/80 as designed). Total shell data invariant at fixed
fraction (fixed 50k dataset); class presence grows with N_full. Predictions: presence-
binding => sparse toxicity shrinks as K grows at fixed fraction; quantity-binding (IID
control suggests this) => toxicity persists. Either outcome sharpens the paper vs the
scale-confound attack.

## K-scaling VERDICT (2026-07-04): law tracks FRACTION not COUNT; binding variable = shell data VOLUME

18 runs done (~33 min each, 3/GPU). R1 core+CCVR, n=3 paired (K=50 rows = surface5 s42-44
subset for seed-matching):
  K   p_full N_full  core   full   core-full        argmax
  50  0.10   5       52.24  48.96  +3.28+/-1.47     interior
  100 0.10   10      52.04  48.89  +3.15+/-0.63     interior
  200 0.10   20      50.52  47.61  +2.91+/-0.72     interior
  100 0.05   5       52.24  46.42  +5.82+/-0.95     0.25
  200 0.025  5       51.25  39.13  +12.12+/-2.79    0.25
  50  0.40   20      50.14  54.29  -4.15+/-0.67     1.0
  100 0.40   40      50.34  55.07  -4.73+/-0.55     1.0
  200 0.40   80      47.45  55.12  -7.66+/-1.04     1.0
Fixed fraction: toxicity INVARIANT as K quadruples (presence/count grows, volume fixed).
Fixed count N=5: toxicity nearly QUADRUPLES as data share shrinks (volume drops 4000->1000).
=> binding variable = VOLUME of data reaching the wide band (p_full x fleet data); class
presence ruled out as primary; kills the small-K-artifact attack by intervention. p040:
full optimal at every K. Honest scope: at extreme cross-device scale a tiny fraction of a
huge data-rich fleet can supply enough volume to escape the sparse regime (added to paper).
Paper updated: tab:kscale + law paragraph + mechanism rework (volume primary, presence/
density secondary) + abstract/intro/limitations. 12 pages, compiles clean.

## Review panel round 2 (2026-07-04): AC 5/10 borderline; expert = major revision; consistency PASS

Paper restructured: Algorithm 1 fixed (per-class half-statistics upload), main cut 12->9pp
(technical ~7.3pp + refs), 4pp supplementary created (raw surface, heatmap, alpha sweep,
band map, negmap, FedRolex trail, experimental details). Fresh panel (AC / FL expert /
consistency):
- AC 5/10 weak-reject-borderline; strongest = mechanism (fixed-fraction K-sweep arm);
  path to weak accept = large-fleet scale test OR 2nd backbone; also: selector motivation
  must be "label-free recovery of w*" not "server never observes p_full" (server knows it
  by construction); count-fixed arm confounds volume with skew (class-balanced fixed-count
  arm would isolate); demote Prop 1 wording from "certificate".
- Expert (major revision, nothing critical): (1) oracle bracket numbers untraceable (47.1
  core matches no table - they come from the older skyline runs at a different operating
  point) + IID control has NO numbers anywhere -> must tabulate both at a reconcilable
  point; (2) NEW alternative surviving all probes: core-shell CO-ADAPTATION (both arms
  freeze the FL core; Arm B's 60%-only recovery is the tell) -> separating experiment =
  Arm B with a core frozen from a homogeneous FULL-width fleet; (3) half-statistics
  privacy leak (differencing; n_c=1-2 classes expose raw features) - softened in supp,
  needs main-text treatment; (4) comm cost honestly 2x|W| not 2x; (5) drop the 12pp-gap
  anti-leakage argument (invalid; rely on path audit); (6) fine law claims exceed 5-seed
  resolution at sparse end (w*(0.10) is a 0.06pp near-tie with core) -> soften abstract to
  coarse law, add bootstrap-argmax + monotone-trend test, report interior-beats-endpoints
  as per-seed sign counts; (7) kscale "surface unchanged" too strong (p040 magnitudes grow
  with K; regime/argmax invariant); (8) selector eval semi-circular (CCVR both sides) +
  population composition undisclosed -> per-p_full regret breakout; virtual-vs-real gap is
  8x on c10 (0.26 vs 0.03) - scope the "loses nothing" claim.
- Consistency: compile clean, no stale numbers; fixed H1 (tab:calib heterofl 52.34 vs
  surface 52.15 = independent replication note added), H2 ("every cell" -> core-scoped),
  toxicity defined, homo_large orphan dropped, related-work capping dup trimmed, orphan
  tab_alpha_sweep.tex deleted, supp heatmap/bandmap now referenced.

## AC fleet verdicts (2026-07-04 evening): IID tabulated, volume-sans-skew pinned, K500 regime holds, boundary shifts with D

24 runs (conf/acfleet, R1 core+CCVR %, n=3, argmax over 6 rates):
  fsiid_p010      core-full +5.76+/-0.27  argmax 0.25  (IID CONTROL TABULATED: toxic shell
                  survives zero skew / full presence; delta even larger than dirichlet +3.28)
  fsiidK200_p0025 core-full +17.44+/-0.45 argmax 0.25  (VOLUME WITHOUT SKEW: fixed 5 clients,
                  1/40 data share, IID => explosion bigger than dirichlet +12.1; kills the
                  AC's volume/skew confound - volume pinned, skew ruled out)
  fsK500_p010     core-full +4.97+/-0.32  argmax 0.25  (regime + argmax invariant at 10x fleet;
                  magnitudes not invariant - levels drop with per-client scarcity, consistent
                  with regime-invariance claim)
  fsK500_p040     core-full -8.90+/-1.55  argmax 1.0
  fshalf_p020     core-full +1.00+/-0.76  argmax 0.5   (BOUNDARY SHIFT: at full D p020 was a
                  tie -0.04; halving fleet data makes p020 core-ahead => toxic boundary moves
                  right as volume shrinks, the volume law's directional prediction; shift
                  ~1.6 SE, suggestive not decisive)
  fshalf_p040     core-full -3.33+/-0.95  argmax 1.0   (vs -3.59 at D: nearly unchanged)
  fshalf_p060     core-full -10.22+/-1.08 argmax 1.0   (MORE negative than D's -6.06: at high
                  p_full halving D hurts the small core more than the full => volume law
                  governs the shell-limited sparse regime, not every cell)
  fshomolarge     full 61.17 (homo_large reference number); slices useless (r0.25 = 14.78
                  calibrated!) => slice-deployability itself REQUIRES width-het training -
                  new pro-width-het counterpoint to the "buys nothing" corner.
Bracket (12 shell-training runs, 3 core types x arms) running; data-fraction option added
to fastsim (config/data/loop + 3 new tests, 14/14 pass).

## ORACLE BRACKET RERUN + CO-ADAPTATION VERDICT (2026-07-04 night): traceable ladder; substrate priced at ~12pp; FL core NOT damaged

12 shell-training runs (probe.arm_a_skyline, cores from current fastsim checkpoints, raw
acc, sBN recal, held-out epoch selection, 3 seeds):
  FL-trained full model (p010)            25.1 +/- 2.6   (raw surface 24.98 ✓ traceable)
  Arm A: oracle shell, 5-client pool      32.7 +/- 2.4   (pool ~4.2k ex, 91-92/100 classes,
                                                          median 20/class => volume not
                                                          presence; 5.4pp BELOW the core)
  frozen FL core standalone               38.2 +/- 1.4   (raw surface 37.63 ✓)
  Arm B: oracle shell, all data (FL core) 55.2 +/- 1.6   (coverage component +22.5)
  Arm B, standalone homo025 core          56.1 +/- 0.3   (FL core == standalone core as
                                                          substrate => width-het training
                                                          does NOT damage the core; expert's
                                                          co-adaptation-of-FL-core alternative
                                                          REFUTED in that form)
  Arm B, full-width-trained (clean) core  67.6 +/- 0.4   (substrate component ~12.3pp exists
                                                          but attaches to ANY narrow-trained
                                                          core; unreachable at sparse p_full
                                                          by construction)
Arm B recovers 58% of the bracket range; remainder priced to substrate. homolarge core
standalone raw ~4% (prefix of a homo-full net is not a functional subnet).
PAPER INTEGRATED: tab:bracket + rewritten primary-evidence paragraph; IID rows added to
tab:kscale (+5.76 / +17.44); K500 rows added; half-data boundary shift sentence (supp
tab:halfdata); slice-deployability counterpoint added to calib section (homolarge r0.25
slice 14.78 calibrated vs 52.2 het core); supp gains bracket details + half-data table.
Main 10pp + 5pp supp, both compile clean.

## ViT SECOND-BACKBONE VERDICT (2026-07-04 night, HeteroViT r0.125, 600r AdamW, n=3, R1 %)

  p010: core(0.125) 31.65  full 22.22  core-full +9.43+/-1.25  argmax 0.125
  p030: 29.09 / 27.52  +1.56+/-0.54  argmax 0.125
  p060: 21.93 / 33.37  -11.43+/-0.57  argmax 1.0
  Interior rates (0.25-0.75): VALLEY-shaped, all below both endpoints at p010/p030
  (e.g. p010: 12.19 at r0.25 rising to 20.02 at r0.75).

TWO-PART VERDICT: (1) The coverage LAW's core claims REPLICATE on a second architecture
with tight stats: sparse-regime toxicity (+9.4), monotone argmax drift core->full as p
rises, crossover between p030 and p060, core column degrades with p (31.7->21.9), full
recovers (22.2->33.4). (2) The INTERIOR RIDGE does NOT replicate: ViT rows are valley-
shaped, argmax jumps binary core->full. Mechanism consistent with the diagnosis campaign's
zeroed-shell finding: LayerNorm normalizes ACROSS the embed dim, so any partially-trained
band corrupts the core's normalization, unlike BatchNorm's per-channel separation ->
graded interior ridge is a per-channel-normalization (BN-family) property; drift+endpoints
are architecture-general. Paper: keep the law as drift+endpoints (2 architectures),
explicitly scope row-unimodality/interior optimum to BN-family backbones with the
LayerNorm-coupling explanation. BONUS for the selector: a THIRD crossover location
(~p015 ResNet-C100, ~p030 C10, in (0.30,0.60) ViT) - thresholds transfer even worse across
architectures; selector eval on vitfleet checkpoints still pending (MF1).

## SUBSTRATE-AXIS METHOD SEARCH: BOTH ROUTES FALSIFIED (2026-07-05, Opus/ultracode)

Motivation: the oracle bracket found a +12.3pp SUBSTRATE component (narrow-trained core caps a
later oracle shell at ~56 raw; embedded-trained core supports 67.6). The core is trained by
every client every round (never starved), so the substrate axis looked like the one open door
sparse participation does not close. Two O(core) methods tried to walk through it; both fail.

(1) GHOST-INPUT (SE1 stop-gradient 2x2, centralized, same basis, Arm B substrate):
    S_fwd(fwd-on,bwd-off)=67.83  S_bwd(off,on)=69.44  S_onon=67.26  S_offoff=68.10.
    Pre-registered stop: S_fwd - S_offoff = -0.27 < 2.0 => forward hidden injection carries NO
    capturable substrate share. (Also: rank-16 R^2 of the cross-term was 0.5-0.81 cross-channel
    - structure is real and reproducible - but structure is not the lever.)

(2) PHANTOM-HEAD (extend the core's classifier with frozen shell columns + shell-feature
    Gaussians; alternate bare/phantom to hold slice). 4 arms, slice / substrate(max-over-80ep):
    G0 standalone 63.69/63.94 ; P1 placebo 62.10/62.79 ; P2 always-phantom 42.93/61.31 ;
    P3 alternating-real 63.47/63.92. KILL: P3 substrate 63.92 == G0 63.94 (+/-0), phantom head
    adds ZERO substrate benefit; P1 placebo ~ P3 real; every good-slice core caps ~62-64, none
    near 68. (Confound handled: skyline picks best-VAL epoch = epoch0=slice for good-slice
    cores; max-over-epochs test used instead, verdict unchanged.)

DOUBLE BOUNDARY RESULT (paper: strengthens mechanism + extends negative-map to substrate axis):
The +12pp substrate benefit is NOT transportable by any O(core) surrogate - not forward-hidden
(ghost-input) nor head-level (phantom-head). It requires the shell's actual presence during core
training (hidden forward contribution AND backward gradient through hidden layers), which sparse
participation withholds. This is the SUBSTRATE-AXIS analog of the data-axis Arm A impossibility:
two axes, two walls, both mapped. NO positive method; the paper's method contribution stays the
deployment-side selector (+ untested progressive widening). Full detail: FLoRE-fastsim/results/
ghost_gate/{GATE_RESULTS.md,THEORY_ATTACK.md,FALSIFIER_REPORT.md}. Single-seed (s42).

## PAPER ANALYSES VERIFIED (2026-07-05, GPU-free workflow, results/paper_analysis/)

Independent adversarial audit of all headline numbers vs raw JSON/logs: ZERO numerical
mismatches (72 surface cells, both argmax drifts, 12 K-sweep/IID core-full deltas incl +5.76 /
+17.44 IID, 4 bracket rungs in strict order, 4 SE1 quadrants + ladder-stop, 3 ViT rows with
interior VALLEY vs ResNet RIDGE). One LOW label nuance: K=50 p010 "interior" is core-argmax at
n=3 (interior only at n=5) - footnote it.

Deployment-policy BASELINES (regret vs oracle, per-surface, pp): always-core 4.41/1.61/3.81
(C100/C10/ViT), always-full 0.62/2.49/3.66, p_full-threshold tuned p*=0.15 gives 0.166/0.535/
0.521. KEY: C100 optimal-threshold interval (0.10,0.20] is DISJOINT from ViT's (0.30,0.60], so
NO fixed threshold serves both architectures (0.521pp pure transfer regret on ViT, 1.564pp at
p030 where ViT still wants core). C100->C10 transfers only by luck (intervals overlap). Volume-
threshold == p_full rule under constant D=50000 (report honestly; differs only in variable-D /
half-data). Selector (0.11/0.26 cited) beats the best threshold even at home => selector is
NECESSARY across architectures. [Same-convention recompute + ViT selector + diagonal ablation
pending as of this write.]

COMM/PRIVACY (exact, HeteroPreResNet18, fp32, 300 rounds): full-cov selector 482.6MB one-shot =
3.76% of one full client's 12.5GB training bill = 4.82x one deploy calibration - a real second
channel, affordable, must be reported. Diagonal variant 2.73MB = 0.021% = 176x cheaper.
=> diagonal ablation is pivotal (does diagonal selection hold regret? running on GPUs 0-2).
Privacy: two half-stats reveal their difference; 1-2-example classes -> near-raw features;
secure-agg of pooled halves needed; diagonal shrinks bytes 176x but not the two-halves leak.

FREQ-vs-VOLUME (reviewer attack closed): fixed p010, K 50->500 raises expected full-clients/
round 10x (0.5->5.0) at fixed shell volume (5000 ex); toxicity flat +3.28/+3.15/+2.91 (K50/100/
200), +4.97 at K500 (UP, wrong direction for a frequency account) => p_full acts through data
VOLUME not update frequency.

## SUBSTRATE PREMISE REFINEMENT (2026-07-05, builder+orchestrator converged)

Builder agent completed and its verdicts MATCH the orchestrator's independent grades exactly
(phantom P3 substrate 63.9 == G0 63.9; ViT selector 0.52). One refinement to the substrate story
for accuracy: the "+12.3pp substrate component" (FL core 56 -> embedded 67.6) is basis-dependent.
On a CLEAN CENTRALIZED basis a plain standalone rate-0.25 core is already a ~64 substrate, so the
FL core's 56 was largely an FL-TRAINING ARTIFACT (sparse participation + sBN drift + aggregation),
and the true embedded-substrate premium is only ~4pp (64 -> 68), not +12. The double-boundary KILL
is unchanged (phantom head does not lift 64; the residual ~4pp needs the shell's hidden-layer
presence during training, which no O(core) surrogate reproduces). Implication if the substrate
result is ever used in the paper: frame the transportable-benefit ceiling as ~4pp on the clean
basis, not 12. Builder also confirmed: the ~19:30 machine-wide process sweep (killed all 6 GPUs,
all users) was recovered with identical seeds, no result lost. ViT selector detail in
results/vitfleet/SELECTOR_VIT.md; U-shaped ViT crossover, selector nails p010/p060, misses p030
(large-width bias, 1.56pp) - matches the RIDGE-vs-VALLEY finding.

## FEMNIST CROSS-DEVICE SCALE EXPERIMENT (2026-07-05, launched) - the AC's highest-ceiling test

Tests whether the participation-coverage law holds at genuine cross-device scale (the regime the
paper motivates but never measured; AC/R3's key significance gap). Uses flwrlabs/femnist: 3597
NATURAL writers (~= "3500 clients"), 692k train / 122k test, 62 classes, 50 clients/round. Built
FEMNIST support in fastsim (GpuFemnist: 28x28 grayscale -> pad32 + replicate 3ch to reuse
HeteroPreResNet18; natural writer partitioning; hflip disabled; data/femnist_cache.pt) + probe
(FemnistDataset in width_aware_calibration_oracle.py; validated: calib mean~0, 62 classes, 122k
test). ~16.6+ s/round, 500 rounds, ~2.3-3h/config on GPUs 0-2.

THE DESIGN (critical): our mechanism says toxicity is governed by shell VOLUME = p_full x D, not
fraction. At CIFAR (D=50k) crossover sits at p_full~0.15-0.20 => critical volume ~7500-10000. At
FEMNIST (D=692k, 14x more data) the SAME volume threshold predicts the crossover MOVES DOWN to
p_full ~ 0.012-0.015 (~1.2%). So the sweep goes LOW: p_full in {0.005,0.01,0.02,0.05,0.10,0.30}
(17/35/71/180/360/1079 full writers; shell volume 3.5k/6.9k/13.8k/34.6k/69k/208k).

THE TEST: if the toxic-shell crossover (core+CCVR beats full+CCVR) sits near p_full~0.01-0.02
(volume-predicted) rather than ~0.15 (fraction), the LAW tracks VOLUME across a 14x scale change
+ a different dataset/task => turns the paper's Limitations hedge into a headline cross-device
confirmation, directly answering the significance-ceiling critique. If it sits at ~0.15 or there
is no crossover, the mechanism is challenged (honest either way). Probe+analysis ready
(scripts/probe_femnist.sh); training self-completes -> logs/femnist/train.marker.

## FEMNIST + HeteroCNN: ADVERSARIAL REVIEW PASSED + refinements (2026-07-05)

Built HeteroCNN (right-sized width-sliceable BN CNN, base [64,128,256,512], 1.58M full/0.106M core,
128/512 feat dims matching ResNet). Slicing gate PASSED (max|core-full_prefix|=0, strict-load all 6
rates). Running FEMNIST cross-device sweep on GPUs 4-5 (~37min/config) alongside the ResNet sweep on
0-2. 3-agent adversarial review (results/paper_reviews/CNN_REVIEW_*.md) VERDICT = sweeps trustworthy,
DO NOT STOP; all 3 critical gates clear:
- NORMALIZATION train-vs-eval BIT-IDENTICAL in fp32 (5.6e-4 diff = fp16 storage only) - experiment valid.
- SCALER inert-not-buggy (ScalerV2 before BN with no residual => BN cancels the 1/rate scale; with
  core-force-scaler-eval it's bit-equiv to no scaler; dead code, remove later; ResNet scalers are LIVE
  via residual add). CNN CCVR path WORKS (forward_features dispatch in AAAI-baseline probe copy;
  model-reviewer's "would crash" was the stale fastsim copy, not the one the sweep runs).
- CCVR quick-test (CNN p010 30r): core 70.21 / full 73.39, +3.18pp gap, monotone => no washout.
REFINEMENTS APPLIED: (1) added p_full=0.015 grid point to both sweeps (predicted crossover had no point
on it); (2) REALIZED shell volume != nominal - the first-N (writer-id-sorted) full-width writers are
~1.58x LARGER than average, so realized vol = 1.58 x nominal. Corrected prediction: crossover at
p_full ~0.007-0.01 (realized vol ~8-11k), not nominal ~0.012. Analysis now plots realized volume.
CLAIMS SCOPING (from review): GO for qualitative "crossover at low single-digit fraction, not ~15%"
in both arches; the exact LOCATION needs a seed+FULL-BAND-RANDOMIZATION follow-up (rate assignment is a
pure fn of partition_id so the full band is a FIXED writer set across seeds - seeds don't resample the
shell writers); NO-GO on "CNN confirms SAME critical volume as ResNet" (shell 1.48M CNN vs 10.49M ResNet,
7.1x) => report as TWO independent qualitative replications (matched 128/512 feat dims remove readout
confound). DISCLOSE: example-level global-test holdout (writers overlap train/test; fair for core-vs-full
delta since both hit the same set). Minor cleanup: norm_type ignored (CNN uses plain BN not sBN; cosmetic
since probe recalibrates BN at eval); worktree code divergence (fastsim vs aaai-baseline probe copies).

## FEMNIST CROSS-DEVICE RESULT (both architectures, 2026-07-05 19:15)

K=3597 writers, 50/round, 500 rounds, natural writer partitioning, 62 classes. core=rate0.25, full=1.0,
core+CCVR vs full+CCVR (3 head-seeds avg). Realized shell volume (first-N writers 1.58x avg).

ResNet (11M, shell 10.49M):          CNN (1.58M, shell 1.48M):
 p_full realVol core  full  gap        core  full  gap
 0.005   5461  81.78 80.10 +1.68 (4.7s) 79.61 79.24 +0.37 (2.6s)
 0.010  10928  81.92 81.27 +0.65 (5.2s) 79.71 79.67 +0.04 (0.1s)
 0.015  16274  81.85 81.28 +0.57 (1.3s) 79.75 79.31 +0.44 (2.1s)
 0.020  21875  82.30 81.60 +0.70 (6.3s) 79.55 79.87 -0.32
 0.050  53820  81.83 81.73 +0.10        79.55 80.12 -0.57 (2.4s)
 0.100 108006  81.32 81.77 -0.45        79.51 80.50 -0.99 (5.9s)
 0.300 305028  80.35 82.75 -2.40 (11.7s)78.27 81.45 -3.18 (8.0s)
(s = gap / head-seed-noise; head-seed noise ONLY, NOT training-run/band variance)

FINDINGS:
- BOTH architectures show the participation-coverage crossover: core+CCVR optimal at low participation,
  full at high. Monotone gap in both. Argmax width drifts narrow(0.25-0.375)@low-p -> full(1.0)@high-p.
- ResNet is the STRONG/CLEAN instance: core wins thru p0.05 (5/7 points), peak +1.68pp, crosses at
  realized vol 54k-108k (p~5-10%). CNN is a MUTED replication (near width-saturated, only ~2pp headroom):
  core wins thru p0.015, peak +0.44pp, crosses at vol 16k-22k (p~1.5-2%).
- CRITICAL VOLUME SCALES WITH SHELL CAPACITY: ResNet crossover volume / CNN crossover volume ~ 3.4-5x,
  tracking the 7.1x shell-size ratio. Bigger shell needs more volume to be net-beneficial. This is a
  NEW quantitative confirmation of the volume mechanism (stronger than "same volume"; matches review NO-GO).
- BOTH crossovers << 15% fraction-hypothesis, AND at DIFFERENT fractions (ResNet ~5-10%, CNN ~1.5-2%)
  => refutes fixed-fraction law, confirms volume+capacity scaling at 14x cross-device scale.
- Both mechanisms visible at p0.30: core DEGRADES (ResNet 81.9->80.35, toxic-shell interference) while
  full IMPROVES (80.1->82.75, shell finally gets volume).
CAVEATS: n=1 training seed, FIXED full-band (pids 0..N-1, same writers every seed - review's concern);
head-seed sigma is NOT the real error bar; exact crossover LOCATION needs multi-seed + band-randomization
follow-up. Qualitative claim (crossover at low fraction, tracks volume+capacity, 2-arch) is defensible now.

## AUDIT VERDICT (2026-07-05) - CORRECTS the section above

3-auditor adversarial audit of the two-arch result. Verdict: TRUSTWORTHY-WITH-CAVEATS.
Full: results/paper_reviews/AUDIT_{ccvr,volume,stats}.md + AUDIT_SYNTHESIS.md.

CROSSOVER IS REAL (not a CCVR/checkpoint artifact) - STRONGER than first reported:
- The sign flip is present in R0 (pre-CCVR standalone) too, with LARGER magnitude. ResNet R0 gap
  +5.09(p005) -> +1.67(p10) -> -4.62(p30); CNN R0 +3.66 -> +0.99 -> -4.64. CCVR DEFLATES the core
  lead and shifts the crossover to lower p. => REPORT R0 as primary; R1_ccvr is a conservative,
  core-deflating readout.
- Diagonal-cov re-probe: gap GROWS (+3.37), so "CCVR favors low-dim core" confound is WRONG SIGN.
- One checkpoint/config, core+full sliced from the SAME final state (deployable-slice contract). Clean.
- Robust statistical result: core-full gap decreases MONOTONICALLY with shell volume in both arches,
  Spearman rho=-0.89 each, exact perm p=0.012 each, combined Fisher p=0.0015.
- p0.30 core degradation real in R0 (ResNet -5.3pp, CNN -4.3pp toxic interference); CCVR masks part.

RETRACTED / DEMOTED (I over-reached in the section above):
- "Critical volume scales with shell capacity ~3.4-5x matching 7.1x" is NOT DEFENSIBLE -> DROP.
  Measured ratio ~3.2x (not a match to 7.1x); READOUT-DEPENDENT (under R0 both cross at ~130-142k,
  ratio ~1.1x, capacity ordering VANISHES); bootstrap CI 2.3x-17x; n=2 arches cannot establish a
  scaling law; shell=94-95% of total params so shell-vs-total inseparable. Do NOT publish at any
  strength (would need >=4-5 arches decoupling width from depth/total-params).
- Headline "+1.68pp @p005" is NOT representative: full is band-depressed 1.43pp below its own
  p001-p010 plateau at the fixed 17-18 writer band. Quote p001-p002 gap (+0.6-0.7pp R1, +2.1-2.5pp
  R0) as the representative low-p effect. Sign is signal; that magnitude is the most band-sensitive #.
- Exact location "p0.05-0.10" -> report as "low single-digit-% regime", not a point (R0/R1 disagree
  2-7x on location). CNN low-p CCVR win fragile -> lead CNN with R0 (+3.66pp).

SURVIVING ARTIFACT-RISK: the fixed size-biased writer band at small p (writers 1.5-1.7x avg). Threatens
magnitude/location/volume-vs-identity, NOT existence/sign. Fix = rate_assignment="random" (implemented).

FOLLOW-UP (audit: hard gate before publishing beyond qualitative n=1):
- Band-randomized n=5 ResNet + n=5 CNN at p_full {0.005,0.01,0.02,0.05,0.10}; average 3 head-seeds
  per run then take across-run std; t-test head-avg gap vs 0 + Spearman trend; probe rates {0.25,1.0}
  only (3x cheaper); report R0+R1 side by side. Cost ~75 GPU-h (~1 day on 3 GPUs).
- Headline-grade option: ResNet n=10 (~155 GPU-h, ~2 days). NO-GO: capacity-scaling program (needs 4-5 arches).

## CENTRAL CROSS-WIDTH TRAINING GATE (2026-07-05 22:30) - the last untried lever

The 34th method, only training-side one (all 33 prior were eval-side). SlimResNet18, CIFAR-100, 120ep, 2 seeds.
probe/central_crosswidth_gate.py; results/crosswidth_gate/.
BARRIER's CAPACITY CLAIM HOLDS: even best-case (full participation, inplace distillation full->core-prefix),
core = 71.0(s42)/71.4(s43) vs full 77.8, ~6.5pp gap, only ~20% of headroom captured. Shell capacity does
NOT transfer into the deployable core. joint (sandwich, no KD) core = +0.15pp = nothing.
BUT a real cross-width-SPECIFIC, seed-robust lift EXISTS (I mispredicted generic-reg washout):
joint_kd core beats standalone core +1.35(s42)/+2.00(s43) = +1.67pp mean; label-smoothing control only
+0.55pp => cross-width-specific component ~+1.1pp. FIRST training-side positive signal in the program.
OPEN QUESTION -> FL TEST: does the +1.1pp survive SPARSE participation (weak/under-trained full teacher)?
Prior: likely shrinks (under-trained shell = poor teacher). Win-win: survives=modest method; vanishes=
sharpens barrier (participation wall bounds even the training-side lever). Building crosswidth_fl_gate.py.

## FL CROSS-WIDTH TEST (2026-07-06 00:07) - training-side lever in sparse FL: BARRIER COMPLETE

Does the central +1.1pp inplace-distillation core lift survive SPARSE participation? probe/crosswidth_fl_gate.py
(SlimResNet18, CIFAR-100 N=100 Dirichlet a=0.5, cpr=10, 400 rounds; correct HeteroFL prefix aggregation
VERIFIED numerically; FL sim adversarially reviewed = clean, arms byte-identical except KD gradient). 2 arms
(heterofl vs crosswidth=full clients do sandwich+KD), p_full {0.1,0.3,0.5}, 2 seeds.

CORE lift (XW-HFL), cleanly participation-gated: p0.1 +0.14 / p0.3 +1.32 / p0.5 +5.03pp. Positive control
(p0.5) CONFIRMS the mechanism works with a trained teacher. BUT the number that matters, DEPLOYED accuracy
= max(core,full) per arm (selector picks best width): p0.1 +0.14 / p0.3 -0.33 / p0.5 -1.95pp. NO deployment
win, NEGATIVE at moderate+ participation. Why the core lift is a TRAP: (1) it is large only at p0.3-0.5 where
you deploy the FULL model anyway; (2) the KD TAXES the full (XW full < HFL full everywhere, -2.8pp @p0.5) so
the deployed model gets worse; (3) at p0.1 where the core IS deployed, the lift is ~0 (teacher too under-trained).
=> the participation wall gates the training-side lever to vanish exactly where the core is deployed.

VERDICT: BARRIER COMPLETE. 34 methods now (33 eval-side + this training-side one), none improve DEPLOYED
accuracy at sparse participation, with a mechanistic reason for the training-side failure (not just a null).
Robust at n=2 (p0.5 loss large+consistent); KD tuning cannot rescue (no weight fixes an under-trained teacher).
Firm to 5 seeds + KD-HP note ONLY if featured in paper (tightens numbers, won't flip verdict). Review flagged
S1(seeds) S2(KD-HP) - see results/paper_reviews/FL_GATE_REVIEW.md.

## OBJECTIVE-INCONSISTENCY IDEAS ATTACKED - BOTH DEAD, DIAGNOSIS SHARPENED (2026-07-06)

Red-teamed 2 ideas from the "objective inconsistency" framing (role-adapter/shared-trunk decoupling; Ghost
Width local-proxy r_psi). results/paper_reviews/ATTACK_*.md. VERDICT: NO-GO as method; DIAGNOSIS.

DECISIVE EMPIRICAL FULCRUM (probe/grad_conflict_gate.py, grad_conflict.json): measured cos(g_core^core,
g_core^full) on shared core prefix (713k params), SAME held-out minibatches, real HeteroFL p0.3 run, 3 stages.
cos = +0.54/+0.63/+0.52 (early/mid/converged), NEVER negative, min 0.478 over 16 minibatches. By PCGrad def
(conflict iff dot<0): ZERO conflict; the two width forwards agree at ~55deg. Norm ratio ~1.9x (core pushes
harder) but CO-DIRECTIONAL = extra help, not opposition. => the width-conflict premise BOTH ideas need is
FALSE. Correction to earlier read: the core degradation 51.5->36.5 is NOT a conflict wound but HEALTHY
REPURPOSING - at p1.0 core standalone=1.77% but full=62.28%; the full model's optimum WANTS the destroyed core.

Both ideas: IDEA A (role-adapter) dead on premise (oracle bracket: FL core == clean standalone core as
substrate, 55.2 vs 56.1), mechanism (linear head can't absorb representational conflict), novelty (NeFL +
multi-task + softer map#12 which failed 3-seed). IDEA B (Ghost Width) already falsified (r_psi = O(core)
surrogate = substrate wall; = phantom-head#24 + aux-core#14; all couplings pre-killed; even unrun wide-r_psi
variant escape-prob ~0). Deployment catch-22 confirmed: core recovery largest where you deploy FULL; cross-
width KD taxes full; at sparse p the bar is homo-small 51.45 (beats deployed HeteroFL, needs no shell).

SHARPENED DIAGNOSIS (the publishable upgrade): the objective inconsistency is REAL but resolving it is
UNDESIRABLE. The shared prefix interpolates between two objectives (standalone core at low p, pure feature
substrate at high p) as a monotone fn of full-client volume; that SINGLE interpolation is the common cause of
the core-full crossover, the monotone w*(p_full), the participation/volume law, and all impossibility results
(Arm A, ghost, phantom, zero-conflict gradient). Any "fix the core" method optimizes against the deployed
objective = premise-level kill of the whole family. 4-axis evidence: gradient / causal / substrate / deployment.
=> METHOD SEARCH DEFINITIVELY CLOSED (now at gradient level). Paper = the diagnosis, airtight.

## RESEARCH-VS-ENGINEERING VERDICT (2026-07-06) - venue + fork decision

Stress-tested (steelman / harsh-AC / venue-grounded, results/paper_reviews/RES_*.md) whether this is research
or engineering, prompted by the user's (correct) critique that the width selector is post-hoc + setting-sensitive.
VERDICT (b): the SELECTOR is engineering (unanimous concede: argmax over 6 widths of a biased label-free proxy,
output BN/dataset-specific, no theorem, beats fixed threshold by 0.05pp in seed noise). The RESEARCH is the
VOLUME-ISOLATION DIAGNOSIS + oracle-bounded barrier (K-scaling + IID + Arm A; "best thing in the paper" per the
hostile AC). BUT it is scoped/incremental: the smooth interior width-law is a BN artifact (collapses to binary
valley on ViT); cross-device is n=1 + retracted scaling claim. => clears BENCHMARK/CHARACTERIZATION bar, below
MAIN-TRACK bar. This is why it kept scoring ~5: a D&B-tier contribution aimed at a main technical track.

STRONGEST FRAMING: lead with the VOLUME LAW (helps-or-hurts governed by data volume p_full x D, not fraction/
skew/gradient-conflict, each falsified by a dedicated intervention; confirmed OOD at 14x; oracle bounds =>
unrecoverable by low-comm single-core methods; deploy-side selection not shell repair). Selector = final-sentence
corollary only. Demote objective-inconsistency to organizing lens (carries the "repair is undesirable" claim).

VENUE (as-is): NeurIPS/ICML Datasets&Benchmarks = best fit ~35-45% (->55-60% w/ rigor upgrade); FL workshop/
ICBINB floor ~80% non-archival; AAAI main ~20-30% modal reject; ICLR/NeurIPS/ICML main <10% (needs a formal
impossibility THEOREM - not attempted; method + transferable law both foreclosed by our record).

RECOMMENDATION = A + rigor-slice-of-B, then C:
1. ONE experiment: FEMNIST cross-device band-randomized n>=5 + uniform-rigor harness w/ the 2 missing baselines
   (calibrated core, homo-small+CCVR) across all 3 datasets/archs. The shared gate all 3 reviews flag; compute
   not new science; removes the n=1/retracted-scaling liability; makes it the reproducible-protocol D&B artifact.
2. Write the honest paper (volume-law thesis, selector corollary); ship to FL workshop floor now + D&B archival.
3. Do NOT chase the selector theorem (dead: proxy 12pp-biased) or a new method (foreclosed by 34-method search).
   Time-box formal-impossibility to 2-3 wks; if no theorem, drop main-track target.
4. Redirect PRIMARY research energy to the next thread now, in parallel with writing.
The honest ceiling of this thread is a strong benchmark/characterization paper. Harvest it; stop the method-hunt tax.
