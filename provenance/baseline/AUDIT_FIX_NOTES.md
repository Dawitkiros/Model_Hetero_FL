# Competitor-fidelity audit + fixes (AAAI missing-baseline table)

Date: 2026-06-30.

A multi-agent adversarial audit (4 dimensions: FedRolex, FIARSE, CCVR-eval, harness-fairness)
ran before any competitor checkpoint was trained, hunting specifically for a crippled competitor
that would falsely inflate the homo-small+CCVR headline. Verdict: FIX_FIRST. One HIGH and two
MEDIUM issues, all biased favors_homo_small.

## HIGH (FIXED) - FedRolex hard-blocked from +CCVR

`probe/width_aware_calibration_oracle.py` raised ValueError for the `fedrolex` core extractor, so
FedRolex could never receive a +CCVR row while homo-small/HeteroFL/FIARSE all do. The stated
justification ("rolling window not saved") is false for the canonical config: overlap=-1.0 maps to
None, so the roll is fully deterministic (`roll = rounds % output_size`, `model_idx = arange`),
reconstructible from the round count and per-layer widths.

Fix: implemented a deterministic FedRolex core extractor mirroring the FIARSE branch. It rebuilds
the strategy with the config's params, loads the final checkpoint, sets `strategy.rounds =
num_server_rounds - 1`, and returns the rate-0.25 rolled window a core client receives at the final
round (via `_split_model` / `_distribute`). Validated on the 2-round smoke checkpoint: extracts
without error, core final linear = 128 features (rate-0.25), param ratio 0.064 (= 0.25^2). The
extracted core is the correct rolled window, not the wrong static [0,128) slice.

## MEDIUM (open) - FedRolex round budget below roll period

num_server_rounds=300 but the widest layer is 512 channels and the roll advances 1 channel/round,
so layer4 never completes a sweep; mid-shell channels (~[127,213) of 512) are trained only by the
~10% full clients. This depresses FedRolex's RAW full-model column (the +CCVR core headline is
UNAFFECTED: the final-round core window is well-trained). Options: run >=512 rounds for all
width-het extractors (fair, costly), or keep 300 and disclose that the raw-full column conflates
sparse participation with sub-roll-period rounds. Decision pending; does not block the +CCVR
headline.

## MEDIUM (open) - FIARSE in-training acc_core uses static prefix

`flore/server_app.py` logs FIARSE per-round acc_core via the static prefix extractor, but FIARSE
trains importance-top-k channels, so the FL-curve raw-core is deflated. The +CCVR FIARSE row uses
the correct importance extractor; the raw-core column needs the importance-extracted core for all
seeds (the existing override is hardcoded to s42). Affects the FIARSE raw-core column for s43/s44
only. Fix before reporting the raw-core column.

## Net

The +CCVR headline (homo-small vs HeteroFL vs FedRolex vs FIARSE, all +CCVR) is now trustworthy
once the HIGH fix is in (done). The two MEDIUM issues affect only the secondary RAW columns and are
tracked for fix/disclosure before publishing the full table.
