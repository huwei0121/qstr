# Data and statistical reproduction

The reference dataset contains all 510 completed runs from the preserved
thirty-seed comparison, with 17 problem/method combinations. It is a lossless
export of the numerical records needed for the four-problem figures, together
with every paid query coordinate and scalar observation. The optimizer is never
rerun to draw those figures.

## Files

- data/reference_runs.json.gz: per-seed initial quality, checkpoint recommendations
  and coordinates, final quality, actual call counts, original evaluation limits,
  native-result availability and separate training/validation costs.
- data/trajectories/PROBLEM/METHOD/seed_N.npz: every paid parameter vector x and
  objective feedback value. These are numeric arrays loadable with allow_pickle=False.
- data/reference_curves.json: previously displayed means, quartiles and initial
  means, used to check recomputation. It is not the input used to draw the figure.
- data/source_hashes.json: paths relative to the original research repository and
  hashes of the source records. These provenance labels are not runtime dependencies.

Pulse trajectory files also contain individual circuit failure probabilities;
QAOA contains the 32-state count histograms and simulator seeds; gate calibration
contains per-circuit counts and circuit seeds; Ising contains sampled couplings.
The original research archive retains additional diagnostic logs and large
independent-validation internals. Their reduced quality statistics and the exact
procedure required to regenerate them are included here.

## Quality fields

The stored quality is the independently assessed error used in the original
comparison: sequence failure probability for pulse calibration, 6 minus expected
cut for QAOA, ORBIT failure for gate calibration, and expected state-transfer
infidelity for Ising. analysis.py transforms these to log failure, negative
expected cut, ORBIT failure and percentage fidelity respectively, then takes the
arithmetic mean and 25th/75th percentiles across matching seed records.

Different methods may first have a native recommendation at different costs.
Each method uses checkpoints present in all of its runs. A native early-stop
return can remain available at later budget checkpoints; the actual paid call
count is retained separately. No hindsight-best true-quality envelope is substituted.

## New experiments

Each run directory contains configuration.json, execution.log, record.json on
success, and backend-specific trajectory and validation details. A failure writes
failure.json. The outer execution.json retains completion/failure statuses.
record.json includes the full paid history, frozen recommendations, offline
quality, environment versions and the executed backend hash. These records can
be analyzed by the same plot command as the reference data.

The reference seeds are already observed research data. Rerunning this package
checks reproducibility; it does not constitute a new independent confirmation.
