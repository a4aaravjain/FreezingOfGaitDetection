# Completed development CV review

Original run: C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027\runs\20260922_202818_cv

All 4 folds completed. 2 disabled their policies on inner validation.
Outer episodes predicted: 10/210 (4.76%).
Mean fold sensitivity: 5.10%.
Successful warnings came from: S03.

| Outer subjects | Policy | Episodes warned | Window F1 |
|---|---|---:|---:|
| S01,S02 | disabled_no_feasible_policy | 0/47 | 0.000 |
| S03,S04 | feasible | 10/49 | 0.438 |
| S05,S06 | disabled_no_feasible_policy | 0/76 | 0.000 |
| S07,S08 | feasible | 0/38 | 0.000 |

The low aggregate false-alarm rate partly reflects disabled policies; it is not evidence of a useful wearable.
Per-subject burden is in outer_subject_summary.csv. Pooled burden can hide poor outcomes for an individual.

validation_rejection_diagnostics.csv reproduces the original threshold constraints from INNER validation only.
It separately checks continuous duration against the current default 5-second engineering budget when the old manifest has no duration limit.
This is a diagnostic flag, not a retroactive threshold selection or re-evaluation of outer predictions.
A duration guard rejects selection; it does not truncate alarms, prove clinical acceptability, or guarantee bounded actuator behavior on new data.

The completed v3 run is preserved. No model was refit, no original threshold was changed, and no outer result was used to optimize a threshold.
