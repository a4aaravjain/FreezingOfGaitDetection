# Latest research status

A new right-ankle research candidate warns 6/16 episodes (37.5%), up from 5/16, at 3.12-4.77 seconds before annotated onset. There were zero scored unsuccessful alarm starts during 10.9 minutes and one boundary-censored start. This is exploratory repeated-validation tuning, not perfect prediction or an independent final test. It is not better in every metric: warnings last longer, mean lead is slightly shorter, and the same policy fails the duration guard on other training seeds.

Use the **right ankle**, with acceleration and gyroscope, for this specific model. A matched left/right comparison supports that provisional choice; it does not establish a universal best side. The adapters and new tests verify channel identity, units and sampling rates. More Stanford raw data were downloaded and tested; pooled models did not improve performance.

See [ANKLE_AND_NEW_DATA_RESULTS.md](ANKLE_AND_NEW_DATA_RESULTS.md) for the full model comparison, source documentation, timing limitations and reproducible commands. 47 regression tests passed. NumPy inference avoids TensorFlow in the streaming path and is checked against saved model predictions across all 2,622 validation windows.

Actual modified project: `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`.

Run `py -3.11 ankles.py summary` there to inspect and replay the new candidate. Saved candidate: `runs/right_ankle_timing_candidate`. Original model: `runs/long_history_round_1`. Newly added datasets, failed experiments, threshold grids and previous reports remain preserved.
