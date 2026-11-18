# Latest research status

The episode-coverage round did not improve beyond 6/16. All six current warnings came from FS15 (6/10); FS16 had 0/1 and FS17 had 0/5. The short validation duration, participant concentration and historical tuning mean this is not reliable lab or real-world prediction.

Added disjoint training (people 1-10), calibration (11-14) and assessment (15-18) within new experiments. Five trained variants, including episode-focused learning and causal local gait contrast, produced no useful assessment coverage at their calibration-selected policies. An exploratory existing-model combination still covered the same six episodes. Reserved people 19-22 remain unused in these experiments. 51 regression tests passed.

See [EPISODE_COVERAGE_RESULTS.md](EPISODE_COVERAGE_RESULTS.md) for the new results and [SHORT_ALERT_RESULTS.md](SHORT_ALERT_RESULTS.md) for the retained candidate's reduced alert burden. The current candidate remains 6/16 warnings at 3.12-4.77 seconds nominal lead, zero scored unsuccessful starts over 10.9 minutes and one boundary-censored alarm; ten episodes are missed.

Actual project: `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`. `py -3.11 ankles.py summary` retains the prior right-ankle research candidate; this round did not replace it with a worse or disabled model.
