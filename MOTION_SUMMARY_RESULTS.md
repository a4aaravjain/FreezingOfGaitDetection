# Motion-summary prediction results

The requested >=90% episode coverage with <=1 unsuccessful warning/hour and at least 2 seconds of lead is NOT achieved. There is a promising development improvement, with substantial training variability.

A 24-feature summary of the last three seconds of right-ankle acceleration and rotation feeds a 32-unit dense neural network. Features describe motion magnitude, variation, change, and spectral energy. Inputs are accelerometer g and gyroscope degrees/second at 60 Hz. Training uses FoG-STAR people 1-14; validation uses 15-18. No reserved people 19-22 were evaluated in this round. Repeated use of the validation people means these are exploratory results, not an independent final test.

Warnings must start 2-5 seconds before annotated onset. All 16 validation onsets remain in the denominator. Alarm persistence, cooldown, smoothing and occupancy guards are unchanged. Thresholds are development-selected; repeat seeds also used the original fixed 0.700 threshold, with the same results shown below.

| Run | Threshold | Early episodes | Unsuccessful starts | Boundary-censored starts |
|---|---:|---:|---:|---:|
| long_history_round_1 | 0.700 | 5/16 | 0 | 2 |
| summary_repeat_17 | 0.700 | 5/16 | 0 | 0 |
| summary_repeat_73 | 0.700 | 1/16 | 0 | 1 |
| summary_repeat_101 | 0.700 | 3/16 | 0 | 1 |
| Equal average of all four | 0.700 | 2/16 | 0 | 0 |

The original seed's five warnings preceded annotated onset by 3.12-4.77 seconds (mean 4.07). Only 653.6 seconds (10.9 minutes) of monitored validation time were available. Zero scored unsuccessful starts over this interval does not establish a real-world false-alarm rate near zero. Boundary-censored starts are unknown outcomes, not successful warnings. Some active alarm duration falls outside useful pre-onset intervals even when its start is successful.

The four seeds were 42, 17, 73 and 101. Coverage ranged from 1/16 to 5/16 at the same threshold. The equally weighted four-model average was evaluated without selecting seeds; it warned only 2/16 at its guard-compliant threshold. This variability prevents treating the strongest single run as a robust solution. No model was silently promoted to the default ankles.py entry point.

## Verification

43 regression tests passed, including causal history padding and segment boundaries. All fitted models passed save/reload checks. summary_stream.py accepts six sensor values and timestamps without labels. verify_summary_stream.py replayed every validation sample: all 2,622 probability, smoothed-score, alarm-state and alarm-start outputs matched saved offline replay. Invalid samples, backwards timestamps and gaps reset state. Desktop feature extraction plus inference/alarm processing measured 1.42 ms median and 8.74 ms maximum; this excludes sensor transport and actuator delays. Reported lead times remain annotation-relative and do not claim delivered hardware warning times.

## Other hypotheses tested

- Thirty-second summary-history LSTM: selected 1/16 with zero unsuccessful starts; did not beat the current-summary network.
- Left-ankle raw acceleration/rotation LSTM: selected 1/16 original onsets, zero unsuccessful starts. Right-ankle raw LSTM and four-state model did not yield a useful guard-compliant policy.
- Extra data: downloaded and verified the complete 159,910,531-byte turning IMU archive. Auxiliary motion pretraining on 71 trials from 35 people did not produce a successful early-warning policy on raw FoG-STAR validation. Processed turning data were not used for causal evaluation.
- Separate-recording personalization did not improve the useful false-alarm/coverage tradeoff.
- A 100 ms causal sample-hold experiment affected no model input windows; its score variation is not evidence that holding samples helps.

## Reproduction

Actual project: C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027.

Run `py -3.11 verify_summary_stream.py` for the saved original summary-model streaming check and `py -3.11 test_pipeline.py` for regression tests. For a fresh preserved experiment: `py -3.11 long_history_experiments.py --summary-only --seed 42 --output YOUR_NEW_RUN_NAME`. Output directories must not already exist. Run manifests, source snapshots, normalization, checkpoints, training logs, threshold grids and predictions are saved under runs. summary_robustness.py records fixed-threshold repeat performance and the equal ensemble; it also refuses to overwrite its output folder.
