# Waveform ensemble results - 2026-10-24

The verified research ensemble improves early episode coverage from 6/16 (37.5%) to 7/16 (43.75%) on repeatedly examined development recordings. Zero scored false starts and one boundary-censored start were observed. It is not perfect, still misses nine episodes, and is not independently validated.

## Candidate

The existing dense network keeps its six warning starts unchanged. A non-neural ridge classifier adds 384 fixed random temporal patterns from gravity-relative ankle acceleration/rotation plus the 24 motion summaries. Alert starts are combined with a three-second refractory period and one-second pulses. Input remains a single RIGHT ankle, six channels at native 60 Hz: acceleration in g and angular velocity in degrees/second. No labels or future samples enter inference.

| Metric | Previous | Combined |
|---|---:|---:|
| Early episodes | 6/16 | 7/16 |
| Scored false starts | 0 | 0 |
| Unknown boundary starts | 1 | 1 |
| Mean lead, seconds | 4.033 | 3.967 |
| Median lead, seconds | 4.067 | 3.867 |
| Maximum pulse, seconds | 1.00 | 1.00 |
| Total alert seconds including unknown outcome | 7.00 | 8.00 |

The new success is FS17 task 1, onset 3951.9167 seconds, warned 3.5667 seconds early. FS15 remains 6/10, FS16 0/1, FS17 improves from 0/5 to 1/5; FS18 has no episodes. Every original successful warning retains its exact lead. The average/median fall because the added episode has a different lead. The extra useful pulse and additional computation mean this is not better in every conceivable metric.

Zero scored false starts over only 653.62 scored seconds, after development tuning, cannot establish a near-zero real-world rate. The boundary-censored alert has unknown outcome.

## Run and verification

From this actual project folder:

    py -3.11 ankles.py ensemble

This command replays the saved candidate and verifies it. Live integration is via waveform_ensemble.WaveformEnsemble.step(timestamp, six_channels); reset between sessions. The command does not connect a sensor or deliver cues. The previous candidate is retained via ankles.py summary.

All 2,622 native-data prediction windows passed probability, smoothing, component-start and combined-alert parity. Gaps, repeated timestamps and invalid samples reset state. Maximum waveform score difference from batch reference: 1.33e-6. Inference uses plain JSON kernel parameters and NumPy ridge weights with artifact checksum checks, rather than unpickling an experiment bank.

Desktop feature/inference/alarm time: median 21.55 ms, 95th percentile 30.58 ms, maximum 93.02 ms. No measured inference exceeded the 250 ms interval. This is slower than the prior dense-only model. Transport and actuator delays are excluded; these are nominal signal-based leads, not measured delivered warning times.

70 tests passed, covering scalar-convolution parity, batch independence, temporal-order sensitivity, seed reproducibility, invalid input, merged pulses, refractory behavior and session reset.

## All completed comparisons

23 fits completed in this round: six small-bank models with separate calibration, six on the original development split, six larger-bank models and five pooled-ankle dense models. Each passed checkpoint-reload score parity.

Small bank: native six-axis and gravity-relative signals, each with ridge regularization 1/1000 and logistic regression. Separate train/calibration/assessment people were 1-10/11-14/15-18. No useful assessment gain resulted: native logistic generated three scored false starts without a successful early warning; the other policies were disabled. Disabled alerts are failures, not useful zero-false-alarm models.

The same six fits on train 1-14/tune 15-18 yielded the gravity-ridge FS17 warning. Selecting and combining models used this repeatedly examined development cohort, so 7/16 is an exploratory gain with selection bias, not a final test. People 19-22 were neither fitted nor evaluated in this round.

A 1,024-pattern bank and amplitude-normalized gravity shape variant added no newly covered episode. The large gravity ridge-1 solve emitted an ill-conditioning warning and was not included in the candidate.

Pooling left/right TRAINING ankles while deploying on one right ankle did not extend episode coverage. Separate calibration yielded no assessment warnings. The pooled development model warned about the same six FS15 episodes; fine-tuning warned about three.

## Artifacts and references

Candidate: runs/right_ankle_waveform_candidate, containing self-contained model artifacts, predictions.csv, results.json, per_subject.csv, manifest.json, source snapshots, verification.log, stream_verification.json and regression_verification.json. Previous candidate: runs/right_ankle_pulse_candidate.

Experiments: runs/waveform_bank_separate_round_1, runs/waveform_bank_development_round_1, runs/waveform_bank_expanded_development_round_1, runs/pooled_ankle_separate_round_1 and runs/pooled_ankle_development_round_1.

This is a custom small multivariate bank inspired by [ROCKET](https://arxiv.org/abs/1910.13051), not the official ROCKET/MiniRocket implementation. Published detection scores cannot replace the early-warning task: [FM-FoG](https://arxiv.org/html/2509.24176v1) labels windows containing existing freezing, rather than requiring two-second pre-onset warnings. No unavailable dataset or published detection score was counted as new early-warning evidence.
