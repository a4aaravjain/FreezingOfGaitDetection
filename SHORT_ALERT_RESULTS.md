# Short-alert research candidate

This change reduces unnecessary notification time while preserving exactly the same early-warning starts and lead times. It does not improve the classifier's episode coverage, establish clinical cue effectiveness, or make this a successful real-world predictor.

The right-ankle model still warns 6 of 16 episodes (37.5%) at 3.12-4.77 seconds before annotated onset, with zero scored unsuccessful starts over 10.9 minutes and one boundary-censored start. It still misses 10 episodes. No final test or reserved participant data were used.

## What changed

A fixed one-second output pulse replaces continuous notification during elevated risk. The underlying risk state remains latched after the pulse ends. Thus a persistent high score cannot trigger repeated alerts. Rearming still requires the original score drop, hysteresis and cooldown. This decision uses only scores and time; no labels enter the alarm state machine. Gaps and recording boundaries still reset state. One second is an experimental engineering choice, not a clinically validated cue prescription.

Config.alert_pulse_seconds defaults to zero to preserve earlier experiments. The new saved candidate explicitly sets it to 1.0 seconds. Positive pulse durations must be integer multiples of the prediction interval. At the unchanged 0.675 threshold, EMA 0.25 and persistence 2, every start time was identical before and after across all four repeat models. The output can only shorten or remain the same; it cannot create or suppress a warning start relative to the same underlying risk policy.

| Training run | Early warnings | Unsuccessful starts | Total alert seconds: old -> new | Non-useful alert seconds: old -> new |
|---|---:|---:|---:|---:|
| long_history_round_1 | 6/16 | 0 | 28.50 -> 7.00 | 14.20 -> 0.00 |
| summary_repeat_17 | 4/16 | 0 | 18.25 -> 4.00 | 10.72 -> 0.37 |
| summary_repeat_73 | 2/16 | 0 | 10.75 -> 3.00 | 5.48 -> 0.00 |
| summary_repeat_101 | 5/16 | 0 | 24.25 -> 6.00 | 12.08 -> 0.00 |

For the current candidate, total alert time fell 75.4%, from 28.5 to 7 seconds. Six seconds belong to successful early-warning pulses; one second belongs to a boundary-censored alert. Non-useful scored active duration fell from 14.20 to 0 seconds. This is distinct from false-start count, which was already zero and remains zero. Unknown boundary outcomes remain unknown, not counted as successful or assumed false-free.

A shared-threshold comparison required every guardrail to pass on every seed and maximized worst-seed coverage, then mean coverage. It retained 0.675 and produced 6, 4, 2 and 5 warnings. All four now satisfy the maximum-output-duration guard. These are repeated-validation development results; the weakest run still warns only 2/16. The prior continuous version exceeded the 5-second duration limit for three seeds.

## Additional learning experiment

Two right-ankle summary networks were trained with ongoing FoG explicitly included as negative examples. Labels were used only for training; live inference was not label-gated. Their selected continuous policies warned 4/16 and 2/16, and their highest-coverage threshold diagnostics reached only 5/16 with excessive unsuccessful alarms. They did not improve the existing candidate and were retained as unsuccessful experiments rather than promoted.

## Verification and execution

49 regression tests passed, including pulse duration, persistent-risk no-retrigger behavior, unchanged starts/cooldown, and existing leakage/time/unit checks. The actual CLI replay matched all 2,622 saved probabilities, smoothed scores, alarm outputs and starts. NumPy feature extraction plus prediction/state processing measured 0.37 ms median and 26.04 ms maximum on this desktop run. These values exclude sensor transport and actuator delay and do not guarantee target-hardware latency. Earlier TensorFlow stalls remain documented in ANKLE_AND_NEW_DATA_RESULTS.md.

Run `py -3.11 ankles.py summary` from `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`. It now replays `runs/right_ankle_pulse_candidate`; the earlier timing-only candidate is preserved. `py -3.11 test_pipeline.py` runs the regression suite. New source: pulse_experiment.py, pulse_threshold_selection.py, Config/Alarm changes in fog_pipeline.py, and the ongoing-FoG training option in long_history_experiments.py.

Hardware sensor model, processor and alert modality are still unspecified. Use right-ankle accelerometer g plus gyroscope degrees/second at 60 Hz for this saved model. A short notification's perceptibility and the total delivered warning margin need measurement on the actual device before real-world performance can be claimed.
