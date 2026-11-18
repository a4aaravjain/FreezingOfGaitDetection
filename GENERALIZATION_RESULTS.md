# Generalization experiments — 2026-10-16

The model is not perfect. This round did not produce a model better across episode coverage, unsuccessful warnings and warning lead. No failed candidate replaced the existing research stream.

The previous 6/16 result remains an exploratory result on repeatedly used FoG-STAR assessment people. All six successes came from FS15, while FS16 and FS17 had no early warnings. Its zero scored unsuccessful starts over 10.9 minutes does not establish a dependable near-zero real-world rate.

## Completed interrupted work

All three gravity-relative variants finished: a dense summary network, gradient-boosted classifier and sequence-shape GRU. None found a useful policy within the calibration guardrails. Their selected thresholds are disabled, not successful zero-false-alarm predictors. Artifacts: runs/gravity_gait_round_1.

## Earlier-recording labelled personalization

For each person, the earliest complete recording was chosen from timestamps without using its labels to choose it. Every segment belonging to that recording stayed together. The latest setup sample had to precede the earliest assessment sample. The setup's labels were used only after that recording was complete. Setup episodes were never counted as predictions.

The generic model and normalization were trained on people 1-10. Personalization methods either froze that model, adapted only its output layer, or adapted all layers. Each adaptation used eight fixed epochs, combining generic training examples with that person's setup examples. When setup lacked both classes, the fixed fallback was the generic model. Later recordings from people 11-14 calibrated the common threshold; this threshold was saved before predictions on later recordings from people 15-18.

All three methods failed to find a useful calibration policy and consequently selected disabled thresholds. Assessment contained 12 episodes after excluding four setup episodes, so its denominator is not interchangeable with the original 16. This experiment requires labelled setup data and is not a model for an uncalibrated new wearer. Results, recording boundaries, checkpoints and predictions are preserved in runs/earlier_recording_adaptation_round_1.

## Group-held-out calibration

Four folds partitioned development people 1-14 into [1,2,3,4], [5,6,7,8], [9,10,11] and [12,13,14]. Each calibration score came from a model trained on the other groups. Normalization was fit only on that fold's training data. Assertions checked group disjointness and that every development row received exactly one out-of-fold score. Epoch count, architecture and seed were fixed. A shared threshold was selected from these scores before the full-development model was evaluated on people 15-18.

For the 2-5-second interval, no useful policy met the guardrails. A diagnostic threshold of 0.675 warned before 7 of 56 development onsets but produced 14 unsuccessful warnings, approximately 14.33/hour. Zero-unsuccessful-warning thresholds warned before no development episodes. The selected threshold was disabled, so the assessment result is 0/16, not a success.

A separately declared 2-8-second interval retained the two-second minimum but widened the maximum acceptable lead. It selected threshold 0.875, with 1/56 development episodes warned and zero scored unsuccessful starts; the fully trained model then warned before 0/16 assessment episodes. Original 2-5-second scoring was also saved for this candidate and likewise had no episode success. A wider scoring window was not presented as better performance on the original task.

Calibration scores from smaller fold models can differ from the full-data model's scores; this limitation is explicit in the manifest. Assessment people were historically exposed in previous experiments, so this is grouped development assessment, not a pristine final test. Reserved people 19-22 were not evaluated or fitted in this round.

Artifacts: runs/group_calibration_round_1 and runs/group_calibration_8s_round_1. Both contain fold checkpoints, normalization, training logs, out-of-fold predictions, candidate thresholds, frozen policy records and assessment predictions. Earlier runs were preserved.

## Verification and current state

53 regression tests passed, including two new tests that prevent label-based setup-recording selection and overlapping setup/assessment timelines. Fitted checkpoints were compared after reload. These are implementation checks, not evidence of perfect predictive accuracy.

New code: earlier_recording_adaptation.py and group_calibration.py. The original `py -3.11 ankles.py summary` research replay remains available. Its episode coverage has not improved this round. All project changes are in C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027, not the similarly named active workspace.
