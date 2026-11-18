# Privileged-sensor distillation development experiment

No model is promoted unless every prespecified gate passes. See promotion_gate.json for the computed decision.

This follows [PhysioGPN research on transferring knowledge from multiple sensors to fewer sensors](https://pubmed.ncbi.nlm.nih.gov/40030782/). It is a small boosted-tree adaptation of that principle, not a replication of the published neural network or a claim of matching its results.

Four fixed participant outer folds each separate fitting, calibration and evaluation people. The teacher uses trailing right-ankle, left-ankle and back features from fitting people only. The distilled student receives right-ankle features and learns a fixed 50/50 mixture of the true target and teacher risk. The matched direct student learns the original target with the same regressor and right-ankle features. There are 12 fits total: four teachers and eight students. Teacher fit predictions are in-sample supervised distillation targets, not evidence of held-participant accuracy.

Both calibration and evaluation use only the right ankle. Extra-sensor windows align to exactly the same trailing timestamps, including after right-sensor gaps. Missing extra windows remain missing and never remove a reference right-ankle window or original episode. Training, normalizers, thresholds and runtime never access reserved data.

| Policy | Early episodes | Unsuccessful starts | Unknown starts | Monitored hours |
|---|---:|---:|---:|---:|
| direct | 0/72 | 0 | 0 | 1.1587 |
| distilled | 0/72 | 0 | 0 | 1.1587 |

All distilled policies disabled by calibration: True. A disabled policy has no useful detections; its zero alarms are not evidence of useful prediction. The calibrated false-warning limit is strictly below one per monitored hour, with no unknown starts. No threshold changes followed outer evaluation.

On the historical v2 comparison people15-18, distilled student warns 0/16 episodes versus original v2 7/16. The nested 72-episode aggregate is a different population from this 16-episode operational comparison; do not directly compare those sensitivities.

Native replay checks 16,726 decision windows, with probability tolerance rtol1e-5/atol1e-6 and exact alarm decisions. Worst measured desktop prediction computation: 130.14ms. Teacher files and extra sensor channels are not loaded by the native student. Computation measurements exclude sensor/transport/actuator delays, so actual delivered lead is unverified.

Only the verified development export was read; reserved participants remained excluded. These are repeatedly used historical development recordings, not independent assessment. Right ankle remains the supported runtime placement; this does not establish that the right side is best for every person.

Artifacts include fixed plans/source snapshots, frozen student/teacher models, all calibration grids, full outer prediction traces, original episode inventories, per-participant metrics, calibration tradeoff diagnostics and promotion gates.

Final project test suite: **140 tests passed**. Frozen v2 artifact hashes and development export hash verified; see final_verification.json.
