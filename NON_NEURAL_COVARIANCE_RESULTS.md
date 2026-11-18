# Non-neural covariance experiments — 2026-10-16

Eight fitted alternatives completed: logistic regression, RBF SVM, histogram gradient boosting and Extra Trees, each with two training representations. None improved the previous research candidate.

Both rounds used right-ankle acceleration and angular velocity, trailing 3-second inputs at 60 Hz, and the unchanged 2–5-second early-warning target. Training people 1–10; calibration 11–14; assessment 15–18. Thresholds were frozen on calibration before assessment predictions; people 19–22 were not used. The assessment cohort has been examined in earlier work, so this is not a pristine final test.

Round 1 used 87 features including log-covariances at 0, 0.25 and 0.5-second delays. Round 2 appended 24 motion summaries and balanced training mass across people, classes within people, and positive episodes within each person. Because both representation and weighting changed together, their separate effects cannot be inferred.

| Round | Model | Threshold | Early episodes /16 | False starts |
|---|---|---:|---:|---:|
| covariance_non_neural_round_1 | logistic | 1.1 | 0 | 0 |
| covariance_non_neural_round_1 | rbf_svm | 1.1 | 0 | 0 |
| covariance_non_neural_round_1 | hist_gradient_boosting | 1.1 | 0 | 0 |
| covariance_non_neural_round_1 | extra_trees | 0.25 | 0 | 13 |
| balanced_non_neural_round_1 | logistic | 1.1 | 0 | 0 |
| balanced_non_neural_round_1 | rbf_svm | 1.1 | 0 | 0 |
| balanced_non_neural_round_1 | hist_gradient_boosting | 1.1 | 0 | 0 |
| balanced_non_neural_round_1 | extra_trees | 0.25 | 0 | 11 |

Threshold 1.1 explicitly disables alerts because no feasible calibration policy was found; zero false starts in these rows is not useful prediction. Extra Trees had one calibration warning but failed on assessment: 13 and 11 false starts respectively over 653.62 scored seconds. Both also had two boundary-censored starts and one late warning (included among unsuccessful starts).

58 tests passed: the existing 53 plus tests for finite stationary covariance, offset invariance of covariance, independence between input windows, rejection of nonfinite signals, and equal person/class/episode training weights. All eight saved models passed reload prediction parity checks.

Models, frozen policies, source snapshots, hashes, calibration grids and assessment predictions are in the two named runs directories. No new model replaced the previous candidate. The earlier 6/16 result remains exploratory, with all six successes from one participant.
