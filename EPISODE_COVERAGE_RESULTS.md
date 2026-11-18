# Episode coverage and separate calibration

**This round did not increase the 6/16 episode count.** Five newly trained variants and an exploratory combination of existing models were tested. The previous research candidate remains available, but the evidence does not support treating its 6/16 development result as reliable across people.

## Important finding: where the six successes occurred

| Assessment person | Annotated episodes | Current candidate early warnings |
|---|---:|---:|
| FS15 | 10 | 6 |
| FS16 | 1 | 0 |
| FS17 | 5 | 0 |
| FS18 | 0 | 0 |

All 16 onsets have input coverage under the current protocol. The ten misses are not simply omitted because of missing history. Smoothed scores before the other participants' episodes remain below the alert threshold. A threshold-only sweep of the current model did not exceed six valid 2-5-second warnings, even when allowing more unsuccessful starts. Lowering the threshold can move starts too early or leave risk latched, so more positive windows do not necessarily yield more useful episode warnings.

Exact onset times, maximum raw/smoothed early-window scores and warning outcomes are saved in runs/episode_calibration_round_1/previous_candidate_episode_audit.csv. This is a diagnostic use of labels, not a signal available to the predictor.

## Additional validation layer

People 1-10 were used for training. People 11-14 were used only for threshold calibration. People 15-18 were used for assessment after each threshold and checkpoint hash were saved. Training ran a fixed 24 epochs rather than selecting checkpoints on assessment results. People 19-22 were not used. The assessment participants were historically exposed in prior experiments, so this is separated development assessment, not a recovered pristine final test. It also uses fewer training people than the old candidate, so the results are not a controlled claim that a training method alone caused the decrease.

The fixed warning requirement remains 2-5 seconds before onset, with one-second alert pulses and the existing unsuccessful-warning budget. A disabled threshold of 1.100 signifies failure to find a useful calibration policy; zero alerts is not success.

| Model | Calibration threshold | Calibration early warnings | Assessment early warnings | Assessment unsuccessful starts |
|---|---:|---:|---:|---:|
| pointwise | 0.775 | 1/16 | 0/16 | 0 |
| episode_max_pair | 1.100 | 0/16 | 0/16 | 0 |
| episode_max_pair_weighted | 1.100 | 0/16 | 0/16 | 0 |
| local_only | 1.100 | 0/16 | 0/16 | 0 |
| raw_plus_local | 0.875 | 1/16 | 0/16 | 0 |

The episode-focused models learn from bags of eligible pre-onset windows rather than rewarding every window equally. A positive episode requires an adjacent pair of high scores, while background bags penalize their largest score. Weighted and unweighted versions were tested. Two further models use past-only local gait contrast: current summaries minus the preceding summaries' mean, and clipped standard-deviation-normalized differences. No subject identifier, future sample or FoG label enters those contrast features.

## Existing-model combination

The saved summary network, two gradient-boosted models, raw LSTM and four-state model were checked for complementary warnings. Each policy used the same short-pulse scoring rules; merged starts used a fixed three-second refractory interval. The combined result was still 6/16, zero scored unsuccessful starts and one boundary-censored start. It improved one warning's timing but found no additional episode. This exploratory comparison reused the old assessment recordings for threshold selection and is explicitly separate from the calibration-only training experiments. It was not promoted to the streaming default.

## Verification and artifacts

51 regression tests passed. New tests verify that episode-positive bags end at least two seconds before onset, background bags exclude positive windows, and local contrast uses only past data within the current segment. All five new fitted checkpoints passed reload comparisons. No existing checkpoint or data reservation was replaced.

New code: episode_calibration.py, local_contrast.py, complementarity_experiment.py. Results, manifests, frozen thresholds, training logs and source snapshots: runs/episode_calibration_round_1 and runs/local_contrast_round_1. Combination details: runs/complementarity_round_1.

`py -3.11 ankles.py summary` still replays the existing right-ankle short-pulse research candidate. The remaining problem is useful prediction across participants, not a demonstrated need for a different threshold alone. Neither more validation nor repeated searching can itself establish higher episode coverage; a new model must demonstrate that improvement on appropriately separated recordings.
