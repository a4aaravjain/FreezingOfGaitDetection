# Ankle choice and latest model experiments

For the current best research model, use the **right ankle**, with three-axis acceleration in g and three-axis gyroscope in degrees/second at native 60 Hz. This is a provisional choice based on this model's development results, not a universal medical recommendation. The sensor must be secured to the ankle/lower-shank region; a shoe/foot sensor is not the same measurement location. Match the study mounting position when reproducing experiments; orientation invariance does not make different anatomical locations interchangeable.

## Data audit

FoG-STAR's official schema explicitly identifies `ankleL` and `ankleR`. The current best model takes only `ankleR_acc_x/y/z` and `ankleR_gyro_x/y/z`. No back, wrist, task, subject-ID, or FoG-label fields enter the features. The loader groups recordings by subject/session/task and resets at missing samples; those identifiers define boundaries rather than predictive inputs. Native timing is checked numerically at 1/60 s; the README's timestamp-ms description conflicts with the actual file's second-scale increments.

Stanford's right ankle columns are `imu_ankle_r_ax/ay/az/gx/gy/gz`, not `imu_foot_r_*`. Raw accelerometer m/s^2 is divided by 9.80665 and rad/s is multiplied by 180/pi. The 128 Hz samples are kept native for window timing. The feature extractor averages only within the trailing supplied window when mapping its motion summaries to a common representation. No future interpolation, labels, or whole-recording normalization enter inference.

New regression tests verify explicit left/right channel order, bilateral missing-data splitting, Stanford unit conversions and native rate, and rigid-rotation invariance of the new features. The streaming entry point now rejects explicitly incompatible ankle metadata/settings. Software cannot detect a physically misplaced sensor from a caller falsely declaring it to be on the right ankle.

## Fair left/right comparison and new models

The matched trials use only intervals where BOTH ankles have valid readings, with the same people, seed, labels, boundaries and model structure. They differ only in which six sensor channels are used. Their coverage is reported against all 16 original validation episodes, including those lost through gaps. Other right-side trials use all available right-side intervals. Disabled policies catch zero episodes and are failures, not successful zero-false-alarm solutions.

| Model | Early warnings / original episodes | Scored unsuccessful starts | Boundary-censored starts |
|---|---:|---:|---:|
| matched_left_42 | 0/16 | 0 | 0 |
| matched_right_42 | 4/16 | 0 | 1 |
| matched_left_73 | 0/16 | 0 | 0 |
| matched_right_73 | 1/16 | 0 | 1 |
| right_multiscale_42 | 1/16 | 0 | 0 |
| right_multiscale_73 | 0/16 | 0 | 0 |
| right_summary_hgb_42 | 3/16 | 0 | 2 |
| right_multiscale_hgb_42 | 2/16 | 0 | 0 |

Both matched left-ankle runs failed to find a useful policy within the existing guardrails. Matched right runs warned 4 and 1 episodes. This supports retaining the right ankle for this prototype, while the difference between seeds and missing data preclude a general conclusion that right is intrinsically superior.

The new multiscale features add 0.5-, 1- and 2-second motion changes and stepping-period autocorrelations within the same trailing 3-second input. Gradient boosting and a nonlinear RBF support-vector classifier were also tested. The three RBF settings warned 1, 0 and 1 episodes, respectively, under the existing guards. None of these models beat the previous 5/16 summary network in every measure.

## Additional raw dataset and transfer result

Downloaded the complete Stanford six-IMU/seven-person raw directory: 60 recording files plus README, license and preprocessing source, 392,947,843 bytes total. All files matched publisher Git blob checksums; SHA-256 values and the pinned repository revision are preserved in external_downloads/stanford/download_manifest.json. Downloaded source was inspected, not executed. `python-calamine` was installed to read the XLSX data.

People 1-4 were allocated to training, person 5 to calibration, and people 6-7 were reserved without parsing or evaluation. The five development people provide 36 recordings and 144 observed onsets across about 59.4 minutes. FoG-STAR people 19-22 also remain outside these new model fits and evaluations.

The previously saved 5/16 FoG-STAR model, with its frozen 0.7 threshold, warned 0/18 episodes on Stanford person 5 and made one unsuccessful warning in 716.8 monitored seconds (5.02/hour). This is an external transfer diagnostic on one calibration person, not an independent final benchmark.

Two additional pooled models, a dense network and regularized logistic regression, trained on Stanford people 1-4 plus FoG-STAR people 1-14. Neither found a single threshold that passed the warning guardrails on both calibration cohorts. Their selected policies are disabled. More data did not automatically improve prediction.

## Limits and evidence

The best previous single run remains 5/16 (31.25%) with nominal 3.12-4.77-second lead, zero scored unsuccessful starts over only 10.9 minutes, and two boundary-censored starts. Four training seeds gave 5, 5, 1 and 3 warnings. All these FoG-STAR comparisons repeatedly reuse development people; they cannot establish prospective reliability or perfection, even in a lab. A high training score would not fix this limitation.

47 regression tests passed. Checkpoint reload checks passed for the new fitted models. All model predictions and rejected thresholds are saved; no failed experiment was deleted. Current code and experiments reside in `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`, not the similarly named current app workspace.

Sources:

- FoG-STAR schema and units: https://zenodo.org/records/17838806
- Stanford raw file schema and units: https://github.com/stanfordnmbl/imu-fog-detection
- Stanford study and lateral-shank sensor placement: https://link.springer.com/article/10.1186/s12984-022-00992-x

The Stanford paper studies detection of existing FoG, not proof of >=2-second prediction. Its published detection accuracy must not be presented as this project's early-warning performance.

## Alarm-timing follow-up and deployable calculation

A bounded 3-by-3 EMA/persistence comparison across four saved seeds found a new single-model development candidate: EMA 0.25, persistence 2, threshold 0.675. It warns 6/16 episodes (37.5%) versus the prior 5/16, with zero scored unsuccessful starts and one boundary-censored start. Lead times are 3.12-4.77 seconds (mean 4.03). This is NOT better in every way: mean lead fell from 4.07 to 4.03 seconds, maximum alarm duration increased from 4.25 to 4.75 seconds, and active alarm time increased from 19.75 to 28.5 seconds. Non-useful active alarm duration increased from 6.33 to 14.20 seconds even though no scored alarm start was unsuccessful. Other seeds yielded 4, 2 and 5 warnings and each exceeded the 5-second continuous-alarm guard. Therefore this is a saved research candidate, not a robust replacement or a perfect lab predictor.

The setting passing guards on all four seeds used persistence 3 instead, yielding 5, 3, 2 and 4 warnings. It raises worst-seed coverage but does not improve every seed or every metric. Full grids and outcomes are preserved in runs/alarm_timing_round_1.

The model's dense layers were exported to checked NumPy weights, removing TensorFlow from live inference. This keeps the learned function and makes runtime lighter; checkpoint and export hashes are verified on load. Full 2,622-window streaming parity passed. A TensorFlow timing run under load had a 4.35-second maximum stall despite a 1.39 ms median. That outlier is preserved in runs/ankle_multiscale_round_1/stream_verification_under_load.json. A subsequent NumPy replay of the original policy measured 0.43 ms median and 2.55 ms maximum. Desktop timing cannot guarantee sensor-to-alert latency on the eventual device.

The six-warning candidate is saved in runs/right_ankle_timing_candidate. Run `py -3.11 ankles.py summary` from the actual project folder to print its development metrics and replay the entire validation stream without TensorFlow inference. It does not train on evaluation people or rerun a final test. The original five-warning checkpoint and policy remain in runs/long_history_round_1. The existing default training command remains available.

Final six-warning CLI replay passed all 2,622 comparisons, with NumPy processing median 0.28 ms and maximum 1.95 ms on this desktop run. The final regression rerun passed 47 tests.
