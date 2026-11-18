# Gait regularity development experiment

The new entropy models did not improve the warning system. No candidate was promoted; waveform v2 remains the default.

## Research and bounded experiment

Sample entropy and freeze-index features were motivated by [the 2019 Sensors study](https://www.mdpi.com/1424-8220/19/18/3898). That paper is not evidence that this project meets a two-second early-warning requirement. This is an adaptation, not a reproduction of its results.

Six descriptors (sample, permutation and spectral entropy, autocorrelation peak, freeze-band ratio and amplitude) are calculated from gravity-relative right-ankle signals in the trailing 3 seconds and its two halves. Half-to-half differences describe changes in regularity. Sample entropy uses m=2, r=.2 SD, 20Hz block means and a .5 count correction for sparse matches. No feature uses future samples, labels or participant ID.

16 fits: two fixed model families (unweighted logistic regression and histogram gradient boosting), with 24 existing features versus those features plus 96 regularity descriptors. Four participant outer folds; each has separate fitting and calibration participants. No extra seeds or threshold search followed evaluation. Calibration alone selects family and alarm threshold.

## Results

| Frozen policy | Early episodes | Unsuccessful starts | Unknown starts | Monitored hours |
|---|---:|---:|---:|---:|
| motion_selected | 0/72 | 0 | 0 | 1.1587 |
| entropy_selected | 0/72 | 0 | 0 | 1.1587 |

Every fitted family failed to find a calibration setting with a useful detection, strictly fewer than one unsuccessful warning per hour, zero unknown starts and one-second pulses. The predefined fallback disabled all candidate alarms (threshold1.1). Zero alerts therefore mean failure to find a useful model, not successful alarm suppression. All 72 original onsets remain in the denominator, including onsets hidden by missing sensors.

Calibration tradeoffs are retained in calibration_tradeoff_diagnostic.csv and the complete per-fold grids. They are diagnostic results only; thresholds were not changed after seeing outer outcomes.

On the original v2 historical comparison cohort, the new policy warned 0/16 episodes versus v2's 7/16. It loses previously warned episodes and fails promotion. The 72-episode nested aggregate is a different evaluation from the original 16-episode v2 result; their sensitivities are not directly comparable.

## Verification and limitations

Native replay checked 16,726 prediction windows: probability agreement within rtol1e-5/atol1e-6 and exact output decisions in all four folds. Maximum measured prediction computation was 52.51 ms. Predictions are computed even when the frozen alarm policy is disabled. Sensor, transport and actuator delays are unmeasured; no delivered-lead claim is made.

The first execution retained all fitted artifacts but stopped evaluation in each fold because a batch-versus-single float32 check demanded excessive precision. Differences were below 1e-8 in the failed checks. The separate completion run uses precisely those saved weights and policies, with the native replay tolerance. No retraining, calibration changes, folds or failed artifacts were discarded.

These recordings have been repeatedly studied, so this is historical development evidence rather than independent assessment. Strictly zero unsuccessful starts here cannot establish a population rate below one per hour; the zero-count descriptive Poisson upper limit over this short exposure is 3.18/h and is not a validation of a disabled policy.

Only the verified development-only export was read. Reserved participants remained excluded. The original combined file and locked assessment were not accessed. Right-ankle input remains the supported placement; this trial did not compare left versus right.

The evidence does not justify claiming near-perfect early prediction. A useful next study would need a distinct, prespecified source of information and substantially more participant-diverse continuous monitoring; repeated fitting to these same outcomes is not independent evidence.

See promotion_gate.json, original_v2_comparison.json, per-participant results, native_parity.json and preregistered_plan.json for the reviewable details.

Final project test suite: **140 tests passed**. Frozen v2 artifact hashes and development export hash verified; see final_verification.json.
