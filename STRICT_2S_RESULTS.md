# Strict two-second prediction: measured development results

None of the five fitted models produced a feasible warning policy on its validation split. The objective remains unmet. All automatically selected policies were disabled; their zero false alarms do not represent a successful predictor.

The following are diagnostic candidates with the highest episode sensitivity in each fixed threshold grid, breaking ties by warning precision. They violate deployment constraints and are not recommended operating points. FoG-STAR denominators include original annotations lost across sensor gaps.

| Dataset | Model | Episodes warned 2-3 s ahead | Unsuccessful warnings/hour | Feasible thresholds |
|---|---|---:|---:|---:|
| Daphnet | logistic | 7/38 (18.4%) | 86.9 | 0 |
| Daphnet | gradient_boosting | 6/38 (15.8%) | 153.9 | 0 |
| FoG-STAR | logistic | 2/16 (12.5%) | 103.6 | 0 |
| FoG-STAR | gradient_boosting | 1/16 (6.2%) | 31.1 | 0 |
| Daphnet | CNN (18 epochs) | 5/38 (13.2%) | 118.9 | 0 |

Validation: Daphnet S07/S08; FoG-STAR subjects 15-18, left ankle. Development results cannot establish generalization to new patients. No S09/S10 or reserved FoG-STAR model evaluation was performed.

FoG-STAR validation has 16 original annotated onsets; valid ankle segments retain 12. Missing-data events must not silently disappear from the all-episode denominator. The raw run artifacts report valid-segment metrics; the comparison above supplies the conservative original-annotation denominator.

Verification: 34 automated scientific regression checks passed. The CNN completed 18 epochs and passed saved-model reload and real-stream probability parity checks. Both baseline model types passed saved-model reload checks. Native 60 Hz online/offline state parity, feature causality and missing-data boundaries are tested.

Evidence limits: Daphnet validation monitoring is about 39 minutes; FoG-STAR validation monitoring is about 12 minutes after gaps and warm-up. Neither establishes a reliable one-warning-per-hour rate for real deployment. Inference/actuator latency and annotation uncertainty are not included in nominal warning times.

Data: the local FoG-STAR file matches the published v3 checksum, so no duplicate download was necessary. The processed turning dataset is excluded because its published preprocessing uses future samples. See STRICT_2S_PROTOCOL.md for sources and the full protocol.

Next evidence needed: additional continuous raw ankle recordings with precise synchronized onset annotations, ample non-FoG walking/turning, and genuinely unseen people. Changing thresholds or repeatedly scoring these same people cannot demonstrate near-perfect performance. New candidate models should be evaluated under a preregistered development protocol before any reserved evaluation.

Artifacts: runs/strict_2s_baselines, runs/strict_2s_cnn, runs/strict_2s_fogstar, runs/external_provenance. Each fitted run contains its exact source snapshot; do not substitute a different revision when reproducing its results.

All 185 saved threshold candidates were replayed with the final recording-tail fix; warning counts were unchanged. See runs/strict_2s_scoring_verification.json.
