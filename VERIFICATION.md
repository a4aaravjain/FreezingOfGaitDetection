# Executed verification

Tested with Python 3.11.9, TensorFlow 2.21.0, Keras 3.15.1, NumPy 2.4.6, pandas 3.0.5 and scikit-learn 1.9.0 on Windows CPU.

## Passed

- All six application/test Python files compiled: ankles.py, fog_pipeline.py, analyze.py, app.py, inspect_fogstar.py, test_pipeline.py.
- All five application modules imported without triggering training.
- 23 adversarial unittest cases passed. These cover target endpoints, censoring, no-FoG recordings, past-only windows, fixed normalization, invalid labels/NaNs, timestamp and annotation gaps, subject separation, validation-only threshold selection, continuous alarms, duplicate warnings, nearby episodes, active-FoG replay, stream resets, final-test guards, and online/offline probability/EMA/alarm-state equality.
- All 17 Daphnet files loaded and checked; 10 subjects, 237 observed onsets. Dataset inspection is not model evaluation of S09/S10.
- All 20 external IMU files inspected; FoG-STAR fully read and checked; the clinical CSV read with its actual Windows-1252 encoding. None was used for training.
- Real-data preprocessing/window generation produced 55,112 windows of shape (192,3) across S01-S08: 2,391 positive, 46,542 negative, 6,179 active/censored endpoints excluded from binary supervision. Sensor features were finite.
- Initial one-epoch/512-window smoke training completed, with full validation replay. Its infeasible policy correctly disabled alarms. This preliminary smoke run preceded the final censoring refinements; use verification_v3 for the tested final source.
- Final-code two-epoch training completed on all 40,675 eligible S01-S06 windows, with 9,454 S07/S08 validation replay windows (8,258 binary-eligible).
- All four grouped-CV smoke folds completed in runs/cv_smoke_v3. Each fit used 512 training windows for one epoch, selected its policy only on inner validation, then exported frozen-policy outer-subject results. All four policies were disabled; these are plumbing tests, not scientific CV performance estimates.
- Checkpointing, CSV training logs, full validation threshold sweep, model export/reload and report generation completed.
- Reloaded-model predictions matched, and the real validation stream matched batched inference within rtol=1e-5, atol=1e-6. Synthetic full-stream tests also matched alarm starts/state.
- Tested source hashes and saved model hash exactly match runs/verification_v3/manifest.json. That manifest includes no S09/S10 recording.
- Read-only report HTML rendered successfully from the final run; Python launcher py -3.11 resolves to 3.11.9.

## Two-epoch validation diagnostics (not final-test results)

Run: runs/verification_v3. No architectural, threshold-budget, or training choices were changed in response to these scores.

| Metric | Value |
|---|---:|
| Raw window precision | 0.2234 |
| Raw window recall | 0.2675 |
| Raw window F1 | 0.2435 |
| Average precision (PR-AUC definition) | 0.1389 |
| Confusion matrix [[TN,FP],[FN,TP]] | [[7378,424],[334,122]] |
| Validation-selected threshold | 0.300 |
| Episodes warned before onset | 9 / 38 (23.7%) |
| False alarm starts | 36 |
| False alarms/minute | 0.914 |
| Alarm-active time | 226.75 seconds (9.60%) |
| Mean warning lead | 1.453 seconds |
| Median warning lead | 1.547 seconds |
| Maximum continuous alarm | 21.75 seconds |

All 9 successful warnings came from S07 (9/24); S08 had 0/14. One successful warning had only 0.062 seconds lead, before any hardware latency deduction. S07 individually exceeded the pooled false-start budget (1.085/min). Feasibility is currently defined on pooled validation burden, not guaranteed for every subject. The model is not established as generalizing or practically useful; do not present pooled diagnostics as success on unseen people.

Native Windows TensorFlow reported CPU-only execution and an ignored MapDataset attribute warning. Training, exports and parity checks nonetheless completed without exceptions. Numerical bitwise reproducibility on another machine was not established.

## Still needed

- Full-length (up to 18-epoch) development/grouped-CV training and review of every subject/fold, including disabled policies.
- Prospective evaluation on genuinely unseen participants under a frozen protocol; none of S01-S10 qualifies.
- Hardware axis calibration, bias/range matching, dropped-sample behavior, on-device latency, memory/power and actuator timing. The software state reset must also cancel any physical actuator state in a future hardware adapter.
- External-dataset placement/axis/label provenance and justified resampling, before any combined-data experiment.
- UI HTTP serving was not exercised; its import and report rendering were checked.
- No physical device or clinical benefit was tested, and no final-test performance claim is made.



## Follow-up verification after completed full CV

The user completed runs/20260922_202818_cv (17/13/18/18 epochs). Its original source hashes match the preserved files in legacy/v3_before_duration_guard. Full-run findings are in runs/20260922_202818_cv_review; the earlier statement that full v3 CV remained untested is now superseded. The revised duration-selection protocol has only a smoke training check, not a full-length CV rerun.

27 regression tests pass, including selection-level rejection of a 20-second alarm that otherwise meets all aggregate limits, untruncated burden reporting, raw-rate diagnostics, and invalid duration limits. Modified Python files compiled. review_cv.py reproduced all 148 original candidate feasibility decisions from their inner-validation reports. It wrote summaries separately and did not fit models, change thresholds, or modify the original run.

runs/policy_guard_smoke completed one epoch on a 512-window training subset, full validation threshold replay, save/reload and real-data streaming parity. Candidate logs include rejection reasons and raw-window metrics. The saved exact source snapshots match both current source and manifest hashes. This smoke run is implementation verification only. No new outer-subject performance claim is made.
