# Scientific and engineering audit

## Verified findings and changes

1. **Future leakage:** ankles.py centered its five-step probability average with symmetric padding. At 0.25-second spacing this could use predictions up to 0.5 seconds in the future. Replaced with forward-only EMA shared with streaming inference.
2. **Oracle-dependent replay:** active-FoG windows were deleted before smoothing/persistence. Labels therefore determined which predictions existed and stitched disjoint time periods together. Now every valid sensor window is inferred, including active FoG and censored tails. Only loss/window metrics use an eligibility mask.
3. **Continuous-alarm metric failure:** historical event code looked at any positive point in a cluster, let one cluster cover multiple episodes, and reported little cost for minutes of continuous warning. Now match actual alarm starts one-to-one; report occupancy, false occupancy, durations, censored alarms, and warning lead times separately.
4. **Test contamination:** experiment_log.csv explicitly says a model was accepted because it beat a prior test F1. Code comments document further test-informed changes; multiple logs evaluate S09/S10. The claim “untouched ... evaluated exactly once” was false. Reclassified all ten local subjects as historically exposed. Default development avoids S09/S10; grouped nested train/validation/outer assessment is available, honestly labeled development.
5. **Wrong sample rate:** all 17 Daphnet recordings alternate 15/16 ms steps, averaging 15.625 ms (64 Hz). Taking the median gave 16 ms and 62.5 Hz, hence 187-row “3-second” histories and 15-row prediction strides. Now use documented/verified 64 Hz, 192 rows, 16-row stride.
6. **Boundary/target problems:** labels were filtered before episode construction, allowing false starts and future targets across unobserved regions. Recording tails were labeled negative without full future follow-up. Now segment before targets; exclude unknown-onset left-censored FoG and incomplete supervision horizons. Preserve no-FoG recordings. Strictly future onsets include exactly t+3 seconds and exclude t itself.
7. **Normalization:** expanding statistics were causal, but nonstationary, tied to the particular recording history, and advanced over label-filtered samples. analyze.py additionally used whole-recording future statistics. Fixed training-only per-axis normalization gives a reproducible wearable transform without dependence on evaluation data.
8. **Class handling:** old code combined negative undersampling with positive weights, and had an external mixed-Sequence path whose effective prevalence depended on sources. Replaced with one capped sample-weight correction, natural-prevalence validation, standard modern Keras array inputs.
9. **Threshold fallbacks:** old fallbacks could violate requested budgets; choosing max probability still flags some samples. New policy never relaxes constraints silently: no feasible candidate means alarms disabled. The budgets are declared assumptions, not selected using historical test results.
10. **Model/checkpoints:** retained the idea of compact temporal convolutions; reduced capacity and removed BatchNorm/pooling complexity. PR-AUC checkpointing avoids threshold-0.5 F1 as a misleading ranking signal. No claim of improved predictive accuracy; comparison requires development CV.
11. **Reproducibility:** old artifacts overwrote prior runs, config omitted timing/split/environment details, and loading depended on the working directory. New unique run folders store source/data/model hashes, full configuration, versions, architecture, normalization, training history, threshold candidates and timestamped predictions.
12. **Broken auxiliary paths:** app.py imported nonexistent ml_ready_data and Flask is not installed in the tested interpreter. Replaced with a read-only standard-library report viewer. analyze.py now calls the audited workflow rather than offering a second leaking trainer. inspect_fogstar.py now measures timestamps within subject/session/task.
13. **Preservation:** original code/config/result files are copied to legacy/. Existing .keras files, raw data and historical logs are unchanged. They are historical artifacts, not compatible v3 deployment bundles.

## Dataset inspection

All local Daphnet sensor files were parsed: 17 recordings, 10 subjects, 11 numeric columns; labels 0/1/2; no nonfinite values; strictly increasing timestamps. Observed normal-to-FoG transitions total 237. S04 and S10 have no FoG and must remain represented as negative subjects. File-level measurements/hashes are saved in runs/audit_v3/dataset_inventory.csv.

UCI defines milliseconds, accelerations in mg, ankle/shank forward/vertical/lateral axes, and labels 0 outside experiment, 1 normal experimental activity, 2 freeze. Label 1 includes standing/turning as well as walking; this project cannot assume every negative window is gait.
Source: https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait

External imu/: 20 files, each 15,360 rows (120 seconds), 128 Hz, headers specify acceleration in g and ML/AP/SI axes. All inspected numeric values are finite; labels are 0/1. Headers alone do not establish sensor placement, mounting/sign conventions, subject overlap, or the exact annotation protocol. The prior mapping [ML, SI, AP] does not establish correspondence to Daphnet [forward, vertical, lateral]. PDFEinfo.csv is a separate 41-row/71-column clinical table in Windows-1252, not UTF-8. Clinical attributes are not wearable-time sensor features. Keep these sources disabled until provenance is established.

FoG-STAR sensor_data.csv: 329,027 rows, 31 columns, 22 subjects, 106 subject/session/task groups; 264,700 normal and 64,327 FoG rows; 356,792 missing cells across sensor channels. Local within-group timestamp steps are approximately 0.01666666667, consistent with seconds at 60 Hz. The upstream README says milliseconds, so blindly following it would introduce a factor-1000 error. The old local seconds assumption agrees with local sampling, but provenance must resolve this discrepancy before integration. Do not sort/merge tasks, interpolate across missing blocks, or infer axis alignment from column order.
Primary sources: https://github.com/CristianTuretta/FoG-STAR and https://pmc.ncbi.nlm.nih.gov/articles/PMC12946194/

No external sensor data is used for training, normalization, or threshold selection in v3. The decision is based on unresolved compatibility, not test-score optimization.

## Limits and research decisions

The v3 configuration is a defensible initial protocol, not an empirically optimal model. Keep raw XYZ initially to preserve directional information and waveform content. Extra magnitude/frequency/freeze-index features, rotation augmentation and architecture alternatives require separate predeclared development comparisons; detection-band evidence alone does not prove early predictive value. Fixed normalization preserves physical amplitude relative to a training scale; calibration/orientation transfer remains untested.

Ankle-only inference has no oracle to suppress active-FoG alarms. Binary targets explicitly apply outside active FoG; device metrics expose behavior during freezing. Post-FoG histories may predict recurrence; subgroup window metrics distinguish this from histories without FoG. Burden is reported on monitored annotated segments, not all-day unsupervised living.

Model selection uses inner validation, including both epoch and policy choice. Inner validation metrics are optimistic development diagnostics, not generalization estimates. Outer CV evaluates a locked fold model/policy but cannot undo historical exposure. Report all folds and subjects; do not treat highly overlapping windows as independent participants or compute confidence intervals by bootstrapping windows. New prospectively held-out participants are needed for credible final claims.

See README.md for exact commands, metric definitions and hardware requirements. Verification outcomes are recorded in VERIFICATION.md after execution.


## File changes

Modified: ankles.py (training/CLI), analyze.py (safe compatibility entry point), app.py (report viewer), inspect_fogstar.py (grouped inventory), run_ankles.bat (explicit development mode).

Added: fog_pipeline.py (causal data/targets/streaming/evaluation), test_pipeline.py (23 regression tests), requirements.txt (tested versions), README.md (commands/protocol), AUDIT.md (this audit), VERIFICATION.md (executed checks and limitations).

Added output directories: runs/audit_v3, runs/smoke_v3, runs/verification_v3, runs/cv_smoke_v3. Original trainer/viewer/launcher and configuration/result history were copied to legacy before modification. Raw datasets, historical logs, existing root-level models, template and editor settings were not changed.


## Follow-up: completed CV and duration selection gap

The full 20260922_202818_cv run completed all four folds (17, 13, 18, 18 epochs including early stopping). Outer predictions warned 10/210 episodes, all from S03; two policies were disabled by inner validation. This is weak development generalization, not final-test evidence. The review is saved separately under runs/20260922_202818_cv_review.

The original policy correctly measured occupancy and duration but only constrained aggregate occupancy. Its sole feasible inner-validation candidates in folds 2 and 4 had continuous alarms of 139.25 and 14.75 seconds. I should not have treated the aggregate feasibility check as sufficient for practical warning behavior. A new configurable 5-second maximum validation-alarm duration rejects such policies while preserving full replay durations. It is an engineering assumption, not a clinically validated vibration duration. It does not guarantee short alarms on unseen data or add a physical actuator timeout.

New candidate logs give all rejection reasons and raw-window rates/metrics. review_cv.py reproduces original feasibility from each old manifest and flags the new duration guard separately. The completed fold thresholds, models and outer predictions are unchanged; no architecture or training hyperparameters were changed in response to outer scores. The code that produced that run is preserved under legacy/v3_before_duration_guard. New runs snapshot exact source files to improve reproducibility.
