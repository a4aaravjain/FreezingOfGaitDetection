2026-11-11 published-model cascade completed: all four nested development folds, 84 upstream neural fits, 12 full verifier fits and 20 ablations. Matched v2 recipe 15/72 early episodes; selected final OR 15/72, 58 unsuccessful starts over 1.1739 known monitored hours (49.41/hour), 0 unknown starts. No promotion. 169 tests passed; 17380 native windows passed replay checks. Original saved v2 weights unchanged; its historical cohort is reported separately. Actual sensor-to-cue lead remains unmeasured. Reserved participants excluded; no locked assessment. See [completed study](runs/published_cascade_20261111_01/REPORT.md).

Historical entries below describe earlier protocols; the continuous-session correction supersedes task-reset performance claims.

2026-11-11 CONTINUOUS-SESSION CORRECTION: all63 task boundaries within sessions were native-time-contiguous. Artificial task resets and follow-up censoring overstated prior performance. Corrected unchanged-v2 replay:6/16 early warnings,3 unsuccessful starts,1 unknown,692.40 monitored seconds (15.60 unsuccessful/hour). Task-reset control under full-session labels:7/16,1 unsuccessful,0 unknown on the same support. Eight corrected-session context/control fits found no useful safe policy; no promotion.155 tests pass; real ensemble CLI and native replay checked. See [protocol correction](CONTINUOUS_SESSION_CORRECTION.md).

2026-11-11 research iteration: 28 new fixed fits tested entropy/regularity features and privileged-sensor teacher/student training. Both studies completed four participant folds and all 72 development onsets. Neither found a useful policy satisfying calibration warning limits; disabled policies detected 0/72 and failed promotion. Original v2 remains 7/16 on its historical cohort, with zero scored unsuccessful starts and one unknown start. 140 tests pass; two native replays each matched 16,726 prediction windows. Reserved participants remained excluded. See [research iteration report](RESEARCH_ITERATION_20261111_RESULTS.md).

2026-11-11 nested rescue feasibility update: verified development-only export created under the authorized participant-ID filter. Four outer folds attempted, three completed, one single-class upstream failure. No Level 1 rule met the 25% cap in completed calibration folds, so real-data verifiers/ablations remained disabled. No additional detections. Completed-fold matched baseline/pass-through: 5/44 onsets, 18 unsuccessful starts over 0.7897h (22.79/h), 3 unknown starts; remaining 28/72 permitted-development onsets unassessed. Not comparable to original v2 7/16. No promotion; original default unchanged. 111 tests passed, 11,396 completed-fold native windows and 2,622 original-v2 native windows passed replay checks. See [nested report](runs/nested_rescue_20261111_02/REPORT.md).

2026-11-11 cascade update: same-tick rescue prototype and strict candidate-ledger fitter implemented; 105 tests pass. Real nested experiment is incomplete because development-only raw input/full nested ledgers are missing. No raw/reserved data accessed, no rescue selected, v2 remains default. See [cascade report](runs/rescue_cascade_20261111_01/REPORT.md).

2026-11-11 assessment-integrity update: frozen review bundle 02 contains a synthetic-tested reserved-safe runner and original-onset inventory. 26 tests pass; 2,622 saved-score alarm decisions reproduced. No raw/reserved data accessed; model remains unchanged. See [runner report](runs/scientific_review_20261111_02/RUNNER_DEVELOPMENT_REPORT.md). Execution remains disabled pending explicit review and an approved subject-only partition.

2026-11-11 research-directed update: DMD features, a shared temporal encoder and agreement policies did not beat v2. See [RESEARCH_DIRECTED_RESULTS.md](RESEARCH_DIRECTED_RESULTS.md) for sources, adaptations and results. Three new DMD tests passed.

2026-11-11 update: nonlinear waveform classifiers and faster response policies did not earn replacement of v2. See [NONLINEAR_WAVEFORM_RESULTS.md](NONLINEAR_WAVEFORM_RESULTS.md). Six new comparison-gate tests passed.

# Latest research status

2026-10-24: verified waveform v2 retains 7/16 early warnings and advances one warning from 3.12 to 3.62 seconds before onset. Mean lead improves from 3.97 to 4.04 seconds; shortest successful lead improves from 3.12 to 3.57 seconds. Scored false starts remain zero, unknown boundary alerts remain one, and total alert time remains eight seconds.

Paired median desktop streaming computation improves from 18.52 ms to 4.38 ms (4.23x faster). Full v2 replay: 4.54 ms median, 19.61 ms maximum; transport and actuator delays excluded.

80 tests passed. All 2,622 native prediction windows passed probability, smoothing and alert parity, including the new head. The earlier warning was confirmed by two higher-precision numerical solvers.

Run `py -3.11 ankles.py ensemble` for v2. The prior 7/16 candidate remains via `--run runs/right_ankle_waveform_candidate`; the original 6/16 model remains via `ankles.py summary`.

Nine episodes remain missed. These repeatedly used development recordings do not establish independent validation or real-world reliability. See [WAVEFORM_V2_RESULTS.md](WAVEFORM_V2_RESULTS.md) for measurements, limitations and all 13 new fits.
