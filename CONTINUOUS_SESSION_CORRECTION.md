# Continuous-session protocol correction

**The previous zero-false-alarm claim must not be used for continuous operation.** The model weights are unchanged, but protocol task labels were incorrectly acting as sensor resets and follow-up boundaries.

The development audit found all 63 within-session task boundaries have consecutive native timestamps. Session preparation now preserves input history, alarm state and annotation follow-up across those boundaries. Real time gaps, invalid sensor samples and separate sessions still reset the predictor. Missing sensor readings never remove an annotated onset.

| Evaluation | Early episodes | Unsuccessful alerts | Unknown alerts | Monitored seconds |
|---|---:|---:|---:|---:|
| Historical task-reset scoring | 7/16 | 0 | 1 | 653.62 |
| Same saved alerts, full-session follow-up | 7/16 | 1 | 0 | 653.62 |
| Task-reset control, common native support | 7/16 | 1 | 0 | 692.40 |
| Continuous-session model, same native support | 6/16 | 3 | 1 | 692.40 |

Follow-up correction alone changes one formerly unknown alert into an unsuccessful warning. The full session replay additionally changes the input history and prediction alignment, so this is not a model improvement. On common support the control and continuous predictor use exactly the same timestamps, monitoring time and original onsets; a larger denominator is not used to manufacture a lower false-warning rate.

The continuous replay warns 6/16 episodes, with 3 unsuccessful warnings (15.60/monitored hour) and 1 unknown warning. Successful nominal leads range from 3.57 to 4.77s. Actual sensor-to-cue latency is unmeasured; the requirement for delivered lead >=2s is unverified.

## Follow-on model study

Eight new fixed fits tested raw gait summaries against changes relative to the previous 20 seconds of movement. Past median/IQR context excludes the current window from its reference and uses fitting-only scale floors. Both arms use the same fixed gradient-boosting classifier, participant folds and calibration grid. Windows, labels and alarm state use corrected physical sessions throughout.

Both calibrated policies detected 0/72 development onsets over 1.2059 monitored hours: neither found a useful calibration setting with fewer than one unsuccessful warning/hour, no unknown starts and bounded pulses. Their alarms were disabled. Zero alerts are a failed candidate, not useful suppression. No model was promoted.

These are new model fits on a corrected protocol, not an independent clinical assessment. Previous entropy and teacher/student studies remain historical task-based negative experiments and must not be used to support continuous-session performance.

## Implementation and checks

`ankles.py ensemble` now runs continuous-session development replay using the verified development export. It no longer prints cached historical performance as the current result. Saved summary/ensemble parity verifiers also use the approved export. Experimental models have not replaced the saved waveform v2 weights.

The real CLI run reproduced all 2,622 saved control prediction probabilities and alarm decisions before assessing 2,773 continuous predictions. This isolates the protocol change from model-file changes. The follow-up mapping includes each actual session ID and is checked for out-of-session timestamps.

Native context checks cover 34,760 model decision rows across both arms, with probability agreement and exact alarm decisions. These are repeat replays of the same development recordings, not added monitoring exposure.

**155 tests passed.** Tests cover task-renaming invariance, a FoG onset at a task boundary, follow-up across a task boundary, real-gap resets, missing-sensor onset retention, session-key/time validation, causal local context and export restrictions.

Only the verified development export was accessed. Reserved participants remained excluded and the locked assessment was not run. Old frozen review bundles use the historical protocol and require correction before any future assessment; their contents and access markers were not altered.

The initial continuous-session diagnostic preserved a bad session-ID mapping in its optional follow-up-only comparison. Its native replay results are unaffected. The subsequent real CLI run in continuous_session_20261111_02 fixes the mapping and validates session time ranges; that second run is the source of every table here.

Additional audit: holding missing readings for <=100ms created no added decision opportunities in the approved export. Shortening history improved availability only slightly. Availability does not establish which ankle has better prediction accuracy; right ankle remains the supported runtime placement.

Evidence: runs/continuous_session_20261111_02, runs/session_context_20261111_01, runs/session_boundary_audit_20261111_01. Each run retains plans, source snapshots, predictions, original episode ledgers, participant metrics and verification records.
