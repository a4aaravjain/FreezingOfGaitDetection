> Historical task-based results below are superseded for continuous operation. See [the protocol correction](CONTINUOUS_SESSION_CORRECTION.md). The earlier zero-unsuccessful-warning figure must not be used as current continuous-session performance.

# Research iteration: entropy and privileged-sensor training

Neither new approach improved the early-warning system under the predefined promotion requirements. Waveform v2 remains the default; no experimental model was substituted for it.

The work is in `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`, in separate experiment files. `ankles.py` retains its existing default because the candidates failed promotion.

## Completed work

- 16 fits compared existing motion summaries with gait regularity features, using logistic regression and gradient boosting. The new features include sample, permutation and spectral entropy, autocorrelation and changes within the trailing three seconds. [Research motivation](https://www.mdpi.com/1424-8220/19/18/3898).
- 12 further fits tested a teacher trained with right ankle, left ankle and back sensors, and students that receive only the right ankle. The matched control uses the same student architecture without teacher targets. [Research motivation](https://pubmed.ncbi.nlm.nih.gov/40030782/).
- Both studies completed four fixed participant folds covering all 72 original development onsets. Fitting, calibration and evaluation participants were separated within each fold. These are repeatedly used development recordings, not fresh validation.

All model families failed to find a calibration setting that caught an episode while satisfying the full calibration limits: fewer than one unsuccessful warning per monitored hour, no unknown warning starts and bounded pulses. The fallback disabled their alarms. Consequently, the frozen new systems warned **0/72 episodes**. Their zero alerts are a failed candidate, not an improvement. Calibration tradeoffs and every fixed threshold are saved in the detailed reports.

## Retained model

On the original historical comparison recordings, saved waveform v2 still warns **7/16 episodes**, with **zero scored unsuccessful warnings and one unknown boundary warning** over **653.62 monitored seconds**. Successful nominal leads range from **3.57 to 4.77 seconds**, averaging **4.04 seconds**. Sensor, communication and actuator delays are unmeasured. These short, reused recordings do not establish reliable performance in daily use.

The new candidate has no added detections and loses the previous warnings, so it fails the promotion gate. The 72-episode nested results and the original 16-episode result have different evaluation populations and must not be compared as a direct gain or loss in sensitivity.

Right-ankle input remains supported. This iteration does not establish whether the left or right side is better for an individual person.

## Verification and reproducibility

- **140 tests passed**, including time-gap handling, past-only features, label-independent alarm generation, extra-sensor window alignment after gaps, and export participant restrictions.
- Each new candidate passed native streaming checks on **16,726 prediction windows**: probability agreement within the specified tolerance and exact alarm decisions. The same recordings were replayed for each candidate; this is not additional monitoring exposure.
- Frozen v2 model hashes and the verified development export hash were checked. Reserved participants remained excluded; the original combined CSV and locked assessment were not accessed.
- The first entropy execution encountered an overly strict float32 numerical check. Its models, policies and failed-run record were preserved. Evaluation was completed from those exact saved models with the corrected tolerance, without retraining or changing thresholds.

Detailed results: [regularity experiment](runs/regularity_20261111_02/REPORT.md) and [teacher/student experiment](runs/privileged_distillation_20261111_01/REPORT.md). Each directory contains fixed plans, source snapshots, calibration grids, prediction traces, participant results, native replay evidence and promotion gates.

Implementation: `regularity_features.py`, `regularity_experiment.py`, `complete_regularity_experiment.py`, `privileged_distillation.py`, `distilled_risk.py`, `regularity_stream.py` and `verify_regularity_stream.py`.
