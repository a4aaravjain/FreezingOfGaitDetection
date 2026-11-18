# Pooled ankle-motion experiment

The model combines Daphnet training subjects S01-S06 with FoG-STAR training subjects 1-14 using motion magnitudes. Each cohort and participant receives balanced total training weight. Raw axis directions are not assumed to match. Validation uses S07/S08 and FoG-STAR 15-18.

Six seconds of trailing input are summarized into 120 physical-time bins. A causal convolution plus GRU predicts an onset 2-5 seconds ahead. Gravity-relative magnitude is computed from the available history only.

Completed epochs: 13. Selected policy: disabled; threshold: 1.1. A disabled policy does not meet the sensitivity target.

Diagnostic highest-sensitivity thresholds below are NOT recommended operating points:

| Cohort | Episodes warned / all original annotations | Unsuccessful warnings/hour | Feasible thresholds |
|---|---:|---:|---:|
| daphnet | 9/38 | 78.3 | 0 |
| fogstar | 2/16 | 68.0 | 0 |

Model reload and label-free online/offline predictions, smoothing and alarm-state checks passed on both native sampling rates. 36 scientific regression tests passed.

Missing-data caution: FoG-STAR validation has 16 original onsets, but only 12 remain observable after ankle gaps. The ideal-score diagnostic with six-second history warns 9/16. Reported comparison denominators above retain all 16. Raw candidate tables use observed continuous-segment denominators and are labeled accordingly.

Timing remains nominal relative to annotation and sensor-window end. Workstation inference measurements are saved in stream_verification.json, but do not establish device end-to-end delay or guarantee a two-second delivered warning.

No final or reserved-subject evaluation was performed. None of these development results establishes reliable real-world performance.
