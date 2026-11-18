> Historical task-based results below are superseded for continuous operation. See [the protocol correction](CONTINUOUS_SESSION_CORRECTION.md). The earlier zero-unsuccessful-warning figure must not be used as current continuous-session performance.

# Waveform version 2 results - 2026-10-24

Version 2 warns earlier for one already-caught episode and reduces desktop inference time, while retaining 7/16 early episode coverage. It is not perfect or independently validated. Nine episodes remain missed.

## Measured comparison to the verified 7/16 candidate

| Metric | Previous ensemble | Version 2 |
|---|---:|---:|
| Early warnings | 7/16 | 7/16 |
| Scored false starts | 0 | 0 |
| Unknown boundary alerts | 1 | 1 |
| Mean lead (s) | 3.967 | 4.038 |
| Shortest successful lead (s) | 3.117 | 3.567 |
| Total alert seconds including unknown outcomes | 8.00 | 8.00 |
| Maximum pulse (s) | 1.00 | 1.00 |
| Paired median streaming computation (ms) | 18.52 | 4.38 |

The FS15 task-3 onset at 3568.0667 seconds is now warned 3.6167 seconds before onset, compared with 3.1167 seconds previously. Every other successful warning keeps its original lead. The set of caught episodes is unchanged. False-start count, unknown alerts and total output time are unchanged. Zero observed false starts over only 653.62 scored seconds of reused development data cannot establish a near-zero real-world rate.

## Algorithm change

The existing dense network and 384-pattern waveform head are retained. A stable extended waveform head uses 1,024 patterns. Its first 384 patterns exactly match the smaller bank, so those features are computed once and shared. Kernels with equal length/dilation are grouped into vectorized operations without changing the arithmetic within each kernel. Small-bank feature results are preserved exactly in prefix tests and the paired feature benchmark. The additional linear head adds model parameters, so memory/complexity are not reduced even though computation is faster.

The previous large-bank fit had a numerical-conditioning warning. Version 2 uses a float64 Cholesky fit and independently checks it against a float64 iterative LSQR fit. Numerical warnings were treated as errors; neither stable fit emitted one. Both chose the same warning policy and retained the earlier warning. Maximum score difference between the two solvers was 4.30e-7. The original small-bank warning also survived an independent stable refit.

## Validation and timing

All 2,622 prediction windows passed streaming probability, smoothing, component-start and merged-alert checks against the batch reference. This includes the new extended head. Gaps, repeated timestamps and nonfinite samples reset state. All 80 regression tests passed, including causal trend/context behavior and exact small-bank prefix preservation inside the larger bank.

Full v2 replay feature/inference/alarm latency: median 4.54 ms, 95th percentile 5.66 ms, maximum 19.61 ms; zero calls exceeded the 250 ms prediction interval.
The paired benchmark used the first 160 source-order prediction windows, with no label-based selection, alternating which version executes first. Median speedup was 4.23x. Desktop timing excludes sensor transport and actuator latency. These measurements do not establish target-hardware performance or delivered warning lead.

All training and policy comparisons used people 1-14 for training and repeatedly examined people 15-18 for development. Candidate/head choice also used those development results. People 19-22 were not fitted or evaluated in this round. This is development optimization with selection bias; it is not a new independent test. Per-person coverage remains FS15 6/10, FS16 0/1, FS17 1/5; FS18 has no FoG onsets.

## Other completed work

13 additional fits completed: six seed-diversity ridge fits, one small-bank numerical control, four longer-context fits, and two stable extended-bank solvers. Six causal score-trend policies were replayed. Longer history and seed diversity did not add an episode under the guardrails. All saved fitted models passed reload score parity.

A trend correction and the pooled-ankle head each advanced a warning by 0.25 seconds but added an unknown boundary alert. Both were excluded because unknown outcomes must not be silently treated as harmless alerts. No false-alarm limit or minimum warning requirement was relaxed.

## Run and artifacts

From the actual project folder:

    py -3.11 ankles.py ensemble

This defaults to runs/right_ankle_waveform_v2 and replays/verifies the research candidate. The previous 7/16 version remains available with:

    py -3.11 ankles.py ensemble --run runs/right_ankle_waveform_candidate

The original 6/16 dense model remains via ankles.py summary. For live integration, use waveform_ensemble.WaveformEnsemble with the v2 run path and native right-ankle acceleration in g plus angular velocity in degrees/second at 60 Hz. This CLI command itself does not connect a physical sensor.

Artifacts: runs/right_ankle_waveform_v2, including weights, kernel JSON, source snapshots, manifest.json, results.json, predictions.csv, stream_verification.json, paired_stream_benchmark.json and regression_verification.json. Experiment folders: runs/waveform_diversity_round_1, runs/waveform_numeric_check_round_1, runs/context_waveform_round_1, runs/trend_warning_round_1 and runs/stable_extended_waveform_round_1. Earlier reports and candidates are retained.
