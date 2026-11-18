2026-11-11 assessment-integrity update: frozen review bundle 02 contains a synthetic-tested reserved-safe runner and original-onset inventory. 26 tests pass; 2,622 saved-score alarm decisions reproduced. No raw/reserved data accessed; model remains unchanged. See [runner report](runs/scientific_review_20261111_02/RUNNER_DEVELOPMENT_REPORT.md). Execution remains disabled pending explicit review and an approved subject-only partition.

2026-11-11 research-directed update: DMD features, a shared temporal encoder and agreement policies did not beat v2. See [RESEARCH_DIRECTED_RESULTS.md](RESEARCH_DIRECTED_RESULTS.md) for sources, adaptations and results. Three new DMD tests passed.

2026-11-11 update: nonlinear waveform classifiers and faster response policies did not earn replacement of v2. See [NONLINEAR_WAVEFORM_RESULTS.md](NONLINEAR_WAVEFORM_RESULTS.md). Six new comparison-gate tests passed.

# Latest research status

2026-10-24: verified waveform v2 retains 7/16 early warnings and advances one warning from 3.12 to 3.62 seconds before onset. Mean lead improves from 3.97 to 4.04 seconds; shortest successful lead improves from 3.12 to 3.57 seconds. Scored false starts remain zero, unknown boundary alerts remain one, and total alert time remains eight seconds.

Paired median desktop streaming computation improves from 18.52 ms to 4.38 ms (4.23x faster). Full v2 replay: 4.54 ms median, 19.61 ms maximum; transport and actuator delays excluded.

80 tests passed. All 2,622 native prediction windows passed probability, smoothing and alert parity, including the new head. The earlier warning was confirmed by two higher-precision numerical solvers.

Run `py -3.11 ankles.py ensemble` for v2. The prior 7/16 candidate remains via `--run runs/right_ankle_waveform_candidate`; the original 6/16 model remains via `ankles.py summary`.

Nine episodes remain missed. These repeatedly used development recordings do not establish independent validation or real-world reliability. See [WAVEFORM_V2_RESULTS.md](WAVEFORM_V2_RESULTS.md) for measurements, limitations and all 13 new fits.
