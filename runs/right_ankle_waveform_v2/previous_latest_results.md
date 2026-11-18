# Latest research status

2026-10-24: the verified right-ankle ensemble improves early warnings from 6/16 to 7/16 (43.75%) on reused development recordings. All six original warning times are retained; one FS17 episode is newly warned 3.57 seconds early. There were zero scored false starts and one unknown boundary start over about 10.9 scored minutes. Nine episodes remain missed. This is not independently validated or near perfect.

The ensemble combines the existing dense network with a non-neural waveform classifier. Mean lead changes from 4.03 to 3.97 seconds because the additional episode has a different lead. Input: one right-ankle accelerometer and gyroscope at native 60 Hz.

70 tests passed. All 2,622 prediction windows passed raw-sample streaming checks. Desktop inference: 21.55 ms median, 93.02 ms maximum; transport and actuator delays excluded.

Run `py -3.11 ankles.py ensemble` to replay and verify. The earlier candidate remains via `py -3.11 ankles.py summary`. See [WAVEFORM_ENSEMBLE_RESULTS.md](WAVEFORM_ENSEMBLE_RESULTS.md) for tradeoffs, validation limits, all 23 fits and artifacts.
