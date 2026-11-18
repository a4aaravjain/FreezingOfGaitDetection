# Latest research status

The new short-alert candidate preserves 6/16 early warnings (37.5%), 3.12-4.77-second nominal lead, and zero scored unsuccessful starts over 10.9 minutes. It reduces total notification time from 28.5 to 7 seconds and caps each pulse at one second. One boundary-censored alarm remains. Coverage has not increased this round; 10 of 16 episodes are still missed, and real-world reliability is not established.

The shortened output preserves every warning start across four repeat models. A fixed shared threshold passes the warning guardrails on all four, with 6, 4, 2 and 5 warnings. Two additional training runs did not beat the existing model. 49 regression tests and the full 2,622-prediction streaming replay passed.

See [SHORT_ALERT_RESULTS.md](SHORT_ALERT_RESULTS.md) for before/after burden, verification and limitations. [ANKLE_AND_NEW_DATA_RESULTS.md](ANKLE_AND_NEW_DATA_RESULTS.md) documents the right-ankle choice, additional datasets and earlier experiments.

Actual project: `C:\Users\victo\Desktop\Aarav\Science Fair 2026-2027`. Run `py -3.11 ankles.py summary` to inspect/replay `runs/right_ankle_pulse_candidate`. The sensor and target processor still need to be specified for hardware timing and cue testing.
