# Dense-model improvement experiments — 2026-10-16

16 fits completed, covering balanced training, feature noise, four training durations, direct fine-tuning of the original saved model, and earlier positive training targets to allow for alarm delay. The original checkpoint was preserved.

Scoring stays at 2–5 seconds before onset, with the existing false-alarm budget and one-second pulses. All models use right-ankle inputs and the original 24-feature dense-network architecture. Gaussian noise is training-only in that variant.

The first round trains on people 1–10, calibrates on 11–14, and assesses on 15–18. The remaining rounds intentionally use the original exploratory split: train 1–14, tune on 15–18. In these latter saved JSON files, calibration and assessment fields duplicate the SAME development evaluation; they are not independent results. The original saved model also used this development cohort. People 19–22 were not used.

| Run | Variant | Early /16 | False starts | Mean lead (s) | Censored starts |
|---|---|---:|---:|---:|---:|
| dense_balanced_round_1 | pointwise | 0 | 0 | — | 0 |
| dense_balanced_round_1 | person_episode_balanced | 1 | 0 | 4.22 | 0 |
| dense_balanced_round_1 | balanced_feature_noise | 0 | 1 | — | 0 |
| dense_balanced_development_round_1 | pointwise | 2 | 0 | 3.62 | 1 |
| dense_balanced_development_round_1 | person_episode_balanced | 1 | 0 | 4.47 | 0 |
| dense_balanced_development_round_1 | balanced_feature_noise | 1 | 0 | 4.47 | 0 |
| dense_duration_development_round_1 | balanced_8 | 2 | 0 | 4.17 | 0 |
| dense_duration_development_round_1 | balanced_12 | 1 | 0 | 4.47 | 0 |
| dense_duration_development_round_1 | balanced_16 | 1 | 0 | 4.47 | 0 |
| dense_duration_development_round_1 | balanced_32 | 1 | 0 | 4.47 | 0 |
| dense_finetune_development_round_1 | finetune_slow | 5 | 0 | 3.97 | 1 |
| dense_finetune_development_round_1 | finetune_medium | 4 | 0 | 3.80 | 1 |
| dense_finetune_development_round_1 | finetune_fast | 2 | 0 | 3.84 | 1 |
| dense_lead_development_round_1 | lead_2.5 | 4 | 0 | 4.05 | 1 |
| dense_lead_development_round_1 | lead_3.0 | 4 | 0 | 4.12 | 1 |
| dense_lead_development_round_1 | lead_3.5 | 4 | 0 | 4.12 | 1 |

Balanced training improved its matched separated-calibration baseline from zero to one early warning, at 4.22 seconds and zero scored false starts. This is not an improvement over the old 6/16 candidate and catches an episode that candidate already warned about.

All 58 regression tests passed in this session. Every fitted checkpoint passed reload prediction parity. Results, calibration grids, prediction replays, source snapshots, normalization and frozen policies are in the corresponding run folders. Repeated development comparisons increase overfitting risk; no result here establishes real-world performance.
