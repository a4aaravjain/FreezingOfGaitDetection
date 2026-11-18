# Nonlinear waveform experiments - 2026-11-11

Three additional classifiers were fitted to the existing 384-pattern gravity-relative waveform features plus the 24 motion summaries: small gradient boosting, Extra Trees, and gradient boosting with ongoing FoG treated as a negative training example. Training people: 1-14. Development: 15-18. Reserved people 19-22 were not used. This is reused development data, not independent validation.

| Model | Individual early warnings | Combined early warnings /16 | Scored false starts | Unknown starts |
|---|---:|---:|---:|---:|
| hgb | 0 | 7 | 0 | 1 |
| extra_trees | 1 | 7 | 0 | 1 |
| hgb_active_negatives | 0 | 7 | 0 | 1 |

Disabled individual policies are failures, not evidence of useful prediction. The combined results include the retained v2 baseline. Each fitted checkpoint passed reload score parity. All candidate thresholds and merged replays are saved.

Three additional response policies tested waveform EMA alpha 0.5, 0.75 and 1.0, with persistence two. All failed to find a useful standalone policy within the guardrails; none improved the combined 7/16 result. Artifacts: runs/waveform_response_round_1.

The per-episode score audit is saved as episode_score_audit.csv. It uses only complete early windows and reports maximum scores of the three current heads, not successful predictions. A high instantaneous score alone does not satisfy the persistence and actual-alert-start rules.

A new comparison gate rejects changed scoring denominators/exposure, extra false starts or unknown alerts, lost or later existing warnings, and candidates without episode/lead improvement. Its six tests passed. Gate output only qualifies a candidate for further streaming verification; it does not establish generalization.

The objective remains unmet: v2 catches 7/16 development episodes, with zero scored false starts and one unknown boundary alert over about 10.9 scored minutes. Nine episodes remain missed. These observations do not establish near-zero real-world false alarms. The verified candidate was preserved.
