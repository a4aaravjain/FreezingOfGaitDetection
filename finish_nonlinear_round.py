from pathlib import Path
import json,shutil,pandas as pd
from candidate_gate import compare_candidates
from fog_pipeline import save_json
r=Path.cwd();out=r/'runs/nonlinear_waveform_round_1';rows=json.loads((out/'results.json').read_text());assert len(rows)==3
previous=json.loads((r/'runs/right_ankle_waveform_v2/results.json').read_text());evaluations=[]
for row in rows:
 evaluations.append(dict(model=row['model'],**compare_candidates(previous['combined'],previous['events'],row['combined'],row['events'])))
save_json(out/'promotion_checks.json',evaluations)
assert not any(x['eligible_for_streaming_verification'] for x in evaluations), 'Promising candidate requires streaming verification before final report'
lines=['# Nonlinear waveform experiments - 2026-11-11','','Three additional classifiers were fitted to the existing 384-pattern gravity-relative waveform features plus the 24 motion summaries: small gradient boosting, Extra Trees, and gradient boosting with ongoing FoG treated as a negative training example. Training people: 1-14. Development: 15-18. Reserved people 19-22 were not used. This is reused development data, not independent validation.','','| Model | Individual early warnings | Combined early warnings /16 | Scored false starts | Unknown starts |','|---|---:|---:|---:|---:|']
for row in rows:
 c=row['combined'];lines.append(f"| {row['model']} | {row['individual']['device']['true_warning_events']} | {c['true_warning_events']} | {c['total_false_alarms']} | {c['censored_alarm_starts']} |")
lines+=['','Disabled individual policies are failures, not evidence of useful prediction. The combined results include the retained v2 baseline. Each fitted checkpoint passed reload score parity. All candidate thresholds and merged replays are saved.','','Three additional response policies tested waveform EMA alpha 0.5, 0.75 and 1.0, with persistence two. All failed to find a useful standalone policy within the guardrails; none improved the combined 7/16 result. Artifacts: runs/waveform_response_round_1.','','The per-episode score audit is saved as episode_score_audit.csv. It uses only complete early windows and reports maximum scores of the three current heads, not successful predictions. A high instantaneous score alone does not satisfy the persistence and actual-alert-start rules.','','A new comparison gate rejects changed scoring denominators/exposure, extra false starts or unknown alerts, lost or later existing warnings, and candidates without episode/lead improvement. Its six tests passed. Gate output only qualifies a candidate for further streaming verification; it does not establish generalization.','','The objective remains unmet: v2 catches 7/16 development episodes, with zero scored false starts and one unknown boundary alert over about 10.9 scored minutes. Nine episodes remain missed. These observations do not establish near-zero real-world false alarms. The verified candidate was preserved.']
(r/'NONLINEAR_WAVEFORM_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
shutil.copy2(r/'LATEST_RESULTS.md',out/'previous_latest_results.md')
text=(r/'LATEST_RESULTS.md').read_text(encoding='utf-8');(r/'LATEST_RESULTS.md').write_text('2026-11-11 update: nonlinear waveform classifiers and faster response policies did not earn replacement of v2. See [NONLINEAR_WAVEFORM_RESULTS.md](NONLINEAR_WAVEFORM_RESULTS.md). Six new comparison-gate tests passed.\n\n'+text,encoding='utf-8')
print(evaluations)
