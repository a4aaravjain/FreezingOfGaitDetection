from pathlib import Path
import json,shutil
r=Path.cwd()
runs=['dense_balanced_round_1','dense_balanced_development_round_1','dense_duration_development_round_1','dense_finetune_development_round_1','dense_lead_development_round_1']
lines=['# Dense-model improvement experiments — 2026-10-16','','16 fits completed, covering balanced training, feature noise, four training durations, direct fine-tuning of the original saved model, and earlier positive training targets to allow for alarm delay. The original checkpoint was preserved.','','Scoring stays at 2–5 seconds before onset, with the existing false-alarm budget and one-second pulses. All models use right-ankle inputs and the original 24-feature dense-network architecture. Gaussian noise is training-only in that variant.','','The first round trains on people 1–10, calibrates on 11–14, and assesses on 15–18. The remaining rounds intentionally use the original exploratory split: train 1–14, tune on 15–18. In these latter saved JSON files, calibration and assessment fields duplicate the SAME development evaluation; they are not independent results. The original saved model also used this development cohort. People 19–22 were not used.','','| Run | Variant | Early /16 | False starts | Mean lead (s) | Censored starts |','|---|---|---:|---:|---:|---:|']
rows=[]
for run in runs:
 for x in json.loads((r/'runs'/run/'results.json').read_text()):
  assert x['model_reload_verified']
  d=x['assessment']['device'];assert d['minimum_lead_seconds']==2 and d['total_fog_episodes']==16
  lead=d['average_warning_seconds'];lead='—' if lead is None else f'{lead:.2f}'
  lines.append(f"| {run} | {x['model']} | {d['true_warning_events']} | {d['total_false_alarms']} | {lead} | {d['censored_alarm_starts']} |")
  rows.append(dict(run=run,model=x['model'],device=d))
lines+=['','Balanced training improved its matched separated-calibration baseline from zero to one early warning, at 4.22 seconds and zero scored false starts. This is not an improvement over the old 6/16 candidate and catches an episode that candidate already warned about.','','All 58 regression tests passed in this session. Every fitted checkpoint passed reload prediction parity. Results, calibration grids, prediction replays, source snapshots, normalization and frozen policies are in the corresponding run folders. Repeated development comparisons increase overfitting risk; no result here establishes real-world performance.']
(r/'DENSE_IMPROVEMENT_RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
(r/'runs/dense_lead_development_round_1/comparison.json').write_text(json.dumps(rows,indent=2))
shutil.copy2(r/'LATEST_RESULTS.md',r/'runs/dense_lead_development_round_1/previous_latest_results.md')
print([(x['model'],x['device']['true_warning_events'],x['device']['total_false_alarms']) for x in rows])
