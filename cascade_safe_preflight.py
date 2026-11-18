from pathlib import Path
import json,hashlib,shutil,math
import pandas as pd
from rescue_cascade import poisson_interval
ROOT=Path(__file__).resolve().parent;out=ROOT/'runs/rescue_cascade_20261111_01'
if any((out/name).exists() for name in ['preflight.json','inner_calibration_single_head_feasibility.csv']):raise FileExistsError('Preflight artifacts already exist; preserve partial/completed runs')
with (out/'PREFLIGHT_STARTED.json').open('x') as marker:marker.write('{}')
# This file is a saved development-only ledger. No raw CSV is opened.
p=ROOT/'runs/group_calibration_round_1/calibration_predictions.csv';f=pd.read_csv(p);assert set(f.subject).issubset({f'FS{i:02}' for i in range(1,15)})
cal=f[f.subject.isin(['FS11','FS12','FS13','FS14'])];known=cal.target.ge(0)|cal.active_fog
rows=[]
for t in [.15,.30,.45]:
 positive=cal.probability.ge(t);hours=cal.loc[known,'duration'].sum()/3600
 starts=sum(int((g.probability.ge(t)&~g.probability.ge(t).shift(1,fill_value=False)).sum()) for _,g in cal.groupby('segment',sort=False))
 rows.append(dict(threshold=t,known_score_occupancy=float(cal.loc[known].probability.ge(t).mul(cal.loc[known,'duration']).sum()/cal.loc[known,'duration'].sum()),candidate_ticks_per_known_hour=float((positive&known).sum()/hours),candidate_run_starts_per_monitored_hour=float(starts/(cal.duration.sum()/3600)),opportunity_recall=None,occupancy_gate_passed=bool(cal.loc[known].probability.ge(t).mul(cal.loc[known,'duration']).sum()/cal.loc[known,'duration'].sum()<=.25)))
pd.DataFrame(rows).to_csv(out/'inner_calibration_single_head_feasibility.csv',index=False)
v2=json.loads((ROOT/'runs/right_ankle_waveform_v2/results.json').read_text())['combined'];hours=v2['monitored_seconds']/3600
report=dict(status='BLOCKED_PROPER_NESTED_RAW_EXPERIMENT',development_preflight_only=True,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),known_monitored_hours=float(cal.loc[known,'duration'].sum()/3600),reason='No development-only FoG-STAR raw partition or complete nested three-head score/motion ledgers found. Existing OOF scores have one head and do not exclude each proposed outer group from all verifier-training upstream fits.',selected_level1_threshold=None,opportunity_recall=None,verifier_fitted_on_real_people=False,baseline=dict(scored_unsuccessful_starts=0,monitored_hours=hours,false_rate95_poisson_per_hour=poisson_interval(0,hours),additional_zero_event_hours_for_two_sided95_upper_below1=max(0,-math.log(.025)-hours),limitation='Homogeneous Poisson calculation is descriptive; development tuning and within-person dependence invalidate treating it as guaranteed real-world coverage'),reserved_or_raw_opened=False)
(out/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
