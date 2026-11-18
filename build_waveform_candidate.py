from pathlib import Path
from dataclasses import replace,asdict
import json,shutil,joblib
import numpy as np,pandas as pd
from waveform_bank import TemporalBank
from waveform_ensemble import PulseUnion,load_bank
from fog_pipeline import Config,build_windows,event_metrics,save_json,sha256,Windows
from fogstar_adapter import load_fogstar
r=Path(__file__).resolve().parent;out=r/'runs/right_ankle_waveform_candidate';out.mkdir(exist_ok=False)
a=r/'runs/right_ankle_pulse_candidate';b=r/'runs/waveform_bank_development_round_1'
cfg=Config(**json.loads((a/'manifest.json').read_text())['config'])
policy=json.loads((b/'gravity_ridge_1_frozen_policy.json').read_text());assert cfg==Config(**policy['config'])
(out/'base').mkdir()
for name in ['manifest.json','results.json','normalization.json','numpy_export.json','current_summary_mlp.keras','current_summary_mlp_weights.npz']:
 shutil.copy2(a/name,out/'base'/name)
bank=joblib.load(b/'gravity_bank.joblib')
save_json(out/'bank.json',dict(seed=bank.seed,scale=bank.scale.tolist(),kernels=[dict(length=l,dilation=d,weights=w.tolist(),channels=c.tolist(),bias=v) for l,d,w,c,v in bank.kernels]))
model=joblib.load(b/'gravity_ridge_1.joblib');scaler=model.steps[0][1];ridge=model.steps[1][1]
np.savez(out/'ridge_weights.npz',mean=scaler.mean_,scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)
left=pd.read_csv(a/'current_summary_mlp_predictions.csv');right=pd.read_csv(b/'gravity_ridge_1_assessment_predictions.csv')
np.testing.assert_array_equal(left.segment,right.segment);np.testing.assert_allclose(left.time,right.time)
f=left.copy();f['base_probability']=left.probability;f['waveform_probability']=right.probability;f['base_smoothed']=left.smoothed;f['waveform_smoothed']=right.smoothed;f['base_start']=left.alarm_start;f['waveform_start']=right.alarm_start;f['alarm_start']=False;f['alarm_active']=False
pulse=PulseUnion()
for _,g in f.groupby('segment',sort=False):
 pulse.reset()
 for i in g.index:f.loc[i,'alarm_active'],f.loc[i,'alarm_start']=pulse.step(f.loc[i,'time'],[left.loc[i,'alarm_start'],right.loc[i,'alarm_start']])
d=build_windows(load_fogstar(r/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
np.testing.assert_array_equal(d.meta.segment,f.segment);np.testing.assert_allclose(d.meta.time,f.time)
metrics,events=event_metrics(d,f,cfg);baseline,old_events=event_metrics(d,left,cfg)
old={ (e['segment'],e['onset']):e['warning_seconds'] for e in old_events if e['warning_seconds'] is not None};new={ (e['segment'],e['onset']):e['warning_seconds'] for e in events if e['warning_seconds'] is not None}
assert set(old).issubset(new)
for k,v in old.items():assert abs(new[k]-v)<1e-6
assert metrics['true_warning_events']>baseline['true_warning_events'] and metrics['total_false_alarms']<=baseline['total_false_alarms']
assert metrics['maximum_alarm_duration_seconds']<=1+1e-6
subjects=[]
for subject in sorted(d.meta.subject.unique()):
 mask=d.meta.subject.eq(subject).to_numpy();sub=Windows(d.x[mask],d.y[mask],d.meta.loc[mask].reset_index(drop=True),[s for s in d.segments if s.subject==subject]);m,_=event_metrics(sub,f.loc[mask].reset_index(drop=True),cfg);subjects.append(dict(subject=subject,**m))
pd.DataFrame(subjects).to_csv(out/'per_subject.csv',index=False);f.to_csv(out/'predictions.csv',index=False)
save_json(out/'results.json',dict(baseline=baseline,combined=metrics,events=events,original_warnings_retained_with_same_lead=True,interpretation='Exploratory development gain only; not independent validation; mean lead changes because additional episodes have different leads'))
artifacts=[p for p in out.rglob('*') if p.is_file() and p.name not in ['predictions.csv','results.json','per_subject.csv']]
source=['waveform_ensemble.py','waveform_bank.py','build_waveform_candidate.py','summary_stream.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py','fog_pipeline.py','fogstar_adapter.py']
(out/'source').mkdir()
for name in source:shutil.copy2(r/name,out/'source'/name)
save_json(out/'manifest.json',dict(config=asdict(cfg),ankle='ankleR',base_threshold=.675,waveform_threshold=policy['threshold'],train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),status='research only; thresholds and model combination examined on reused development people',artifact_sha256={str(p.relative_to(out)).replace(chr(92),'/'):sha256(p) for p in artifacts},source_sha256={n:sha256(r/n) for n in source},dataset_sha256=sha256(r/'sensor_data.csv'),parent_runs=[str(a),str(b)]))
restored=load_bank(out/'bank.json');raw=d.x[:5]
from gravity_gait import gravity_signals
np.testing.assert_array_equal(bank.transform(gravity_signals(raw)),restored.transform(gravity_signals(raw)))
print('Combined',metrics['true_warning_events'],'/',metrics['total_fog_episodes'],'false',metrics['total_false_alarms'],flush=True)
