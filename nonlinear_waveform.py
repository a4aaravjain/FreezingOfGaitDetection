"""Nonlinear classifiers on verified waveform features; exploratory development."""
from pathlib import Path
import json,shutil,joblib
import numpy as np,pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256
from fogstar_adapter import load_fogstar
from waveform_ensemble import load_bank,PulseUnion
from fast_temporal_bank import FastTemporalBank
from waveform_bank import TemporalBank
from gravity_gait import gravity_signals
from long_history_experiments import summaries
r=Path(__file__).resolve().parent;out=r/'runs/nonlinear_waveform_round_1';out.mkdir(exist_ok=False);parent=r/'runs/right_ankle_waveform_v2';cfg=Config(**json.loads((parent/'manifest.json').read_text())['config'])
save_json(out/'manifest.json',dict(train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),purpose='Test nonlinear interactions in waveform patterns rather than new random seeds',config=json.loads((parent/'manifest.json').read_text())['config'],limitation='Repeated development selection, not independent validation',source_sha256={n:sha256(r/n) for n in ['nonlinear_waveform.py','waveform_bank.py','fast_temporal_bank.py','waveform_ensemble.py','gravity_gait.py','fogstar_adapter.py','fog_pipeline.py','long_history_experiments.py','pretraining_experiments.py']}));(out/'source').mkdir()
for n in json.loads((out/'manifest.json').read_text())['source_sha256']:shutil.copy2(r/n,out/'source'/n)
td,vd=[build_windows(load_fogstar(r/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]]
bank=FastTemporalBank.from_bank(TemporalBank(count=384,seed=42).fit(gravity_signals(td.x)))
with threadpool_limits(limits=2):
 x=np.concatenate([bank.transform(gravity_signals(td.x)),summaries(td.x)],1);v=np.concatenate([bank.transform(gravity_signals(vd.x)),summaries(vd.x)],1)
 baseline=pd.read_csv(parent/'predictions.csv');np.testing.assert_array_equal(baseline.segment,vd.meta.segment);np.testing.assert_allclose(baseline.time,vd.meta.time)
 audit=[]
 for s in vd.segments:
  g=baseline[baseline.segment==s.key]
  for onset in s.onsets:
   early=g[(g.time>=onset-5)&(g.time<=onset-2)&~g.active_fog]
   audit.append(dict(subject=s.subject,segment=s.key,onset=onset,early_windows=len(early),**{c+'_max':None if early.empty else float(early[c].max()) for c in ['base_probability','waveform_probability','extended_probability']}))
 pd.DataFrame(audit).to_csv(out/'episode_score_audit.csv',index=False)
 results=[]
 for name in ['hgb','extra_trees','hgb_active_negatives']:
  labels=td.y.copy()
  if name=='hgb_active_negatives':labels[td.meta.active_fog.to_numpy(bool)]=0
  mask=labels>=0;y=labels[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
  model=ExtraTreesClassifier(n_estimators=400,min_samples_leaf=20,max_features=.5,n_jobs=2,random_state=42) if name=='extra_trees' else HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42)
  model.fit(x[mask],y,sample_weight=weight);p=model.predict_proba(v)[:,1];joblib.dump(model,out/(name+'.joblib'));np.testing.assert_allclose(p,joblib.load(out/(name+'.joblib')).predict_proba(v)[:,1])
  threshold,grid=select_threshold(vd,p,cfg);m,f,events=evaluate(vd,p,threshold,cfg);save_json(out/(name+'_policy.json'),dict(threshold=threshold,checkpoint_sha256=sha256(out/(name+'.joblib'))));f.to_csv(out/(name+'_predictions.csv'),index=False);pd.DataFrame(grid).to_csv(out/(name+'_candidates.csv'),index=False)
  merged=baseline.copy();union=PulseUnion()
  for _,g in merged.groupby('segment',sort=False):
   union.reset()
   for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=union.step(merged.loc[i,'time'],[bool(baseline.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
  cm,ce=event_metrics(vd,merged,cfg);merged.to_csv(out/(name+'_combined_predictions.csv'),index=False);results.append(dict(model=name,threshold=threshold,individual=m,combined=cm,events=ce,model_reload_verified=True));save_json(out/'results.json',results);print(name,'combined',cm['true_warning_events'],'false',cm['total_false_alarms'],'unknown',cm['censored_alarm_starts'],flush=True)
