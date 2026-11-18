"""Longer past waveform context with unchanged startup and event scoring."""
from pathlib import Path
from dataclasses import asdict,replace
import json,shutil,joblib
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from waveform_bank import TemporalBank
from fast_temporal_bank import FastTemporalBank
from waveform_ensemble import PulseUnion
from gravity_gait import gravity_signals
from long_history_experiments import summaries
from fogstar_adapter import load_fogstar
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256
ROOT=Path(__file__).resolve().parent

def context_windows(data,seconds):
 factor=round(seconds/3)
 if seconds not in [6,9]:raise ValueError('Expected six or nine seconds')
 rows=180*factor;lookup={s.key:s for s in data.segments};out=np.empty_like(data.x);fraction=np.empty(len(data.y),np.float32)
 for i,row in data.meta.iterrows():
  s=lookup[row['segment']];end=int(row['end_row'])+1;start=max(0,end-rows);x=s.xyz_g[start:end];fraction[i]=len(x)/rows
  if len(x)<rows:x=np.concatenate([np.repeat(s.xyz_g[:1],rows-len(x),axis=0),x])
  out[i]=x.reshape(180,factor,6).mean(1)
 return out,fraction[:,None]

def main():
 out=ROOT/'runs/context_waveform_round_1';out.mkdir(exist_ok=False);cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 save_json(out/'manifest.json',dict(config=asdict(cfg),train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),past_seconds=[6,9],ridge_alpha=[1,100],kernels=384,seed=42,padding='repeat first observed sample before available history; observed fraction included',sampling='average adjacent past pairs/triples; latest endpoint unchanged',startup_seconds=3.,limitation='Reused development cohort; no independent validation',source_sha256={n:sha256(ROOT/n) for n in ['context_waveform.py','fast_temporal_bank.py','waveform_bank.py','gravity_gait.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']}));(out/'source').mkdir()
 for n in json.loads((out/'manifest.json').read_text())['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 td,vd=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]];summary=[summaries(d.x) for d in [td,vd]]
 mask=td.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean();baseline=pd.read_csv(ROOT/'runs/right_ankle_waveform_candidate/predictions.csv');results=[]
 np.testing.assert_array_equal(baseline.segment,vd.meta.segment);np.testing.assert_allclose(baseline.time,vd.meta.time)
 with threadpool_limits(limits=2):
  for seconds in [6,9]:
   context=[context_windows(d,seconds) for d in [td,vd]];signals=[gravity_signals(a[0]) for a in context];bank=FastTemporalBank.from_bank(TemporalBank(count=384,seed=42).fit(signals[0]));features=[np.concatenate([bank.transform(z),s,a[1]],axis=1).astype(np.float64) for z,s,a in zip(signals,summary,context)]
   save_json(out/f'bank_{seconds}s.json',dict(seed=42,scale=bank.scale.tolist(),kernels=[dict(length=l,dilation=d,weights=w.tolist(),channels=c.tolist(),bias=b) for l,d,w,c,b in bank.kernels]))
   print('FEATURES',seconds,features[0].shape,flush=True)
   for alpha in [1,100]:
    name=f'context_{seconds}s_ridge_{alpha}';model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha,solver='lsqr',tol=1e-8,max_iter=5000));model.fit(features[0][mask],y,ridgeclassifier__sample_weight=weight)
    p=expit(3*model.decision_function(features[1]));joblib.dump(model,out/(name+'.joblib'));np.testing.assert_allclose(p,expit(3*joblib.load(out/(name+'.joblib')).decision_function(features[1])))
    t,grid=select_threshold(vd,p,cfg);m,f,events=evaluate(vd,p,t,cfg);merged=baseline.copy();union=PulseUnion()
    for _,g in merged.groupby('segment',sort=False):
     union.reset()
     for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=union.step(merged.loc[i,'time'],[bool(baseline.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
    cm,ce=event_metrics(vd,merged,cfg);results.append(dict(model=name,threshold=t,individual=m,combined=cm,events=ce,model_reload_verified=True));save_json(out/'results.json',results);pd.DataFrame(grid).to_csv(out/(name+'_candidates.csv'),index=False);f.to_csv(out/(name+'_predictions.csv'),index=False);merged.to_csv(out/(name+'_combined_predictions.csv'),index=False)
    print('RESULT',name,'combined',cm['true_warning_events'],'false',cm['total_false_alarms'],'lead',cm['average_warning_seconds'],flush=True)
if __name__=='__main__':main()
