from pathlib import Path
import json,joblib,shutil,warnings
import numpy as np,pandas as pd
from scipy.special import expit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeClassifier
from scipy.linalg import LinAlgWarning
from threadpoolctl import threadpool_limits
from waveform_bank import TemporalBank
from fast_temporal_bank import FastTemporalBank
from waveform_ensemble import load_bank,PulseUnion
from gravity_gait import gravity_signals
from long_history_experiments import summaries
from fogstar_adapter import load_fogstar
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256
r=Path(__file__).resolve().parent;out=r/'runs/stable_extended_waveform_round_1';out.mkdir(exist_ok=False);parent=r/'runs/right_ankle_waveform_candidate';cfg=Config(**json.loads((parent/'manifest.json').read_text())['config'])
save_json(out/'manifest.json',dict(config=json.loads((parent/'manifest.json').read_text())['config'],train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),kernel_count=1024,seed=42,solvers=['cholesky','lsqr'],fit_precision='float64; numerical warnings treated as errors',limitation='Development selection and comparison, not independent validation',source_sha256={n:sha256(r/n) for n in ['stable_extended_waveform.py','fast_temporal_bank.py','waveform_bank.py','waveform_ensemble.py','gravity_gait.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']}));(out/'source').mkdir()
for n in json.loads((out/'manifest.json').read_text())['source_sha256']:shutil.copy2(r/n,out/'source'/n)
bank=joblib.load(r/'runs/waveform_bank_expanded_development_round_1/gravity_bank.joblib');small=load_bank(parent/'bank.json');np.testing.assert_array_equal(bank.scale,small.scale)
for a,b in zip(bank.kernels,small.kernels):
 assert a[0:2]==b[0:2] and a[-1]==b[-1];np.testing.assert_array_equal(a[2],b[2]);np.testing.assert_array_equal(a[3],b[3])
save_json(out/'bank.json',dict(seed=bank.seed,scale=bank.scale.tolist(),kernels=[dict(length=l,dilation=d,weights=w.tolist(),channels=c.tolist(),bias=b) for l,d,w,c,b in bank.kernels]));fast=FastTemporalBank.from_bank(bank)
td,vd=[build_windows(load_fogstar(r/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]];base=pd.read_csv(parent/'predictions.csv');results=[];scores=[]
with threadpool_limits(limits=2):
 x=np.concatenate([fast.transform(gravity_signals(td.x)),summaries(td.x)],1).astype(np.float64);v=np.concatenate([fast.transform(gravity_signals(vd.x)),summaries(vd.x)],1).astype(np.float64)
 mask=td.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
 for solver in ['cholesky','lsqr']:
  model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1.,solver=solver,tol=1e-10,max_iter=10000))
  with warnings.catch_warnings():
   warnings.simplefilter('error',LinAlgWarning);model.fit(x[mask],y,ridgeclassifier__sample_weight=weight)
  p=expit(3*model.decision_function(v));scores.append(p);joblib.dump(model,out/(solver+'.joblib'));np.testing.assert_allclose(p,expit(3*joblib.load(out/(solver+'.joblib')).decision_function(v)))
  t,grid=select_threshold(vd,p,cfg);m,f,events=evaluate(vd,p,t,cfg);f.to_csv(out/(solver+'_predictions.csv'),index=False);pd.DataFrame(grid).to_csv(out/(solver+'_candidates.csv'),index=False);merged=base.copy();u=PulseUnion()
  for _,g in merged.groupby('segment',sort=False):
   u.reset()
   for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=u.step(merged.loc[i,'time'],[bool(base.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
  cm,ce=event_metrics(vd,merged,cfg);merged.to_csv(out/(solver+'_combined_predictions.csv'),index=False)
  result=dict(model=solver,threshold=t,individual=m,combined=cm,events=ce,model_reload_verified=True,numerical_warning=False);results.append(result);save_json(out/'results.json',results)
  print('RESULT',solver,'combined',cm['true_warning_events'],'false',cm['total_false_alarms'],'censored',cm['censored_alarm_starts'],'lead',cm['average_warning_seconds'],flush=True)
 agreement=dict(max_score_difference=float(np.max(abs(scores[0]-scores[1]))));np.testing.assert_allclose(scores[0],scores[1],rtol=1e-4,atol=1e-5);save_json(out/'solver_agreement.json',agreement);print(agreement,flush=True)
