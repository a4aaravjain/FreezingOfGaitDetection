from pathlib import Path
import json,time
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
from waveform_ensemble import load_bank
from fast_temporal_bank import FastTemporalBank
from gravity_gait import gravity_signals
from long_history_experiments import summaries
r=Path(__file__).resolve().parent;out=r/'runs/waveform_numeric_check_round_1';out.mkdir(exist_ok=False)
parent=r/'runs/right_ankle_waveform_candidate';cfg=Config(**json.loads((parent/'manifest.json').read_text())['config'])
save_json(out/'manifest.json',dict(config=json.loads((parent/'manifest.json').read_text())['config'],train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),purpose='Numerical stability of the same 384-kernel seed-42 bank',source_sha256=sha256(Path(__file__))))
td,vd=[build_windows(load_fogstar(r/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]]
bank=load_bank(parent/'bank.json');fast=FastTemporalBank.from_bank(bank)
with threadpool_limits(limits=2):
 tx=np.concatenate([fast.transform(gravity_signals(td.x)),summaries(td.x)],1).astype(np.float64);vx=np.concatenate([fast.transform(gravity_signals(vd.x)),summaries(vd.x)],1).astype(np.float64)
 mask=td.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
 model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1.,solver='lsqr',tol=1e-10,max_iter=10000));model.fit(tx[mask],y,ridgeclassifier__sample_weight=weight)
 p=expit(3*model.decision_function(vx));joblib.dump(model,out/'stable.joblib');np.testing.assert_allclose(p,expit(3*joblib.load(out/'stable.joblib').decision_function(vx)))
 threshold,grid=select_threshold(vd,p,cfg);metrics,frame,events=evaluate(vd,p,threshold,cfg);frame.to_csv(out/'predictions.csv',index=False);pd.DataFrame(grid).to_csv(out/'candidates.csv',index=False)
 old=pd.read_csv(r/'runs/waveform_bank_development_round_1/gravity_ridge_1_assessment_predictions.csv')
 results=dict(threshold=threshold,metrics=metrics,events=events,score_max_difference_from_float32=float(max(abs(old.probability-p))),warning_start_parity_at_selected_threshold=bool(np.array_equal(old.alarm_start,frame.alarm_start)))
 z=gravity_signals(vd.x[np.random.default_rng(123).choice(len(vd.x),120,replace=False)]);times={'reference':[],'grouped':[]};maxdiff=0.
 for i,sample in enumerate(z):
  outputs={}
  for name,fn in ([('reference',bank.transform),('grouped',fast.transform)] if i%2 else [('grouped',fast.transform),('reference',bank.transform)]):
   tic=time.perf_counter();outputs[name]=fn(sample[None]);times[name].append((time.perf_counter()-tic)*1000)
  np.testing.assert_allclose(outputs['reference'],outputs['grouped'],rtol=1e-6,atol=1e-6);maxdiff=max(maxdiff,float(np.max(abs(outputs['reference']-outputs['grouped']))))
 results['paired_feature_benchmark']={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(max(v))) for k,v in times.items()};results['feature_max_difference']=maxdiff;results['benchmark_windows']=len(z)
 save_json(out/'results.json',results);print(json.dumps(results,indent=2))
