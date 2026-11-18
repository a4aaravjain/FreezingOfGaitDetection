"""DMD-inspired past-only features, not a replication of patient-specific published thresholds."""
from pathlib import Path
import json,shutil,joblib
import numpy as np,pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from gravity_gait import gravity_signals
from long_history_experiments import summaries
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256
from fogstar_adapter import load_fogstar
from waveform_ensemble import PulseUnion
from candidate_gate import compare_candidates
ROOT=Path(__file__).resolve().parent

def dmd_features(raw):
 z=gravity_signals(raw)[:,:,:2].astype(np.float64);features=[]
 for signal in z:
  signal-=signal.mean(0);delay=12
  h=np.concatenate([signal[i:i+169].T for i in range(delay)],axis=0)
  x=h[:,:-1];y=h[:,1:];u,s,v=np.linalg.svd(x,full_matrices=False);rank=min(6,int((s>max(s[0]*1e-5,1e-8)).sum()))
  if rank==0:features.append(np.zeros(10));continue
  u=u[:,:rank];s=s[:rank];v=v[:rank].T;operator=(u.T@y@v)/s[None,:];values,w=np.linalg.eig(operator);modes=(y@v/s[None,:])@w
  amplitude=np.linalg.lstsq(modes,h[:,0],rcond=1e-6)[0];maximum=float(max(abs(amplitude)));mode_norm=float(np.max(abs(modes),axis=0).mean());radius=abs(values);frequency=abs(np.angle(values))*60/(2*np.pi)
  reconstruction=modes@np.linalg.lstsq(modes,h,rcond=1e-6)[0];error=np.linalg.norm(h-reconstruction)/max(np.linalg.norm(h),1e-8)
  weight=abs(amplitude);weighted_frequency=float(np.dot(weight,frequency)/max(weight.sum(),1e-8))
  features.append([np.log1p(maximum*mode_norm),np.log1p(maximum),np.log1p(mode_norm),error,float(radius.max()),float(np.median(radius)),weighted_frequency,float(frequency.max()),float(np.log(np.maximum(radius,1e-8)).max()),float(np.std(signal[-30:])/max(np.std(signal[:30]),1e-5))])
 out=np.asarray(features,np.float32)
 if not np.isfinite(out).all():raise ValueError('Nonfinite DMD features')
 return out

def main():
 out=ROOT/'runs/dmd_prediction_round_1';out.mkdir(exist_ok=False);parent=ROOT/'runs/right_ankle_waveform_v2';original=json.loads((parent/'results.json').read_text());cfg=Config(**json.loads((parent/'manifest.json').read_text())['config'])
 save_json(out/'manifest.json',dict(reference='https://doi.org/10.1038/s41598-025-88110-4',method='DMD-inspired multifeature supervised adaptation, not reproduction',delay_samples=12,rank_max=6,train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),limitation='Reused development policy selection, not independent validation',source_sha256={n:sha256(ROOT/n) for n in ['dmd_prediction.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py','fog_pipeline.py','fogstar_adapter.py','waveform_ensemble.py','candidate_gate.py']}));(out/'source').mkdir()
 for n in json.loads((out/'manifest.json').read_text())['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 td,vd=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]];base=pd.read_csv(parent/'predictions.csv');results=[]
 with threadpool_limits(limits=2):
  x=np.concatenate([dmd_features(td.x),summaries(td.x)],1);v=np.concatenate([dmd_features(vd.x),summaries(vd.x)],1);print('Features',x.shape,flush=True)
  mask=td.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
  for name,model in [('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500))),('boosting',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42))]:
   kwargs=dict(logisticregression__sample_weight=weight) if name=='logistic' else dict(sample_weight=weight);model.fit(x[mask],y,**kwargs);p=model.predict_proba(v)[:,1];joblib.dump(model,out/(name+'.joblib'));np.testing.assert_allclose(p,joblib.load(out/(name+'.joblib')).predict_proba(v)[:,1])
   threshold,grid=select_threshold(vd,p,cfg);m,f,e=evaluate(vd,p,threshold,cfg);merged=base.copy();pulse=PulseUnion()
   for _,g in merged.groupby('segment',sort=False):
    pulse.reset()
    for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=pulse.step(merged.loc[i,'time'],[bool(base.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
   cm,ce=event_metrics(vd,merged,cfg);gate=compare_candidates(original['combined'],original['events'],cm,ce);results.append(dict(model=name,threshold=threshold,individual=m,combined=cm,events=ce,gate=gate,model_reload_verified=True));save_json(out/'results.json',results);pd.DataFrame(grid).to_csv(out/(name+'_candidates.csv'),index=False);f.to_csv(out/(name+'_predictions.csv'),index=False);merged.to_csv(out/(name+'_combined.csv'),index=False);print(name,cm['true_warning_events'],cm['total_false_alarms'],gate,flush=True)
if __name__=='__main__':main()
