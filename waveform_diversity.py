"""Predeclared seed-diversity test using only development people."""
from pathlib import Path
from dataclasses import asdict,replace
import json,shutil
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier
from threadpoolctl import threadpool_limits
from waveform_bank import TemporalBank
from waveform_ensemble import PulseUnion
from gravity_gait import gravity_signals
from long_history_experiments import summaries
from fogstar_adapter import load_fogstar
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256,verify_split
ROOT=Path(__file__).resolve().parent

def main():
 out=ROOT/'runs/waveform_diversity_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 verify_split(range(1,15),range(15,19),range(19,23))
 manifest=dict(config=asdict(cfg),train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),seeds=[17,73],kernel_count=384,ridge_alpha=[1.,10.,100.],fit_precision='float64, lsqr solver tol 1e-8',limitation='All thresholds and combinations use reused development recordings; not independent validation',source_sha256={n:sha256(ROOT/n) for n in ['waveform_diversity.py','waveform_bank.py','waveform_ensemble.py','gravity_gait.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']},dataset_sha256=sha256(ROOT/'sensor_data.csv'))
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 td=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(1,15),ankle='ankleR',include_gyro=True),cfg);vd=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
 ts=gravity_signals(td.x);vs=gravity_signals(vd.x);tf=summaries(td.x);vf=summaries(vd.x)
 mask=td.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
 baseline=pd.read_csv(ROOT/'runs/right_ankle_waveform_candidate/predictions.csv');np.testing.assert_array_equal(baseline.segment,vd.meta.segment);np.testing.assert_allclose(baseline.time,vd.meta.time)
 results=[]
 with threadpool_limits(limits=2):
  for seed in manifest['seeds']:
   bank=TemporalBank(count=384,seed=seed).fit(ts)
   save_json(out/f'bank_{seed}.json',dict(seed=seed,scale=bank.scale.tolist(),kernels=[dict(length=l,dilation=d,weights=w.tolist(),channels=c.tolist(),bias=b) for l,d,w,c,b in bank.kernels]))
   x=np.concatenate([bank.transform(ts),tf],axis=1).astype(np.float64);v=np.concatenate([bank.transform(vs),vf],axis=1).astype(np.float64)
   print('FEATURES',seed,x.shape,flush=True)
   for alpha in manifest['ridge_alpha']:
    name=f'seed_{seed}_ridge_{int(alpha)}';model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=alpha,solver='lsqr',tol=1e-8))
    model.fit(x[mask],y,ridgeclassifier__sample_weight=weight);p=expit(3*model.decision_function(v))
    joblib.dump(model,out/(name+'.joblib'));restored=joblib.load(out/(name+'.joblib'));np.testing.assert_allclose(p,expit(3*restored.decision_function(v)))
    threshold,grid=select_threshold(vd,p,cfg);m,f,events=evaluate(vd,p,threshold,cfg)
    save_json(out/(name+'_policy.json'),dict(config=asdict(cfg),threshold=threshold,checkpoint_sha256=sha256(out/(name+'.joblib'))))
    pd.DataFrame(grid).to_csv(out/(name+'_candidates.csv'),index=False);f.to_csv(out/(name+'_predictions.csv'),index=False)
    merged=baseline.copy();union=PulseUnion()
    for _,g in merged.groupby('segment',sort=False):
     union.reset()
     for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=union.step(merged.loc[i,'time'],[bool(baseline.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
    combined,combined_events=event_metrics(vd,merged,cfg)
    results.append(dict(model=name,seed=seed,threshold=threshold,individual=m,events=events,combined=combined,combined_events=combined_events,model_reload_verified=True));save_json(out/'results.json',results)
    print('RESULT',name,'individual',m['device']['true_warning_events'],'combined',combined['true_warning_events'],'false',combined['total_false_alarms'],flush=True)
if __name__=='__main__':main()
