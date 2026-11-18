from pathlib import Path
from dataclasses import replace
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json
from fogstar_adapter import load_fogstar
r=Path.cwd();out=r/'runs/complementarity_round_1';out.mkdir(exist_ok=False)
cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
d=build_windows(load_fogstar(r/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
paths=['right_ankle_pulse_candidate/current_summary_mlp_predictions.csv','ankle_multiscale_round_1/right_summary_hgb_42_predictions.csv','ankle_multiscale_round_1/right_multiscale_hgb_42_predictions.csv','gyro_right_round_1/lstm_acc_and_gyro_predictions.csv','state_forecast_round_1/predictions.csv']
results=[];starts=[];base=None
save_json(out/'manifest.json',dict(paths=paths,warning='Exploratory complementarity on repeatedly used validation people; not independent calibration'))
for path in paths:
 f=pd.read_csv(r/'runs'/path);np.testing.assert_array_equal(f.segment,d.meta.segment);np.testing.assert_allclose(f.time,d.meta.time)
 p=f.probability.to_numpy();threshold,grid=select_threshold(d,p,cfg);m,replayed,events=evaluate(d,p,threshold,cfg)
 results.append(dict(model=path,threshold=threshold,metrics=m,events=events));starts.append(replayed.alarm_start.to_numpy());base=replayed
 print(path,m['device']['true_warning_events'],m['device']['total_false_alarms'],flush=True)
union=np.any(starts,axis=0);base['alarm_start']=False;base['alarm_active']=False
for _,g in base.groupby('segment',sort=False):
 last=-np.inf
 for i in g.index:
  t=base.loc[i,'time']
  if union[i] and t>=last+3:
   last=t;base.loc[i,'alarm_start']=True
  base.loc[i,'alarm_active']=t<last+1-1e-9
metrics,events=event_metrics(d,base,cfg)
save_json(out/'results.json',dict(individual=results,combined=metrics,combined_events=events));base.to_csv(out/'combined_predictions.csv',index=False)
print('COMBINED',metrics,flush=True)
