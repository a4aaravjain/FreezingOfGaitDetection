"""Causal pre-emphasis of recent score trend; exploratory development only."""
from pathlib import Path
from collections import deque
from dataclasses import replace,asdict
import numpy as np,pandas as pd,json
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256
from fogstar_adapter import load_fogstar
from waveform_ensemble import PulseUnion
ROOT=Path(__file__).resolve().parent

class ScoreTrend:
 def __init__(self,gain,steps=4):self.gain=gain;self.steps=steps;self.reset()
 def reset(self):self.history=deque(maxlen=self.steps)
 def step(self,p):
  if not np.isfinite(p) or not 0<=p<=1:raise ValueError('Expected finite score in [0,1]')
  previous=self.history[0] if len(self.history)==self.steps else p
  value=np.clip(p+self.gain*(p-previous),0,1);self.history.append(p);return float(value)

def main():
 out=ROOT/'runs/trend_warning_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 save_json(out/'manifest.json',dict(config=asdict(cfg),development=list(range(15,19)),gains=[.5,1.,2.],history_seconds=1.,source_sha256=sha256(Path(__file__)),limitation='Frozen trained scores, new policy selection on reused development; not independent validation'))
 d=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
 baseline=pd.read_csv(ROOT/'runs/right_ankle_waveform_candidate/predictions.csv');results=[]
 for source,column in [('dense','base_probability'),('waveform','waveform_probability')]:
  for gain in [.5,1.,2.]:
   trend=ScoreTrend(gain);p=np.zeros(len(baseline))
   for _,g in baseline.groupby('segment',sort=False):
    trend.reset()
    for i in g.index:p[i]=trend.step(baseline.loc[i,column])
   t,grid=select_threshold(d,p,cfg);m,f,events=evaluate(d,p,t,cfg);name=f'{source}_gain_{gain}'
   merged=baseline.copy();union=PulseUnion()
   for _,g in merged.groupby('segment',sort=False):
    union.reset()
    for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=union.step(merged.loc[i,'time'],[bool(baseline.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
   cm,ce=event_metrics(d,merged,cfg);results.append(dict(model=name,threshold=t,individual=m,combined=cm,events=ce))
   pd.DataFrame(grid).to_csv(out/(name+'_candidates.csv'),index=False);f.to_csv(out/(name+'_predictions.csv'),index=False);merged.to_csv(out/(name+'_combined_predictions.csv'),index=False);save_json(out/'results.json',results)
   print(name,'combined',cm['true_warning_events'],'false',cm['total_false_alarms'],'lead',cm['average_warning_seconds'],flush=True)
if __name__=='__main__':main()
