"""Bounded alarm-timing ablation on saved development scores, all four seeds reported."""
from pathlib import Path
from dataclasses import replace,asdict
import json
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent
out=ROOT/'runs/alarm_timing_round_1';out.mkdir(exist_ok=False)
base=replace(Config(),sample_rate=60,horizon_seconds=5)
d=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),base)
names=['long_history_round_1','summary_repeat_17','summary_repeat_73','summary_repeat_101']
ps=[]
for name in names:
 f=pd.read_csv(ROOT/'runs'/name/'current_summary_mlp_predictions.csv')
 np.testing.assert_array_equal(f.segment,d.meta.segment);np.testing.assert_allclose(f.time,d.meta.time);ps.append(f.probability.to_numpy())
save_json(out/'manifest.json',dict(runs=names,ema=[.25,.5,1.],persistence=[1,2,3],minimum_lead_seconds=2,warning='Exploratory repeated-validation tuning; no test claim'))
rows=[]
for alpha in [.25,.5,1.]:
 for persistence in [1,2,3]:
  cfg=replace(base,ema_alpha=alpha,persistence=persistence)
  grids=[]
  for name,p in zip(names,ps):
   _,grid=select_threshold(d,p,cfg);grids.append(grid)
  for candidates in zip(*grids):
   record=dict(alpha=alpha,persistence=persistence,threshold=candidates[0]['threshold'],all_seeds_feasible=all(a['feasible'] for a in candidates),min_coverage=min(a['episode_sensitivity'] for a in candidates),mean_coverage=float(np.mean([a['episode_sensitivity'] for a in candidates])),seed_results=candidates)
   rows.append(record)
  print(alpha,persistence,'finished',flush=True)
save_json(out/'all_candidates.json',rows)
feasible=[a for a in rows if a['all_seeds_feasible']]
best=max(feasible,key=lambda a:(a['min_coverage'],a['mean_coverage'])) if feasible else None
save_json(out/'robust_selected.json',best)
print('ROBUST',None if best is None else {k:v for k,v in best.items() if k!='seed_results'},flush=True)
# Single-seed upper result is diagnostic only; no hiding the other seeds.
f=[a for a in rows if a['seed_results'][0]['feasible']]
a=max(f,key=lambda a:a['seed_results'][0]['episode_sensitivity'])
save_json(out/'single_seed_diagnostic.json',a)
print('SINGLE',[b['true_warning_events'] for b in a['seed_results']],a['alpha'],a['persistence'],a['threshold'],flush=True)
