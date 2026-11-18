from pathlib import Path
import numpy as np,pandas as pd
from dataclasses import replace
from fog_pipeline import Config,build_windows,select_threshold,save_json
from fogstar_adapter import load_fogstar
r=Path.cwd();cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
d=build_windows(load_fogstar(r/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
names=['long_history_round_1','summary_repeat_17','summary_repeat_73','summary_repeat_101'];grids=[]
for n in names:
 p=pd.read_csv(r/'runs'/n/'current_summary_mlp_predictions.csv').probability.to_numpy();_,grid=select_threshold(d,p,cfg);grids.append(grid)
 pd.DataFrame(grid).to_csv(r/'runs/pulse_round_1'/(n+'_candidates.csv'),index=False)
feasible=[group for group in zip(*grids) if all(a['feasible'] for a in group)]
best=max(feasible,key=lambda group:(min(a['episode_sensitivity'] for a in group),np.mean([a['episode_sensitivity'] for a in group]))) if feasible else None
save_json(r/'runs/pulse_round_1/shared_threshold_selection.json',dict(runs=names,selection='maximize minimum coverage then average across all four seeds, all guards required',selected=best))
print(None if best is None else [(a['threshold'],a['true_warning_events'],a['total_false_alarms']) for a in best])
