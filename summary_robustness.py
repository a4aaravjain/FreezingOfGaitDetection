"""Report all four predeclared seeds, fixed threshold, and unweighted ensemble."""
from pathlib import Path
from dataclasses import replace
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,evaluate,select_threshold,save_json
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent
cfg=replace(Config(),sample_rate=60,horizon_seconds=5)
d=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
names=['long_history_round_1','summary_repeat_17','summary_repeat_73','summary_repeat_101']
ps=[];results=[]
for name in names:
 frame=pd.read_csv(ROOT/'runs'/name/'current_summary_mlp_predictions.csv')
 np.testing.assert_array_equal(frame.segment,d.meta.segment)
 np.testing.assert_allclose(frame.time,d.meta.time)
 p=frame.probability.to_numpy();ps.append(p)
 m,_,_=evaluate(d,p,.7,cfg)
 results.append(dict(run=name,fixed_threshold=.7,metrics=m))
p=np.mean(ps,axis=0)
t,candidates=select_threshold(d,p,cfg);m,replayed,events=evaluate(d,p,t,cfg)
out=ROOT/'runs/summary_robustness';out.mkdir(exist_ok=False)
pd.DataFrame(candidates).to_csv(out/'ensemble_candidates.csv',index=False)
replayed.to_csv(out/'ensemble_predictions.csv',index=False)
results.append(dict(run='equal_mean_four_seeds',threshold=t,metrics=m))
save_json(out/'results.json',results)
print([(a['run'],a['metrics']['device']['true_warning_events'],a['metrics']['device']['total_false_alarms']) for a in results])
