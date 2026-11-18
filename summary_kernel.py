"""Smooth nonlinear decision boundary on existing causal summary features."""
from pathlib import Path
from dataclasses import replace
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent
out=ROOT/'runs/summary_kernel_round_1';out.mkdir(exist_ok=False)
cfg=replace(Config(),sample_rate=60,horizon_seconds=5)
save_json(out/'manifest.json',dict(train=list(range(1,15)),validation=list(range(15,19)),ankle='ankleR',C=[.1,1,10],score='sigmoid of SVM decision margin, NOT calibrated probability',source_sha256=sha256(__file__)))
td,vd=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]]
a=summaries(td.x);b=summaries(vd.x);mask=td.y>=0;y=td.y[mask];results=[]
for c in [.1,1,10]:
 model=make_pipeline(StandardScaler(),SVC(C=c,kernel='rbf',gamma='scale',class_weight={0:1,1:min((y==0).sum()/max(1,(y==1).sum()),10)},cache_size=128))
 model.fit(a[mask],y);p=expit(model.decision_function(b));joblib.dump(model,out/f'kernel_{c}.joblib');restored=joblib.load(out/f'kernel_{c}.joblib');np.testing.assert_allclose(p,expit(restored.decision_function(b)))
 threshold,grid=select_threshold(vd,p,cfg);m,r,_=evaluate(vd,p,threshold,cfg)
 pd.DataFrame(grid).to_csv(out/f'kernel_{c}_candidates.csv',index=False);r.to_csv(out/f'kernel_{c}_predictions.csv',index=False)
 results.append(dict(C=c,threshold=threshold,metrics=m,model_reload_verified=True));save_json(out/'results.json',results)
 print(c,m['device']['true_warning_events'],m['device']['total_false_alarms'],flush=True)
