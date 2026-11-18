"""Compare non-neural temporal-feature trees with a simple signal-change rule."""
from pathlib import Path
from dataclasses import replace,asdict
import json,shutil
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,load_subjects,build_windows,select_threshold,evaluate,save_json,sha256
from strict_baselines import features
ROOT=Path(__file__).resolve().parent

def temporal_features(x,rate=64):
    if x.ndim!=3 or x.shape[1]!=6*rate or x.shape[2]!=3:raise ValueError('Expected six seconds of triaxial input')
    recent=features(x[:,-3*rate:],rate);previous=features(x[:,:3*rate],rate)
    return np.concatenate([recent,previous,recent-previous],axis=1)

def main():
    cfg=replace(Config(),history_seconds=6,horizon_seconds=5)
    out=ROOT/'runs/non_neural_round_1';out.mkdir(exist_ok=False)
    manifest=dict(config=asdict(cfg),mode='development',train=['S01','S02','S03','S04','S05','S06'],validation=['S07','S08'],
        trials=['extra_trees_300_leaf20','normalized_signal_change'],
        source_sha256={n:sha256(ROOT/n) for n in ['non_neural_experiments.py','strict_baselines.py','fog_pipeline.py']})
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    ts,tp=load_subjects(ROOT,manifest['train'],cfg);vs,vp=load_subjects(ROOT,manifest['validation'],cfg)
    manifest['dataset_sha256']={p.name:sha256(p) for p in tp+vp};save_json(out/'manifest.json',manifest)
    td=build_windows(ts,cfg);vd=build_windows(vs,cfg)
    tx=temporal_features(td.x);vx=temporal_features(vd.x);mask=td.y>=0;y=td.y[mask]
    print('Features extracted:',tx.shape,vx.shape,flush=True)
    model=ExtraTreesClassifier(n_estimators=300,min_samples_leaf=20,max_features=.5,class_weight='balanced',random_state=42,n_jobs=4)
    with threadpool_limits(limits=4):model.fit(tx[mask],y)
    joblib.dump(model,out/'extra_trees.joblib');p=model.predict_proba(vx)[:,1]
    np.testing.assert_allclose(joblib.load(out/'extra_trees.joblib').predict_proba(vx[:10])[:,1],p[:10])
    width=tx.shape[1]//3
    # Unsupervised training reference for a generic change alarm, not a FoG classifier.
    scale=np.maximum(np.median(abs(tx[:,-width:]),axis=0),.01)
    change=np.median(abs(vx[:,-width:])/scale,axis=1)
    rule=change/(1+change)
    save_json(out/'signal_change_reference.json',dict(scale=scale.tolist(),score='median absolute recent-minus-previous feature change / training median, mapped s/(1+s)'))
    results=[]
    for name,prob in [('extra_trees',p),('signal_change_rule',rule)]:
        threshold,candidates=select_threshold(vd,prob,cfg)
        metrics,replayed,events=evaluate(vd,prob,threshold,cfg)
        pd.DataFrame(candidates).to_csv(out/(name+'_candidates.csv'),index=False)
        replayed.to_csv(out/(name+'_predictions.csv'),index=False)
        best=max(candidates,key=lambda r:(r['episode_sensitivity'],r['warning_precision']))
        result=dict(name=name,threshold=threshold,metrics=metrics,diagnostic_highest_sensitivity_candidate=best)
        results.append(result);save_json(out/(name+'_results.json'),result)
        print(name,'best',best['true_warning_events'],'/38','unsuccessful/hour',best['false_alarms_per_hour'],'selected threshold',threshold,flush=True)
    save_json(out/'results.json',results)
    print('NON-NEURAL ROUND COMPLETE',flush=True)
if __name__=='__main__':main()
