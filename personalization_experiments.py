"""Separate-recording personalization experiment. Not an unseen-person benchmark."""
from pathlib import Path
from dataclasses import replace,asdict
import shutil
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,read_recording,build_windows,Windows,save_json,sha256,evaluate,select_threshold
from non_neural_experiments import temporal_features
ROOT=Path(__file__).resolve().parent

def main():
    cfg=replace(Config(),history_seconds=6,horizon_seconds=5)
    out=ROOT/'runs/personalization_round_1';out.mkdir(exist_ok=False)
    subjects=['S01','S02','S03','S05','S06','S07']
    manifest=dict(mode='development separate-recording personalization',config=asdict(cfg),subjects=subjects,
        calibration='R01 only',validation='R02 and R03 when available; not used in fitting',
        chronology='Recording identifiers define the split; actual acquisition ordering is not independently established.',
        methods=['leave_person_out_R01_training','include_own_R01_with_5x_weight'],
        policy='one shared threshold per method on concatenated validation recordings',
        source_sha256={n:sha256(ROOT/n) for n in ['personalization_experiments.py','non_neural_experiments.py','strict_baselines.py','fog_pipeline.py']})
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    calibration={};validation={};cx={};vx={};files=[]
    for subject in subjects:
        cp=ROOT/(subject+'R01.txt');vp=sorted(p for p in ROOT.glob(subject+'R*.txt') if p!=cp)
        if not vp:raise ValueError('Missing separate validation recording')
        files += [cp]+vp
        calibration[subject]=build_windows(read_recording(cp,cfg),cfg)
        validation[subject]=build_windows([s for p in vp for s in read_recording(p,cfg)],cfg)
        cx[subject]=temporal_features(calibration[subject].x);vx[subject]=temporal_features(validation[subject].x)
    manifest['dataset_sha256']={p.name:sha256(p) for p in files};save_json(out/'manifest.json',manifest)
    metas=pd.concat([validation[s].meta for s in subjects],ignore_index=True)
    combined=Windows(np.empty((len(metas),0,3)),np.concatenate([validation[s].y for s in subjects]),metas,[seg for s in subjects for seg in validation[s].segments])
    all_predictions={False:[],True:[]}
    with threadpool_limits(limits=4):
      for subject in subjects:
        for personal in [False,True]:
          train=[s for s in subjects if personal or s!=subject]
          x=np.concatenate([cx[s] for s in train]);y=np.concatenate([calibration[s].y for s in train]);owners=np.concatenate([np.full(len(calibration[s].y),s) for s in train])
          mask=y>=0;x=x[mask];y=y[mask];owners=owners[mask]
          if set(y)!={0,1}:raise ValueError('Calibration split lacks both classes')
          weights=np.where(y==1,10.,1.)*np.where((owners==subject)&personal,5.,1.);weights/=weights.mean()
          model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,l2_regularization=1,learning_rate=.05,early_stopping=False,random_state=42)
          model.fit(x,y,sample_weight=weights);p=model.predict_proba(vx[subject])[:,1]
          name=subject+('_personalized' if personal else '_unseen_person');joblib.dump(model,out/(name+'.joblib'))
          np.testing.assert_allclose(joblib.load(out/(name+'.joblib')).predict_proba(vx[subject][:5])[:,1],p[:5])
          all_predictions[personal].append(p)
        print('Completed calibration comparison',subject,flush=True)
    results=[]
    for personal,parts in all_predictions.items():
        name='personalized' if personal else 'unseen_person';p=np.concatenate(parts)
        threshold,candidates=select_threshold(combined,p,cfg);m,replayed,events=evaluate(combined,p,threshold,cfg)
        pd.DataFrame(candidates).to_csv(out/(name+'_candidates.csv'),index=False);replayed.to_csv(out/(name+'_predictions.csv'),index=False)
        best=max(candidates,key=lambda r:(r['episode_sensitivity'],r['warning_precision']))
        results.append(dict(method=name,threshold=threshold,metrics=m,highest_coverage_diagnostic=best))
        print(name,'best',best['true_warning_events'],'/',best['total_fog_episodes'],'unsuccessful/hour',best['false_alarms_per_hour'],'selected',threshold,flush=True)
    save_json(out/'results.json',results);print('PERSONALIZATION ROUND COMPLETE',flush=True)
if __name__=='__main__':main()
