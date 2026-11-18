"""Bounded causal feature baselines. Development only; no S09/S10 or external tuning."""
from pathlib import Path
from dataclasses import asdict, replace
import json, shutil, time
import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config, load_subjects, build_windows, select_threshold, evaluate, save_json, sha256
ROOT=Path(__file__).resolve().parent

def features(x, rate=64):
    # Every statistic uses only the trailing supplied window, with no labels.
    x=x.astype(np.float64)
    magnitude=np.linalg.norm(x,axis=2,keepdims=True)
    z=np.concatenate([x,magnitude],axis=2)
    result=[]
    for seconds in [1,3]:
        a=z[:,-round(seconds*rate):,:]
        centered=a-a.mean(axis=1,keepdims=True)
        result.extend([a.mean(1),np.log1p(a.std(1)),np.log1p(np.ptp(a,axis=1)),
                       np.log1p(np.sqrt(np.mean(np.diff(a,axis=1)**2,axis=1))*rate)])
        power=abs(np.fft.rfft(centered*np.hanning(a.shape[1])[None,:,None],axis=1))**2
        freq=np.fft.rfftfreq(a.shape[1],1/rate)
        bands=[]
        for lo,hi in [(.5,3),(3,8),(8,16)]:
            energy=power[:,(freq>=lo)&(freq<hi)].sum(1)/a.shape[1]**2
            bands.append(energy);result.append(np.log1p(energy))
        result.append(np.log((bands[1]+1e-6)/(bands[0]+1e-6)))
    output=np.concatenate(result,axis=1).astype(np.float32)
    if not np.isfinite(output).all(): raise ValueError('Nonfinite feature')
    return output

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',choices=['daphnet','fogstar'],default='daphnet')
    args=parser.parse_args();external=args.dataset=='fogstar'
    cfg=replace(Config(),sample_rate=60) if external else Config();cfg.validate()
    out=ROOT/'runs'/('strict_2s_fogstar' if external else 'strict_2s_baselines')
    out.mkdir(exist_ok=False)
    train=['S01','S02','S03','S04','S05','S06'];validation=['S07','S08']
    if external:
        train=list(range(1,15));validation=list(range(15,19))
    manifest=dict(dataset=args.dataset,ankle='left' if external else 'Daphnet ankle',
        reserved_subjects=list(range(19,23)) if external else ['S09','S10'],
        protocol='strict_2_to_3_seconds_development_v4',config=asdict(cfg),
        train_subjects=train,validation_subjects=validation,
        warning='Historically exposed subjects: development results, not an untouched final test.',
        success='episode sensitivity >= 0.90 AND unsuccessful warnings <= 1/hour; measured device latency not included',
        models=['standardized_logistic_C1','hist_gradient_boosting_100_iterations_15_leaves'],
        source_sha256={n:sha256(ROOT/n) for n in ['fog_pipeline.py','strict_baselines.py','fogstar_adapter.py']})
    save_json(out/'manifest.json',manifest)
    (out/'source').mkdir()
    for name in manifest['source_sha256']:shutil.copy2(ROOT/name,out/'source'/name)
    if external:
        from fogstar_adapter import load_fogstar
        ts=load_fogstar(ROOT/'sensor_data.csv',train);vs=load_fogstar(ROOT/'sensor_data.csv',validation)
        tp=vp=[ROOT/'sensor_data.csv']
    else:
        ts,tp=load_subjects(ROOT,train,cfg);vs,vp=load_subjects(ROOT,validation,cfg)
    manifest['dataset_sha256']={p.name:sha256(p) for p in tp+vp};save_json(out/'manifest.json',manifest)
    td=build_windows(ts,cfg);vd=build_windows(vs,cfg)
    print('Extracting trailing features',len(td.y),len(vd.y),flush=True)
    tx=features(td.x,cfg.sample_rate);vx=features(vd.x,cfg.sample_rate);mask=td.y>=0;y=td.y[mask]
    if set(y)!={0,1} or set(vd.y[vd.y>=0])!={0,1}:raise ValueError('Split lacks both classes; do not silently change subjects')
    weight=min((y==0).sum()/max(1,(y==1).sum()),cfg.positive_weight_cap)
    weights=np.where(y==1,weight,1.);weights/=weights.mean()
    total_annotations=sum(len(s.onsets) for s in vs)
    if external:
        inventory=pd.read_csv(ROOT/'runs'/'external_provenance'/'fogstar_recordings.csv')
        total_annotations=int(inventory.loc[inventory.subject.isin(validation),'observed_onsets'].sum())
    manifest['validation_original_annotated_onsets']=total_annotations
    manifest['validation_onsets_retained_after_gap_splitting']=sum(len(s.onsets) for s in vs)
    save_json(out/'manifest.json',manifest)
    models=[('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=42))),
            ('gradient_boosting',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=15,
                learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42))]
    summaries=[]
    with threadpool_limits(limits=4):
        for name,model in models:
            start=time.perf_counter();folder=out/name;folder.mkdir();print('Fitting',name,flush=True)
            kwargs={'logisticregression__sample_weight':weights} if name=='logistic' else {'sample_weight':weights}
            model.fit(tx[mask],y,**kwargs)
            p=model.predict_proba(vx)[:,1]
            joblib.dump(model,folder/'model.joblib')
            np.testing.assert_allclose(joblib.load(folder/'model.joblib').predict_proba(vx[:20])[:,1],p[:20])
            threshold,candidates=select_threshold(vd,p,cfg)
            pd.DataFrame(candidates).to_csv(folder/'validation_candidates.csv',index=False)
            metrics,replayed,events=evaluate(vd,p,threshold,cfg)
            metrics['device']['original_annotated_onsets']=total_annotations
            metrics['device']['sensitivity_all_original_annotations']=metrics['device']['episodes_predicted_before_onset']/total_annotations if total_annotations else None
            save_json(folder/'metrics.json',metrics);save_json(folder/'events.json',events)
            replayed.to_csv(folder/'predictions.csv',index=False)
            best=max(candidates,key=lambda a:(a['episode_sensitivity'],a['warning_precision']))
            result=dict(model=name,threshold=threshold,policy_status='disabled' if threshold>1 else 'feasible',
                elapsed_seconds=time.perf_counter()-start,metrics=metrics,
                highest_sensitivity_grid_candidate=best)
            save_json(folder/'result.json',result);summaries.append(result)
            print(name,'threshold',threshold,'AP',metrics['window']['pr_auc_average_precision'],
                  'best grid sensitivity',best['episode_sensitivity'],'false/hour',best['false_alarms_per_hour'],flush=True)
    save_json(out/'results.json',summaries)
    print('COMPLETE',out,flush=True)
if __name__=='__main__':main()
