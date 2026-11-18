"""Non-neural classifiers on past-only delayed covariance of ankle motion."""
from pathlib import Path
from dataclasses import replace,asdict
import json,shutil
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier,ExtraTreesClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256,verify_split
from fogstar_adapter import load_fogstar
from gravity_gait import gravity_signals
ROOT=Path(__file__).resolve().parent

def covariance_features(signals,scale):
    z=np.asarray(signals,np.float64)/np.asarray(scale,np.float64)[None,None,:]
    if z.ndim!=3 or z.shape[1:]!=(180,4) or not np.isfinite(z).all():raise ValueError('Expected finite 180x4 signals')
    delayed=np.concatenate([z[:,30:],z[:,15:-15],z[:,:-30]],axis=2)
    centered=delayed-delayed.mean(1,keepdims=True)
    covariance=np.einsum('bti,btj->bij',centered,centered)/149
    trace=np.trace(covariance,axis1=1,axis2=2)
    covariance=.95*covariance+(.05*trace/12+1e-4)[:,None,None]*np.eye(12)[None]
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    if np.any(eigenvalues<=0):raise ValueError('Covariance must be positive definite')
    logarithm=(eigenvectors*np.log(eigenvalues)[:,None,:])@np.swapaxes(eigenvectors,1,2)
    row,col=np.triu_indices(12);matrix=logarithm[:,row,col]*np.where(row==col,1,np.sqrt(2))
    result=np.concatenate([matrix,z.mean(1),z.std(1),np.log1p(trace)[:,None]],axis=1).astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite features')
    return result

def balanced_weights(data):
    # Give each person equal total mass, each class equal mass within person,
    # and each positive episode equal mass within that person's positives.
    weight=np.zeros(len(data.y),dtype=float)
    lookup={s.key:s for s in data.segments}
    for person,g in data.meta.groupby('subject'):
        ids=g.index.to_numpy();neg=ids[data.y[ids]==0];pos=ids[data.y[ids]==1]
        if len(neg):weight[neg]=1/len(neg)
        groups={}
        for i in pos:
            row=data.meta.loc[i];segment=lookup[row['segment']]
            episode=int(np.searchsorted(segment.onsets,row['time'],side='right'))
            groups.setdefault((segment.key,episode),[]).append(i)
        for event_ids in groups.values():weight[event_ids]=1/(len(groups)*len(event_ids))
        total=weight[ids].sum()
        if total:weight[ids]/=total
    mask=data.y>=0
    return weight[mask]/weight[mask].mean()


def main():
    out=ROOT/'runs/balanced_non_neural_round_1';out.mkdir(exist_ok=False)
    cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
    train=list(range(1,11));cal=list(range(11,15));assessment=list(range(15,19));verify_split(train,cal,assessment)
    manifest=dict(config=asdict(cfg),train=train,calibration=cal,assessment=assessment,reserved=list(range(19,23)),ankle='ankleR',weighting='equal person, equal class within person, equal positive episode',features='87 covariance plus 24 motion summaries',lags_seconds=[0,.25,.5],shrinkage=.05,models=['logistic','rbf_svm','hist_gradient_boosting','extra_trees'],source_sha256={n:sha256(ROOT/n) for n in ['balanced_non_neural.py','covariance_non_neural.py','long_history_experiments.py','pretraining_experiments.py','gravity_gait.py','fogstar_adapter.py','fog_pipeline.py']},dataset_sha256=sha256(ROOT/'sensor_data.csv'))
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    datasets=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [train,cal,assessment]]
    signals=[gravity_signals(d.x) for d in datasets];scale=np.maximum(signals[0].std(axis=(0,1)),.01)
    save_json(out/'signal_scale.json',dict(scale=scale.tolist(),fitted_on='training people 1-10 only'))
    from long_history_experiments import summaries
    features=[np.concatenate([covariance_features(z,scale),summaries(d.x)],axis=1) for z,d in zip(signals,datasets)];td,cd,ad=datasets
    mask=td.y>=0;y=td.y[mask];weight=balanced_weights(td)
    pd.DataFrame(dict(subject=td.meta.loc[mask,"subject"],label=y,weight=weight)).to_csv(out/"training_weights.csv",index=False)
    classifiers=[('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,random_state=42))),('rbf_svm',make_pipeline(StandardScaler(),SVC(C=1,gamma='scale',cache_size=128))),('hist_gradient_boosting',HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42)),('extra_trees',ExtraTreesClassifier(n_estimators=300,min_samples_leaf=15,max_features=.7,n_jobs=2,random_state=42))]
    results=[]
    with threadpool_limits(limits=2):
      for name,model in classifiers:
        kwargs={list(model.named_steps)[-1]+'__sample_weight':weight} if hasattr(model,'named_steps') else dict(sample_weight=weight)
        model.fit(features[0][mask],y,**kwargs)
        def scores(m,x):return expit(m.decision_function(x)) if name=='rbf_svm' else m.predict_proba(x)[:,1]
        p=scores(model,features[1]);joblib.dump(model,out/(name+'.joblib'));model=joblib.load(out/(name+'.joblib'));np.testing.assert_allclose(p,scores(model,features[1]))
        threshold,grid=select_threshold(cd,p,cfg);cm,cr,_=evaluate(cd,p,threshold,cfg)
        save_json(out/(name+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),checkpoint_sha256=sha256(out/(name+'.joblib')),score_note='SVM sigmoid margin is not calibrated probability'))
        ap=scores(model,features[2]);am,ar,events=evaluate(ad,ap,threshold,cfg)
        cr.to_csv(out/(name+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(name+'_assessment_predictions.csv'),index=False);pd.DataFrame(grid).to_csv(out/(name+'_calibration_candidates.csv'),index=False)
        results.append(dict(model=name,threshold=threshold,calibration=cm,assessment=am,events=events,model_reload_verified=True));save_json(out/'results.json',results)
        print(name,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],'false',am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
