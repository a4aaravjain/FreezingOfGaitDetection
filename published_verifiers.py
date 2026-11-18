"""Participant-balanced candidate-only fitting and separate score calibration."""
from pathlib import Path
import gc,json
import numpy as np,pandas as pd,joblib
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from scipy.special import logit,expit
from fog_pipeline import save_json,sha256
from published_neural import verifier_tcn,predict


def fit_verifier(family,x,seq,candidate,meta,folder):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    if (folder/'FIT.json').exists():
        info=json.loads((folder/'FIT.json').read_text())
        if all(sha256(folder/n)==h for n,h in info['artifact_hashes'].items()):return
        raise ValueError('Completed verifier artifact changed')
    y=meta.target.to_numpy().copy();y[meta.active_fog]=0
    mask=candidate&meta.known_followup.to_numpy()&((meta.target.to_numpy()>=0)|meta.active_fog.to_numpy())
    if set(y[mask])!={0,1}:raise ValueError('Candidate fit lacks classes; retain failure, never fit excluded people')
    owner=meta.subject.to_numpy();counts={s:int(sum(mask&(owner==s))) for s in np.unique(owner[mask])}
    weight=np.array([1/counts[s] for s in owner[mask]]);weight/=weight.mean()
    pd.DataFrame(dict(subject=owner,time=meta.time,candidate=candidate,known=meta.known_followup,label=y,used=mask,hard_category=meta.hard_category)).to_csv(folder/'training_candidate_ledger.csv',index=False)
    if family=='logistic':
        model=make_pipeline(StandardScaler(),LogisticRegression(C=.05,max_iter=1500,random_state=42));model.fit(x[mask],y[mask],logisticregression__sample_weight=weight)
        joblib.dump(model,folder/'model.joblib')
        np.testing.assert_allclose(model.predict_proba(x[:16]),joblib.load(folder/'model.joblib').predict_proba(x[:16]))
    elif family=='hgb':
        model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42).fit(x[mask],y[mask],sample_weight=weight)
        joblib.dump(model,folder/'model.joblib')
        np.testing.assert_array_equal(model.predict_proba(x[:16]),joblib.load(folder/'model.joblib').predict_proba(x[:16]))
    else:
        import tensorflow as tf
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42)
        mean=x[mask].mean(0);std=np.maximum(x[mask].std(0),1e-3)
        np.savez(folder/'normalization.npz',mean=mean,std=std)
        model,_=verifier_tcn(x.shape[1]);model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy')
        z=((seq[mask]-mean)/std).astype(np.float32)
        model.fit(z,y[mask],sample_weight=weight,epochs=12,batch_size=128,verbose=0,
            callbacks=[tf.keras.callbacks.CSVLogger(str(folder/'training.csv')),tf.keras.callbacks.TerminateOnNaN()])
        model.save(folder/'model.keras');loaded=tf.keras.models.load_model(folder/'model.keras',compile=False)
        np.testing.assert_allclose(predict(model,z[:16]),predict(loaded,z[:16]),rtol=1e-5,atol=1e-6)
        del loaded
    save_json(folder/'FIT.json',dict(family=family,subjects=sorted(set(owner[mask])),known_candidates=int(mask.sum()),positive_candidates=int(y[mask].sum()),reload_parity=True,
        artifact_hashes={p.name:sha256(p) for p in folder.iterdir() if p.is_file() and p.name!='FIT.json'}))
    del model;gc.collect()


def verifier_predict(family,folder,x,seq):
    folder=Path(folder)
    if family!='tcn':return joblib.load(folder/'model.joblib').predict_proba(x)[:,1]
    import tensorflow as tf
    with np.load(folder/'normalization.npz',allow_pickle=False) as z:mean,std=z['mean'],z['std']
    model=tf.keras.models.load_model(folder/'model.keras',compile=False);p=predict(model,((seq-mean)/std).astype(np.float32));del model;gc.collect();return p


def fit_platt(p,meta,candidate,path):
    mask=candidate&meta.known_followup.to_numpy()&((meta.target.to_numpy()>=0)|meta.active_fog.to_numpy())
    y=meta.target.to_numpy().copy();y[meta.active_fog]=0
    if set(y[mask])!={0,1}:
        save_json(Path(path).with_suffix('.failure.json'),dict(error='Separate score calibration lacks classes',fallback='raw uncalibrated arm retained'));return None
    model=LogisticRegression(C=1,max_iter=1000,random_state=42).fit(logit(np.clip(p[mask],1e-6,1-1e-6))[:,None],y[mask])
    joblib.dump(model,path);return model


def calibrated(model,p):
    return p if model is None else model.predict_proba(logit(np.clip(p,1e-6,1-1e-6))[:,None])[:,1]
