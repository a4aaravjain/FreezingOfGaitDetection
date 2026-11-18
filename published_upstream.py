"""Checkpointed fold-only upstream fits; downloaded code is never imported."""
from pathlib import Path
import json,gc
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from fog_pipeline import sha256,save_json
from nested_rescue_experiment import train_heads
from published_neural import NEURAL,train_one,predict
from published_cascade_core import HEADS
from gravity_gait import gravity_features
from regularity_features import regularity_features


def baseline_predict(fit,part):
    w1,b1,w2,b2=fit['weights'];m=part['motion'];z=fit['bank'].transform(part['signals'])
    dp=expit(np.maximum(((m-fit['mean'])/fit['std'])@w1+b1,0)@w2+b2).ravel()
    small=np.concatenate([z[:,:768],m],1).astype(float);large=np.concatenate([z,m],1).astype(float)
    return np.column_stack([dp,expit(3*fit['models'][0].decision_function(small)),expit(3*fit['models'][1].decision_function(large))]),z[:,:64]


def load_baseline(folder):
    from waveform_ensemble import load_bank
    from fast_temporal_bank import FastTemporalBank
    w=np.load(folder/'dense_numpy.npz',allow_pickle=False)
    return dict(bank=FastTemporalBank.from_bank(load_bank(folder/'bank.json')),
        weights=tuple(w[k] for k in ['w1','b1','w2','b2']),mean=w['mean'],std=w['std'],
        models=[joblib.load(folder/(n+'.joblib')) for n in ['small','extended']])


def shape_features(part,wave):
    return np.concatenate([part['motion'],gravity_features(part['data'].x),regularity_features(part['data'].x),wave],1)


def fit_upstream(train,targets,folder,held,outer,calibration):
    folder=Path(folder);folder.mkdir(exist_ok=True)
    trainids=sorted(set(train['meta'].subject));assert not set(trainids)&(set(held)|set(outer)|set(calibration))
    done=folder/'DONE.json'
    if done.exists():
        info=json.loads(done.read_text());assert info['fit_subjects']==trainids
        assert all(sha256(folder/n)==h for n,h in info['artifacts'].items())
        result={}
        for name,part in targets.items():
            frame=pd.read_csv(folder/(name+'_scores.csv'));np.testing.assert_allclose(frame.time,part['meta'].time,rtol=0,atol=1e-8)
            np.testing.assert_array_equal(frame.chunk,part['meta'].chunk)
            with np.load(folder/(name+'_features.npz'),allow_pickle=False) as z:shape=z['shape'].copy()
            result[name]=(frame[HEADS].to_numpy(),shape)
        return result
    save_json(folder/'provenance.json',dict(fit_subjects=trainids,held_subjects=held,outer_subjects=outer,calibration_subjects=calibration,
        normalization_subjects=trainids,pretraining_subjects=trainids,hard_negative_selection_subjects=trainids,
        external_weights_used=False,seed=42))
    basefolder=folder/'matched_v2'
    if (basefolder/'fit.json').exists():fit=load_baseline(basefolder)
    else:fit=train_heads(train,basefolder)
    outputs={};shapes={}
    for name,part in targets.items():
        p,wave=baseline_predict(fit,part);outputs[name]=np.zeros((len(p),len(HEADS)));outputs[name][:,:3]=p
        shapes[name]=shape_features(part,wave)
    # Fit-channel statistics use each native training sample once, never an outer window.
    native=np.concatenate([s.xyz_g for s in train['data'].segments]);mean=native.mean(0);std=np.maximum(native.std(0),1e-3)
    normal=dict(mean=mean,std=std);save_json(folder/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),fit_subjects=trainids,each_native_sample_once=True))
    for j,key in enumerate(NEURAL,3):
        print('UPSTREAM',folder.name,key,'fit',trainids,flush=True)
        model=train_one(key,train,normal,folder)
        for name,part in targets.items():outputs[name][:,j]=predict(model,((part['data'].x-mean)/std).astype(np.float32))
        del model;gc.collect()
    for name,part in targets.items():
        frame=part['meta'].copy()
        for j,key in enumerate(HEADS):frame[key]=outputs[name][:,j]
        frame.to_csv(folder/(name+'_scores.csv'),index=False)
        np.savez_compressed(folder/(name+'_features.npz'),shape=shapes[name])
    artifacts={str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.is_file() and p.name!='DONE.json'}
    save_json(done,dict(fit_subjects=trainids,artifacts=artifacts,all_targets=list(targets)))
    del fit;gc.collect();return {name:(outputs[name],shapes[name]) for name in targets}


def predict_saved(folder,part):
    folder=Path(folder);fit=load_baseline(folder/'matched_v2');p,wave=baseline_predict(fit,part);scores=np.zeros((len(p),10));scores[:,:3]=p
    normal=json.loads((folder/'normalization.json').read_text());x=((part['data'].x-np.array(normal['mean'],np.float32))/np.array(normal['std'],np.float32)).astype(np.float32)
    import tensorflow as tf
    for j,key in enumerate(NEURAL,3):
        model=tf.keras.models.load_model(folder/(key+'.keras'),compile=False);scores[:,j]=predict(model,x);del model;gc.collect()
    return scores,shape_features(part,wave)
