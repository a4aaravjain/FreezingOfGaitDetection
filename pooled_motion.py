"""Pooled native-rate ankle motion experiment; held-out people stay separate.
Only rotation-invariant features are combined; physical sample timing is retained.
"""
from pathlib import Path
from dataclasses import replace,asdict
import os,sys,time,shutil,json
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np
import pandas as pd
from fog_pipeline import Config,load_subjects,build_windows,evaluate,select_threshold,save_json,sha256,Windows
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent

def invariant_motion(x,rate):
    """Six-second histories -> 120 causal 50-ms summaries.
    Magnitude, gravity-relative magnitude, jerk magnitude. No guessed axis mapping.
    The full supplied history ends at decision time; there are no later samples.
    """
    x=np.asarray(x,np.float32)
    centered=x-x.mean(axis=1,keepdims=True)
    magnitude=np.linalg.norm(x,axis=2)
    dynamic=np.linalg.norm(centered,axis=2)
    jerk=np.linalg.norm(np.diff(x,axis=1),axis=2)*rate
    jerk=np.concatenate([np.zeros((len(x),1),np.float32),jerk],axis=1)
    z=np.stack([magnitude,dynamic,np.log1p(jerk)],axis=2)
    # Bin edges derived from physical history length, no cross-window resampling.
    boundaries=np.rint(np.linspace(0,x.shape[1],121)).astype(int)
    return np.stack([z[:,a:b].mean(axis=1) for a,b in zip(boundaries[:-1],boundaries[1:])],axis=1)

def load(dataset,subjects,cfg):
    segments=load_subjects(ROOT,subjects,cfg)[0] if dataset=='daphnet' else load_fogstar(ROOT/'sensor_data.csv',subjects)
    return build_windows(segments,cfg)

def concatenate_metadata(data):
    meta=pd.concat([d.meta for d in data],ignore_index=True)
    return Windows(np.empty((len(meta),0,3),np.float32),np.concatenate([d.y for d in data]),meta,[s for d in data for s in d.segments])

def main():
    import tensorflow as tf
    from threadpoolctl import threadpool_limits
    tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
    out=ROOT/'runs/pooled_motion_round_1';out.mkdir(exist_ok=False)
    configs=[replace(Config(),history_seconds=6,horizon_seconds=5),replace(Config(),sample_rate=60,history_seconds=6,horizon_seconds=5)]
    specs=[('daphnet',['S01','S02','S03','S04','S05','S06'],['S07','S08']),('fogstar',list(range(1,15)),list(range(15,19)))]
    manifest=dict(mode='development',minimum_lead=2,maximum_lead=5,history=6,input='single ankle, invariant magnitudes, 120 bins',
        splits=specs,epochs_max=24,seed=42,normalization='fit on training windows only',
        weighting='equal cohort/subject total weight, positive factor 10 then normalize per subject',
        source_sha256={n:sha256(ROOT/n) for n in ['pooled_motion.py','fog_pipeline.py','fogstar_adapter.py']})
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    training=[];validation=[];tx=[];vx=[]
    for (dataset,train,val),cfg in zip(specs,configs):
        a=load(dataset,train,cfg);b=load(dataset,val,cfg)
        training.append(a);validation.append(b)
        tx.append(invariant_motion(a.x,cfg.sample_rate));vx.append(invariant_motion(b.x,cfg.sample_rate))
        print(dataset,'training',len(a.y),'validation',len(b.y),flush=True)
    train_meta=concatenate_metadata(training);val_meta=concatenate_metadata(validation)
    x=np.concatenate(tx);v=np.concatenate(vx);mask=train_meta.y>=0;y=train_meta.y[mask].astype(np.float32)
    x=x[mask];mean=x.mean((0,1));std=np.maximum(x.std((0,1)),1e-3);x=(x-mean)/std;v=(v-mean)/std
    subjects=train_meta.meta.subject.to_numpy()[mask];weights=np.zeros(len(y),np.float32)
    for subject in np.unique(subjects):
        m=subjects==subject;w=np.where(y[m]==1,10.,1.)
        cohort_count=14 if subject.startswith('FS') else 6
        weights[m]=w/w.sum()/cohort_count
    weights/=weights.mean()
    manifest['normalization_parameters']=dict(mean=mean.tolist(),std=std.tolist())
    manifest['dataset_sha256']={'sensor_data.csv':sha256(ROOT/'sensor_data.csv'),**{p.name:sha256(p) for p in ROOT.glob('S0[1-8]R*.txt')}}
    save_json(out/'manifest.json',manifest)
    L=tf.keras.layers
    model=tf.keras.Sequential([L.Input((120,3)),L.Conv1D(24,5,padding='causal',activation='relu'),
        L.GRU(32),L.Dropout(.25),L.Dense(1,activation='sigmoid')],name='invariant_ankle_gru')
    model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
    vm=val_meta.y>=0
    model.fit(x,y,sample_weight=weights,validation_data=(v[vm],val_meta.y[vm]),epochs=24,batch_size=256,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True),
          tf.keras.callbacks.CSVLogger(str(out/'training.csv')),tf.keras.callbacks.TerminateOnNaN()],verbose=2)
    model.save(out/'model.keras');restored=tf.keras.models.load_model(out/'model.keras')
    predict=lambda a:np.concatenate([np.asarray(restored(a[i:i+256],training=False)).ravel() for i in range(0,len(a),256)])
    p=predict(v);np.testing.assert_allclose(p[:16],np.asarray(model(v[:16],training=False)).ravel(),atol=1e-6)
    # Both cohorts share 0.25s prediction cadence. Score native sample durations separately.
    # A fixed threshold is chosen with constraints required in EACH validation cohort.
    rows=[];per_cohort=[]
    offset=0
    for d,cfg in zip(validation,configs):
        pred=p[offset:offset+len(d.y)];offset+=len(d.y)
        _,candidates=select_threshold(d,pred,cfg);per_cohort.append(candidates)
    for i,t in enumerate(np.linspace(.05,.95,37)):
        a,b=per_cohort[0][i],per_cohort[1][i]
        rows.append(dict(threshold=float(t),feasible=bool(a['feasible'] and b['feasible']),
            detections=a['true_warning_events']+b['true_warning_events'],
            daphnet_sensitivity=a['episode_sensitivity'],fogstar_valid_segment_sensitivity=b['episode_sensitivity'],
            daphnet_false_per_hour=a['false_alarms_per_hour'],fogstar_false_per_hour=b['false_alarms_per_hour']))
    feasible=[r for r in rows if r['feasible']];threshold=max(feasible,key=lambda r:r['detections'])['threshold'] if feasible else 1.1
    pd.DataFrame(rows).to_csv(out/'joint_candidates.csv',index=False)
    offset=0;summary=[]
    for (dataset,_,val),d,cfg,candidates in zip(specs,validation,configs,per_cohort):
        pred=p[offset:offset+len(d.y)];offset+=len(d.y)
        m,replayed,events=evaluate(d,pred,threshold,cfg)
        if dataset=='fogstar':
            inventory=pd.read_csv(ROOT/'runs/external_provenance/fogstar_recordings.csv');total=int(inventory.loc[inventory.subject.isin(val),'observed_onsets'].sum())
            m['device']['all_original_onsets']=total;m['device']['all_original_episode_sensitivity']=m['device']['true_warning_events']/total
        save_json(out/(dataset+'_metrics.json'),m);replayed.to_csv(out/(dataset+'_predictions.csv'),index=False)
        pd.DataFrame(candidates).to_csv(out/(dataset+'_candidates.csv'),index=False)
        summary.append(dict(dataset=dataset,metrics=m))
    manifest.update(threshold=threshold,policy_status='feasible' if feasible else 'disabled',model_sha256=sha256(out/'model.keras'))
    save_json(out/'manifest.json',manifest);save_json(out/'results.json',summary)
    save_json(out/'verification.json',dict(model_reload=True,finite_probabilities=bool(np.isfinite(p).all())))
    print('POOLED ROUND COMPLETE',threshold,flush=True)
if __name__=='__main__':main()
