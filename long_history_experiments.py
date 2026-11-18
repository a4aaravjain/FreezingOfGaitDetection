"""Test 30 seconds of past gait summaries without increasing the 3-second startup history."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
from pretraining_experiments import motion_features
ROOT=Path(__file__).resolve().parent

def summaries(raw,rate=60):
    z=motion_features(raw,rate).astype(np.float64)
    centered=z-z.mean(1,keepdims=True)
    power=abs(np.fft.rfft(centered*np.hanning(180)[None,:,None],axis=1))**2/180**2
    f=np.fft.rfftfreq(180,1/60)
    lo=power[:,(f>=.5)&(f<3)].sum(1);hi=power[:,(f>=3)&(f<8)].sum(1)
    return np.concatenate([z.mean(1),z.std(1),z[:,-60:].mean(1)-z[:,:60].mean(1),np.log1p(lo),np.log1p(hi),np.log((hi+1e-6)/(lo+1e-6))],axis=1).astype(np.float32)

def bilateral_summaries(raw):
    if raw.shape[-1]!=12:raise ValueError('Expected both six-axis ankles')
    left=summaries(raw[:,:,:6]);right=summaries(raw[:,:,6:])
    # All features use the same trailing 3-second window, with no future samples.
    l=motion_features(raw[:,:,:6],60);r=motion_features(raw[:,:,6:],60)
    lc=l-l.mean(1,keepdims=True);rc=r-r.mean(1,keepdims=True)
    correlation=(lc*rc).mean(1)/(l.std(1)*r.std(1)+1e-6)
    return np.concatenate([left,right,abs(left-right),correlation],axis=1).astype(np.float32)


def histories(f,meta,steps=60,stride=2):
    augmented=np.concatenate([f,np.ones((len(f),1),np.float32)],axis=1)
    output=np.zeros((len(f),steps,f.shape[1]+1),np.float32)
    for _,g in meta.groupby('segment',sort=False):
        ids=g.index.to_numpy();indices=np.arange(len(g))[:,None]-np.arange((steps-1)*stride,-1,-stride)[None,:]
        valid=indices>=0;output[ids]=augmented[ids[np.maximum(indices,0)]]*valid[:,:,None]
    return output

def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument("--active-negatives",action="store_true")
    parser.add_argument("--ankle",choices=["ankleL","ankleR","both"],default="ankleR")
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--output",default="long_history_round_1")
    parser.add_argument("--summary-only",action="store_true")
    args=parser.parse_args()
    import tensorflow as tf
    out=ROOT/'runs'/args.output;out.mkdir(exist_ok=False)
    cfg=replace(Config(),sample_rate=60,horizon_seconds=5,seed=args.seed)
    manifest=dict(active_negatives=args.active_negatives,config=asdict(cfg),ankle=args.ankle,train=list(range(1,15)),validation=list(range(15,19)),reserved=list(range(19,23)),
        variants=['current_summary_mlp'] if args.summary_only else ['current_summary_mlp','thirty_second_lstm'],history='60 previous summary positions spaced 0.5s, including current; pad unavailable history with an explicit mask',epochs_max=24,
        source_sha256={n:sha256(ROOT/n) for n in ['long_history_experiments.py','pretraining_experiments.py','fogstar_adapter.py','fog_pipeline.py']},dataset_sha256=sha256(ROOT/'sensor_data.csv'))
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    td=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['train'],ankle=args.ankle,include_gyro=True),cfg)
    vd=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['validation'],ankle=args.ankle,include_gyro=True),cfg)
    feature_fn=bilateral_summaries if args.ankle=="both" else summaries
    tfraw=feature_fn(td.x);vfraw=feature_fn(vd.x);mask=td.y>=0;vm=vd.y>=0
    labels=td.y.copy()
    if args.active_negatives:
        active=td.meta.active_fog.to_numpy(bool);labels[active]=0;mask|=active
    y=labels[mask]
    mean=tfraw.mean(0);std=np.maximum(tfraw.std(0),1e-3);tx=(tfraw-mean)/std;vx=(vfraw-mean)/std
    save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()))
    weights=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weights/=weights.mean()
    results=[]
    for name in manifest['variants']:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(args.seed);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
        if name=='current_summary_mlp':
            x=tx[mask];v=vx
            model=tf.keras.Sequential([L.Input((tx.shape[1],)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
        else:
            x=histories(tx,td.meta)[mask];v=histories(vx,vd.meta)
            model=tf.keras.Sequential([L.Input((60,tx.shape[1]+1)),L.Masking(mask_value=0),L.LSTM(32),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
        model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
        model.fit(x,y,sample_weight=weights,validation_data=(v[vm],vd.y[vm]),epochs=24,batch_size=128,verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv'))),tf.keras.callbacks.TerminateOnNaN()])
        model.save(out/(name+'.keras'));restored=tf.keras.models.load_model(out/(name+'.keras'))
        p=np.concatenate([np.asarray(restored(v[i:i+128],training=False)).ravel() for i in range(0,len(v),128)])
        np.testing.assert_allclose(p[:10],np.asarray(model(v[:10],training=False)).ravel(),atol=1e-6)
        threshold,candidates=select_threshold(vd,p,cfg);m,replayed,events=evaluate(vd,p,threshold,cfg)
        pd.DataFrame(candidates).to_csv(out/(name+'_candidates.csv'),index=False);replayed.to_csv(out/(name+'_predictions.csv'),index=False)
        best=max(candidates,key=lambda a:(a['episode_sensitivity'],a['warning_precision']))
        results.append(dict(model=name,threshold=threshold,metrics=m,highest_coverage_diagnostic=best,model_reload_verified=True));save_json(out/'results.json',results)
        print(name,'selected',m['device']['true_warning_events'],'best',best['true_warning_events'],'/16; unsuccessful/hour',best['false_alarms_per_hour'],flush=True)
        del x,v,model,restored
    print('LONG HISTORY ROUND COMPLETE',flush=True)
if __name__=='__main__':main()
