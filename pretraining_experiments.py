"""Paired scratch vs auxiliary motion pretraining; raw FoG-STAR validation only.
The turning source is processed and is NEVER used to establish causal prediction accuracy.
"""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil,re
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent

def motion_features(x,rate):
    x=np.asarray(x,np.float32)
    if x.ndim!=3 or x.shape[2]!=6 or x.shape[1]!=round(3*rate):raise ValueError('Expected 3 seconds, six channels')
    acc=x[:,:,:3];gyro=x[:,:,3:]
    magnitude=np.linalg.norm(acc,axis=2)
    dynamic=np.linalg.norm(acc-acc.mean(1,keepdims=True),axis=2)
    rotation=np.log1p(np.linalg.norm(gyro,axis=2))
    jerk=np.log1p(np.linalg.norm(np.diff(acc,axis=1),axis=2)*rate)
    jerk=np.concatenate([np.zeros((len(x),1)),jerk],axis=1)
    features=np.stack([magnitude,dynamic,rotation,jerk],axis=2)
    if rate==60:return features.astype(np.float32)
    edges=np.rint(np.linspace(0,x.shape[1],181)).astype(int)
    return np.stack([features[:,a:b].mean(1) for a,b in zip(edges[:-1],edges[1:])],axis=1).astype(np.float32)

def auxiliary_windows():
    xs=[];ys=[];inventory=[]
    for p in sorted((ROOT/'external_downloads/turning_full/IMU').glob('*.txt')):
        if not re.fullmatch(r'SUB\d+_\d+',p.stem):continue
        a=pd.read_csv(p,sep='\t').to_numpy(float)
        if a.shape[1]!=9 or not np.isfinite(a).all() or not np.isin(a[:,-1],[0,1]).all() or np.any(abs(np.diff(a[:,1])-1/128)>1e-7):raise ValueError(str(p))
        ends=np.arange(383,len(a),32)
        raw=np.lib.stride_tricks.sliding_window_view(a[:,2:8].astype(np.float32),384,axis=0).transpose(0,2,1)[::32]
        xs.append(motion_features(raw,128));ys.append(a[ends,-1].astype(np.float32))
        inventory.append(dict(file=p.name,subject=p.stem.split('_')[0],rows=len(a),windows=len(ends),sha256=sha256(p)))
    return np.concatenate(xs),np.concatenate(ys),inventory

def main():
    import tensorflow as tf
    out=ROOT/'runs/pretraining_round_1';out.mkdir(exist_ok=False)
    cfg=replace(Config(),sample_rate=60,horizon_seconds=5)
    manifest=dict(config=asdict(cfg),target_dataset='FoG-STAR right ankle',train=list(range(1,15)),validation=list(range(15,19)),reserved=list(range(19,23)),
      trials=['scratch','processed_auxiliary_pretraining'],auxiliary_epochs=5,finetune_epochs_max=24,
      auxiliary_warning='Zero-phase-filtered turning data only initializes representation through current-FoG detection. No auxiliary prediction/evaluation claim. Final validation uses raw FoG-STAR.',
      source_sha256={n:sha256(ROOT/n) for n in ['pretraining_experiments.py','fogstar_adapter.py','fog_pipeline.py']},target_data_sha256=sha256(ROOT/'sensor_data.csv'))
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    td=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['train'],ankle='ankleR',include_gyro=True),cfg)
    vd=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['validation'],ankle='ankleR',include_gyro=True),cfg)
    tx=motion_features(td.x,60);v=motion_features(vd.x,60);mask=td.y>=0;x=tx[mask];y=td.y[mask].astype(np.float32)
    mean=x.mean((0,1));std=np.maximum(x.std((0,1)),1e-3);x=(x-mean)/std;v=(v-mean)/std
    save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()))
    ax,ay,inventory=auxiliary_windows();ax=(ax-mean)/std
    pd.DataFrame(inventory).to_csv(out/'auxiliary_inventory.csv',index=False)
    print('Auxiliary trials',len(inventory),'subjects',len(set(i['subject'] for i in inventory)),'windows',len(ay),flush=True)
    def weights(y):
        w=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);return (w/w.mean()).astype(np.float32)
    def encoder():
        L=tf.keras.layers
        return tf.keras.Sequential([L.Input((180,4)),L.AveragePooling1D(2),L.LSTM(32)],name='motion_encoder')
    def attach(enc):
        model=tf.keras.Sequential([enc,tf.keras.layers.Dropout(.3),tf.keras.layers.Dense(1,activation='sigmoid')])
        model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')]);return model
    results=[]
    for name in manifest['trials']:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
        enc=encoder()
        if name!='scratch':
            pre=attach(enc);pre.fit(ax,ay,sample_weight=weights(ay),epochs=5,batch_size=128,verbose=2,
                callbacks=[tf.keras.callbacks.CSVLogger(str(out/'auxiliary_training.csv')),tf.keras.callbacks.TerminateOnNaN()])
            enc.save(out/'pretrained_encoder.keras')
        tf.keras.utils.set_random_seed(42);model=attach(enc)
        model.fit(x,y,sample_weight=weights(y),validation_data=(v[vd.y>=0],vd.y[vd.y>=0]),epochs=24,batch_size=128,verbose=2,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv'))),tf.keras.callbacks.TerminateOnNaN()])
        model.save(out/(name+'.keras'));restored=tf.keras.models.load_model(out/(name+'.keras'))
        p=np.concatenate([np.asarray(restored(v[i:i+128],training=False)).ravel() for i in range(0,len(v),128)])
        np.testing.assert_allclose(p[:10],np.asarray(model(v[:10],training=False)).ravel(),atol=1e-6)
        threshold,candidates=select_threshold(vd,p,cfg);m,replayed,events=evaluate(vd,p,threshold,cfg)
        pd.DataFrame(candidates).to_csv(out/(name+'_candidates.csv'),index=False);replayed.to_csv(out/(name+'_predictions.csv'),index=False)
        best=max(candidates,key=lambda a:(a['episode_sensitivity'],a['warning_precision']))
        results.append(dict(model=name,threshold=threshold,metrics=m,highest_coverage_diagnostic=best,model_reload_verified=True));save_json(out/'results.json',results)
        print(name,'selected',m['device']['true_warning_events'],'best',best['true_warning_events'],'/16; unsuccessful/hour',best['false_alarms_per_hour'],flush=True)
    print('PRETRAINING ROUND COMPLETE',flush=True)
if __name__=='__main__':main()
