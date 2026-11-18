"""Past-only local gait contrast; separate threshold calibration and assessment."""
from pathlib import Path
from dataclasses import replace,asdict
import os,shutil,json
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

def local_contrast(features,meta,history=60):
    # Prior prediction summaries only. No future, subject ID or FoG labels.
    output=np.zeros((len(features),features.shape[1]*2+1),np.float32)
    for _,g in meta.groupby('segment',sort=False):
        ids=g.index.to_numpy()
        for j,i in enumerate(ids):
            previous=features[ids[max(0,j-history):j]]
            output[i,-1]=len(previous)/history
            if len(previous)>=2:
                mean=previous.mean(0);std=np.maximum(previous.std(0),.25)
                output[i,:features.shape[1]]=features[i]-mean
                output[i,features.shape[1]:-1]=np.clip((features[i]-mean)/std,-10,10)
    return output

def main():
 import tensorflow as tf
 out=ROOT/'runs/local_contrast_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 manifest=dict(config=asdict(cfg),train=list(range(1,11)),calibration=list(range(11,15)),assessment=list(range(15,19)),reserved=list(range(19,23)),epochs=24,variants=['local_only','raw_plus_local'],source_sha256={n:sha256(ROOT/n) for n in ['local_contrast.py','long_history_experiments.py','fog_pipeline.py','fogstar_adapter.py','pretraining_experiments.py']})
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 datasets=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [manifest['train'],manifest['calibration'],manifest['assessment']]]
 features=[summaries(d.x) for d in datasets];mean=features[0].mean(0);std=np.maximum(features[0].std(0),1e-3);z=[(a-mean)/std for a in features]
 local=[local_contrast(a,d.meta) for a,d in zip(z,datasets)];td,cd,ad=datasets
 save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()))
 mask=td.y>=0;y=td.y[mask];weights=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weights/=weights.mean();results=[]
 for variant in manifest['variants']:
  f=local if variant=='local_only' else [np.concatenate([a,b],1) for a,b in zip(z,local)]
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
  model=tf.keras.Sequential([L.Input((f[0].shape[1],)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
  model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy')
  model.fit(f[0][mask],y,sample_weight=weights,epochs=24,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.CSVLogger(str(out/(variant+'_training.csv')))])
  model.save(out/(variant+'.keras'));p=np.asarray(model(f[1],training=False)).ravel();restored=tf.keras.models.load_model(out/(variant+'.keras'));np.testing.assert_allclose(p,np.asarray(restored(f[1],training=False)).ravel(),atol=1e-6)
  threshold,grid=select_threshold(cd,p,cfg);cm,cr,_=evaluate(cd,p,threshold,cfg)
  save_json(out/(variant+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),checkpoint_sha256=sha256(out/(variant+'.keras'))))
  ap=np.asarray(restored(f[2],training=False)).ravel();am,ar,_=evaluate(ad,ap,threshold,cfg)
  cr.to_csv(out/(variant+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(variant+'_assessment_predictions.csv'),index=False);pd.DataFrame(grid).to_csv(out/(variant+'_calibration_candidates.csv'),index=False)
  results.append(dict(model=variant,threshold=threshold,calibration=cm,assessment=am,model_reload_verified=True));save_json(out/'results.json',results)
  print('RESULT',variant,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
