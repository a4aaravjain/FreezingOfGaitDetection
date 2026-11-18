"""Gravity-relative gait signals preserve directional motion without axis dependence."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent

def gravity_signals(raw):
    raw=np.asarray(raw,np.float32)
    if raw.ndim!=3 or raw.shape[1:]!=(180,6):raise ValueError('Expected native 60 Hz three-second six-channel windows')
    acc=raw[:,:,:3];gyro=raw[:,:,3:];g=acc.mean(1);length=np.linalg.norm(g,axis=1,keepdims=True)
    axis=g/np.maximum(length,1e-6)
    vertical=np.einsum('bti,bi->bt',acc,axis);yaw=np.einsum('bti,bi->bt',gyro,axis)
    horizontal=np.sqrt(np.maximum((acc*acc).sum(2)-vertical**2,0))
    rotation=np.sqrt(np.maximum((gyro*gyro).sum(2)-yaw**2,0))
    # Signed logs reduce scale differences while retaining direction.
    return np.stack([vertical,horizontal,np.sign(yaw)*np.log1p(abs(yaw)),np.log1p(rotation)],2).astype(np.float32)

def gravity_features(raw):
    z=gravity_signals(raw).astype(float);parts=[]
    for n in [60,180]:
        a=z[:,-n:];c=a-a.mean(1,keepdims=True)
        parts.extend([a.mean(1),a.std(1),np.ptp(a,axis=1),np.mean(abs(np.diff(a,axis=1)),axis=1),a[:,-n//2:].mean(1)-a[:,:n//2].mean(1)])
        p=abs(np.fft.rfft(c*np.hanning(n)[None,:,None],axis=1))**2/n**2;f=np.fft.rfftfreq(n,1/60)
        for lo,hi in [(.5,3),(3,8),(8,15)]:parts.append(np.log1p(p[:,(f>=lo)&(f<hi)].sum(1)))
    return np.concatenate(parts,1).astype(np.float32)

def main():
 import tensorflow as tf
 from sklearn.ensemble import HistGradientBoostingClassifier
 from threadpoolctl import threadpool_limits
 import joblib
 out=ROOT/'runs/gravity_gait_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 manifest=dict(config=asdict(cfg),ankle='ankleR',train=list(range(1,11)),calibration=list(range(11,15)),assessment=list(range(15,19)),reserved=list(range(19,23)),epochs=24,variants=['gravity_mlp','gravity_hgb','gravity_shape_gru'],source_sha256={n:sha256(ROOT/n) for n in ['gravity_gait.py','fog_pipeline.py','fogstar_adapter.py']})
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 datasets=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [manifest['train'],manifest['calibration'],manifest['assessment']]]
 td,cd,ad=datasets;mask=td.y>=0;y=td.y[mask];weights=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weights/=weights.mean()
 features=[gravity_features(d.x) for d in datasets];mean=features[0].mean(0);std=np.maximum(features[0].std(0),1e-3);normalized=[(a-mean)/std for a in features]
 save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()));results=[]
 for variant in manifest['variants']:
  f=normalized
  if variant=='gravity_shape_gru':
   f=[]
   for d in datasets:
    z=gravity_signals(d.x);f.append((z-z.mean(1,keepdims=True))/np.maximum(z.std(1,keepdims=True),.05))
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
  if variant=='gravity_hgb':
   with threadpool_limits(limits=4):
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42).fit(f[0][mask],y,sample_weight=weights)
    p=model.predict_proba(f[1])[:,1];joblib.dump(model,out/(variant+'.joblib'));model=joblib.load(out/(variant+'.joblib'));np.testing.assert_allclose(p,model.predict_proba(f[1])[:,1])
   checkpoint=out/(variant+'.joblib');predict=lambda a:model.predict_proba(a)[:,1]
  else:
   layers=[L.Input((180,4)),L.AveragePooling1D(3),L.GRU(24),L.Dropout(.3),L.Dense(1,activation='sigmoid')] if variant=='gravity_shape_gru' else [L.Input((f[0].shape[1],)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')]
   model=tf.keras.Sequential(layers);model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy')
   model.fit(f[0][mask],y,sample_weight=weights,epochs=24,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.CSVLogger(str(out/(variant+'_training.csv')))])
   checkpoint=out/(variant+'.keras');model.save(checkpoint);p=np.asarray(model(f[1],training=False)).ravel();model=tf.keras.models.load_model(checkpoint);np.testing.assert_allclose(p,np.asarray(model(f[1],training=False)).ravel(),atol=1e-6);predict=lambda a:np.asarray(model(a,training=False)).ravel()
  threshold,grid=select_threshold(cd,p,cfg);cm,cr,_=evaluate(cd,p,threshold,cfg)
  save_json(out/(variant+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),checkpoint_sha256=sha256(checkpoint)))
  ap=predict(f[2]);am,ar,_=evaluate(ad,ap,threshold,cfg)
  cr.to_csv(out/(variant+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(variant+'_assessment_predictions.csv'),index=False);pd.DataFrame(grid).to_csv(out/(variant+'_calibration_candidates.csv'),index=False)
  results.append(dict(model=variant,threshold=threshold,calibration=cm,assessment=am,model_reload_verified=True));save_json(out/'results.json',results)
  print('RESULT',variant,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
