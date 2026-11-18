"""Matched ankle comparison and new causal multiscale features; development only."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
from pretraining_experiments import motion_features
ROOT=Path(__file__).resolve().parent

def multiscale(x):
 z=motion_features(x,60);parts=[summaries(x)]
 for n in (30,60,120):
  a=z[:,-n:];parts.extend([a.mean(1),a.std(1),a[:,-n//2:].mean(1)-a[:,:n//2].mean(1)])
 # Gait periodicity at delays covering common stepping intervals, all within past window.
 c=z-z.mean(1,keepdims=True)
 for lag in (15,30,45,60):
  a=c[:,lag:];b=c[:,:-lag]
  parts.append((a*b).mean(1)/(np.sqrt((a*a).mean(1)*(b*b).mean(1))+1e-6))
 return np.concatenate(parts,1).astype(np.float32)

def main():
 import tensorflow as tf
 from sklearn.ensemble import HistGradientBoostingClassifier
 from threadpoolctl import threadpool_limits
 import joblib
 out=ROOT/'runs/ankle_multiscale_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5)
 planned=[('matched_left',42),('matched_right',42),('matched_left',73),('matched_right',73),('right_multiscale',42),('right_multiscale',73),('right_summary_hgb',42),('right_multiscale_hgb',42)]
 save_json(out/'manifest.json',dict(config=asdict(cfg),train=list(range(1,15)),validation=list(range(15,19)),original_validation_onsets=16,planned=planned,source_sha256={n:sha256(ROOT/n) for n in ['ankle_multiscale.py','fogstar_adapter.py','fog_pipeline.py','long_history_experiments.py','pretraining_experiments.py']}))
 (out/'source').mkdir()
 for n in json.loads((out/'manifest.json').read_text())['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 paired=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='both',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]]
 right=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]]
 results=[]
 for variant,seed in planned:
  name=variant+'_'+str(seed);td,vd=paired if variant.startswith('matched') else right
  transform=multiscale if 'multiscale' in variant else summaries
  rawtrain=td.x[:,:,:6] if variant=='matched_left' else td.x[:,:,6:] if variant=='matched_right' else td.x
  rawval=vd.x[:,:,:6] if variant=='matched_left' else vd.x[:,:,6:] if variant=='matched_right' else vd.x
  a=transform(rawtrain);b=transform(rawval);mean=a.mean(0);std=np.maximum(a.std(0),1e-3);a=(a-mean)/std;b=(b-mean)/std
  mask=td.y>=0;vm=vd.y>=0;y=td.y[mask];w=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);w/=w.mean()
  save_json(out/(name+'_normalization.json'),dict(mean=mean.tolist(),std=std.tolist()))
  if variant.endswith('hgb'):
   with threadpool_limits(limits=4):
    model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=seed).fit(a[mask],y,sample_weight=w)
    p=model.predict_proba(b)[:,1];joblib.dump(model,out/(name+'.joblib'));reloaded=joblib.load(out/(name+'.joblib'));np.testing.assert_allclose(p,reloaded.predict_proba(b)[:,1])
  else:
   tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(seed);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
   model=tf.keras.Sequential([L.Input((a.shape[1],)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
   model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
   model.fit(a[mask],y,sample_weight=w,validation_data=(b[vm],vd.y[vm]),epochs=24,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv')))])
   p=np.asarray(model(b,training=False)).ravel();model.save(out/(name+'.keras'));reloaded=tf.keras.models.load_model(out/(name+'.keras'));np.testing.assert_allclose(p,np.asarray(reloaded(b,training=False)).ravel(),atol=1e-6)
  threshold,grid=select_threshold(vd,p,cfg);m,r,events=evaluate(vd,p,threshold,cfg)
  pd.DataFrame(grid).to_csv(out/(name+'_candidates.csv'),index=False);r.to_csv(out/(name+'_predictions.csv'),index=False)
  rec=dict(model=name,threshold=threshold,metrics=m,original_onsets=16,original_episode_sensitivity=m['device']['true_warning_events']/16,model_reload_verified=True)
  results.append(rec);save_json(out/'results.json',results)
  print(name,'SELECTED',m['device']['true_warning_events'],'/16',m['device']['total_false_alarms'],'false',flush=True)
if __name__=='__main__':main()
