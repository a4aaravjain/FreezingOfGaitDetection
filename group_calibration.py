"""Group-held-out score calibration, then full development fit; no assessment tuning."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,Windows,select_threshold,evaluate,save_json,sha256,verify_split
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

def subset(data,features,subjects):
 mask=data.meta.subject.isin([f'FS{i:02d}' for i in subjects]).to_numpy()
 return Windows(np.empty((sum(mask),0,0),np.float32),data.y[mask],data.meta.loc[mask].reset_index(drop=True),[s for s in data.segments if s.subject in {f'FS{i:02d}' for i in subjects}]),features[mask]

def main():
 import argparse
 parser=argparse.ArgumentParser()
 parser.add_argument('--horizon',type=float,choices=[5.,8.],default=5.)
 parser.add_argument('--output',default='group_calibration_round_1')
 args=parser.parse_args()
 import tensorflow as tf
 out=ROOT/'runs'/args.output;out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=args.horizon,ema_alpha=.25,alert_pulse_seconds=1)
 folds=[[1,2,3,4],[5,6,7,8],[9,10,11],[12,13,14]]
 manifest=dict(config=asdict(cfg),development=list(range(1,15)),assessment=list(range(15,19)),folds=folds,epochs=24,seed=42,reserved=list(range(19,23)),threshold_selection='out-of-fold pooled development predictions only; assessment excluded',limitation='scores of final model may differ in calibration from smaller fold models',source_sha256={n:sha256(ROOT/n) for n in ['group_calibration.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']})
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 development=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(1,15),ankle='ankleR',include_gyro=True),cfg);features=summaries(development.x);development.x=np.empty((len(development.y),0,0),np.float32)
 oof=np.full(len(development.y),np.nan);assignments=[]
 def fit(trainx,y,name):
  mean=trainx.mean(0);std=np.maximum(trainx.std(0),1e-3);mask=y>=0;labels=y[mask];x=((trainx-mean)/std)[mask]
  weights=np.where(labels==1,min((labels==0).sum()/max(1,(labels==1).sum()),10),1.);weights/=weights.mean()
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
  model=tf.keras.Sequential([L.Input((24,)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')]);model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy')
  model.fit(x,labels,sample_weight=weights,epochs=24,batch_size=128,verbose=0,callbacks=[tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv'))),tf.keras.callbacks.TerminateOnNaN()])
  model.save(out/(name+'.keras'));before=np.asarray(model(x[:10],training=False));model=tf.keras.models.load_model(out/(name+'.keras'),compile=False);np.testing.assert_allclose(before,np.asarray(model(x[:10],training=False)),atol=1e-6)
  save_json(out/(name+'_normalization.json'),dict(mean=mean.tolist(),std=std.tolist()))
  return model,mean,std
 for k,held in enumerate(folds):
  train=sorted(set(range(1,15))-set(held));verify_split(train,held,range(15,19))
  td,tx=subset(development,features,train);_,vx=subset(development,features,held)
  model,mean,std=fit(tx,td.y,f'fold_{k+1}')
  held_mask=development.meta.subject.isin([f'FS{i:02d}' for i in held]).to_numpy()
  assert np.isnan(oof[held_mask]).all();oof[held_mask]=np.asarray(model((vx-mean)/std,training=False)).ravel()
  assignments.append(dict(fold=k+1,training=train,held_out=held));print('Fold',k+1,'complete',flush=True)
 assert np.isfinite(oof).all()
 threshold,grid=select_threshold(development,oof,cfg);cm,cr,_=evaluate(development,oof,threshold,cfg)
 pd.DataFrame(grid).to_csv(out/'calibration_candidates.csv',index=False);cr.to_csv(out/'calibration_predictions.csv',index=False)
 save_json(out/'frozen_policy.json',dict(threshold=threshold,config=asdict(cfg),fold_assignments=assignments))
 model,mean,std=fit(features,development.y,'full_development')
 assessment=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
 p=np.asarray(model((summaries(assessment.x)-mean)/std,training=False)).ravel();am,ar,events=evaluate(assessment,p,threshold,cfg)
 strict=build_windows(assessment.segments,replace(cfg,horizon_seconds=5))
 strict_metrics,_,_=evaluate(strict,p,threshold,replace(cfg,horizon_seconds=5))
 save_json(out/'assessment_original_2_to_5.json',strict_metrics)
 ar.to_csv(out/'assessment_predictions.csv',index=False);save_json(out/'results.json',dict(threshold=threshold,calibration=cm,assessment=am,events=events,model_reload_verified=True))
 print('RESULT',threshold,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],'false',am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
