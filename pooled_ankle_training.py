"""Pool contralateral TRAINING ankle examples; deployment uses one right ankle."""
from pathlib import Path
from dataclasses import replace,asdict
import argparse,os,shutil,json
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256,verify_split
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--development',action='store_true');parser.add_argument('--output',default='pooled_ankle_separate_round_1');args=parser.parse_args()
 import tensorflow as tf
 out=ROOT/'runs'/args.output;out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 train=list(range(1,15)) if args.development else list(range(1,11));cal=list(range(15,19)) if args.development else list(range(11,15));assessment=list(range(15,19));verify_split(train,cal,[] if args.development else assessment)
 names=['right_only','pooled_ankles']+(['pooled_finetune'] if args.development else [])
 manifest=dict(config=asdict(cfg),train=train,calibration=cal,assessment=assessment,development_reuse=args.development,models=names,epochs=24,seed=42,deployment_ankle='ankleR',additional_training_ankle='ankleL',reserved=list(range(19,23)),limitation='Previously examined assessment people. Development assessment duplicates calibration.',source_sha256={n:sha256(ROOT/n) for n in ['pooled_ankle_training.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']},dataset_sha256=sha256(ROOT/'sensor_data.csv'))
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 right=build_windows(load_fogstar(ROOT/'sensor_data.csv',train,ankle='ankleR',include_gyro=True),cfg);left=build_windows(load_fogstar(ROOT/'sensor_data.csv',train,ankle='ankleL',include_gyro=True),cfg)
 cd=build_windows(load_fogstar(ROOT/'sensor_data.csv',cal,ankle='ankleR',include_gyro=True),cfg);ad=cd if args.development else build_windows(load_fogstar(ROOT/'sensor_data.csv',assessment,ankle='ankleR',include_gyro=True),cfg)
 rx=summaries(right.x);lx=summaries(left.x);cx=summaries(cd.x);ax=cx if args.development else summaries(ad.x);results=[]
 for name in names:
  f=rx if name=='right_only' else np.concatenate([rx,lx]);labels=right.y if name=='right_only' else np.concatenate([right.y,left.y]);mask=labels>=0;y=labels[mask]
  mean=f.mean(0);std=np.maximum(f.std(0),1e-3)
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
  if name=='pooled_finetune':
   norm=json.loads((ROOT/'runs/right_ankle_pulse_candidate/normalization.json').read_text());mean=np.array(norm['mean']);std=np.array(norm['std']);model=tf.keras.models.load_model(ROOT/'runs/right_ankle_pulse_candidate/current_summary_mlp.keras',compile=False)
  else:model=tf.keras.Sequential([L.Input((24,)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
  tx=((f-mean)/std)[mask];v=(cx-mean)/std;av=(ax-mean)/std;weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
  model.compile(optimizer=tf.keras.optimizers.Adam(1e-5 if name=='pooled_finetune' else .0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
  model.fit(tx,y,sample_weight=weight,validation_data=(v[cd.y>=0],cd.y[cd.y>=0]),epochs=8 if name=='pooled_finetune' else 24,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv'))),tf.keras.callbacks.TerminateOnNaN()])
  model.save(out/(name+'.keras'));p=np.asarray(model(v,training=False)).ravel();restored=tf.keras.models.load_model(out/(name+'.keras'),compile=False);np.testing.assert_allclose(p,np.asarray(restored(v,training=False)).ravel(),atol=1e-6)
  save_json(out/(name+'_normalization.json'),dict(mean=mean.tolist(),std=std.tolist()))
  threshold,grid=select_threshold(cd,p,cfg);cm,cr,_=evaluate(cd,p,threshold,cfg);save_json(out/(name+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),checkpoint_sha256=sha256(out/(name+'.keras'))))
  ap=p if args.development else np.asarray(restored(av,training=False)).ravel();am,ar,events=evaluate(ad,ap,threshold,cfg)
  pd.DataFrame(grid).to_csv(out/(name+'_calibration_candidates.csv'),index=False);cr.to_csv(out/(name+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(name+'_assessment_predictions.csv'),index=False)
  results.append(dict(model=name,threshold=threshold,calibration=cm,assessment=am,events=events,model_reload_verified=True,training_windows=int(len(y))));save_json(out/'results.json',results)
  print('RESULT',name,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],'false',am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
