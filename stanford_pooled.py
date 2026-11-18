"""Predeclared pooled raw-data experiment. Stanford 6/7 remain reserved."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,evaluate,select_threshold,save_json,sha256
from fogstar_adapter import load_fogstar
from stanford_adapter import load_stanford
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

def main():
 import tensorflow as tf
 from sklearn.linear_model import LogisticRegression
 import joblib
 out=ROOT/'runs/stanford_pooled_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5);scfg=replace(cfg,sample_rate=128)
 manifest=dict(fogstar_train=list(range(1,15)),fogstar_calibration=list(range(15,19)),stanford_train=[1,2,3,4],stanford_calibration=[5],stanford_reserved=[6,7],models=['pooled_summary_mlp','pooled_logistic'],config=asdict(cfg),source_sha256={n:sha256(ROOT/n) for n in ['stanford_pooled.py','stanford_adapter.py','long_history_experiments.py','fog_pipeline.py','fogstar_adapter.py','pretraining_experiments.py']})
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 datasets=[];features=[]
 for cohort,ids,config in [('FS',range(1,15),cfg),('FS',range(15,19),cfg),('ST',range(1,5),scfg),('ST',[5],scfg)]:
  segments=load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True) if cohort=='FS' else load_stanford(ids)
  d=build_windows(segments,config);f=summaries(d.x,config.sample_rate)
  # Release large raw windows after feature extraction; continuous replay uses segments/meta.
  d.x=np.empty((len(d.y),0,0),np.float32);datasets.append(d);features.append(f)
  print(cohort,list(ids),'windows',len(d.y),'onsets',sum(len(s.onsets) for s in segments),flush=True)
 ft,fv,st,sv=datasets;fx,fvx,sx,svx=features
 # Frozen previous model/threshold, no new fitting: transfer diagnostic on Stanford calibration person.
 norm=json.loads((ROOT/'runs/long_history_round_1/normalization.json').read_text())
 frozen=tf.keras.models.load_model(ROOT/'runs/long_history_round_1/current_summary_mlp.keras')
 p=np.asarray(frozen(((svx-np.array(norm['mean']))/np.array(norm['std'])).astype(np.float32),training=False)).ravel()
 m,r,_=evaluate(sv,p,.7,scfg);save_json(out/'frozen_transfer.json',m);r.to_csv(out/'frozen_transfer_predictions.csv',index=False)
 trainx=np.concatenate([fx,sx]);y=np.concatenate([ft.y,st.y]);subjects=np.concatenate([ft.meta.subject,st.meta.subject]);mask=y>=0
 mean=trainx.mean(0);std=np.maximum(trainx.std(0),1e-3);x=((trainx-mean)/std)[mask];labels=y[mask]
 validations=[(fv,(fvx-mean)/std,cfg),(sv,(svx-mean)/std,scfg)]
 save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()))
 # Each cohort contributes half; within each cohort each person's total weight is equal.
 weights=np.zeros(len(y))
 for cohort in ['FS','ST']:
  people=np.unique(subjects[np.char.startswith(subjects.astype(str),cohort)])
  for person in people:
   idx=(subjects==person)&mask;pos=(y[idx]==1).sum();neg=(y[idx]==0).sum()
   w=np.where(y[idx]==1,min(neg/max(pos,1),10),1.)
   weights[idx]=w/w.sum()/len(people)
 weights=weights[mask];weights/=weights.mean()
 results=[]
 for name in manifest['models']:
  if name=='pooled_summary_mlp':
   tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
   model=tf.keras.Sequential([L.Input((24,)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
   model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy')
   # Fixed 24 epochs avoids checkpoint selection on new calibration person.
   model.fit(x,labels,sample_weight=weights,epochs=24,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv'))),tf.keras.callbacks.TerminateOnNaN()])
   before=np.asarray(model(x[:10],training=False));model.save(out/(name+'.keras'));model=tf.keras.models.load_model(out/(name+'.keras'))
   np.testing.assert_allclose(before,np.asarray(model(x[:10],training=False)),atol=1e-6)
   predict=lambda z:np.asarray(model(z.astype(np.float32),training=False)).ravel()
  else:
   model=LogisticRegression(C=.1,max_iter=1000,random_state=42).fit(x,labels,sample_weight=weights)
   before=model.predict_proba(x[:10]);joblib.dump(model,out/(name+'.joblib'));model=joblib.load(out/(name+'.joblib'))
   np.testing.assert_allclose(before,model.predict_proba(x[:10]));predict=lambda z:model.predict_proba(z)[:,1]
  probabilities=[predict(v) for _,v,_ in validations];grids=[]
  for (d,v,c),p,cohort in zip(validations,probabilities,['FS','ST']):
   _,grid=select_threshold(d,p,c);grids.append(grid);pd.DataFrame(grid).to_csv(out/(name+'_'+cohort+'_candidates.csv'),index=False)
  feasible=[(a,b) for a,b in zip(*grids) if a['feasible'] and b['feasible']]
  chosen=max(feasible,key=lambda ab:min(ab[0]['episode_sensitivity'],ab[1]['episode_sensitivity'])) if feasible else None
  threshold=chosen[0]['threshold'] if chosen else 1.1
  record=dict(model=name,threshold=threshold,model_reload_verified=True,cohorts={})
  for (d,v,c),p,cohort in zip(validations,probabilities,['FS','ST']):
   m,r,_=evaluate(d,p,threshold,c);r.to_csv(out/(name+'_'+cohort+'_predictions.csv'),index=False);record['cohorts'][cohort]=m
  results.append(record);save_json(out/'results.json',results)
  print(name,'threshold',threshold,[(k,v['device']['true_warning_events'],v['device']['total_false_alarms']) for k,v in record['cohorts'].items()],flush=True)
if __name__=='__main__':main()
