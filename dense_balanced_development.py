"""Episode-focused learning with disjoint training/calibration/assessment people."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256,verify_split
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

def episode_bags(data,features,length=13):
    bags=[];valid=[];labels=[]
    def add(ids,label):
        if label and len(ids)<2:return
        a=np.zeros((length,features.shape[1]),np.float32);v=np.zeros(length,np.float32)
        a[:len(ids)]=features[ids];v[:len(ids)]=1
        bags.append(a);valid.append(v);labels.append(label)
    for s in data.segments:
        g=data.meta[data.meta.segment==s.key];ids=g.index.to_numpy();t=g.time.to_numpy()
        for onset in s.onsets:
            chosen=ids[(t>=onset-5-1e-9)&(t<=onset-2+1e-9)&(data.y[ids]==1)]
            # y==1 can refer to a nearer onset, so explicitly require next-onset identity.
            chosen=np.array([i for i in chosen if s.onsets[np.searchsorted(s.onsets,data.meta.loc[i,'time'],side='right')]==onset],int)
            if len(chosen)>length:raise ValueError('Unexpected bag size')
            add(chosen,1)
        negative=(data.y[ids]==0)|g.active_fog.to_numpy(bool)
        starts=np.flatnonzero(negative&np.r_[True,~negative[:-1]])
        for start in starts:
            end=start
            while end<len(ids) and negative[end]:end+=1
            for pos in range(start,end,length):add(ids[pos:min(pos+length,end)],0)
    return np.asarray(bags),np.asarray(valid),np.asarray(labels,np.float32)

def main():
 import tensorflow as tf
 out=ROOT/'runs/dense_balanced_development_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 train=list(range(1,15));cal=list(range(15,19));assessment=cal;verify_split(train,cal,[])
 manifest=dict(config=asdict(cfg),train=train,calibration=cal,assessment=assessment,reserved=list(range(19,23)),epochs=24,models=['pointwise','person_episode_balanced','balanced_feature_noise'],warning='EXPLORATORY DEVELOPMENT ONLY: calibration and assessment are the same people 15-18, deliberately matching original development split; assessment fields are duplicate development metrics, NOT independent validation',source_sha256={n:sha256(ROOT/n) for n in ['dense_balanced_development.py','balanced_non_neural.py','long_history_experiments.py','pretraining_experiments.py','fog_pipeline.py','fogstar_adapter.py']})
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 td,cd,ad=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [train,cal,assessment]]
 raw=summaries(td.x);mean=raw.mean(0);std=np.maximum(raw.std(0),1e-3)
 x=(raw-mean)/std;cx=(summaries(cd.x)-mean)/std;ax=(summaries(ad.x)-mean)/std
 save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()))
 bags,valid,by=episode_bags(td,x);save_json(out/'training_bags.json',dict(positive=int(by.sum()),negative=int((by==0).sum()),length=13))
 results=[]
 for name in manifest['models']:
  tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
  model=tf.keras.Sequential([L.Input((24,)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')]);opt=tf.keras.optimizers.Adam(.0005,clipnorm=1)
  mask=td.y>=0;y=td.y[mask]
  if name=='pointwise':
   weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
  else:
   from balanced_non_neural import balanced_weights
   weight=balanced_weights(td)
  if name=='balanced_feature_noise':
   model=tf.keras.Sequential([L.Input((24,)),L.GaussianNoise(.15),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')])
  model.compile(optimizer=opt,loss='binary_crossentropy')
  model.fit(x[mask],y,sample_weight=weight,epochs=24,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.CSVLogger(str(out/(name+'_training.csv'))),tf.keras.callbacks.TerminateOnNaN()])
  model.save(out/(name+'.keras'));p=np.asarray(model(cx,training=False)).ravel();restored=tf.keras.models.load_model(out/(name+'.keras'));np.testing.assert_allclose(p,np.asarray(restored(cx,training=False)).ravel(),atol=1e-6)
  threshold,grid=select_threshold(cd,p,cfg);cm,cr,_=evaluate(cd,p,threshold,cfg)
  # Threshold is frozen BEFORE producing assessment scores.
  save_json(out/(name+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),checkpoint_sha256=sha256(out/(name+'.keras'))))
  ap=np.asarray(restored(ax,training=False)).ravel();am,ar,_=evaluate(ad,ap,threshold,cfg)
  cr.to_csv(out/(name+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(name+'_assessment_predictions.csv'),index=False);pd.DataFrame(grid).to_csv(out/(name+'_calibration_candidates.csv'),index=False)
  results.append(dict(model=name,threshold=threshold,calibration=cm,assessment=am,model_reload_verified=True));save_json(out/'results.json',results)
  print('RESULT',name,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
