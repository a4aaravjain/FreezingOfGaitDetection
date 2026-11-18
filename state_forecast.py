"""Four-state LSTM: normal, useful precursor, too-late precursor, active FoG.
Labels are supervision only. The warning score comes only from sensor inputs.
"""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent

def state_labels(data,cfg):
    y=np.full(len(data.y),-1,np.int8)
    y[data.y>=0]=0;y[data.y==1]=1
    for s in data.segments:
        ids=np.flatnonzero(data.meta.segment.to_numpy()==s.key);t=data.meta.time.to_numpy()[ids]
        j=np.searchsorted(s.onsets,t,side='right');valid=j<len(s.onsets)
        lead=np.full(len(t),np.inf);lead[valid]=s.onsets[j[valid]]-t[valid]
        active=data.meta.active_fog.to_numpy()[ids]
        late=(lead<cfg.minimum_lead_seconds-1e-9)&(~active)
        y[ids[late]]=2;y[ids[active]]=3
    return y

def main():
 import tensorflow as tf
 tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5)
 out=ROOT/'runs/state_forecast_round_1';out.mkdir(exist_ok=False)
 manifest=dict(config=asdict(cfg),ankle='ankleR',channels='acceleration_g + angular_velocity_degrees_per_second',
  train=list(range(1,15)),validation=list(range(15,19)),reserved=list(range(19,23)),
  states=['normal','onset_2_to_5_seconds_ahead','onset_under_2_seconds_ahead','active_fog'],
  checkpoint='validation average precision of useful-precursor score on binary-scorable normal windows',
  source_sha256={n:sha256(ROOT/n) for n in ['state_forecast.py','fog_pipeline.py','fogstar_adapter.py']},dataset_sha256=sha256(ROOT/'sensor_data.csv'))
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 td=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['train'],ankle='ankleR',include_gyro=True),cfg)
 vd=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['validation'],ankle='ankleR',include_gyro=True),cfg)
 labels=state_labels(td,cfg);vl=state_labels(vd,cfg);mask=labels>=0
 mean=td.x[mask].mean((0,1));std=np.maximum(td.x[mask].std((0,1)),1e-3)
 x=(td.x[mask]-mean)/std;v=(vd.x-mean)/std;y=labels[mask]
 counts=np.bincount(y,minlength=4)
 if np.any(counts==0):raise ValueError('Missing supervised state')
 weights=np.minimum(len(y)/(4*counts),10)[y];weights/=weights.mean()
 save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),state_counts=counts.tolist()))
 L=tf.keras.layers
 model=tf.keras.Sequential([L.Input((180,6)),L.AveragePooling1D(2),L.LSTM(32),L.Dropout(.3),L.Dense(4,activation='softmax')],name='four_state_ankle_lstm')
 model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='sparse_categorical_crossentropy')
 class PrecursorCheckpoint(tf.keras.callbacks.Callback):
  def on_epoch_end(self,epoch,logs=None):
   score=np.concatenate([np.asarray(self.model(v[i:i+128],training=False))[:,1] for i in range(0,len(v),128)])
   m=vd.y>=0;logs['val_precursor_ap']=float(average_precision_score(vd.y[m],score[m]))
 model.fit(x,y,sample_weight=weights,validation_data=(v[vl>=0],vl[vl>=0]),epochs=24,batch_size=128,
  callbacks=[PrecursorCheckpoint(),tf.keras.callbacks.EarlyStopping(monitor='val_precursor_ap',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/'training.csv')),tf.keras.callbacks.TerminateOnNaN()],verbose=2)
 model.save(out/'model.keras');restored=tf.keras.models.load_model(out/'model.keras')
 pred=np.concatenate([np.asarray(restored(v[i:i+128],training=False)) for i in range(0,len(v),128)])
 np.testing.assert_allclose(pred[:10],np.asarray(model(v[:10],training=False)),atol=1e-6)
 threshold,candidates=select_threshold(vd,pred[:,1],cfg);m,replayed,events=evaluate(vd,pred[:,1],threshold,cfg)
 pd.DataFrame(candidates).to_csv(out/'candidates.csv',index=False);replayed.to_csv(out/'predictions.csv',index=False)
 np.save(out/'state_probabilities.npy',pred)
 best=max(candidates,key=lambda a:(a['episode_sensitivity'],a['warning_precision']))
 result=dict(threshold=threshold,metrics=m,highest_coverage_diagnostic=best,model_reload_verified=True)
 save_json(out/'results.json',result)
 print('FOUR-STATE COMPLETE',threshold,'selected',m['device']['true_warning_events'],'/16','best',best['true_warning_events'],'unsuccessful/hour',best['false_alarms_per_hour'],flush=True)
if __name__=='__main__':main()
