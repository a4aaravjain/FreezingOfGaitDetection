"""Past-window twin encoder inspired by gait self-similarity; not PhysioGPN reproduction."""
from pathlib import Path
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,select_threshold,evaluate,event_metrics,save_json,sha256
from fogstar_adapter import load_fogstar
from gravity_gait import gravity_signals
from waveform_ensemble import PulseUnion
from candidate_gate import compare_candidates
ROOT=Path(__file__).resolve().parent

def main():
 import tensorflow as tf
 out=ROOT/'runs/twin_gait_prediction_round_1';out.mkdir(exist_ok=False);parent=ROOT/'runs/right_ankle_waveform_v2';old=json.loads((parent/'results.json').read_text());cfg=Config(**json.loads((parent/'manifest.json').read_text())['config'])
 save_json(out/'manifest.json',dict(reference='https://doi.org/10.1109/JBHI.2024.3522664',method='Single-ankle GRU twin encoder comparing earlier/later observed motion; not original paper architecture or performance replication',train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),epochs_max=16,early_stop_patience=4,seed=42,limitation='Reused development people select checkpoint and policy; not independent validation',source_sha256={n:sha256(ROOT/n) for n in ['twin_gait_prediction.py','gravity_gait.py','fog_pipeline.py','fogstar_adapter.py','waveform_ensemble.py','candidate_gate.py']}));(out/'source').mkdir()
 for n in json.loads((out/'manifest.json').read_text())['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 td,vd=[build_windows(load_fogstar(ROOT/'sensor_data.csv',ids,ankle='ankleR',include_gyro=True),cfg) for ids in [range(1,15),range(15,19)]]
 tx=gravity_signals(td.x);vx=gravity_signals(vd.x);mean=tx.mean((0,1));std=np.maximum(tx.std((0,1)),1e-3);tx=(tx-mean)/std;vx=(vx-mean)/std;save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist()))
 mask=td.y>=0;vm=vd.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
 tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
 left=L.Input((90,4));right=L.Input((90,4));encoder=tf.keras.Sequential([L.Input((90,4)),L.AveragePooling1D(3),L.GRU(16)])
 a=encoder(left);b=encoder(right);difference=L.Subtract()([b,a]);product=L.Multiply()([a,b]);z=L.Concatenate()([a,b,difference,product]);z=L.Dense(32,activation='relu')(z);z=L.Dropout(.3)(z);prediction=L.Dense(1,activation='sigmoid')(z);model=tf.keras.Model([left,right],prediction)
 model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')]);model.fit([tx[mask,:90],tx[mask,90:]],y,sample_weight=weight,validation_data=([vx[vm,:90],vx[vm,90:]],vd.y[vm]),epochs=16,batch_size=128,verbose=2,callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=4,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/'training.csv')),tf.keras.callbacks.TerminateOnNaN()])
 model.save(out/'model.keras');p=np.asarray(model([vx[:,:90],vx[:,90:]],training=False)).ravel();restored=tf.keras.models.load_model(out/'model.keras',compile=False);np.testing.assert_allclose(p,np.asarray(restored([vx[:,:90],vx[:,90:]],training=False)).ravel(),atol=1e-6)
 t,grid=select_threshold(vd,p,cfg);m,f,events=evaluate(vd,p,t,cfg);pd.DataFrame(grid).to_csv(out/'candidates.csv',index=False);f.to_csv(out/'predictions.csv',index=False);base=pd.read_csv(parent/'predictions.csv');np.testing.assert_array_equal(base.segment,vd.meta.segment);np.testing.assert_allclose(base.time,vd.meta.time);merged=base.copy();pulse=PulseUnion()
 for _,g in merged.groupby('segment',sort=False):
  pulse.reset()
  for i in g.index:merged.loc[i,'alarm_active'],merged.loc[i,'alarm_start']=pulse.step(merged.loc[i,'time'],[bool(base.loc[i,'alarm_start']),bool(f.loc[i,'alarm_start'])])
 cm,ce=event_metrics(vd,merged,cfg);gate=compare_candidates(old['combined'],old['events'],cm,ce);save_json(out/'results.json',dict(threshold=t,individual=m,combined=cm,events=ce,gate=gate,model_reload_verified=True));merged.to_csv(out/'combined_predictions.csv',index=False);print('RESULT',cm,gate,flush=True)
if __name__=='__main__':main()
