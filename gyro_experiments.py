"""Single-ankle gyroscope ablation: paired trees and a pure LSTM, development only."""
from pathlib import Path
from dataclasses import asdict,replace
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,build_windows,select_threshold,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
from strict_baselines import features
ROOT=Path(__file__).resolve().parent

def gyro_features(x):
    if x.ndim!=3 or x.shape[2] not in (6,7):raise ValueError('Expect separate acceleration and angular velocity axes')
    result=np.concatenate([features(x[:,:,:3],60),features(x[:,:,3:6],60)],axis=1)
    return np.concatenate([result,x[:,:,6].mean(1)[:,None]],axis=1) if x.shape[2]==7 else result

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--ankle',choices=['ankleL','ankleR'],default='ankleL');parser.add_argument('--max-hold-seconds',type=float,choices=[0,.1],default=0);args=parser.parse_args()
    hold=args.max_hold_seconds
    out=ROOT/'runs'/('gyro_right_round_1' if args.ankle=='ankleR' else ('gyro_hold_round_1' if hold else 'gyro_round_1'));out.mkdir(exist_ok=False)
    cfg=replace(Config(),sample_rate=60,history_seconds=3,horizon_seconds=5)
    manifest=dict(max_hold_seconds=hold,config=asdict(cfg),train=list(range(1,15)),validation=list(range(15,19)),reserved=list(range(19,23)),
      dataset='FoG-STAR',ankle=args.ankle,units=['g']*3+['degrees/second']*3+(['held_sample_flag'] if hold else []),epochs_max=24,seed=42,
      planned_models=['trees_acc_only','trees_acc_and_gyro','lstm_acc_and_gyro'],
      source_sha256={n:sha256(ROOT/n) for n in ['gyro_experiments.py','fogstar_adapter.py','fog_pipeline.py','strict_baselines.py','causal_hold.py']},
      dataset_sha256=sha256(ROOT/'sensor_data.csv'))
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    td=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['train'],ankle=args.ankle,include_gyro=True,max_hold_seconds=hold),cfg)
    vd=build_windows(load_fogstar(ROOT/'sensor_data.csv',manifest['validation'],ankle=args.ankle,include_gyro=True,max_hold_seconds=hold),cfg)
    mask=td.y>=0;y=td.y[mask];weights=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.).astype(np.float32);weights/=weights.mean()
    tx=gyro_features(td.x);vx=gyro_features(vd.x);predictions=[]
    print('Gyro paired training and validation windows:',len(td.y),len(vd.y),flush=True)
    with threadpool_limits(limits=4):
      for name,cols in [('trees_acc_only',slice(0,tx.shape[1]//2)),('trees_acc_and_gyro',slice(None))]:
        model=HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,learning_rate=.05,l2_regularization=1,early_stopping=False,random_state=42)
        model.fit(tx[mask,cols],y,sample_weight=weights);joblib.dump(model,out/(name+'.joblib'))
        p=model.predict_proba(vx[:,cols])[:,1];np.testing.assert_allclose(p[:10],joblib.load(out/(name+'.joblib')).predict_proba(vx[:10,cols])[:,1]);predictions.append((name,p))
        print('Fitted',name,flush=True)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers
    mean=td.x[mask].mean((0,1));std=np.maximum(td.x[mask].std((0,1)),1e-3)
    x=(td.x[mask]-mean)/std;v=(vd.x-mean)/std
    model=tf.keras.Sequential([L.Input((180,td.x.shape[2])),L.AveragePooling1D(2),L.LSTM(32),L.Dropout(.3),L.Dense(1,activation='sigmoid')],name='ankle_lstm_acc_gyro')
    model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy',metrics=[tf.keras.metrics.AUC(curve='PR',name='pr_auc')])
    model.fit(x,y,sample_weight=weights,validation_data=(v[vd.y>=0],vd.y[vd.y>=0]),epochs=24,batch_size=128,
      callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=5,restore_best_weights=True),tf.keras.callbacks.CSVLogger(str(out/'training.csv')),tf.keras.callbacks.TerminateOnNaN()],verbose=2)
    model.save(out/'lstm.keras');restored=tf.keras.models.load_model(out/'lstm.keras')
    p=np.concatenate([np.asarray(restored(v[i:i+128],training=False)).ravel() for i in range(0,len(v),128)])
    np.testing.assert_allclose(p[:10],np.asarray(model(v[:10],training=False)).ravel(),atol=1e-6);predictions.append(('lstm_acc_and_gyro',p))
    save_json(out/'normalization.json',dict(mean=mean.tolist(),std=std.tolist(),units=manifest['units']))
    results=[]
    for name,p in predictions:
      threshold,candidates=select_threshold(vd,p,cfg);metrics,replayed,events=evaluate(vd,p,threshold,cfg)
      metrics['device']['original_annotated_onsets']=16;metrics['device']['original_episode_sensitivity']=metrics['device']['true_warning_events']/16
      pd.DataFrame(candidates).to_csv(out/(name+'_candidates.csv'),index=False);replayed.to_csv(out/(name+'_predictions.csv'),index=False)
      best=max(candidates,key=lambda r:(r['episode_sensitivity'],r['warning_precision']))
      result=dict(model=name,threshold=threshold,metrics=metrics,highest_coverage_diagnostic=best,model_reload_verified=True);results.append(result)
      print(name,'best',best['true_warning_events'],'/16 original onsets','unsuccessful/hour',best['false_alarms_per_hour'],'selected',threshold,flush=True)
    save_json(out/'results.json',results);print('GYRO ROUND COMPLETE',flush=True)
if __name__=='__main__':main()
