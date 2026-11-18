"""Earlier-recording labelled adaptation; held-out later recordings only."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,Windows,select_threshold,evaluate,save_json,sha256,verify_split
from fogstar_adapter import load_fogstar
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

def chronological_split(segments):
    groups={}
    for s in segments:groups.setdefault(':'.join(s.key.split(':')[:3]),[]).append(s)
    ordered=sorted(groups.values(),key=lambda group:min(s.time[0] for s in group))
    if len(ordered)<2:raise ValueError('Need separate recordings')
    setup=ordered[0];later=[s for group in ordered[1:] for s in group]
    if max(s.time[-1] for s in setup)>=min(s.time[0] for s in later):raise ValueError('Recording chronology overlaps')
    return setup,later

def combine(datasets):
    return Windows(np.empty((sum(len(d.y) for d in datasets),0,0),np.float32),np.concatenate([d.y for d in datasets]),pd.concat([d.meta for d in datasets],ignore_index=True),[s for d in datasets for s in d.segments])

def main():
 import tensorflow as tf
 out=ROOT/'runs/earlier_recording_adaptation_round_1';out.mkdir(exist_ok=False)
 cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
 baseline=ROOT/'runs/episode_calibration_round_1'
 verify_split(range(1,11),range(11,15),range(15,19))
 manifest=dict(config=asdict(cfg),generic_training=list(range(1,11)),policy_calibration=list(range(11,15)),assessment=list(range(15,19)),reserved=list(range(19,23)),methods=['frozen_base','adapt_head','adapt_all'],epochs=8,setup='first complete chronological recording, chosen without inspecting labels; setup events are NEVER scored as predicted',requires='expert labels for completed setup recording',baseline_sha256=sha256(baseline/'pointwise.keras'),source_sha256={n:sha256(ROOT/n) for n in ['earlier_recording_adaptation.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']})
 save_json(out/'manifest.json',manifest);(out/'source').mkdir()
 for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 norm=json.loads((baseline/'normalization.json').read_text());mean=np.array(norm['mean'],np.float32);std=np.array(norm['std'],np.float32)
 td=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(1,11),ankle='ankleR',include_gyro=True),cfg)
 tx=((summaries(td.x)-mean)/std)[td.y>=0];ty=td.y[td.y>=0]
 def weight(y):
    w=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10.),1.);return w/w.sum()
 generic_weight=weight(ty)
 all_segments=load_fogstar(ROOT/'sensor_data.csv',range(11,19),ankle='ankleR',include_gyro=True)
 setup={};later={};features={};audit=[]
 for i in range(11,19):
    subject=f'FS{i:02d}';early,late=chronological_split([s for s in all_segments if s.subject==subject])
    setup[i]=build_windows(early,cfg);later[i]=build_windows(late,cfg)
    features[i]=((summaries(setup[i].x)-mean)/std,(summaries(later[i].x)-mean)/std)
    audit.append(dict(subject=i,setup_keys=[s.key for s in early],assessment_keys=[s.key for s in late],setup_onsets=sum(len(s.onsets) for s in early),later_onsets=sum(len(s.onsets) for s in late),setup_end=max(s.time[-1] for s in early),later_start=min(s.time[0] for s in late)))
 save_json(out/'recording_split.json',audit)
 cd=combine([later[i] for i in range(11,15)]);ad=combine([later[i] for i in range(15,19)])
 all_results=[]
 for method in manifest['methods']:
    predictions={}
    # Complete calibration people and freeze threshold before assessment people.
    for group in [list(range(11,15)),list(range(15,19))]:
      for i in group:
        tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
        model=tf.keras.models.load_model(baseline/'pointwise.keras',compile=False)
        mask=setup[i].y>=0;sy=setup[i].y[mask];sx=features[i][0][mask]
        adapted=method!='frozen_base' and bool((sy==1).any()) and bool((sy==0).any())
        if adapted:
          if method=='adapt_head':model.layers[0].trainable=False
          x=np.concatenate([tx,sx]);y=np.concatenate([ty,sy]);w=np.concatenate([generic_weight,weight(sy)]);w/=w.mean()
          model.compile(optimizer=tf.keras.optimizers.Adam(.0001,clipnorm=1),loss='binary_crossentropy')
          model.fit(x,y,sample_weight=w,epochs=8,batch_size=128,verbose=0,callbacks=[tf.keras.callbacks.TerminateOnNaN()])
        path=out/f'{method}_FS{i:02d}.keras';model.save(path)
        p=np.asarray(model(features[i][1],training=False)).ravel();loaded=tf.keras.models.load_model(path,compile=False);np.testing.assert_allclose(p,np.asarray(loaded(features[i][1],training=False)).ravel(),atol=1e-6)
        predictions[i]=p;print(method,i,'adapted',adapted,flush=True)
      if group[0]==11:
        cp=np.concatenate([predictions[i] for i in group]);threshold,grid=select_threshold(cd,cp,cfg)
        save_json(out/(method+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),calibration_subjects=group))
        pd.DataFrame(grid).to_csv(out/(method+'_calibration_candidates.csv'),index=False)
    cp=np.concatenate([predictions[i] for i in range(11,15)]);ap=np.concatenate([predictions[i] for i in range(15,19)])
    cm,cr,_=evaluate(cd,cp,threshold,cfg);am,ar,events=evaluate(ad,ap,threshold,cfg)
    cr.to_csv(out/(method+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(method+'_assessment_predictions.csv'),index=False)
    all_results.append(dict(model=method,threshold=threshold,calibration=cm,assessment=am,assessment_events=events,model_reload_verified=True));save_json(out/'results.json',all_results)
    print('RESULT',method,'assessment',am['device']['true_warning_events'],'/',am['device']['total_fog_episodes'],'false',am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
