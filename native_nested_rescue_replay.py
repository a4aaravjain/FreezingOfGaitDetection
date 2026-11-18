"""Native-sample replay of already-frozen outer-fold models, no tuning."""
from pathlib import Path
from collections import deque
from dataclasses import replace
import json,time
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from threadpoolctl import threadpool_limits
from fogstar_adapter import segments_from_frame
from waveform_ensemble import load_bank,PulseUnion
from fast_temporal_bank import FastTemporalBank
from gravity_gait import gravity_signals
from long_history_experiments import summaries
from fog_pipeline import Config,Alarm,save_json,sha256
from rescue_cascade import RescueFeatures,PhysicalOR
r=Path(__file__).resolve().parent;out=r/'runs/nested_rescue_20261111_02';assert (out/'execution_completed.json').exists();source=r/'development_only_20261111_01.csv';manifest=json.loads(source.with_suffix('.manifest.json').read_text());assert sha256(source)==manifest['output_sha256'];frame=pd.read_csv(source);assert set(frame.subjectID)==set(range(1,19));allrows=[];reports=[]
with threadpool_limits(limits=2):
 for folder in sorted(out.glob('fold_*')):
  if not (folder/'results.json').exists():continue
  policy=json.loads((folder/'FROZEN_CASCADE.json').read_text());outer=policy['outer'];up=folder/'upstream_final';bank=FastTemporalBank.from_bank(load_bank(up/'bank.json'));small=joblib.load(up/'small.joblib');extended=joblib.load(up/'extended.joblib');weights=np.load(up/'dense_numpy.npz',allow_pickle=False);family=policy['selected_family'];vf=folder/'full'/(family+'.joblib');verifier=joblib.load(vf) if vf.exists() else None;threshold=policy['policies'].get('full_'+family,1.1)
  expected=pd.read_csv(folder/'baseline_OR_selected_rescue_predictions.csv');scores=pd.read_csv(folder/'outer_scores.csv');base_expected=pd.read_csv(folder/'matched_v2_predictions.csv');count=0;latencies=[]
  cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
  for segment in segments_from_frame(frame[frame.subjectID.isin(outer)],ankle='ankleR',include_gyro=True):
   buffer=deque(maxlen=180);tick=0;headalarms=[Alarm(t,cfg) for t in [.675,.775,.875]];baseline=PulseUnion(1,3);features=RescueFeatures(policy['level1_threshold']);rescue=Alarm(threshold,replace(cfg,ema_alpha=1,persistence=1));physical=PhysicalOR();indices=expected.index[expected.chunk==segment.key].to_list();position=0
   for timestamp,sample in zip(segment.time,segment.xyz_g):
    buffer.append(sample.copy());tick+=1
    if tick<180 or (tick-180)%15:continue
    tic=time.perf_counter();raw=np.array(buffer,np.float32)[None];motion=summaries(raw);z=bank.transform(gravity_signals(raw));normalized=(motion-weights['mean'])/weights['std'];dp=float(expit(np.maximum(normalized@weights['w1']+weights['b1'],0)@weights['w2']+weights['b2']).ravel()[0]);p=[dp,float(expit(3*small.decision_function(np.concatenate([z[:,:768],motion],1).astype(float)))[0]),float(expit(3*extended.decision_function(np.concatenate([z,motion],1).astype(float)))[0])]
    starts=[a.step(timestamp,prob)[2] for a,prob in zip(headalarms,p)];ba,bs=baseline.step(timestamp,starts);candidate,x=features.step(p,motion[0]);prob=float(verifier.predict_proba(x[None])[0,1]) if verifier is not None and candidate else 0.;_,ra,rs=rescue.step(timestamp,prob);active,start=physical.step(ba,ra);elapsed=(time.perf_counter()-tic)*1000;latencies.append(elapsed)
    i=indices[position];position+=1;np.testing.assert_allclose(p,[scores.loc[i,h+'_probability'] for h in ['base','waveform','extended']],rtol=1e-5,atol=1e-6);np.testing.assert_allclose(timestamp,expected.loc[i,'time'],atol=1e-8,rtol=0);assert bs==base_expected.loc[i,'alarm_start'] and ba==base_expected.loc[i,'alarm_active'];assert active==expected.loc[i,'alarm_active'] and start==expected.loc[i,'alarm_start'];assert rs==expected.loc[i,'rescue_start']
    if verifier is not None and candidate:np.testing.assert_allclose(prob,expected.loc[i,'verifier_probability'],atol=1e-6,rtol=1e-5)
    allrows.append(dict(fold=folder.name,segment=expected.loc[i,'segment'],time=timestamp,compute_ms=elapsed,alarm_start=start,baseline_start=bs));count+=1
   assert position==len(indices)
  report=dict(fold=folder.name,native_prediction_windows=count,upstream_score_parity=True,baseline_decision_parity=True,combined_physical_decision_parity=True,rescue_decision_parity=True,median_ms=float(np.median(latencies)),p95_ms=float(np.percentile(latencies,95)),max_ms=float(max(latencies)),over_250ms=int(sum(v>250 for v in latencies)));reports.append(report);save_json(folder/'native_streaming_parity.json',report);print(report,flush=True)
 pd.DataFrame(allrows).to_csv(out/'native_decision_latency.csv',index=False);save_json(out/'native_streaming_parity.json',dict(folds=reports,reserved_used=False,transport_and_actuator_latency_measured=False))
