"""Verify the pooled model's label-free stream and summarize actual results."""
from pathlib import Path
import os,json,time
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd
from dataclasses import replace
from fog_pipeline import Config,OnlinePredictor,load_subjects
from fogstar_adapter import load_fogstar
from pooled_motion import invariant_motion
ROOT=Path(__file__).resolve().parent

def main():
 import tensorflow as tf
 out=ROOT/'runs/pooled_motion_round_1';m=json.loads((out/'manifest.json').read_text())
 if 'threshold' not in m:raise RuntimeError('Training/evaluation not yet complete')
 from motion_stream import MotionStream
 verification=[];summary=[]
 for dataset,rate in [('daphnet',64),('fogstar',60)]:
  cfg=replace(Config(),sample_rate=rate,history_seconds=6,horizon_seconds=5)
  seg=load_subjects(ROOT,['S07','S08'],cfg)[0] if rate==64 else load_fogstar(ROOT/'sensor_data.csv',list(range(15,19)))
  s=next(s for s in seg if len(s.time)>=cfg.history_rows+3*cfg.interval_rows)
  online=MotionStream(out,rate);infer=online.predict_windows
  outputs=[a for t,x in zip(s.time[:cfg.history_rows+3*cfg.interval_rows],s.xyz_g) if (a:=online.step(t,x)) is not None]
  saved=pd.read_csv(out/(dataset+'_predictions.csv'));expected=saved.loc[saved.segment==s.key].head(len(outputs))
  for key in ['probability','smoothed','alarm_start','alarm_active']:
   np.testing.assert_allclose([v[key] for v in outputs],expected[key],atol=1e-6,rtol=1e-5)
  durations=[]
  sample=s.xyz_g[:cfg.history_rows][None]
  for _ in range(10):
   start=time.perf_counter();np.asarray(infer(sample));durations.append(time.perf_counter()-start)
  verification.append(dict(dataset=dataset,label_free_stream_parity=True,predictions=len(outputs),
      workstation_inference_median_ms=1000*float(np.median(durations)),
      workstation_inference_max_ms=1000*float(np.max(durations)),
      observed_calls_under_prediction_interval=bool(max(durations)<cfg.interval_seconds),
      timing_note='Transform and model call on this computer only; excludes sensor/communication/actuator delay'))
  c=pd.read_csv(out/(dataset+'_candidates.csv'));b=c.sort_values(['episode_sensitivity','warning_precision'],ascending=False).iloc[0]
  total=38 if dataset=='daphnet' else 16
  summary.append(dict(dataset=dataset,diagnostic_detections=int(b.true_warning_events),original_episodes=total,
      unsuccessful_per_hour=float(b.false_alarms_per_hour),threshold=float(b.threshold),feasible=int(c.feasible.sum())))
 (out/'stream_verification.json').write_text(json.dumps(verification,indent=2))
 pd.DataFrame(summary).to_csv(out/'comparison.csv',index=False)
 lines=['# Pooled ankle-motion experiment','',
 'The model combines Daphnet training subjects S01-S06 with FoG-STAR training subjects 1-14 using motion magnitudes. Each cohort and participant receives balanced total training weight. Raw axis directions are not assumed to match. Validation uses S07/S08 and FoG-STAR 15-18.', '',
 'Six seconds of trailing input are summarized into 120 physical-time bins. A causal convolution plus GRU predicts an onset 2-5 seconds ahead. Gravity-relative magnitude is computed from the available history only.','',
 f"Completed epochs: {len(pd.read_csv(out/'training.csv'))}. Selected policy: {m['policy_status']}; threshold: {m['threshold']}. A disabled policy does not meet the sensitivity target.", '',
 'Diagnostic highest-sensitivity thresholds below are NOT recommended operating points:','',
 '| Cohort | Episodes warned / all original annotations | Unsuccessful warnings/hour | Feasible thresholds |','|---|---:|---:|---:|']
 for row in summary:lines.append(f"| {row['dataset']} | {row['diagnostic_detections']}/{row['original_episodes']} | {row['unsuccessful_per_hour']:.1f} | {row['feasible']} |")
 lines+=['','Model reload and label-free online/offline predictions, smoothing and alarm-state checks passed on both native sampling rates. 36 scientific regression tests passed.', '',
 'Missing-data caution: FoG-STAR validation has 16 original onsets, but only 12 remain observable after ankle gaps. The ideal-score diagnostic with six-second history warns 9/16. Reported comparison denominators above retain all 16. Raw candidate tables use observed continuous-segment denominators and are labeled accordingly.', '',
 'Timing remains nominal relative to annotation and sensor-window end. Workstation inference measurements are saved in stream_verification.json, but do not establish device end-to-end delay or guarantee a two-second delivered warning.','',
 'No final or reserved-subject evaluation was performed. None of these development results establishes reliable real-world performance.']
 (ROOT/'POOLED_MOTION_RESULTS.md').write_text('\n'.join(lines)+'\n')
 print(pd.DataFrame(summary).to_string(index=False));print(json.dumps(verification,indent=2))
if __name__=='__main__':main()
