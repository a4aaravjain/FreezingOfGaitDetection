from pathlib import Path
import time,json
import numpy as np,pandas as pd
from waveform_ensemble import WaveformEnsemble
from fogstar_adapter import segments_from_frame
from development_data import load_development
from fog_pipeline import save_json
ROOT=Path(__file__).resolve().parent

def verify(run):
 run=Path(run);stream=WaveformEnsemble(run);expected=pd.read_csv(run/'predictions.csv');latencies=[];count=0;worst=0.
 for s in segments_from_frame(load_development(range(15,19)),ankle='ankleR',include_gyro=True):
  stream.reset();rows=[]
  for t,sample in zip(s.time,s.xyz_g):
   tic=time.perf_counter();v=stream.step(float(t),sample)
   if v is not None:latencies.append((time.perf_counter()-tic)*1000);rows.append(v)
  ref=expected[expected.segment==s.key];assert len(rows)==len(ref),(s.key,len(rows),len(ref))
  if rows:
   got=pd.DataFrame(rows)
   for c in ['time','base_probability','waveform_probability','base_smoothed','waveform_smoothed']+(['extended_probability','extended_smoothed'] if 'extended_probability' in ref else []):
    np.testing.assert_allclose(got[c],ref[c],rtol=1e-5,atol=1e-6)
   for c in ['base_start','waveform_start','alarm_active','alarm_start']+(['extended_start'] if 'extended_start' in ref else []):np.testing.assert_array_equal(got[c],ref[c])
   worst=max(worst,float(np.max(abs(got.waveform_probability.to_numpy()-ref.waveform_probability.to_numpy()))))
  count+=len(rows)
 stream.reset()
 for i in range(179):assert stream.step(i/60,np.zeros(6)) is None
 assert stream.step(10.,np.zeros(6)) is None
 try:stream.step(10.,np.zeros(6))
 except ValueError:pass
 else:raise AssertionError('Repeated time accepted')
 try:stream.step(11.,[np.nan]*6)
 except ValueError:pass
 else:raise AssertionError('Invalid sample accepted')
 assert stream.step(12.,np.zeros(6)) is None
 result=dict(windows_checked=count,all_probability_smoothing_and_alarm_checks_passed=True,gap_backwards_invalid_reset_checks=True,waveform_probability_max_absolute_difference=worst,feature_inference_alarm_median_ms=float(np.median(latencies)),feature_inference_alarm_p95_ms=float(np.percentile(latencies,95)),feature_inference_alarm_max_ms=float(max(latencies)),interval_budget_ms=250.,exceeds_interval_count=int((np.array(latencies)>250).sum()),excludes='Sensor transport and actuator delay; desktop only; nominal warning lead is not measured delivered lead')
 save_json(run/'stream_verification.json',result);print(result);return result
if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=ROOT/'runs/right_ankle_waveform_v2');a=parser.parse_args();verify(a.run)
