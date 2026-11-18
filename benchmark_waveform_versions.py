from pathlib import Path
import time,numpy as np
from waveform_ensemble import WaveformEnsemble
from fogstar_adapter import load_fogstar
from fog_pipeline import save_json
r=Path(__file__).resolve().parent;streams={'previous':WaveformEnsemble(r/'runs/right_ankle_waveform_candidate'),'v2':WaveformEnsemble(r/'runs/right_ankle_waveform_v2')};latencies={k:[] for k in streams};count=0
for segment in load_fogstar(r/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True):
 for s in streams.values():s.reset()
 for i,(timestamp,sample) in enumerate(zip(segment.time,segment.xyz_g)):
  output={}
  order=['previous','v2'] if i%2 else ['v2','previous']
  for name in order:
   tic=time.perf_counter();v=streams[name].step(float(timestamp),sample)
   if v is not None:latencies[name].append((time.perf_counter()-tic)*1000);output[name]=v
  assert (len(output)==0 or len(output)==2)
  if output:count+=1
  if count>=160:break
 if count>=160:break
report=dict(windows=count,selection='first 160 prediction windows in source order, no label-based selection',order='alternate which stream runs first each sample',timing={k:dict(median_ms=float(np.median(v)),p95_ms=float(np.percentile(v,95)),max_ms=float(max(v))) for k,v in latencies.items()},excludes='Transport, actuator delay and target-hardware validation; desktop paired timing only')
report['median_speedup']=report['timing']['previous']['median_ms']/report['timing']['v2']['median_ms'];save_json(r/'runs/right_ankle_waveform_v2/paired_stream_benchmark.json',report);print(report)
