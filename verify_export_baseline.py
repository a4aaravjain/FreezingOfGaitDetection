from pathlib import Path
import json,time,hashlib
import numpy as np,pandas as pd
from waveform_ensemble import WaveformEnsemble
from rescue_cascade import RescueCascade
from fogstar_adapter import segments_from_frame
r=Path(__file__).resolve().parent;run=r/'runs/right_ankle_waveform_v2';out=r/'runs/nested_rescue_20261111_02';source=r/'development_only_20261111_01.csv';m=json.loads(source.with_suffix('.manifest.json').read_text());assert hashlib.sha256(source.read_bytes()).hexdigest()==m['output_sha256'];frame=pd.read_csv(source);assert set(frame.subjectID).issubset(set(range(1,19)))
class RejectAll:
 def predict_proba(self,x):return np.tile([1.,0.],(len(x),1))
baseline=WaveformEnsemble(run);cascade=RescueCascade(WaveformEnsemble(run),RejectAll(),.15,.99);expected=pd.read_csv(run/'predictions.csv');count=0;latency=[]
for segment in segments_from_frame(frame[frame.subjectID.isin([15,16,17,18])],ankle='ankleR',include_gyro=True):
 baseline.reset();cascade.reset();rows=[]
 for t,x in zip(segment.time,segment.xyz_g):
  a=baseline.step(float(t),x);tic=time.perf_counter();b=cascade.step(float(t),x)
  if a is None:assert b is None;continue
  latency.append((time.perf_counter()-tic)*1000);assert a==b['baseline'];assert b['rescue_errors']==0;assert (a['alarm_active'],a['alarm_start'])==(b['alarm_active'],b['alarm_start']);rows.append(a)
 reference=expected[expected.segment==segment.key];assert len(rows)==len(reference)
 if rows:
  got=pd.DataFrame(rows)
  for c in ['time','base_probability','waveform_probability','extended_probability']:np.testing.assert_allclose(got[c],reference[c],rtol=1e-5,atol=1e-6)
  for c in ['base_start','waveform_start','extended_start','alarm_start','alarm_active']:np.testing.assert_array_equal(got[c],reference[c])
 count+=len(rows)
report=dict(native_windows=count,independent_baseline_output_dicts_exact=True,disabled_rescue_physical_start_and_pulse_parity=True,saved_reference_decisions_exact=True,source='verified development-only export, people15-18',reserved_data_used=False,wrapper_median_ms=float(np.median(latency)),wrapper_max_ms=float(max(latency)),excludes='Sensor/transport/actuator delay; rescue always rejects in this non-regression test')
(out/'baseline_native_parity.json').write_text(json.dumps(report,indent=2));print(report)
