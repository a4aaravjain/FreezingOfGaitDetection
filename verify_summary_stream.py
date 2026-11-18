"""Verify saved summary-model replay against every native validation sensor sample."""
from pathlib import Path
import time
import numpy as np,pandas as pd
from summary_stream import SummaryStream
from fogstar_adapter import segments_from_frame
from development_data import load_development
from fog_pipeline import save_json
ROOT=Path(__file__).resolve().parent
def verify(run):
    run=Path(run)
    stream=SummaryStream(run)
    expected=pd.read_csv(run/'current_summary_mlp_predictions.csv')
    latencies=[];count=0
    for segment in segments_from_frame(load_development(range(15,19)),ankle='ankleR',include_gyro=True):
        stream.reset();rows=[]
        for timestamp,sample in zip(segment.time,segment.xyz_g):
            t=time.perf_counter();result=stream.step(timestamp,sample)
            if result is not None:latencies.append((time.perf_counter()-t)*1000);rows.append(result)
        ref=expected[expected.segment==segment.key]
        assert len(rows)==len(ref),(segment.key,len(rows),len(ref))
        if rows:
            got=pd.DataFrame(rows)
            for c in ['time','probability','smoothed']:
                np.testing.assert_allclose(got[c],ref[c],rtol=1e-5,atol=1e-6)
            for c in ['alarm_active','alarm_start']:
                np.testing.assert_array_equal(got[c],ref[c])
        count+=len(rows)
    stream.reset()
    for i in range(179):assert stream.step(i/60,np.zeros(6)) is None
    assert stream.step(10.,np.zeros(6)) is None
    try:stream.step(10.,np.zeros(6))
    except ValueError:pass
    else:raise AssertionError('Non-increasing timestamp accepted')
    try:stream.step(11.,[np.nan]*6)
    except ValueError:pass
    else:raise AssertionError('Invalid sample accepted')
    report=dict(windows_checked=count,probability_smoothing_alarm_parity=True,gap_invalid_time_checks=True,
        inference_feature_and_alarm_median_ms=float(np.median(latencies)),inference_feature_and_alarm_max_ms=float(max(latencies)),
        excludes='Sensor transport and actuator delays; desktop timing only')
    save_json(run/'stream_verification.json',report);print(report)
    return report

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',type=Path,default=ROOT/'runs/right_ankle_pulse_candidate')
    args=parser.parse_args();verify(args.run)
