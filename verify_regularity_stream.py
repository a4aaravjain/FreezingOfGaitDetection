"""No tuning: native sample parity for already frozen entropy selections."""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fogstar_adapter import segments_from_frame
from fog_pipeline import save_json,sha256
from regularity_stream import RegularityStream


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--run',default='regularity_20261111_02')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = root/'runs'/args.run
    assert (out/'execution_completed.json').exists()
    source = root/'development_only_20261111_01.csv'
    assert sha256(source) == json.loads((out/'preregistered_plan.json').read_text())['data_sha256']
    frame = pd.read_csv(source)
    assert set(frame.subjectID) == set(range(1,19))
    reports, timings = [], []
    with threadpool_limits(limits=2):
        for folder in sorted(out.glob('fold_*')):
            if not (folder/'FROZEN_POLICY.json').exists():
                continue
            policy = json.loads((folder/'FROZEN_POLICY.json').read_text())
            key = 'entropy_'+policy['selected']['entropy']
            expected = pd.read_csv(folder/(key+'_predictions.csv'))
            stream = RegularityStream(folder)
            latency, count, error = [], 0, 0.
            for segment in segments_from_frame(frame[frame.subjectID.isin(policy['outer'])],ankle='ankleR',include_gyro=True):
                stream.reset()
                rows = expected[expected.chunk==segment.key].reset_index(drop=True)
                position = 0
                for timestamp, sample in zip(segment.time, segment.xyz_g):
                    tic = time.perf_counter()
                    result = stream.step(timestamp,sample)
                    elapsed = (time.perf_counter()-tic)*1000
                    if result is None:
                        continue
                    row = rows.iloc[position]
                    position += 1
                    assert abs(result['time']-row.time) < 1e-8
                    error = max(error,abs(result['probability']-row.probability))
                    np.testing.assert_allclose(result['probability'],row.probability,rtol=1e-5,atol=1e-6)
                    assert result['alarm_start'] == row.alarm_start
                    assert result['alarm_active'] == row.alarm_active
                    latency.append(elapsed)
                    timings.append(dict(fold=folder.name,segment=segment.key,time=timestamp,compute_ms=elapsed))
                    count += 1
                assert position == len(rows)
            report = dict(fold=folder.name,key=key,windows=count,probability_parity=True,decisions_exact=True,
                max_probability_error=error,median_ms=float(np.median(latency)),p95_ms=float(np.percentile(latency,95)),
                max_ms=float(max(latency)),over_250ms=int(sum(x>250 for x in latency)))
            reports.append(report)
            print(report,flush=True)
    pd.DataFrame(timings).to_csv(out/'native_compute_latency.csv',index=False)
    save_json(out/'native_parity.json',dict(folds=reports,transport_actuator_latency_measured=False,
        source_hashes={name:sha256(root/name) for name in ['regularity_stream.py','verify_regularity_stream.py']}))


if __name__ == '__main__':
    main()
