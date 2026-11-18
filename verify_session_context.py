"""Frozen-model native replay, no parameter changes or selection."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from fog_pipeline import save_json,sha256
from development_data import load_development
from session_assessment import prepare_sessions
from session_context_stream import SessionContextStream


def main():
    root=Path(__file__).resolve().parent;out=root/'runs/session_context_20261111_01'
    assert (out/'execution_completed.json').exists()
    frame=load_development(range(1,19));reports=[]
    with threadpool_limits(limits=2):
        for fold in range(1,5):
            folder=out/f'fold_{fold}';policy=json.loads((folder/'FROZEN_POLICY.json').read_text())
            outer=policy['outer'];part=frame[frame.subjectID.isin(outer)]
            _,chunks,_,bad=prepare_sessions(part,outer);assert not bad
            for key in ['raw','context']:
                expected=pd.read_csv(folder/(key+'_predictions.csv'));stream=SessionContextStream(folder,key)
                latency=[];count=0;error=0.
                for chunk in chunks:
                    stream.reset();reference=expected[expected.chunk==chunk['id']].reset_index(drop=True);position=0
                    for t,sample in zip(chunk['time'],chunk['x']):
                        tic=time.perf_counter();result=stream.step(t,sample)
                        elapsed=(time.perf_counter()-tic)*1000
                        if result is None:continue
                        row=reference.iloc[position];position+=1;count+=1;latency.append(elapsed)
                        assert abs(t-row.time)<1e-8
                        error=max(error,abs(result['probability']-row.probability))
                        np.testing.assert_allclose(result['probability'],row.probability,rtol=1e-5,atol=1e-6)
                        assert result['alarm_start']==row.alarm_start and result['alarm_active']==row.alarm_active
                    assert position==len(reference)
                report=dict(fold=fold,model=key,windows=count,probability_parity=True,decisions_exact=True,
                    max_probability_error=float(error),median_ms=float(np.median(latency)),p95_ms=float(np.percentile(latency,95)),max_ms=float(max(latency)),over_250ms=int(sum(x>250 for x in latency)))
                reports.append(report);print(report,flush=True)
    save_json(out/'native_parity.json',dict(reports=reports,physical_delivery_latency_measured=False,
        source_hashes={n:sha256(root/n) for n in ['verify_session_context.py','session_context_stream.py']}))


if __name__=='__main__':main()
