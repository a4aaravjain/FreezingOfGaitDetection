"""Native replay of every frozen upstream head and every fitted full verifier."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from fog_pipeline import save_json,sha256
from published_cascade_core import load_export,HEADS,candidate_mask
from session_assessment import prepare_sessions
from published_cascade_stream import PublishedCascadeStream,NativeVerifier
from waveform_ensemble import PulseUnion,WaveformEnsemble
from rescue_cascade import PhysicalOR


def read_csv(path):return pd.read_csv(path,float_precision='round_trip')


def main():
    root=Path(__file__).resolve().parent;out=root/'runs/published_cascade_20261111_01';assert (out/'execution_completed.json').exists()
    frame=load_export();reports=[];timingrows=[]
    with threadpool_limits(limits=2):
        for fold in range(1,5):
            folder=out/f'fold_{fold}'
            if not (folder/'results.json').exists():continue
            spec=json.loads((folder/'partitions.json').read_text());outer=spec['outer']
            _,chunks,_,bad=prepare_sessions(frame[frame.subjectID.isin(outer)],outer);assert not bad
            stream=PublishedCascadeStream(folder);frozen=stream.frozen
            selected=read_csv(folder/'baseline_OR_best_rescue_predictions.csv');scores=read_csv(folder/'outer_score_ledger.csv')
            expected_candidates=candidate_mask(dict(meta=scores),scores[HEADS].to_numpy(),frozen['level1'])
            branches={};verifiers={};ledgers={};expected={}
            for key,policy in frozen['verifiers'].items():
                if policy['family'] not in verifiers:verifiers[policy['family']]=NativeVerifier(policy['family'],folder/policy['model_dir'])
                ledgers[key]=read_csv(folder/(key+'_verifier_ledger.csv'));expected[key]=read_csv(folder/(key+'_combined_predictions.csv'))
            original=WaveformEnsemble(root/'runs/right_ankle_waveform_v2') if outer==[15,16,17,18] else None
            original_expected=read_csv(folder/'original_saved_v2_OR_rescue_predictions.csv') if original is not None and (folder/'original_saved_v2_OR_rescue_predictions.csv').exists() else None
            count=0;latency=[];maxerror=0.;checked=0
            for chunk in chunks:
                stream.reset()
                if original:original.reset()
                indices=selected.index[selected.chunk==chunk['id']].to_list();position=0
                branches={key:(PulseUnion(1,3),PhysicalOR()) for key in frozen['verifiers']};lastbase=-np.inf
                opulse,ophysical=PulseUnion(1,3),PhysicalOR();lastorig=-np.inf
                for timestamp,sample in zip(chunk['time'],chunk['x']):
                    orig=original.step(timestamp,sample) if original is not None else None
                    callback=[]
                    tic=time.perf_counter();result=stream.step(timestamp,sample,on_baseline=callback.append);elapsed=(time.perf_counter()-tic)*1000
                    if result is None:
                        if original is not None:assert orig is None
                        continue
                    i=indices[position];position+=1;count+=1;latency.append(elapsed)
                    assert result['rescue_errors']==0
                    assert len(callback)==1 and callback[0]['alarm_start']==result['baseline_start'] and callback[0]['compute_ms']<=elapsed
                    assert abs(timestamp-selected.loc[i,'time'])<1e-8
                    reference=scores.loc[i,HEADS].to_numpy(float);error=float(max(abs(result['probabilities']-reference)));maxerror=max(error,maxerror)
                    np.testing.assert_allclose(result['probabilities'],reference,rtol=1e-5,atol=1e-6)
                    assert result['candidate']==expected_candidates[i]
                    for key in ['baseline_active','baseline_start','rescue_active','rescue_start','alarm_active','alarm_start']:
                        assert result[key]==selected.loc[i,key],(fold,i,key,result[key],selected.loc[i,key])
                    if result['candidate'] and stream.verifier:
                        key=frozen['selected'];np.testing.assert_allclose(result['verifier_probability'],ledgers[key].loc[i,'probability'],rtol=1e-5,atol=1e-6)
                    if result['baseline_start']:lastbase=timestamp
                    for key,policy in frozen['verifiers'].items():
                        p=verifiers[policy['family']].step(stream.last_x,stream.last_q,policy['mode']) if result['candidate'] else 0.
                        if result['candidate']:
                            np.testing.assert_allclose(p,ledgers[key].loc[i,'probability'],rtol=1e-5,atol=1e-6);checked+=1
                        pulse,physical=branches[key];allowed=not result['baseline_active'] and timestamp-lastbase>=3-1e-9
                        ra,rs=pulse.step(timestamp,[bool(result['candidate'] and p>=policy['combined'] and allowed)])
                        active,start=physical.step(result['baseline_active'],ra)
                        for name,value in [('alarm_active',active),('alarm_start',start),('rescue_active',ra),('rescue_start',rs)]:assert value==expected[key].loc[i,name]
                    if orig is not None and original_expected is not None:
                        if orig['alarm_start']:lastorig=timestamp
                        p=result['verifier_probability'];threshold=stream.policy['combined'] if stream.policy else 1.1
                        allowed=not orig['alarm_active'] and timestamp-lastorig>=3-1e-9
                        ra,rs=opulse.step(timestamp,[bool(result['candidate'] and p>=threshold and allowed)])
                        active,start=ophysical.step(orig['alarm_active'],ra)
                        assert active==original_expected.loc[i,'alarm_active'] and start==original_expected.loc[i,'alarm_start']
                        assert orig['alarm_start']==original_expected.loc[i,'baseline_start']
                    timingrows.append(dict(fold=fold,segment=selected.loc[i,'segment'],time=timestamp,compute_ms=elapsed,
                        baseline_compute_ms=result['baseline_compute_ms'],effective_output_ms=result['baseline_compute_ms'] if result['baseline_start'] and result['alarm_start'] else elapsed,
                        alarm_start=result['alarm_start'],baseline_start=result['baseline_start']))
                assert position==len(indices)
            stream.reset()
            for tick in range(179):assert stream.step(tick/60,np.zeros(6)) is None
            assert stream.step(10.,np.zeros(6)) is None
            try:stream.step(10.,np.zeros(6))
            except ValueError:pass
            else:raise AssertionError('Repeated timestamp accepted')
            try:stream.step(11.,np.full(6,np.nan))
            except ValueError:pass
            else:raise AssertionError('Invalid sensor sample accepted')
            assert not len(stream.buffer) and not len(stream.feature_history)
            stream.reset();saved_networks=stream.networks
            def fail(_):raise RuntimeError('synthetic auxiliary failure')
            class ForcedAlarm:
                def step(self,t,p):return p,True,True
            stream.networks=[fail]+saved_networks[1:];stream.heads=[ForcedAlarm() for _ in range(3)]
            seen=[]
            for tick in range(180):fault=stream.step(tick/60,np.zeros(6),on_baseline=seen.append)
            assert seen[-1]['alarm_start'] and fault['baseline_start'] and fault['alarm_start'] and fault['rescue_errors']==1
            stream.networks=saved_networks;stream.reset()
            report=dict(fold=fold,windows=count,upstream_heads=len(HEADS),max_probability_error=maxerror,
                gap_invalid_repeated_time_reset_checks=True,
                baseline_callback_before_rescue_work=True,
                auxiliary_failure_preserves_baseline_output=True,
                upstream_score_parity=True,selected_physical_parity=True,all_full_verifier_combined_decisions_exact=True,
                verifier_candidate_checks=checked,original_saved_v2_operational_parity=original_expected is not None,
                setup_seconds=stream.setup_seconds,median_ms=float(np.median(latency)),p95_ms=float(np.percentile(latency,95)),max_ms=float(max(latency)),over_250ms=int(sum(v>250 for v in latency)))
            reports.append(report);save_json(folder/'native_verification.json',report);print(report,flush=True)
    pd.DataFrame(timingrows).to_csv(out/'native_latency.csv',index=False)
    save_json(out/'native_verification.json',dict(folds=reports,source_hashes={n:sha256(root/n) for n in ['published_cascade_stream.py','verify_published_cascade.py']},
        setup_policy='Models compile/warm on synthetic zeros during initialization; reported separately from active-stream timing. No evaluation data used for warmup.',
        transport_and_actuator_delay_measured=False))


if __name__=='__main__':main()
