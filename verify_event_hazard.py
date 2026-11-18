"""Complete native six-axis replay after every model/policy is frozen."""
from pathlib import Path
import json,time
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from event_hazard_audit import OUT,save,sha,ROOT,PRIOR
from event_hazard_stream import EventHazardStream,FrozenScoreEngine
from published_cascade_core import HEADS


def main():
    assert (OUT/'training_completed.json').exists();reports=[];timings=[];adjusted=[];optimizations=[]
    with threadpool_limits(limits=2):
        for fold in range(1,5):
            folder=OUT/f'fold_{fold}'
            if not (folder/'results.json').exists():continue
            frozen=json.loads((folder/'FROZEN_CHALLENGER.json').read_text());data=joblib.load(folder/'outer_data.joblib')
            name=frozen['selected'];expected=pd.read_csv(folder/'challenger_predictions.csv',float_precision='round_trip')
            hazard=pd.read_csv(folder/(name+'_hazards.csv'),float_precision='round_trip');upstream=pd.read_csv(PRIOR/f'fold_{fold}'/'outer_score_ledger.csv',float_precision='round_trip')
            # Engineering-only graph fusion after model/policy freeze. No weight,
            # architecture, preprocessing, cutoff or decision rule changes.
            stream=EventHazardStream(folder,fused=True);reference=FrozenScoreEngine(PRIOR/f'fold_{fold}'/'upstream_final',fused=False)
            probes=np.random.default_rng(42).normal(size=(8,180,6)).astype(np.float32);errors=[]
            for x in probes:
                a,_=reference.step(x);b,_=stream.engine.step(x);np.testing.assert_allclose(a,b,rtol=1e-5,atol=1e-6);errors.append(float(max(abs(a-b))))
            optimizations.append(dict(fold=fold,change='Fused frozen7upstream neural forwards into one traced inference graph',synthetic_probes=8,max_error=max(errors),weights_changed=False,policy_changed=False))
            del reference
            current_person=current_session=None;position=0;max_error=0.;feature_error=0.;reset_count=0;alerts=0
            for record in data['part']['records']:
                person=record.subject;session=':'.join(record.key.split(':')[:2])
                if person!=current_person:stream.reset_wearer()
                elif session!=current_session:stream.reset_session()
                else:stream.reset_history()
                current_person,current_session=person,session
                for t,sample in zip(record.time,record.xyz_g):
                    delivered=[];arrival=time.perf_counter()
                    def cue(event):delivered.append(dict(event,delivered_ms=(time.perf_counter()-arrival)*1000))
                    try:result=stream.step(float(t),sample,arrival_wall=arrival,on_alert=cue)
                    except ValueError:
                        assert not np.isfinite(sample).all();reset_count+=1;continue
                    if result is None:continue
                    i=position;position+=1
                    assert i<len(expected) and abs(t-expected.loc[i,'time'])<1e-8 and expected.loc[i,'segment']==record.key
                    np.testing.assert_allclose(stream.last_scores,upstream.loc[i,HEADS].to_numpy(float),rtol=1e-5,atol=1e-6)
                    reference_x=data['x'][data['row_index'][i]];feature_error=max(feature_error,float(max(abs(stream.last_features-reference_x))))
                    np.testing.assert_allclose(stream.last_features,reference_x,rtol=1e-5,atol=2e-5)
                    for key in ['candidate','alarm_start','alarm_active']:
                        assert result[key]==bool(expected.loc[i,key]),(fold,i,key,result[key],expected.loc[i,key])
                    assert result['hazard_available']==bool(hazard.loc[i,'hazard_available'])
                    if result['hazard_available']:
                        p=hazard.loc[i,[f'probability_{j}' for j in range(5)]].to_numpy(float);err=float(max(abs(result['probabilities']-p)));max_error=max(max_error,err)
                        np.testing.assert_allclose(result['probabilities'],p,rtol=2e-5,atol=2e-6)
                    assert len(delivered)==int(result['alarm_start']);alerts+=len(delivered)
                    output_ms=delivered[0]['delivered_ms'] if delivered else result['effective_output_ms']
                    timings.append(dict(fold=fold,segment=record.key,time=float(t),compute_ms=result['compute_ms'],dispatch_delay_ms=result['dispatch_delay_ms'],
                        effective_output_ms=output_ms,alarm_start=result['alarm_start']))
            assert position==len(expected)
            result=json.loads((folder/'challenger_results.json').read_text())
            for e in result['events']:
                if e['warning_seconds'] is None:continue
                start=e['onset']-e['warning_seconds'];row=next(r for r in timings if r['fold']==fold and r['segment']==e['segment'] and abs(r['time']-start)<1e-8)
                adjusted.append(dict(fold=fold,segment=e['segment'],onset=e['onset'],nominal_lead=e['warning_seconds'],compute_and_dispatch_ms=row['effective_output_ms'],
                    software_output_lead=e['warning_seconds']-row['effective_output_ms']/1000))
            stream.reset_wearer()
            for i in range(30):stream.step(i/60,np.zeros(6,np.float32))
            for invalid_time,invalid_sample in [(29/60,np.zeros(6)),(1.,np.full(6,np.nan)),(np.inf,np.zeros(6)),(2.,np.zeros(24))]:
                try:stream.step(invalid_time,invalid_sample)
                except ValueError:pass
                else:raise AssertionError('Invalid/reversed/repeated/teacher-dimensional native sample was accepted')
                assert not len(stream.buffer) and not len(stream.history) and not len(stream.features.raw)
            for i in range(30):stream.step(i/60,np.zeros(6,np.float32))
            assert stream.step(3.,np.zeros(6,np.float32)) is None and len(stream.buffer)==1
            rows=[r for r in timings if r['fold']==fold];ms=np.array([r['compute_ms'] for r in rows]);dispatch=np.array([r['dispatch_delay_ms'] for r in rows])
            report=dict(fold=fold,windows=position,alerts=alerts,setup_seconds=stream.setup_seconds,score_hazard_decision_parity=True,
                max_hazard_error=max_error,max_feature_absolute_error=feature_error,invalid_native_resets=reset_count,
                explicit_invalid_gap_repeated_reversed_teacher_dimension_checks=True,median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),max_ms=float(ms.max()),over_250ms=int(sum(ms>=250)),
                dispatch_median_ms=float(np.median(dispatch)),dispatch_max_ms=float(dispatch.max()),teacher_channels_used_at_inference=False,
                scheduling_scope='Observed host API dispatch from simulated sample availability; not sensor/OS polling or radio delay',transport_actuator_delay_measured=False)
            reports.append(report);save(folder/'native_verification.json',report);print('HAZARD NATIVE',report,flush=True)
    pd.DataFrame(timings).to_csv(OUT/'native_latency.csv',index=False);pd.DataFrame(adjusted).to_csv(OUT/'software_output_leads.csv',index=False)
    save(OUT/'runtime_optimization_audit.json',dict(changes=optimizations,applied_after_model_policy_freeze=True,new_experiments=False,source_hashes={n:sha(ROOT/n) for n in ['event_hazard_stream.py','verify_event_hazard.py']}))
    save(OUT/'native_verification.json',dict(folds=reports,total_windows=sum(r['windows'] for r in reports),all_folds_complete=len(reports)==4,
        transport_actuator_delay_measured=False,setup_warmup='Complete20s synthetic-zero history only, then full wearer/history reset; all setup time separate'))

if __name__=='__main__':main()
