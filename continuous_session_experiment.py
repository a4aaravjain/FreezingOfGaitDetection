"""Frozen-model experiment isolating task resets from physical-session continuity."""
from pathlib import Path
import json,shutil,time
import numpy as np,pandas as pd
from threadpoolctl import threadpool_limits
from fog_pipeline import sha256,save_json
from waveform_ensemble import WaveformEnsemble
from session_assessment import prepare_sessions,score_sessions
from assessment_core import prepare,score
from rescue_cascade import poisson_interval


def main(output=None,model_run=None):
    root=Path(__file__).resolve().parent
    out=Path(output) if output is not None else root/'runs/continuous_session_20261111_01'
    out.mkdir(exist_ok=False)
    source=root/'development_only_20261111_01.csv'
    expected_hash='2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
    assert sha256(source)==expected_hash
    run=Path(model_run) if model_run is not None else root/'runs/right_ankle_waveform_v2'
    model_manifest=json.loads((run/'manifest.json').read_text())
    names=['continuous_session_experiment.py','session_assessment.py','test_session_assessment.py','ankles.py','development_data.py',
        'assessment_core.py','waveform_ensemble.py','summary_stream.py','fog_pipeline.py',
        'long_history_experiments.py','gravity_gait.py','fast_temporal_bank.py','waveform_bank.py']
    save_json(out/'preregistered_plan.json',dict(
        hypothesis='Protocol task-ID changes must not reset physically continuous native data or censor available annotation follow-up.',
        participants=[15,16,17,18],development_only=True,
        development_export_sha256=expected_hash,model_manifest_sha256=sha256(run/'manifest.json'),
        model_artifact_hashes=model_manifest['artifact_sha256'],
        variants=['task_reset_control','continuous_session'],training=False,threshold_selection=False,
        common_scoring='Whole physical sessions; retain onsets across missing sensors; never bridge real time gaps. Score native 60Hz frames after 180 consecutive finite right-ankle samples, with exactly the same exposure for both variants. Task-control outputs may be absent during its artificial warmup and are inactive there.',
        historical_scoring='Also retain task-reset legacy scoring and follow-up-only corrected scoring on the same saved control trace, to separate rescoring from changed model outputs.',
        useful_change='More original episodes warned, no old true warning lost or later, no additional unsuccessful/unknown starts, no increased false duration or maximum pulse. No promotion without all existing promotion requirements and actual delivered lead measurement.',
        calibration='None. Fixed weights, thresholds and alarm timings unchanged. Historical diagnostic; not independent validation.',
        source_sha256={n:sha256(root/n) for n in names}))
    (out/'source').mkdir()
    for n in names:shutil.copy2(root/n,out/'source'/n)
    frame=pd.read_csv(source)
    assert set(frame.subjectID)==set(range(1,19))
    frame=frame[frame.subjectID.isin([15,16,17,18])].copy()
    records,chunks,inventory,incomplete=prepare_sessions(frame,[15,16,17,18])
    assert not incomplete
    expected=pd.read_csv(run/'predictions.csv')
    controls,continuous=WaveformEnsemble(run),WaveformEnsemble(run)
    common_control,common_continuous,native=[],[],[]
    control_rows,continuous_rows=[],[]
    latency=[];pointer=0
    with threadpool_limits(limits=2):
        for ids,g in frame.groupby(['subjectID','sessionID'],sort=True):
            controls.reset();continuous.reset();streak=0;last_task=None;chunk_start=0
            record=f'FS{int(ids[0]):02}:{ids[1]}:run0'
            columns=['ankleR_'+s+'_'+a for s in ['acc','gyro'] for a in 'xyz']
            for offset,(timestamp,sample,label,task) in enumerate(zip(g.timestamp,g[columns].to_numpy(np.float32),g.fog,g.taskID)):
                if last_task is not None and task!=last_task:controls.reset()
                last_task=task
                if not np.isfinite(sample).all():
                    controls.reset();continuous.reset();streak=0
                    continue
                if streak==0:chunk_start=offset
                streak+=1
                a=controls.step(timestamp,sample)
                tic=time.perf_counter();b=continuous.step(timestamp,sample)
                if b is not None:latency.append((time.perf_counter()-tic)*1000)
                base=dict(segment=record,subject=int(ids[0]),chunk=record+':chunk:'+str(chunk_start),
                    time=float(timestamp),active_fog=bool(label),duration=1/60)
                if a is not None:
                    old=expected.iloc[pointer]
                    assert abs(timestamp-old.time)<1e-8
                    for field in [k for k in ['base_probability','waveform_probability','extended_probability'] if k in expected]:
                        np.testing.assert_allclose(a[field],old[field],rtol=1e-5,atol=1e-6)
                    assert a['alarm_start']==old.alarm_start and a['alarm_active']==old.alarm_active
                    control_rows.append(dict(base,**{k:v for k,v in a.items() if k!='time'}))
                    pointer+=1
                if b is not None:continuous_rows.append(dict(base,**{k:v for k,v in b.items() if k!='time'}))
                if streak>=180:
                    ca=bool(timestamp<controls.union.last+controls.union.pulse-1e-9)
                    cb=bool(timestamp<continuous.union.last+continuous.union.pulse-1e-9)
                    common_control.append(dict(base,alarm_active=ca,alarm_start=bool(a is not None and a['alarm_start'])))
                    common_continuous.append(dict(base,alarm_active=cb,alarm_start=bool(b is not None and b['alarm_start'])))
            print('SESSION',ids,'control ticks',pointer,'continuous ticks',len(continuous_rows),flush=True)
    assert pointer==len(expected)
    a,b=pd.DataFrame(common_control),pd.DataFrame(common_continuous)
    np.testing.assert_array_equal(a[['segment','time','duration']],b[['segment','time','duration']])
    results={}
    for name,trace in [('task_reset_control',a),('continuous_session',b)]:
        result=score_sessions(records,trace,inventory,incomplete,continuous.cfg)
        result['false_rate95_poisson']=poisson_interval(result['metrics']['total_false_alarms'],result['metrics']['monitored_seconds']/3600)
        results[name]=result
        trace.to_csv(out/(name+'_native_predictions.csv'),index=False)
        save_json(out/(name+'_results.json'),result)
    oldrecords,_,oldinventory,oldbad=prepare(frame,[15,16,17,18])
    oldtrace=expected.copy();oldtrace['chunk']=oldtrace.segment;oldtrace['segment']=oldtrace.segment.map(lambda s:':'.join(s.split(':')[:3]))
    legacy=score(oldrecords,oldtrace,oldinventory,oldbad,continuous.cfg)
    relabeled=oldtrace.copy();relabeled['segment']=relabeled.segment.map(lambda s:':'.join(s.split(':')[:2])+':run0')
    followup=score_sessions(records,relabeled,inventory,incomplete,continuous.cfg)
    save_json(out/'legacy_vs_followup_only.json',dict(legacy=legacy,followup_only=followup,
        warning_times_unchanged=True,exposure_unchanged=True))
    pd.DataFrame(control_rows).to_csv(out/'control_decision_scores.csv',index=False)
    pd.DataFrame(continuous_rows).to_csv(out/'continuous_decision_scores.csv',index=False)
    save_json(out/'results.json',results)
    saved={}
    for key,result in results.items():
        saved[key]={(e['segment'],e['onset']):e['warning_seconds'] for e in result['events'] if e['warning_seconds'] is not None}
    previous,current=saved['task_reset_control'],saved['continuous_session']
    ma,mb=results['task_reset_control']['metrics'],results['continuous_session']['metrics']
    gates=dict(more_episodes=len(current)>len(previous),
        previous_episodes_preserved=set(previous).issubset(current),
        previous_leads_not_later=all(k in current and current[k]>=v-1e-6 for k,v in previous.items()),
        false_starts_not_increased=mb['total_false_alarms']<=ma['total_false_alarms'],
        unknown_starts_not_increased=mb['censored_alarm_starts']<=ma['censored_alarm_starts'],
        false_duration_not_increased=mb['false_alarm_seconds']<=ma['false_alarm_seconds']+1e-6,
        pulse_not_longer=mb['maximum_alarm_duration_seconds']<=ma['maximum_alarm_duration_seconds']+1e-6,
        same_original_onsets=ma['original_annotated_onsets']==mb['original_annotated_onsets'],
        same_exposure=abs(ma['monitored_seconds']-mb['monitored_seconds'])<1e-6)
    save_json(out/'diagnostic_gate.json',dict(gates=gates,all_point_checks=all(gates.values()),promoted=False,
        actual_transport_actuator_latency_measured=False,independent_validation=False,default_model_weights_changed=False))
    save_json(out/'native_verification.json',dict(control_windows=pointer,continuous_windows=len(continuous_rows),
        saved_control_probabilities_match=True,saved_control_decisions_exact=True,
        continuous_prediction_ms=dict(median=float(np.median(latency)),max=float(max(latency))),
        shared_native_scoring_rows=len(a),same_exposure=True,reserved_participants_excluded=True))
    print(json.dumps({k:v['metrics'] for k,v in results.items()},indent=2),flush=True)
    return results


if __name__=='__main__':main()
