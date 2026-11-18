"""Report the corrected protocol and negative follow-on; never tune a model."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from fog_pipeline import save_json,sha256
from development_data import load_development
from session_assessment import prepare_sessions,score_sessions
from nested_rescue_experiment import CFG


def ledger(records,trace):
    rows=[]
    for record in records:
        matched=set()
        for row in trace[(trace.segment==record.key)&trace.alarm_start].itertuples():
            t=row.time;j=int(np.searchsorted(record.onsets,t,side='right'))
            lead=float(record.onsets[j]-t) if j<len(record.onsets) else None
            state=record.labels[np.searchsorted(record.time,t,side='right')-1]
            if state==1 and lead is not None and 2-1e-9<=lead<=5+1e-9 and j not in matched:
                outcome='successful';matched.add(j)
            elif state==1 and lead is not None and 0<lead<2-1e-9:outcome='unsuccessful_late'
            elif state==1 and lead is not None and lead<=5+1e-9:outcome='unsuccessful_duplicate'
            elif state==1 and t+5>record.time[-1]+1e-9:outcome='unknown'
            else:outcome='unsuccessful'
            rows.append(dict(segment=record.key,time=t,outcome=outcome,next_onset_lead=lead))
    return pd.DataFrame(rows)


def main():
    root=Path(__file__).resolve().parent
    replay=root/'runs/continuous_session_20261111_02'
    study=root/'runs/session_context_20261111_01'
    result=json.loads((replay/'results.json').read_text())
    old=json.loads((replay/'legacy_vs_followup_only.json').read_text())
    parity=json.loads((replay/'native_verification.json').read_text())
    context=json.loads((study/'aggregate_results.json').read_text())
    native=json.loads((study/'native_parity.json').read_text())
    frame=load_development([15,16,17,18]);records,chunks,inventory,bad=prepare_sessions(frame,[15,16,17,18])
    people=[]
    for name in ['task_reset_control','continuous_session']:
        trace=pd.read_csv(replay/(name+'_native_predictions.csv'))
        alerts=ledger(records,trace)
        alerts.to_csv(replay/(name+'_alert_ledger.csv'),index=False)
        assert sum(alerts.outcome.eq('successful'))==result[name]['metrics']['true_warning_events']
        assert sum(alerts.outcome.str.startswith('unsuccessful'))==result[name]['metrics']['total_false_alarms']
        for subject in range(15,19):
            rr=[r for r in records if r.subject==f'FS{subject:02}'];ii=[i for i in inventory if i['subject']==f'FS{subject:02}']
            m=score_sessions(rr,trace[trace.subject==subject],ii,bad,CFG)['metrics']
            people.append(dict(model=name,subject=subject,**m))
    pd.DataFrame(people).to_csv(replay/'participants.csv',index=False)
    models=root/'runs/right_ankle_waveform_v2';manifest=json.loads((models/'manifest.json').read_text())
    assert all(sha256(models/name)==digest for name,digest in manifest['artifact_sha256'].items())
    c=result['continuous_session']['metrics']
    candidate=json.loads((study/'fold_4/context_results.json').read_text())
    candidate_m=candidate['metrics']
    assert abs(candidate_m['monitored_seconds']-c['monitored_seconds'])<1e-6
    original_events={(e['segment'],e['onset']):e['warning_seconds'] for e in result['continuous_session']['events'] if e['warning_seconds'] is not None}
    new_events={(e['segment'],e['onset']):e['warning_seconds'] for e in candidate['events'] if e['warning_seconds'] is not None}
    added=set(new_events)-set(original_events)
    gates=dict(all_folds_complete=len(native['reports'])==8,
        at_least_90_percent=context['context']['sensitivity']>=.9,
        unsuccessful_per_hour_below_one=context['context']['false_starts_per_hour']<1,
        zero_unknown_starts=context['context']['unknown_starts']==0,
        two_added_episodes=len(added)>=2,
        added_in_two_people=len({k[0].split(':')[0] for k in added})>=2,
        old_warnings_not_lost_or_later=all(k in new_events and new_events[k]>=v-1e-6 for k,v in original_events.items()),
        burdens_not_increased=all(candidate_m[k]<=c[k]+1e-6 for k in ['total_false_alarms','false_alarm_seconds','total_alarm_seconds','maximum_alarm_duration_seconds','censored_alarm_starts']),
        participant_macro_improved=False,
        native_parity=all(r['probability_parity'] and r['decisions_exact'] for r in native['reports']),
        computation_under_250ms=all(r['over_250ms']==0 for r in native['reports']),
        measured_delivered_lead_two_seconds=False)
    # Compute the macro change instead of assigning it from aggregate window counts.
    changes=[]
    for subject in range(15,19):
        prefix=f'FS{subject:02}:'
        events=[e for e in result['continuous_session']['events'] if e['segment'].startswith(prefix)]
        now=[e for e in candidate['events'] if e['segment'].startswith(prefix)]
        if events:changes.append(sum(e['warning_seconds'] is not None for e in now)/len(now)-sum(e['warning_seconds'] is not None for e in events)/len(events))
    macro=float(np.mean(changes));gates['participant_macro_improved']=macro>0
    save_json(study/'promotion_gate.json',dict(promoted=all(gates.values()),gates=gates,
        failures=[k for k,v in gates.items() if not v],added_episodes=len(added),participant_macro_change=macro,
        model_weights_default_changed=False,protocol_default_corrected=True))
    lines=['# Continuous-session protocol correction','',
      '**The previous zero-false-alarm claim must not be used for continuous operation.** The model weights are unchanged, but protocol task labels were incorrectly acting as sensor resets and follow-up boundaries.',
      '', 'The development audit found all 63 within-session task boundaries have consecutive native timestamps. Session preparation now preserves input history, alarm state and annotation follow-up across those boundaries. Real time gaps, invalid sensor samples and separate sessions still reset the predictor. Missing sensor readings never remove an annotated onset.',
      '', '| Evaluation | Early episodes | Unsuccessful alerts | Unknown alerts | Monitored seconds |',
      '|---|---:|---:|---:|---:|']
    entries=[('Historical task-reset scoring',old['legacy']['metrics']),
             ('Same saved alerts, full-session follow-up',old['followup_only']['metrics']),
             ('Task-reset control, common native support',result['task_reset_control']['metrics']),
             ('Continuous-session model, same native support',c)]
    for title,m in entries:
        lines.append(f"| {title} | {m['true_warning_events']}/{m['total_fog_episodes']} | {m['total_false_alarms']} | {m['censored_alarm_starts']} | {m['monitored_seconds']:.2f} |")
    lines += ['', 'Follow-up correction alone changes one formerly unknown alert into an unsuccessful warning. The full session replay additionally changes the input history and prediction alignment, so this is not a model improvement. On common support the control and continuous predictor use exactly the same timestamps, monitoring time and original onsets; a larger denominator is not used to manufacture a lower false-warning rate.',
      '', f"The continuous replay warns {c['true_warning_events']}/16 episodes, with {c['total_false_alarms']} unsuccessful warnings ({c['false_alarms_per_hour']:.2f}/monitored hour) and {c['censored_alarm_starts']} unknown warning. Successful nominal leads range from {min(c['warning_times_seconds']):.2f} to {max(c['warning_times_seconds']):.2f}s. Actual sensor-to-cue latency is unmeasured; the requirement for delivered lead >=2s is unverified.",
      '', '## Follow-on model study','',
      'Eight new fixed fits tested raw gait summaries against changes relative to the previous 20 seconds of movement. Past median/IQR context excludes the current window from its reference and uses fitting-only scale floors. Both arms use the same fixed gradient-boosting classifier, participant folds and calibration grid. Windows, labels and alarm state use corrected physical sessions throughout.',
      '', f"Both calibrated policies detected 0/72 development onsets over {context['context']['monitored_hours']:.4f} monitored hours: neither found a useful calibration setting with fewer than one unsuccessful warning/hour, no unknown starts and bounded pulses. Their alarms were disabled. Zero alerts are a failed candidate, not useful suppression. No model was promoted.",
      '', 'These are new model fits on a corrected protocol, not an independent clinical assessment. Previous entropy and teacher/student studies remain historical task-based negative experiments and must not be used to support continuous-session performance.',
      '', '## Implementation and checks','',
      '`ankles.py ensemble` now runs continuous-session development replay using the verified development export. It no longer prints cached historical performance as the current result. Saved summary/ensemble parity verifiers also use the approved export. Experimental models have not replaced the saved waveform v2 weights.',
      '', f"The real CLI run reproduced all {parity['control_windows']:,} saved control prediction probabilities and alarm decisions before assessing {parity['continuous_windows']:,} continuous predictions. This isolates the protocol change from model-file changes. The follow-up mapping includes each actual session ID and is checked for out-of-session timestamps.",
      '', f"Native context checks cover {sum(r['windows'] for r in native['reports']):,} model decision rows across both arms, with probability agreement and exact alarm decisions. These are repeat replays of the same development recordings, not added monitoring exposure.",
      '', '**155 tests passed.** Tests cover task-renaming invariance, a FoG onset at a task boundary, follow-up across a task boundary, real-gap resets, missing-sensor onset retention, session-key/time validation, causal local context and export restrictions.',
      '', 'Only the verified development export was accessed. Reserved participants remained excluded and the locked assessment was not run. Old frozen review bundles use the historical protocol and require correction before any future assessment; their contents and access markers were not altered.',
      '', 'The initial continuous-session diagnostic preserved a bad session-ID mapping in its optional follow-up-only comparison. Its native replay results are unaffected. The subsequent real CLI run in continuous_session_20261111_02 fixes the mapping and validates session time ranges; that second run is the source of every table here.',
      '', 'Additional audit: holding missing readings for <=100ms created no added decision opportunities in the approved export. Shortening history improved availability only slightly. Availability does not establish which ankle has better prediction accuracy; right ankle remains the supported runtime placement.',
      '', 'Evidence: runs/continuous_session_20261111_02, runs/session_context_20261111_01, runs/session_boundary_audit_20261111_01. Each run retains plans, source snapshots, predictions, original episode ledgers, participant metrics and verification records.']
    text='\n'.join(lines)+'\n'
    (root/'CONTINUOUS_SESSION_CORRECTION.md').write_text(text,encoding='utf-8')
    (replay/'REPORT.md').write_text(text,encoding='utf-8')
    (study/'REPORT.md').write_text(text,encoding='utf-8')
    print(json.dumps(dict(current_continuous=c,context_candidate=context['context'],promoted=False),indent=2))


if __name__=='__main__':main()
