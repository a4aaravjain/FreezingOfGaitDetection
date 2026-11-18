"""Post-freeze reporting; no fitting, policy changes or raw combined-file access."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from rescue_cascade import poisson_interval
from fog_pipeline import save_json,sha256
ROOT=Path(__file__).resolve().parent

def alert_ledger(trace,events,record_ends):
 rows=[];by={}
 for e in events:by.setdefault(e['segment'],[]).append(e)
 for segment,g in trace.groupby('segment',sort=False):
  used=set();onsets=sorted(e['onset'] for e in by.get(segment,[]));end=record_ends[segment]
  for row in g[g.alarm_start].itertuples():
   following=[o for o in onsets if o>row.time];onset=following[0] if following else None;lead=onset-row.time if onset is not None else None
   if not row.active_fog and lead is not None and 2-1e-9<=lead<=5+1e-9 and onset not in used:outcome='successful';used.add(onset)
   elif not row.active_fog and lead is not None and lead<2:outcome='late_unsuccessful'
   elif not row.active_fog and lead is not None and lead<=5:outcome='duplicate_unsuccessful'
   elif not row.active_fog and row.time+5>end:outcome='boundary_censored'
   else:outcome='unsuccessful'
   rows.append(dict(segment=segment,subject=int(row.subject),time=row.time,outcome=outcome,lead_seconds=lead))
 return rows

def main():
 out=ROOT/'runs/nested_rescue_20261111_02'
 if any((out/n).exists() for n in ['REPORT.md','aggregate_results.json','promotion_gate.json']):raise FileExistsError('Preserve existing report artifacts')
 folds=json.loads((out/'all_folds.json').read_text());assert (out/'execution_completed.json').exists();raw=pd.read_csv(ROOT/'development_only_20261111_01.csv',usecols=['subjectID','sessionID','taskID','timestamp','fog']);record_ends={f'FS{int(key[0]):02}:{key[1]}:{key[2]}':float(g.timestamp.max()) for key,g in raw.groupby(['subjectID','sessionID','taskID'])};coverage=[]
 for key,g in raw.groupby('subjectID'):
  original=sum(int(((rr.fog.to_numpy()[1:]==1)&(rr.fog.to_numpy()[:-1]==0)).sum()) for _,rr in g.groupby(['sessionID','taskID']))
  coverage.append(dict(subject=int(key),original_onsets=original,fold_completed=any('results' in f and int(key) in f['outer'] for f in folds)))
 save_json(out/'fold_coverage.json',dict(participants=coverage,all_development_onsets=sum(x['original_onsets'] for x in coverage),completed_fold_onsets=sum(x['original_onsets'] for x in coverage if x['fold_completed']),full_cohort_sensitivity_estimate=None if any(not x['fold_completed'] for x in coverage) else 'see aggregate'))
 aggregate={};gains=[];reasons=[];all_added=[];baseline_events={};combined_events={};participants=[]
 if any('failure' in f for f in folds):reasons.append('one_or_more_incomplete_folds')
 for fold in folds:
  if 'results' not in fold:continue
  folder=out/f"fold_{fold['fold']}";results=fold['results']
  for name,result in results.items():
   m=result['metrics'];a=aggregate.setdefault(name,dict(episodes=0,warned=0,false_starts=0,unknown_starts=0,monitored_seconds=0.,alert_seconds=0.,false_alert_seconds=0.,max_pulse_seconds=0.,leads=[]))
   for dst,src in [('episodes','total_fog_episodes'),('warned','true_warning_events'),('false_starts','total_false_alarms'),('unknown_starts','censored_alarm_starts'),('monitored_seconds','monitored_seconds'),('alert_seconds','total_alarm_seconds'),('false_alert_seconds','false_alarm_seconds')]:a[dst]+=m[src]
   a['max_pulse_seconds']=max(a['max_pulse_seconds'],m['maximum_alarm_duration_seconds']);a['leads']+=m['warning_times_seconds'];trace=pd.read_csv(folder/(name+'_predictions.csv'));
   if name=='matched_v2':
    save_json(folder/'exposure_breakdown.json',dict(total_monitored_hours=float(trace.duration.sum()/3600),known_normal_outside_target_hours=float(trace.loc[trace.target.eq(0),'duration'].sum()/3600),known_preonset_normal_hours=float(trace.loc[trace.target.eq(1),'duration'].sum()/3600),known_active_fog_hours=float(trace.loc[trace.active_fog & trace.known_followup,'duration'].sum()/3600),insufficient_followup_hours=float(trace.loc[~trace.known_followup,'duration'].sum()/3600)))
   pd.DataFrame(alert_ledger(trace,result['events'],record_ends)).to_csv(folder/(name+'_alert_ledger.csv'),index=False);pd.DataFrame(result['events']).to_csv(folder/(name+'_episodes.csv'),index=False)
  b=results['matched_v2'];c=results['baseline_OR_selected_rescue'];bm=b['metrics'];cm=c['metrics'];be={(e['segment'],e['onset']):e['warning_seconds'] for e in b['events'] if e['warning_seconds'] is not None};ce={(e['segment'],e['onset']):e['warning_seconds'] for e in c['events'] if e['warning_seconds'] is not None};baseline_events.update(be);combined_events.update(ce);added=set(ce)-set(be);all_added+=list(added)
  if not c['physical_baseline_preserved']:reasons.append('baseline_physical_start_changed')
  if cm['censored_alarm_starts']>bm['censored_alarm_starts']:reasons.append('unknown_alerts_increased')
  if cm['maximum_alarm_duration_seconds']>1+1e-6:reasons.append('pulse_exceeded_one_second')
  if cm['false_alarm_seconds']>bm['false_alarm_seconds']+1e-6:reasons.append('false_occupancy_increased')
  if abs(cm['monitored_seconds']-bm['monitored_seconds'])>1e-6 or cm['total_fog_episodes']!=bm['total_fog_episodes']:reasons.append('denominator_or_exposure_changed')
  if not all(k in ce and ce[k]>=v-1e-6 for k,v in be.items()):reasons.append('previous_warning_lost_or_delayed')
  byperson={p['subject']:p for p in b['participants']}
  for person in c['participants']:
   previous=byperson[person['subject']];p=person.copy();p['fold']=fold['fold'];p['baseline_warned']=previous['true_warning_events'];p['additional_warned']=p['true_warning_events']-previous['true_warning_events'];participants.append(p)
   if person['false_alarms_per_hour'] is None or person['false_alarms_per_hour']>=1:reasons.append('participant_false_rate_not_below_one')
   if person['episode_sensitivity'] is not None:gains.append(person['episode_sensitivity']-previous['episode_sensitivity'])
 for name,a in aggregate.items():
  a['sensitivity']=a['warned']/a['episodes'] if a['episodes'] else None;a['false_starts_per_hour']=a['false_starts']*3600/a['monitored_seconds'] if a['monitored_seconds'] else None;a['false_rate95_poisson']=poisson_interval(a['false_starts'],a['monitored_seconds']/3600);a['mean_nominal_lead']=float(np.mean(a['leads'])) if a['leads'] else None
 if len(all_added)<2:reasons.append('fewer_than_two_added_episodes')
 added_people={k[0].split(':')[0] for k in all_added}
 if len(added_people)<2:reasons.append('added_episodes_not_in_two_people')
 if not gains or np.mean(gains)<=0:reasons.append('no_participant_macro_gain')
 selected=aggregate.get('baseline_OR_selected_rescue')
 if selected is None or selected['false_starts_per_hour'] is None or selected['false_starts_per_hour']>=1:reasons.append('aggregate_false_rate_not_below_one')
 eligible_for_streaming_check=not reasons
 # Full native-sample replay/latency is mandatory before any promotion.
 
 if not (out/'native_streaming_parity.json').exists():reasons.append('full_native_cascade_streaming_not_verified')
 reasons.append('physical_transport_and_actuator_latency_not_measured')
 gate=dict(promoted=False,eligible_for_streaming_check=eligible_for_streaming_check,reasons=sorted(set(reasons)),additional_episodes=len(all_added),additional_participants=len(added_people),participant_macro_sensitivity_change=float(np.mean(gains)) if gains else None,limited_nested_development=True,clinical_validation=False)
 save_json(out/'aggregate_results.json',aggregate);save_json(out/'promotion_gate.json',gate);pd.DataFrame(participants).to_csv(out/'combined_participants.csv',index=False)
 lines=['# Nested rescue-cascade development results','','Source: verified development_only_20261111_01.csv; original combined CSV was not used for fitting/evaluation. Reserved rows were excluded during the separately authorized metadata-only export. This is historical nested development evidence, not an independent assessment.','','## Comparison (completed folds only; failed folds remain unassessed)','', '| Arm | Early episodes | Scored unsuccessful starts | Monitored hours | Starts/hour | 95% Poisson interval /h | Unknown alerts |','|---|---:|---:|---:|---:|---:|---:|']
 for name,a in aggregate.items():lines.append(f"| {name} | {a['warned']}/{a['episodes']} | {a['false_starts']} | {a['monitored_seconds']/3600:.4f} | {a['false_starts_per_hour']} | {a['false_rate95_poisson']} | {a['unknown_starts']} |")
 lines+=['','The matched baseline is fold-fitted v2 architecture with unchanged alert settings, not the original weights evaluated on people they trained on. Original saved v2 remains unchanged and its historical 7/16 result is not directly comparable to this all-development nested aggregate. Inner models use two disjoint participant groups for cross-fitted verifier training; every upstream fit excludes the scored participant, all outer people and all inner-calibration people. Head normalization is training-only. All final policies and model hashes are written before outer prediction.','', '## Promotion decision','',json.dumps(gate,indent=2),'','No cascade is promoted. Disabled policies are recorded failures to find a rescue, not successful zero-alert predictors. The selected-family ablations were chosen using inner calibration only; outer folds did not drive family/threshold revision. No fallback threshold search was conducted.','','Exact original annotation onset denominators include those hidden by missing sensors. Per-fold candidate opportunity recall, known-followup occupancy, candidate runs/hour, negative candidate labels and boundary exclusions are saved. All alerts are timed at the actual physical OR rising edge. Logical baseline requests are independent, but OR overlap can hide a physical rising edge; these cases are rejected by the preservation check.','','Poisson 95% intervals are descriptive homogeneous-rate calculations and do not correct clustering, historical tuning or distribution shift. Short exposure cannot establish a dependable sub-one/hour rate. Per-person false rates are also subjected to a conservative <1/hour gate; this numerical concentration rule is a conservative reporting check, not an independently preregistered confirmatory threshold. Nominal window-end lead is not physical delivered lead: acquisition/communication/actuator delays are not measured. Any successful matched event obeys the 2ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã¢â‚¬Å“5-second scoring interval; that is not clinical usefulness.','','## Reproducibility','', 'Preregistration, exact partitions, source snapshots, upstream weights/normalization and histories, OOF provenance, training-candidate ledgers, threshold grids, outer score ledgers, per-window decisions, per-alert outcomes, per-episode and per-person results, and reload checks are retained in fold folders. Fold exceptions are preserved in FAILURE.json and make the aggregate incomplete. fold_coverage.json lists the entire development onset inventory and explicitly identifies unassessed participants; partial-fold ratios must not be presented as full-cohort sensitivity. Core causal/reset/leakage/export tests passed before fitting. Batch-versus-one-tick verifier score parity is checked for every outer window; this is not complete native-sample streaming parity.','','Commands: `python export_development_only.py --source sensor_data.csv --output development_only_20261111_01.csv` (already completed; never overwrite export); `python nested_rescue_experiment.py` (new run refuses overwrite); `python nested_rescue_report.py`. Export manifest records allowed IDs, permitted row count, column schema, command, script and output SHA-256. Original file content hash was deliberately not calculated because it would read reserved signal/label bytes.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(gate,indent=2))
if __name__=='__main__':main()
