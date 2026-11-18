"""All prespecified comparison reporting; no model/policy tuning in this module."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from scipy.stats import chi2
from fog_pipeline import save_json,sha256

ROOT=Path(__file__).resolve().parent;OUT=ROOT/'runs/published_cascade_20261111_01'


def estimator_configuration(value):
    if hasattr(value,'get_params'):
        return dict(estimator=type(value).__module__+'.'+type(value).__name__,
            parameters={k:estimator_configuration(v) for k,v in value.get_params(deep=False).items()})
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,(list,tuple)):return [estimator_configuration(v) for v in value]
    if isinstance(value,dict):return {str(k):estimator_configuration(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):return value
    return repr(value)


def markdown_table(rows):
    if not rows:return 'No completed comparisons.'
    keys=list(rows[0]);lines=['| '+' | '.join(keys)+' |','|'+'---|'*len(keys)]
    for row in rows:
        values=[format(row[k],'.5g') if isinstance(row[k],float) else str(row[k]) for k in keys]
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)


def rate_interval(n,hours):
    if hours<=0:return [None,None]
    return [0. if n==0 else float(.5*chi2.ppf(.025,2*n)/hours),float(.5*chi2.ppf(.975,2*(n+1))/hours)]


def aggregate(items,people):
    m=[i['metrics'] for i in items];n=sum(x['total_fog_episodes'] for x in m);tp=sum(x['true_warning_events'] for x in m)
    false=sum(x['total_false_alarms'] for x in m);hours=sum(x['known_monitored_seconds'] for x in m)/3600
    leads=[lead for x in m for lead in x['warning_times_seconds']]
    ps=[p['episode_sensitivity'] for p in people if p['episode_sensitivity'] is not None]
    return dict(episodes=n,detected=tp,episode_sensitivity=tp/n if n else None,
        participant_macro_sensitivity=float(np.mean(ps)) if ps else None,participants=len(people),participants_with_onsets=len(ps),
        unsuccessful_starts=false,known_monitored_hours=hours,total_monitored_hours=sum(x['monitored_seconds'] for x in m)/3600,
        unsuccessful_per_known_hour=false/hours if hours else None,descriptive_poisson95=rate_interval(false,hours),
        unknown_starts=sum(x['censored_alarm_starts'] for x in m),
        false_alert_seconds=sum(x['false_alarm_seconds'] for x in m),
        false_alert_occupancy=sum(x['false_alarm_seconds'] for x in m)/sum(x['monitored_seconds'] for x in m) if sum(x['monitored_seconds'] for x in m) else None,
        warning_precision=tp/(tp+false) if tp+false else None,
        total_alert_seconds=sum(x['total_alarm_seconds'] for x in m),max_pulse_seconds=max(x['maximum_alarm_duration_seconds'] for x in m),
        nominal_leads_seconds=leads,minimum_nominal_lead=min(leads) if leads else None,
        median_nominal_lead=float(np.median(leads)) if leads else None,mean_nominal_lead=float(np.mean(leads)) if leads else None)


def ledger(events,trace,records):
    rows=[]
    successful={(e['segment'],round(e['onset']-e['warning_seconds'],8)):e for e in events if e['warning_seconds'] is not None}
    lookup={r['record']:r for r in records}
    for row in trace[trace.alarm_start].itertuples():
        matched=successful.get((row.segment,round(row.time,8)))
        rows.append(dict(segment=row.segment,time=float(row.time),outcome='successful' if matched else 'unmatched',
            onset=matched['onset'] if matched else None,lead=matched['warning_seconds'] if matched else None))
    return pd.DataFrame(rows)


def main():
    data=json.loads((OUT/'all_folds.json').read_text());completed=[f for f in data if 'results' in f]
    if not completed:raise ValueError('No completed folds; report actual technical failures, never invent metrics')
    native=json.loads((OUT/'native_verification.json').read_text()) if (OUT/'native_verification.json').exists() else dict(folds=[])
    names=sorted({n for f in completed for n in f['results']})
    perpeople={f['fold']:pd.read_csv(OUT/f"fold_{f['fold']}"/'participants.csv').replace({np.nan:None}) for f in completed}
    summary={};tables=[];rejections=[];calibration_constraints=[]
    import zipfile
    configs={}
    for path in OUT.rglob('*.keras'):
        with zipfile.ZipFile(path) as archive:
            configs[str(path.relative_to(OUT))]=json.loads(archive.read('config.json'))
    save_json(OUT/'saved_neural_model_configurations.json',configs)
    import joblib
    save_json(OUT/'saved_classical_model_configurations.json',
        {str(path.relative_to(OUT)):estimator_configuration(joblib.load(path)) for path in OUT.rglob('*.joblib')})
    for name in names:
        folds=[f for f in completed if name in f['results']];people=[];paired=[]
        for f in folds:
            p=perpeople[f['fold']];people.extend(p[p.model==name].to_dict('records'));paired.extend(p[p.model=='matched_v2'].to_dict('records'))
        s=aggregate([f['results'][name] for f in folds],people)
        paired_base=aggregate([f['results']['matched_v2'] for f in folds],paired)
        s.update(covered_folds=[f['fold'] for f in folds],complete_all_outer_folds=len(folds)==4,
            paired_matched_baseline=paired_base)
        summary[name]=s
        tables.append(dict(model=name,folds=len(folds),episodes=s['episodes'],detected=s['detected'],
            macro=s['participant_macro_sensitivity'],unsuccessful=s['unsuccessful_starts'],known_hours=s['known_monitored_hours'],
            unsuccessful_per_known_hour=s['unsuccessful_per_known_hour'],unknown=s['unknown_starts']))
    save_json(OUT/'aggregate_results.json',summary);pd.DataFrame(tables).to_csv(OUT/'all_comparisons.csv',index=False)
    base=summary['matched_v2'];final=summary['baseline_OR_best_rescue']
    baseline_events={};final_events={};baseline_preserved=True;full_failures=[];episode_rows=[];alert_rows=[];exposures=[];candidates=[]
    for fold in completed:
        number=fold['fold'];folder=OUT/f'fold_{number}';old=fold['results']['matched_v2'];new=fold['results']['baseline_OR_best_rescue']
        for key,result,store in [('matched_v2',old,baseline_events),('baseline_OR_best_rescue',new,final_events)]:
            for e in result['events']:
                episode_rows.append(dict(fold=number,model=key,**e))
                if e['warning_seconds'] is not None:store[(e['segment'],e['onset'])]=e['warning_seconds']
        a=pd.read_csv(folder/'matched_v2_predictions.csv');b=pd.read_csv(folder/'baseline_OR_best_rescue_predictions.csv')
        baseline_preserved&=bool(np.all(~a.alarm_start.to_numpy(bool)|b.alarm_start.to_numpy(bool)))
        baseline_preserved&=bool(np.array_equal(a.alarm_start.to_numpy(),b.baseline_start.to_numpy()))
        # Full annotation records are reconstructed only from the verified export, never reserved data.
        from published_cascade_core import load_export,part_from_frame
        from report_session_correction import ledger as classify_alerts
        part=part_from_frame(load_export(),fold['outer'])
        for name,result in fold['results'].items():
            trace=pd.read_csv(folder/(name+'_predictions.csv'))
            classified=classify_alerts(part['records'],trace)
            for r in classified.to_dict('records'):alert_rows.append(dict(fold=number,model=name,**r))
            pd.DataFrame(result['events']).to_csv(folder/(name+'_episodes.csv'),index=False)
            m=result['metrics'];exposures.append(dict(fold=number,model=name,known_seconds=m['known_monitored_seconds'],
                total_seconds=m['monitored_seconds'],unsuccessful_starts=m['total_false_alarms'],unknown_starts=m['censored_alarm_starts']))
        candidates.append(dict(fold=number,**json.loads((folder/'outer_level1_metrics.json').read_text())))
    pd.DataFrame(episode_rows).to_csv(OUT/'paired_episode_ledger.csv',index=False)
    pd.DataFrame(alert_rows).to_csv(OUT/'all_alert_outcomes.csv',index=False)
    pd.DataFrame(exposures).to_csv(OUT/'false_alarm_exposure.csv',index=False)
    pd.DataFrame(candidates).to_csv(OUT/'candidate_metrics.csv',index=False)
    for fold in completed:
        folder=OUT/f"fold_{fold['fold']}";frozen=json.loads((folder/'FROZEN_CASCADE.json').read_text())
        for key,policy in frozen['verifiers'].items():
            grid=json.loads((folder/policy['model_dir']/(policy['mode']+'_thresholds.json')).read_text())['combined']
            floor=next(row for row in grid if row['threshold']>1)
            adding=sum(bool(row['safe'] and row['gain']>0) for row in grid)
            calibration_constraints.append(dict(fold=fold['fold'],verifier=key,selected=key==frozen['selected'],
                combined_threshold=policy['combined'],safe_adding_thresholds=adding,
                baseline_unsuccessful_starts=floor['total_false_alarms'],
                baseline_known_hours=floor['known_monitored_seconds']/3600,
                baseline_unsuccessful_per_known_hour=floor['unsuccessful_per_known_hour'],
                disabled_reason=('preserved_baseline_already_exceeds_warning_budget' if floor['unsuccessful_per_known_hour'] is not None and floor['unsuccessful_per_known_hour']>=1 else 'no_adding_policy_meets_all_constraints') if policy['combined']>1 else None))
            ledger_frame=pd.read_csv(folder/(key+'_verifier_ledger.csv'));mask=ledger_frame.candidate.to_numpy(bool);p=ledger_frame.probability.to_numpy();n=int(sum(mask))
            rejections.append(dict(fold=fold['fold'],verifier=key,candidates=n,
                rejection_fixed05=float(np.mean(p[mask]<.5)) if n else None,
                rejection_standalone=float(np.mean(p[mask]<policy['alone'])) if n else None,
                rejection_combined=float(np.mean(p[mask]<policy['combined'])) if n else None,
                standalone_disabled=policy['alone']>1,combined_disabled=policy['combined']>1))
    pd.DataFrame(rejections).to_csv(OUT/'verifier_rejection_rates.csv',index=False)
    pd.DataFrame(calibration_constraints).to_csv(OUT/'calibration_policy_constraints.csv',index=False)
    added=set(final_events)-set(baseline_events);addedpeople={k[0].split(':')[0] for k in added}
    latency=pd.read_csv(OUT/'native_latency.csv') if (OUT/'native_latency.csv').exists() else pd.DataFrame()
    adjusted=[]
    for key,lead in final_events.items():
        time=key[1]-lead
        matches=latency[(latency.segment==key[0])&np.isclose(latency.time,time,atol=1e-8,rtol=0)] if len(latency) else []
        ms=float(matches.iloc[0].effective_output_ms) if len(matches) else None
        adjusted.append(dict(segment=key[0],onset=key[1],nominal_lead=lead,compute_ms=ms,
            desktop_compute_adjusted_lead=lead-ms/1000 if ms is not None else None))
    pd.DataFrame(adjusted).to_csv(OUT/'compute_adjusted_leads.csv',index=False)
    active_failures={str(f['fold']):json.loads((OUT/f"fold_{f['fold']}"/'FROZEN_CASCADE.json').read_text()).get('failures',{}) for f in completed}
    save_json(OUT/'final_active_fit_failures.json',active_failures)
    gates=dict(all_folds_complete=len(completed)==4,
        all_prespecified_fit_comparisons_completed=len(completed)==4 and not any(active_failures.values()) and all(len(json.loads((OUT/f"fold_{f['fold']}"/'FROZEN_CASCADE.json').read_text())['verifiers'])==6 and len(json.loads((OUT/f"fold_{f['fold']}"/'FROZEN_CASCADE.json').read_text())['ablations'])==5 for f in completed),
        baseline_actual_starts_preserved=baseline_preserved,
        prior_episode_warnings_not_lost_or_later=all(k in final_events and final_events[k]>=v-1e-6 for k,v in baseline_events.items()),
        two_added_episodes=len(added)>=2,additions_in_two_people=len(addedpeople)>=2,
        unsuccessful_rate_below_one=final['unsuccessful_per_known_hour'] is not None and final['unsuccessful_per_known_hour']<1,
        unknown_alerts_not_increased=final['unknown_starts']<=base['unknown_starts'],
        participant_macro_improved=final['participant_macro_sensitivity'] is not None and final['participant_macro_sensitivity']>base['participant_macro_sensitivity'],
        pulses_one_second=final['max_pulse_seconds']<=1+1e-6,
        candidate_occupancy_below_40=all(x['candidate_positive_occupancy']<=.4+1e-6 for x in candidates),
        native_reload_parity=len(native['folds'])==4 and all(f['upstream_score_parity'] and f['selected_physical_parity'] and f['all_full_verifier_combined_decisions_exact'] for f in native['folds']),
        streaming_latency=len(native['folds'])==4 and all(f['over_250ms']==0 for f in native['folds']),
        desktop_adjusted_lead_two_seconds=bool(adjusted) and all(r['desktop_compute_adjusted_lead'] is not None and r['desktop_compute_adjusted_lead']>=2-1e-9 for r in adjusted),
        actual_delivered_lead_verified=False)
    operational={}
    f4=next((f for f in completed if f['outer']==[15,16,17,18]),None)
    if f4 and 'original_saved_v2_OR_rescue' in f4['results']:
        a=f4['results']['original_saved_v2'];b=f4['results']['original_saved_v2_OR_rescue']
        aa={(e['segment'],e['onset']):e['warning_seconds'] for e in a['events'] if e['warning_seconds'] is not None}
        bb={(e['segment'],e['onset']):e['warning_seconds'] for e in b['events'] if e['warning_seconds'] is not None}
        added_orig=set(bb)-set(aa)
        pa=pd.read_csv(OUT/'fold_4/original_saved_v2_predictions.csv');pb=pd.read_csv(OUT/'fold_4/original_saved_v2_OR_rescue_predictions.csv')
        operational=dict(original_v2_metrics=a['metrics'],combined_metrics=b['metrics'],added_episodes=len(added_orig),
            added_people=sorted({k[0].split(':')[0] for k in added_orig}),
            original_actual_starts_preserved=bool(np.all(~pa.alarm_start.to_numpy(bool)|pb.alarm_start.to_numpy(bool))))
        gates['original_saved_v2_starts_preserved']=operational['original_actual_starts_preserved']
        gates['original_saved_v2_combined_below_one']=b['metrics']['unsuccessful_per_known_hour'] is not None and b['metrics']['unsuccessful_per_known_hour']<1
        gates['original_saved_v2_two_additions_two_people']=len(added_orig)>=2 and len(operational['added_people'])>=2
    else:gates['original_saved_v2_operational_evaluation_complete']=False
    save_json(OUT/'original_saved_v2_operational_comparison.json',operational)
    # Test results must be explicitly supplied by the completed suite, not inferred from model scores.
    tests=OUT/'final_tests.log';gates['causality_split_reset_leakage_tests']=tests.exists() and '\nOK' in tests.read_text()
    promotion=dict(promoted=all(gates.values()),gates=gates,failed=[k for k,v in gates.items() if not v],
        added_episodes=len(added),added_people=sorted(addedpeople),locked_assessment_executed=False,default_v2_weights_changed=False)
    save_json(OUT/'promotion_gate.json',promotion)
    # Descriptive cluster bootstrap; repeats do not create independent assessment evidence.
    rng=np.random.default_rng(42);p=perpeople[completed[0]['fold']]
    allp=pd.concat(list(perpeople.values()),ignore_index=True);bp=allp[allp.model=='matched_v2'];fp=allp[allp.model=='baseline_OR_best_rescue']
    ids=sorted(fp.subject.unique());draws=[]
    for _ in range(2000):
        sample=rng.choice(ids,size=len(ids),replace=True);aa=bp.set_index('subject').loc[sample];bb=fp.set_index('subject').loc[sample]
        n=float(bb.total_fog_episodes.sum())
        if n:draws.append(float(bb.true_warning_events.sum()-aa.true_warning_events.sum())/n)
    save_json(OUT/'uncertainty.json',dict(episode_sensitivity_difference_participant_bootstrap95=[float(x) for x in np.quantile(draws,[.025,.975])] if draws else None,
        seed=42,replicates=2000,limitations='Descriptive participant resampling; overlapping fits, few people, clustered episodes and historical development tuning preclude an independent clinical confidence claim.',
        baseline_false_rate_poisson95=base['descriptive_poisson95'],combined_false_rate_poisson95=final['descriptive_poisson95']))
    if promotion['promoted']:opening='The cascade passed every promotion requirement and improved development performance.'
    elif len(added)>0 and final['detected']>base['detected']:opening='The cascade improved some metrics but failed one or more promotion requirements.'
    else:opening='The cascade did not improve development performance.'
    lines=[opening,'', '# Published-model rescue cascade: complete development results','',
        'This is corrected physical-session, nested participant development. Reserved participants remained excluded; no locked assessment was run. All architectures, recipes, folds, calibration splits, thresholds and ablations were frozen before fitting. No additional experiment was selected after outer results.',
        '', '## Main paired comparison','',
        '| System | Early episodes | Participant macro | Unsuccessful starts | Known hours | Starts/known hour | Unknown starts |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in ['matched_v2','baseline_OR_best_rescue']:
        s=summary[name];lines.append(f"| {name} | {s['detected']}/{s['episodes']} | {s['participant_macro_sensitivity']} | {s['unsuccessful_starts']} | {s['known_monitored_hours']:.4f} | {s['unsuccessful_per_known_hour']:.3f} | {s['unknown_starts']} |")
    lines += ['', 'The nested baseline is the fixed v2 architecture/training recipe refitted without each outer group. Original saved v2 trained people1-14 and therefore cannot be a subject-independent baseline on those people. Its unchanged weights are separately compared with the selected rescue only on historical people15-18; see fold4 original_saved_v2 results. The frozen baseline path is logically independent; physical OR collisions and extended pulses are explicitly tested rather than called preserved merely because requests were ORed.',
        '', '## Candidate and verifier results','',
        'candidate_metrics.csv reports opportunity recall against every original onset, eligible-window recall and known-time candidate occupancy. risk_blind_capacity_control.json in each fold exposes whether candidate coverage exceeds a trivial equal-capacity selector. A causal25%/33.3% capacity gate prevents distribution shift from producing an always-positive candidate stream. Failure to reach95% is retained and does not skip verifier training.',
        '', 'All prespecified standalone components, training recipes, fixed.5 diagnostic policies, raw/Platt verifiers, calibrated standalone/combined policies and external/motion/history ablations are tabulated in all_comparisons.csv. verifier_rejection_rates.csv distinguishes classifier rejection at fixed.5 from calibration-disabled policies. Missing arms remain explicitly partial with paired baseline coverage; they are never silently scored as zero or compared to a different denominator. Disabled safe policies are failures to find useful warnings, not successful low-false-alarm models.',
        '', 'calibration_policy_constraints.csv identifies whether the preserved baseline already exceeded the inner warning budget or whether no adding rescue policy met the complete constraints. These are distinct failure modes: an additive verifier cannot repair an already excessive preserved baseline warning rate.',
        '', '## Technical recovery','',
        'Initial fold-one verifier attempts stopped before fitting because parent output directories were absent. Later attempts saved fitted weights and passed reload checks, then failed while serializing NumPy participant IDs to JSON. These failed attempts are retained under technical_attempts. Completion creates the missing directories and converts metadata scalars to native JSON values. It reuses saved weights only when model family and the complete training-membership/label ledger match; any initial fallback-family ablation is archived and the originally prespecified calibration-selected family is fitted. Cached decimal scores are read with round-trip precision. No upstream weights, training labels, hyperparameters, folds, threshold grids or experiment choices were changed. See bookkeeping_recovery_ledger.json.',
        '', '## Source adaptations and training','',
        'The external_source_license_audit.json and EXTERNAL_SOURCE_AUDIT.md record URLs, exact commits, papers, licenses, sensor assumptions and adaptations. Stanford BSD3 and ResNet MIT notices are retained. Unlicensed FoGSTAR/GaitFM source is not imported or copied; architectural/masked-learning ideas are independently implemented. External detection scores and externally trained weights are not used as early-warning evidence.',
        '', 'FoGSTAR/Stanford causal CNNs and compact ResNet variants use right-ankle acceleration(g) and gyroscope(degrees/s) at60Hz, with fit-only normalization. ResNet recipe comparisons add participant/episode-balanced focal training, fit-only hard-negative mining, rotation/noise/scale augmentation, then fit-only masked pretraining. They never change event timing. The two-stage TCN verifier replaces symmetric refinement with left-causal convolutions over current/past score features.',
        '', 'Sensor placement: these fitted models require the right ankle and the same six-channel axis/unit convention used by the export. The official single-ankle architecture originally used a left-ankle signal; adapting its topology does not make left/right inputs interchangeable. This experiment does not compare ankle sides, so it cannot establish which side is intrinsically best. Neither ankle has been established as a dependable real-world early-warning placement by this development experiment.',
        '', 'Each verifier fitting candidate comes from an upstream model excluding that participant, the outer group and both calibration groups. Its candidate cutoff is calibrated using that same upstream model on separate policy-calibration people. Unknown tails remain unknown. Active/late/postFoG/stopping/turning negatives are training annotations only; they are not inference inputs. Platt calibration uses separate people from operating-policy selection.',
        '', '## Promotion and uncertainty','',
        '```json',json.dumps(promotion,indent=2),'```','',
        'All uncertainty estimates are descriptive. Poisson intervals assume a homogeneous warning rate; the participant bootstrap does not undo repeated historical tuning or overlapping fits. Short known-monitor exposure cannot establish a dependable population rate below one/hour even if observed unsuccessful starts are zero.',
        '', 'Training-strategy contrasts are prespecified sequential recipe comparisons, not a factorial decomposition. Balancing/focal loss is one bundle, and the mining recipe includes two additional hard-negative epochs. Improvements cannot be attributed to a single bundled ingredient in isolation.',
        '', 'Lead times are actual algorithm-output start timestamps relative to annotations, with measured desktop computation additionally subtracted in compute_adjusted_leads.csv. Sensor/communication/actuator delays are unmeasured, so the required two-second delivered warning is unverified. Model setup/compilation on synthetic zeros is reported separately from live-stream inference latency.',
        '', 'All failures, training histories, model configurations/provenance, candidate ledgers, per-window scores, per-alert and per-episode outcomes, per-person metrics, threshold grids, exposure totals and reload/native checks are retained in this directory. The saved v2 weights remain unchanged.']
    if len(latency):
        lines+=['','## Streaming latency','',f"Selected live cascade median {float(latency.compute_ms.median()):.3f}ms; p95 {float(np.percentile(latency.compute_ms,95)):.3f}ms; maximum {float(latency.compute_ms.max()):.3f}ms. Setup/compilation and sensor/transport/actuator delay are excluded and reported separately."]
    if operational:
        om=operational['original_v2_metrics']
        lines+=['','## Constraint imposed by preserving baseline warnings','',
            f"The original saved baseline already emits {om['total_false_alarms']} unsuccessful starts on this historical cohort ({om['unsuccessful_per_known_hour']:.3f} per known monitored hour). Under fixed annotations and the same warning horizon, preserving every physical baseline start prevents an additive rescue path from removing that existing unsuccessful-warning burden. A useful rescue can add true warnings, but this OR-only design cannot reach a lower false-start rate than its preserved baseline floor on these recordings. Changing the baseline warning policy would require a separately prespecified study; no such change was made here."]
    lines+=['','## All frozen comparisons','',markdown_table(tables)]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(opening);print(json.dumps(promotion,indent=2))


if __name__=='__main__':main()
