"""Fixed promotion accounting for privileged training; no model selection."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fog_pipeline import sha256,save_json
from nested_rescue_experiment import make_data,scored
from candidate_gate import compare_candidates


def main():
    root = Path(__file__).resolve().parent
    out = root/'runs/privileged_distillation_20261111_01'
    plan = json.loads((out/'preregistered_plan.json').read_text())
    agg = json.loads((out/'aggregate_results.json').read_text())
    parity = json.loads((out/'native_parity.json').read_text())
    source = Path(plan['data'])
    assert sha256(source) == plan['data_sha256']
    frame = pd.read_csv(source)
    assert set(frame.subjectID)==set(range(1,19))
    part = make_data(frame,[15,16,17,18])
    candidate = pd.read_csv(out/'fold_4/distilled_predictions.csv')
    old = pd.read_csv(root/'runs/right_ankle_waveform_v2/predictions.csv')
    np.testing.assert_array_equal(candidate.chunk,old.segment)
    np.testing.assert_allclose(candidate.time,old.time,rtol=0,atol=1e-8)
    baseline = candidate.copy()
    baseline['alarm_active'],baseline['alarm_start'] = old.alarm_active.to_numpy(),old.alarm_start.to_numpy()
    a,b = scored(part,baseline),scored(part,candidate)
    comparison = compare_candidates(a['metrics'],a['events'],b['metrics'],b['events'])
    save_json(out/'original_v2_comparison.json',dict(baseline=a,candidate=b,comparison=comparison,
        interpretation='Operational historical-cohort comparison only. Original v2 fits people1-14; this student fits1-10 and calibrates11-14. Direct/distilled student ablations have identical participant splits.'))
    oldevents = {(r['segment'],r['onset']):r['warning_seconds'] for r in a['events'] if r['warning_seconds'] is not None}
    newevents = {(r['segment'],r['onset']):r['warning_seconds'] for r in b['events'] if r['warning_seconds'] is not None}
    added = set(newevents)-set(oldevents)
    oldpeople,newpeople = {},{}
    for event in a['events']:
        subject = event['segment'].split(':')[0]
        oldpeople.setdefault(subject,[]).append(event['warning_seconds'] is not None)
    for event in b['events']:
        subject = event['segment'].split(':')[0]
        newpeople.setdefault(subject,[]).append(event['warning_seconds'] is not None)
    macro = float(np.mean([np.mean(newpeople[p])-np.mean(v) for p,v in oldpeople.items()]))
    m = agg['distilled']
    ma,mb = a['metrics'],b['metrics']
    gates = dict(all_folds_complete=len(parity['folds'])==4,
        at_least_90_percent_episodes=m['sensitivity']>=.9,
        unsuccessful_rate_below_one=m['false_starts_per_hour']<1,
        zero_unknown_starts=m['unknown_starts']==0,
        two_added_episodes=len(added)>=2,
        additions_in_two_people=len({p[0].split(':')[0] for p in added})>=2,
        previous_warnings_not_lost_or_later=all(k in newevents and newevents[k]>=v-1e-6 for k,v in oldevents.items()),
        burden_not_increased=all(mb[k]<=ma[k]+1e-6 for k in ['total_false_alarms','false_alarm_seconds','total_alarm_seconds','censored_alarm_starts','maximum_alarm_duration_seconds']),
        participant_macro_improved=macro>0,
        native_parity=all(r['probability_parity'] and r['decisions_exact'] for r in parity['folds']),
        computation_under_250ms=all(r['over_250ms']==0 for r in parity['folds']),
        measured_delivered_lead_two_seconds=False)
    save_json(out/'promotion_gate.json',dict(promoted=all(gates.values()),gates=gates,
        failures=[k for k,v in gates.items() if not v],added_episodes=len(added),
        participant_macro_change=macro,default_changed=False,clinical_validation=False))
    tradeoffs = []
    for path in out.glob('fold_*/*_calibration_grid.csv'):
        f = pd.read_csv(path)
        d = f[f.true_warning_events>0]
        if len(d):
            best = d.sort_values(['total_false_alarms','true_warning_events','threshold'],ascending=[True,False,False]).iloc[0]
            tradeoffs.append(dict(fold=path.parent.name,model=path.stem,threshold=float(best.threshold),
                detections=int(best.true_warning_events),episodes=int(best.total_fog_episodes),
                false_starts=int(best.total_false_alarms),unknown_starts=int(best.censored_alarm_starts),
                false_per_hour=float(best.false_alarms_per_hour)))
    pd.DataFrame(tradeoffs).to_csv(out/'calibration_tradeoff_diagnostic.csv',index=False)
    lines = ['# Privileged-sensor distillation development experiment','',
        'No model is promoted unless every prespecified gate passes. See promotion_gate.json for the computed decision.',
        '', 'This follows [PhysioGPN research on transferring knowledge from multiple sensors to fewer sensors](https://pubmed.ncbi.nlm.nih.gov/40030782/). It is a small boosted-tree adaptation of that principle, not a replication of the published neural network or a claim of matching its results.',
        '', 'Four fixed participant outer folds each separate fitting, calibration and evaluation people. The teacher uses trailing right-ankle, left-ankle and back features from fitting people only. The distilled student receives right-ankle features and learns a fixed 50/50 mixture of the true target and teacher risk. The matched direct student learns the original target with the same regressor and right-ankle features. There are 12 fits total: four teachers and eight students. Teacher fit predictions are in-sample supervised distillation targets, not evidence of held-participant accuracy.',
        '', 'Both calibration and evaluation use only the right ankle. Extra-sensor windows align to exactly the same trailing timestamps, including after right-sensor gaps. Missing extra windows remain missing and never remove a reference right-ankle window or original episode. Training, normalizers, thresholds and runtime never access reserved data.',
        '', '| Policy | Early episodes | Unsuccessful starts | Unknown starts | Monitored hours |',
        '|---|---:|---:|---:|---:|']
    for key in ['direct','distilled']:
        r = agg[key]
        lines.append(f"| {key} | {r['detected']}/{r['episodes']} | {r['false_starts']} | {r['unknown_starts']} | {r['monitored_hours']:.4f} |")
    disabled = all(json.loads((out/f'fold_{f}/FROZEN_POLICY.json').read_text())['policies']['entropy_distilled']==1.1 for f in range(1,5))
    lines += ['', f'All distilled policies disabled by calibration: {disabled}. A disabled policy has no useful detections; its zero alarms are not evidence of useful prediction. The calibrated false-warning limit is strictly below one per monitored hour, with no unknown starts. No threshold changes followed outer evaluation.',
        '', f"On the historical v2 comparison people15-18, distilled student warns {mb['true_warning_events']}/{mb['total_fog_episodes']} episodes versus original v2 {ma['true_warning_events']}/{ma['total_fog_episodes']}. The nested 72-episode aggregate is a different population from this 16-episode operational comparison; do not directly compare those sensitivities.",
        '', f"Native replay checks {sum(r['windows'] for r in parity['folds']):,} decision windows, with probability tolerance rtol1e-5/atol1e-6 and exact alarm decisions. Worst measured desktop prediction computation: {max(r['max_ms'] for r in parity['folds']):.2f}ms. Teacher files and extra sensor channels are not loaded by the native student. Computation measurements exclude sensor/transport/actuator delays, so actual delivered lead is unverified.",
        '', 'Only the verified development export was read; reserved participants remained excluded. These are repeatedly used historical development recordings, not independent assessment. Right ankle remains the supported runtime placement; this does not establish that the right side is best for every person.',
        '', 'Artifacts include fixed plans/source snapshots, frozen student/teacher models, all calibration grids, full outer prediction traces, original episode inventories, per-participant metrics, calibration tradeoff diagnostics and promotion gates.']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps(dict(aggregate=agg,promoted=all(gates.values())),indent=2))


if __name__ == '__main__':
    main()
