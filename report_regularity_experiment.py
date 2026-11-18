"""Summarize frozen negative results; never selects or changes a model/policy."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from fog_pipeline import save_json,sha256
from nested_rescue_experiment import make_data,scored
from candidate_gate import compare_candidates


def main():
    root = Path(__file__).resolve().parent
    out = root/'runs/regularity_20261111_02'
    plan = json.loads((out/'preregistered_plan.json').read_text())
    agg = json.loads((out/'aggregate_results.json').read_text())
    parity = json.loads((out/'native_parity.json').read_text())
    source = Path(plan['data'])
    assert sha256(source) == plan['data_sha256']
    frame = pd.read_csv(source)
    assert set(frame.subjectID) == set(range(1,19))
    part = make_data(frame,[15,16,17,18])
    folder = out/'fold_4'
    policy = json.loads((folder/'FROZEN_POLICY.json').read_text())
    new = pd.read_csv(folder/('entropy_'+policy['selected']['entropy']+'_predictions.csv'))
    old = pd.read_csv(root/'runs/right_ankle_waveform_v2/predictions.csv')
    np.testing.assert_array_equal(old.segment,new.chunk)
    np.testing.assert_allclose(old.time,new.time,rtol=0,atol=1e-8)
    baseline = new.copy()
    baseline['alarm_start'],baseline['alarm_active'] = old.alarm_start.to_numpy(),old.alarm_active.to_numpy()
    previous,current = scored(part,baseline),scored(part,new)
    save_json(out/'original_v2_comparison.json',dict(baseline=previous,candidate=current,
        comparison=compare_candidates(previous['metrics'],previous['events'],current['metrics'],current['events']),
        note='Original v2 was trained on people1-14. Its saved weights are only compared on the same historical people15-18; no original-weight score on its own training people is reported. Entropy model here uses people1-10 fit,11-14 calibration. This is an operational candidate comparison, not an isolated feature ablation. Matched feature ablations use identical fit/calibration people in every fold.'))
    people = []
    for subject in range(15,19):
        sub = dict(records=[r for r in part['records'] if r.subject==f'FS{subject:02}'],
            inventory=[r for r in part['inventory'] if r['subject']==f'FS{subject:02}'],incomplete=False)
        a = scored(sub,baseline[baseline.subject==subject])['metrics']
        b = scored(sub,new[new.subject==subject])['metrics']
        people.append(dict(subject=subject,episodes=a['total_fog_episodes'],
            baseline_detected=a['true_warning_events'],candidate_detected=b['true_warning_events'],
            baseline_sensitivity=a['episode_sensitivity'],candidate_sensitivity=b['episode_sensitivity']))
    pd.DataFrame(people).to_csv(out/'original_v2_participants.csv',index=False)
    baseline_events = {(e['segment'],e['onset']):e['warning_seconds'] for e in previous['events'] if e['warning_seconds'] is not None}
    candidate_events = {(e['segment'],e['onset']):e['warning_seconds'] for e in current['events'] if e['warning_seconds'] is not None}
    added = set(candidate_events)-set(baseline_events)
    macro = float(np.mean([p['candidate_sensitivity']-p['baseline_sensitivity'] for p in people if p['episodes']]))
    a,b = previous['metrics'],current['metrics']
    s = agg['entropy_selected']
    gates = dict(all_four_folds_complete=len(parity['folds'])==4,
        at_least_90_percent_all_episodes=s['sensitivity']>=.9,
        unsuccessful_rate_below_one=s['false_starts_per_hour']<1,
        zero_unknown_starts=s['unknown_starts']==0,
        two_added_episodes=len(added)>=2,
        additions_in_two_people=len({k[0].split(':')[0] for k in added})>=2,
        previous_warnings_not_lost_or_later=all(k in candidate_events and candidate_events[k]>=v-1e-6 for k,v in baseline_events.items()),
        burdens_not_increased=all(b[k]<=a[k]+1e-6 for k in ['total_false_alarms','false_alarm_seconds','total_alarm_seconds','censored_alarm_starts','maximum_alarm_duration_seconds']),
        participant_macro_improved=macro>0,
        native_probability_and_decisions=all(r['probability_parity'] and r['decisions_exact'] for r in parity['folds']),
        computation_under_250ms=all(r['over_250ms']==0 for r in parity['folds']),
        measured_delivered_lead_two_seconds=False)
    save_json(out/'promotion_gate.json',dict(promoted=all(gates.values()),gates=gates,
        failed=[k for k,v in gates.items() if not v],added_episodes=len(added),participant_macro_change=macro,
        reason_for_latency_failure='No sensor/transport/actuator measurements; no successful candidate alerts to assess.',
        disabled_candidate_is_not_a_useful_predictor=True,default_changed=False))
    tradeoffs = []
    for fold in range(1,5):
        f = out/f'fold_{fold}'
        for path in sorted(f.glob('*_calibration_grid.csv')):
            grid = pd.read_csv(path)
            detected = grid[grid.true_warning_events>0]
            if len(detected):
                best = detected.sort_values(['total_false_alarms','true_warning_events','threshold'],ascending=[True,False,False]).iloc[0]
                tradeoffs.append(dict(fold=fold,model=path.name.replace('_calibration_grid.csv',''),
                    threshold=float(best.threshold),detections=int(best.true_warning_events),
                    episodes=int(best.total_fog_episodes),false_starts=int(best.total_false_alarms),
                    false_per_hour=float(best.false_alarms_per_hour),unknown_starts=int(best.censored_alarm_starts)))
    pd.DataFrame(tradeoffs).to_csv(out/'calibration_tradeoff_diagnostic.csv',index=False)
    fits = [json.loads(p.read_text()) for p in out.glob('fold_*/*_fit.json')]
    rows = ['# Gait regularity development experiment','',
      'The new entropy models did not improve the warning system. No candidate was promoted; waveform v2 remains the default.',
      '', '## Research and bounded experiment','',
      'Sample entropy and freeze-index features were motivated by [the 2019 Sensors study](https://www.mdpi.com/1424-8220/19/18/3898). That paper is not evidence that this project meets a two-second early-warning requirement. This is an adaptation, not a reproduction of its results.',
      '', 'Six descriptors (sample, permutation and spectral entropy, autocorrelation peak, freeze-band ratio and amplitude) are calculated from gravity-relative right-ankle signals in the trailing 3 seconds and its two halves. Half-to-half differences describe changes in regularity. Sample entropy uses m=2, r=.2 SD, 20Hz block means and a .5 count correction for sparse matches. No feature uses future samples, labels or participant ID.',
      '', f'{len(fits)} fits: two fixed model families (unweighted logistic regression and histogram gradient boosting), with 24 existing features versus those features plus 96 regularity descriptors. Four participant outer folds; each has separate fitting and calibration participants. No extra seeds or threshold search followed evaluation. Calibration alone selects family and alarm threshold.',
      '', '## Results','',
      '| Frozen policy | Early episodes | Unsuccessful starts | Unknown starts | Monitored hours |',
      '|---|---:|---:|---:|---:|']
    for key in ['motion_selected','entropy_selected']:
        m = agg[key]
        rows.append(f"| {key} | {m['detected']}/{m['episodes']} | {m['false_starts']} | {m['unknown_starts']} | {m['monitored_hours']:.4f} |")
    rows += ['', 'Every fitted family failed to find a calibration setting with a useful detection, strictly fewer than one unsuccessful warning per hour, zero unknown starts and one-second pulses. The predefined fallback disabled all candidate alarms (threshold1.1). Zero alerts therefore mean failure to find a useful model, not successful alarm suppression. All 72 original onsets remain in the denominator, including onsets hidden by missing sensors.',
      '', 'Calibration tradeoffs are retained in calibration_tradeoff_diagnostic.csv and the complete per-fold grids. They are diagnostic results only; thresholds were not changed after seeing outer outcomes.',
      '', f"On the original v2 historical comparison cohort, the new policy warned {b['true_warning_events']}/{b['total_fog_episodes']} episodes versus v2's {a['true_warning_events']}/{a['total_fog_episodes']}. It loses previously warned episodes and fails promotion. The 72-episode nested aggregate is a different evaluation from the original 16-episode v2 result; their sensitivities are not directly comparable.",
      '', '## Verification and limitations','',
      f"Native replay checked {sum(r['windows'] for r in parity['folds']):,} prediction windows: probability agreement within rtol1e-5/atol1e-6 and exact output decisions in all four folds. Maximum measured prediction computation was {max(r['max_ms'] for r in parity['folds']):.2f} ms. Predictions are computed even when the frozen alarm policy is disabled. Sensor, transport and actuator delays are unmeasured; no delivered-lead claim is made.",
      '', 'The first execution retained all fitted artifacts but stopped evaluation in each fold because a batch-versus-single float32 check demanded excessive precision. Differences were below 1e-8 in the failed checks. The separate completion run uses precisely those saved weights and policies, with the native replay tolerance. No retraining, calibration changes, folds or failed artifacts were discarded.',
      '', 'These recordings have been repeatedly studied, so this is historical development evidence rather than independent assessment. Strictly zero unsuccessful starts here cannot establish a population rate below one per hour; the zero-count descriptive Poisson upper limit over this short exposure is 3.18/h and is not a validation of a disabled policy.',
      '', 'Only the verified development-only export was read. Reserved participants remained excluded. The original combined file and locked assessment were not accessed. Right-ankle input remains the supported placement; this trial did not compare left versus right.',
      '', 'The evidence does not justify claiming near-perfect early prediction. A useful next study would need a distinct, prespecified source of information and substantially more participant-diverse continuous monitoring; repeated fitting to these same outcomes is not independent evidence.',
      '', 'See promotion_gate.json, original_v2_comparison.json, per-participant results, native_parity.json and preregistered_plan.json for the reviewable details.']
    (out/'REPORT.md').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    registry = {str(p.relative_to(out)):sha256(p) for p in out.rglob('*') if p.is_file() and p.name!='artifact_registry.json'}
    save_json(out/'artifact_registry.json',registry)
    print(json.dumps(dict(fits=len(fits),gates=gates,default_changed=False),indent=2))


if __name__ == '__main__':
    main()
