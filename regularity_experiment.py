"""One bounded, participant-nested follow-on development study, not locked assessment."""
from pathlib import Path
from dataclasses import replace, asdict
import json, shutil, time
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from fog_pipeline import Alarm, save_json, sha256
from nested_rescue_experiment import make_data, scored, CFG
from regularity_features import regularity_features
from rescue_cascade import poisson_interval

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'runs/regularity_20261111_01'
GROUPS = [[1,2,3,4], [5,6,7,8], [9,10,11,12,13,14], [15,16,17,18]]
THRESHOLDS = [.01, .025, .05, .1, .2, .35, .5, .7, .9, .99, 1.1]


def partitions(outer):
    remaining = sorted(set(range(1,19))-set(outer))
    fit, cal = remaining[:-4], remaining[-4:]
    assert not (set(fit)&set(cal) or set(fit)&set(outer) or set(cal)&set(outer))
    return fit, cal


def alarm_trace(part, probabilities, threshold):
    f = part['meta'].copy()
    f['probability'] = probabilities
    active = np.zeros(len(f), bool)
    starts = active.copy()
    for _, group in f.groupby('chunk', sort=False):
        alarm = Alarm(threshold, CFG)
        for i, timestamp in zip(group.index, group.time):
            _, active[i], starts[i] = alarm.step(timestamp, float(probabilities[i]))
    f['alarm_start'], f['alarm_active'] = starts, active
    return f


def select_threshold(part, probabilities):
    """Only calibration labels enter this function. Disable when no safe gain."""
    rows = []
    for threshold in THRESHOLDS:
        result = scored(part, alarm_trace(part, probabilities, threshold))
        m = result['metrics']
        safe = (m['false_alarms_per_hour'] is not None and m['false_alarms_per_hour'] < 1
                and m['censored_alarm_starts'] == 0
                and m['maximum_alarm_duration_seconds'] <= 1+1e-6
                and m['alarm_active_fraction'] <= .15)
        rows.append(dict(threshold=threshold, safe=safe, **m))
    eligible = [r for r in rows if r['safe'] and r['true_warning_events'] > 0]
    if not eligible:
        return 1.1, rows
    best = max(eligible, key=lambda r:(r['true_warning_events'], -r['total_false_alarms'],
                r['median_warning_seconds'], r['threshold']))
    return best['threshold'], rows


def fit_model(family, x, y):
    if set(y) != {0,1}:
        raise ValueError('Fit lacks classes: retain failure, no outcome-driven repartition')
    if family == 'logistic':
        model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=42))
    else:
        model = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=7,
                    min_samples_leaf=40, l2_regularization=5,
                    early_stopping=False, random_state=42)
    return model.fit(x, y)  # Natural prevalence: no resampling/class weighting.


def reduce_results(results):
    metrics = [r['metrics'] for r in results]
    hours = sum(m['monitored_seconds'] for m in metrics)/3600
    episodes = sum(m['total_fog_episodes'] for m in metrics)
    detected = sum(m['true_warning_events'] for m in metrics)
    false = sum(m['total_false_alarms'] for m in metrics)
    leads = [v for m in metrics for v in m['warning_times_seconds']]
    return dict(episodes=episodes, detected=detected, sensitivity=detected/episodes if episodes else None,
                false_starts=false, monitored_hours=hours, false_starts_per_hour=false/hours if hours else None,
                descriptive_poisson95=poisson_interval(false,hours),
                unknown_starts=sum(m['censored_alarm_starts'] for m in metrics),
                alarm_seconds=sum(m['total_alarm_seconds'] for m in metrics),
                false_alarm_seconds=sum(m['false_alarm_seconds'] for m in metrics),
                max_pulse_seconds=max(m['maximum_alarm_duration_seconds'] for m in metrics),
                minimum_nominal_lead=min(leads) if leads else None,
                mean_nominal_lead=float(np.mean(leads)) if leads else None)


def main():
    OUT.mkdir(exist_ok=False)
    source = ROOT/'development_only_20261111_01.csv'
    expected_hash = '2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
    assert sha256(source) == expected_hash
    names = ['regularity_experiment.py','regularity_features.py','test_regularity_features.py','test_regularity_experiment.py',
             'nested_rescue_experiment.py','assessment_core.py','fog_pipeline.py',
             'fogstar_adapter.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py']
    plan = dict(study='Research-motivated entropy features versus matched 24-feature control',
        context='Follow-on historical development only; recordings repeatedly studied. No confirmatory claim.',
        research='https://www.mdpi.com/1424-8220/19/18/3898',
        adaptation='Sample entropy and freeze index motivate a new right-ankle feature set; not a paper replication. Permutation/spectral entropy, autocorrelation, amplitudes and within-window changes are fixed additions.',
        data=str(source), data_sha256=expected_hash, allowed_subjects=list(range(1,19)),
        groups=GROUPS, calibration='last four non-outer participants; all remaining people fit',
        variants=['motion_logistic','motion_hgb','entropy_logistic','entropy_hgb'],
        models=dict(logistic=dict(C=.1,max_iter=1000),hgb=dict(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2=5)),
        weighting='natural prevalence; active freezing is a non-precursor negative; unknown follow-up excluded',
        thresholds=THRESHOLDS, alarm_config=asdict(CFG),
        selection='Per feature set, calibrate each family: maximize episodes with <1 false start/hour, zero unknown starts and <=1s pulses. Ties prefer fewer false starts, greater median lead, higher threshold. If no safe detections disable. Choose family by calibration detections; ties prefer logistic.',
        promotion=['all folds complete','at least 90% of all original episodes warned',
          'strictly below one unsuccessful warning per monitored hour',
          'no unknown starts','at least two added episodes across at least two participants on the original v2 development cohort',
          'no previously warned episode lost or later than original v2',
          'no increase in false starts, false duration, total duration, unknown starts or max pulse versus original v2 on that cohort',
          'positive participant-macro gain versus original v2 on that cohort',
          'native probability/decision parity and no prediction computation >250ms',
          'measured delivered lead >=2 seconds including transport/actuator latency'],
        failed_trials='retained; no extra seeds, windows, thresholds or refits after evaluation',
        source_sha256={n:sha256(ROOT/n) for n in names})
    save_json(OUT/'preregistered_plan.json', plan)
    (OUT/'source').mkdir()
    for name in names:
        shutil.copy2(ROOT/name, OUT/'source'/name)
    frame = pd.read_csv(source)
    assert set(frame.subjectID) == set(range(1,19))
    cache = {}
    def get_part(ids):
        # Deterministic sample-local feature cache. No fitted normalizer or labels used in features.
        for subject in ids:
            if subject not in cache:
                p = make_data(frame, [subject])
                p['entropy'] = np.concatenate([p['motion'],regularity_features(p['data'].x)],1)
                del p['data'], p['signals']
                cache[subject] = p
        parts = [cache[i] for i in ids]
        return dict(meta=pd.concat([p['meta'] for p in parts], ignore_index=True),
            motion=np.concatenate([p['motion'] for p in parts]),
            entropy=np.concatenate([p['entropy'] for p in parts]),
            records=[r for p in parts for r in p['records']],
            inventory=[r for p in parts for r in p['inventory']],
            incomplete=any(p['incomplete'] for p in parts))
    all_results = []
    with threadpool_limits(limits=2):
        for fold, outer in enumerate(GROUPS,1):
            folder = OUT/f'fold_{fold}'
            folder.mkdir()
            fitids, calids = partitions(outer)
            save_json(folder/'partitions.json',dict(fit=fitids,calibration=calids,outer=outer))
            print('FOLD',fold,'fit',fitids,'cal',calids,'outer',outer,flush=True)
            try:
                fit, cal = get_part(fitids), get_part(calids)
                mask = fit['meta'].known_followup & ((fit['meta'].target>=0)|fit['meta'].active_fog)
                y = fit['meta'].target.to_numpy().copy()
                y[fit['meta'].active_fog] = 0
                models, policies, ranks = {}, {}, {}
                for features in ['motion','entropy']:
                    for family in ['logistic','hgb']:
                        key = features+'_'+family
                        started = time.perf_counter()
                        model = fit_model(family,fit[features][mask],y[mask])
                        joblib.dump(model,folder/(key+'.joblib'))
                        model2 = joblib.load(folder/(key+'.joblib'))
                        p = model.predict_proba(cal[features])[:,1]
                        np.testing.assert_allclose(p, model2.predict_proba(cal[features])[:,1],rtol=0,atol=0)
                        threshold, grid = select_threshold(cal,p)
                        pd.DataFrame(grid).to_csv(folder/(key+'_calibration_grid.csv'),index=False)
                        models[key], policies[key] = model2, threshold
                        ranks[key] = next(row['true_warning_events'] for row in grid if row['threshold']==threshold)
                        save_json(folder/(key+'_fit.json'),dict(fit_subjects=fitids,calibration_subjects=calids,
                            rows=int(mask.sum()),positive_rows=int(y[mask].sum()),features=fit[features].shape[1],
                            seconds=time.perf_counter()-started,reload_exact=True))
                        print(key,'threshold',threshold,'cal detections',ranks[key],flush=True)
                selected = {feature:max(['logistic','hgb'],key=lambda f:(ranks[feature+'_'+f],f=='logistic')) for feature in ['motion','entropy']}
                save_json(folder/'FROZEN_POLICY.json',dict(policies=policies,selected=selected,
                    hashes={k:sha256(folder/(k+'.joblib')) for k in models},outer=outer))
                # Only now access this fold's evaluation outcomes.
                test = get_part(outer)
                results, people = {}, []
                for key, model in models.items():
                    feature = key.split('_')[0]
                    p = model.predict_proba(test[feature])[:,1]
                    # Float32 BLAS can round batch/single products differently.
                    np.testing.assert_allclose(p[:20],np.array([model.predict_proba(x[None])[0,1] for x in test[feature][:20]]),rtol=1e-5,atol=1e-6)
                    trace = alarm_trace(test,p,policies[key])
                    trace.to_csv(folder/(key+'_predictions.csv'),index=False)
                    result = scored(test,trace)
                    results[key] = result
                    save_json(folder/(key+'_results.json'),result)
                    for subject in outer:
                        small = get_part([subject])
                        subtrace = trace[trace.subject==subject]
                        m = scored(small,subtrace)['metrics']
                        people.append(dict(fold=fold,subject=subject,model=key,**m))
                for feature, family in selected.items():
                    results[feature+'_selected'] = results[feature+'_'+family]
                pd.DataFrame(people).to_csv(folder/'participants.csv',index=False)
                all_results.append(dict(fold=fold,outer=outer,results=results,selected=selected))
                print('RESULT',fold,{k:(v['metrics']['true_warning_events'],v['metrics']['total_false_alarms']) for k,v in results.items()},flush=True)
            except Exception as error:
                import traceback
                traceback.print_exc()
                save_json(folder/'FAILURE.json',dict(type=type(error).__name__,error=str(error)))
                all_results.append(dict(fold=fold,outer=outer,failure=str(error)))
            save_json(OUT/'all_folds.json',all_results)
    complete = [r for r in all_results if 'results' in r]
    aggregates = {key:reduce_results([r['results'][key] for r in complete]) for key in complete[0]['results']} if complete else {}
    save_json(OUT/'aggregate_results.json',aggregates)
    save_json(OUT/'execution_completed.json',dict(folds_attempted=4,completed=len(complete),
        promoted=False,default_changed=False,reserved_used=False,delivered_latency_measured=False))
    print('AGGREGATE',json.dumps(aggregates,indent=2),flush=True)


if __name__ == '__main__':
    main()
