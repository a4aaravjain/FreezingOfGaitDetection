"""Fixed privileged-sensor teacher/student development study. Single-ankle inference."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier,HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from fog_pipeline import sha256,save_json
from nested_rescue_experiment import make_data,scored
from long_history_experiments import summaries
from regularity_features import regularity_features
from regularity_experiment import GROUPS,partitions,select_threshold,alarm_trace,reduce_results
from distilled_risk import DistilledRisk


def right_features(part):
    return np.concatenate([part['motion'],regularity_features(part['data'].x)],1)


def training_only_extra(frame,part):
    """Extra sensor inputs only for fitting people, at identical trailing timestamps.

    No imputation across time: an unavailable extra-sensor window stays missing.
    Right-ankle windows/targets remain the reference; none are removed here.
    """
    lookup = {}
    columns = [side+'_'+sensor+'_'+axis for side in ['ankleL','back'] for sensor in ['acc','gyro'] for axis in 'xyz']
    for ids,g in frame.groupby(['subjectID','sessionID','taskID'],sort=True):
        lookup[f'FS{int(ids[0]):02}:{ids[1]}:{ids[2]}'] = (g.timestamp.to_numpy(),g[columns].to_numpy(np.float32))
    raw = np.full((len(part['meta']),180,12),np.nan,np.float32)
    for i,row in enumerate(part['meta'].itertuples()):
        t,x = lookup[row.segment]
        end = int(row.chunk.rsplit(':',1)[-1])+row.end_row
        if end<179 or end>=len(x) or abs(t[end]-row.time)>1e-8:
            raise ValueError('Privileged window alignment failed')
        raw[i] = x[end-179:end+1]
    features = []
    for start in [0,6]:
        a = raw[:,:,start:start+6]
        valid = np.isfinite(a).all((1,2))
        f = np.full((len(a),120),np.nan,np.float32)
        if valid.any():
            f[valid] = np.concatenate([summaries(a[valid]),regularity_features(a[valid])],1)
        features.append(f)
    return np.concatenate(features,1)


def main():
    root = Path(__file__).resolve().parent
    out = root/'runs/privileged_distillation_20261111_01'
    out.mkdir(exist_ok=False)
    parent = root/'runs/regularity_20261111_02/preregistered_plan.json'
    plan = json.loads(parent.read_text())
    names = ['privileged_distillation.py','distilled_risk.py','test_privileged_distillation.py','regularity_experiment.py','regularity_features.py',
        'regularity_stream.py','verify_regularity_stream.py','assessment_core.py','nested_rescue_experiment.py','fog_pipeline.py',
        'fogstar_adapter.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py']
    save_json(out/'preregistered_plan.json',dict(plan,
        study='Privileged multisensor teacher and right-ankle-only student; historical development adaptation, not a replication',
        research='https://pubmed.ncbi.nlm.nih.gov/40030782/',
        motivation='PhysioGPN reports knowledge distillation to reduce the number of sensors; this uses fixed small boosted trees, not that published neural architecture.',
        source_sha256={n:sha256(root/n) for n in names},
        teacher='HGB classifier 100 iterations,7 leaves,min40,L2=5,seed42; right+left ankle+back trailing features; natural prevalence; no early stopping.',
        students='HGB regressors with same fixed settings: direct target y control versus .5*y+.5*teacher risk. Both use only 120 right-ankle features. Teacher targets are in-sample fits on fitting participants, standard supervised distillation; no generalization claim from teacher fit accuracy.',
        calibration='Same fixed participant folds and grid; only right-ankle input for both calibration and outer prediction. Zero unknown starts required for a safe selected policy. No family selection.',
        variants=['direct','distilled'],additional_fits=12,
        parent_plan_sha256=sha256(parent)))
    (out/'source').mkdir()
    for n in names:
        shutil.copy2(root/n,out/'source'/n)
    source = Path(plan['data'])
    assert sha256(source) == plan['data_sha256']
    frame = pd.read_csv(source)
    assert set(frame.subjectID)==set(range(1,19))
    results = []
    parameters = dict(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42)
    with threadpool_limits(limits=2):
        for fold,outer in enumerate(GROUPS,1):
            folder = out/f'fold_{fold}'
            folder.mkdir()
            fitids,calids = partitions(outer)
            save_json(folder/'partitions.json',dict(fit=fitids,calibration=calids,outer=outer))
            print('FOLD',fold,flush=True)
            fit,cal = make_data(frame,fitids),make_data(frame,calids)
            x,cx = right_features(fit),right_features(cal)
            # Explicitly restrict the privileged helper to fitting participants.
            extra = training_only_extra(frame[frame.subjectID.isin(fitids)],fit)
            mask = fit['meta'].known_followup & ((fit['meta'].target>=0)|fit['meta'].active_fog)
            y = fit['meta'].target.to_numpy().copy()
            y[fit['meta'].active_fog] = 0
            if set(y[mask])!={0,1}:
                raise ValueError('Fit lacks classes; no corrective repartition authorized by plan')
            teacher = HistGradientBoostingClassifier(**parameters).fit(np.concatenate([x,extra],1)[mask],y[mask])
            joblib.dump(teacher,folder/'teacher.joblib')
            tp = teacher.predict_proba(np.concatenate([x,extra],1)[mask])[:,1]
            del extra
            policies,models = {},{}
            for key,target in [('direct',y[mask]),('distilled',.5*y[mask]+.5*tp)]:
                model = DistilledRisk(HistGradientBoostingRegressor(**parameters).fit(x[mask],target))
                joblib.dump(model,folder/(key+'.joblib'))
                restored = joblib.load(folder/(key+'.joblib'))
                cp = model.predict_proba(cx)[:,1]
                np.testing.assert_array_equal(cp,restored.predict_proba(cx)[:,1])
                threshold,grid = select_threshold(cal,cp)
                pd.DataFrame(grid).to_csv(folder/(key+'_calibration_grid.csv'),index=False)
                policies[key],models[key] = threshold,restored
                print(key,'threshold',threshold,flush=True)
            # Compatibility with the native replay wrapper; aliases point to the same frozen students.
            for key in models:
                shutil.copy2(folder/(key+'.joblib'),folder/('entropy_'+key+'.joblib'))
            save_json(folder/'FROZEN_POLICY.json',dict(policies={'entropy_'+k:v for k,v in policies.items()},
                selected={'entropy':'distilled'},outer=outer,
                hashes={'entropy_'+k:sha256(folder/('entropy_'+k+'.joblib')) for k in models},
                privileged_training_participants=fitids,inference_sensors=['ankleR'],calibration_input_sensors=['ankleR']))
            test = make_data(frame,outer)
            tx = right_features(test)
            r,people = {},[]
            for key,model in models.items():
                p = model.predict_proba(tx)[:,1]
                trace = alarm_trace(test,p,policies[key])
                trace.to_csv(folder/(key+'_predictions.csv'),index=False)
                if key=='distilled':
                    trace.to_csv(folder/'entropy_distilled_predictions.csv',index=False)
                r[key] = scored(test,trace)
                save_json(folder/(key+'_results.json'),r[key])
                for subject in outer:
                    sub = dict(records=[a for a in test['records'] if a.subject==f'FS{subject:02}'],
                        inventory=[a for a in test['inventory'] if a['subject']==f'FS{subject:02}'],incomplete=False)
                    people.append(dict(subject=subject,model=key,**scored(sub,trace[trace.subject==subject])['metrics']))
            pd.DataFrame(people).to_csv(folder/'participants.csv',index=False)
            results.append(dict(fold=fold,outer=outer,results=r))
            save_json(out/'all_folds.json',results)
            print('RESULT',fold,{k:(v['metrics']['true_warning_events'],v['metrics']['total_false_alarms']) for k,v in r.items()},flush=True)
    save_json(out/'aggregate_results.json',{k:reduce_results([r['results'][k] for r in results]) for k in ['direct','distilled']})
    save_json(out/'execution_completed.json',dict(completed=4,additional_fits=12,default_changed=False,reserved_used=False))


if __name__ == '__main__':
    main()
