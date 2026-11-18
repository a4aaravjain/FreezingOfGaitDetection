"""Evaluation-only completion: unchanged frozen models/policies after a numeric check bug."""
from pathlib import Path
import json, shutil
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from fog_pipeline import save_json,sha256
from nested_rescue_experiment import make_data,scored
from regularity_features import regularity_features
from regularity_experiment import alarm_trace,reduce_results


def main():
    root = Path(__file__).resolve().parent
    old = root/'runs/regularity_20261111_01'
    out = root/'runs/regularity_20261111_02'
    out.mkdir(exist_ok=False)
    parent = json.loads((old/'preregistered_plan.json').read_text())
    save_json(out/'preregistered_plan.json',dict(parent,
        original_plan_hash=sha256(old/'preregistered_plan.json'),
        correction='Evaluation-only completion from exactly the frozen weights/policies. Batch/single tolerance rtol1e-5 atol1e-6 (consistent with native replay); previous rtol1e-7 atol1e-12 was inappropriate for float32 differences below 1e-8. No refits, policy selection or architecture change.',
        completion_source_sha256={name:sha256(root/name) for name in ['complete_regularity_experiment.py','regularity_stream.py','verify_regularity_stream.py']}))
    (out/'source').mkdir()
    for path in (old/'source').iterdir():
        shutil.copy2(path,out/'source'/path.name)
    for name in ['complete_regularity_experiment.py','regularity_stream.py','verify_regularity_stream.py']:
        shutil.copy2(root/name,out/'source'/name)
    source = Path(parent['data'])
    assert sha256(source) == parent['data_sha256']
    frame = pd.read_csv(source)
    assert set(frame.subjectID) == set(parent['allowed_subjects'])
    all_results = []
    with threadpool_limits(limits=2):
        for fold, outer in enumerate(parent['groups'],1):
            original = old/f'fold_{fold}'
            folder = out/f'fold_{fold}'
            folder.mkdir()
            for path in original.iterdir():
                if path.name != 'FAILURE.json':
                    shutil.copy2(path,folder/path.name)
            policy = json.loads((folder/'FROZEN_POLICY.json').read_text())
            assert policy['outer'] == outer
            test = make_data(frame,outer)
            test['entropy'] = np.concatenate([test['motion'],regularity_features(test['data'].x)],1)
            people, results = [], {}
            for key, model_hash in policy['hashes'].items():
                path = folder/(key+'.joblib')
                assert sha256(path) == model_hash
                model = joblib.load(path)
                feature = key.split('_')[0]
                p = model.predict_proba(test[feature])[:,1]
                np.testing.assert_allclose(p[:20],np.array([model.predict_proba(x[None])[0,1] for x in test[feature][:20]]),rtol=1e-5,atol=1e-6)
                trace = alarm_trace(test,p,policy['policies'][key])
                trace.to_csv(folder/(key+'_predictions.csv'),index=False)
                result = scored(test,trace)
                save_json(folder/(key+'_results.json'),result)
                results[key] = result
                for subject in outer:
                    small = dict(records=[r for r in test['records'] if r.subject==f'FS{subject:02}'],
                        inventory=[r for r in test['inventory'] if r['subject']==f'FS{subject:02}'],incomplete=False)
                    m = scored(small,trace[trace.subject==subject])['metrics']
                    people.append(dict(fold=fold,subject=subject,model=key,**m))
            for feature, family in policy['selected'].items():
                results[feature+'_selected'] = results[feature+'_'+family]
            pd.DataFrame(people).to_csv(folder/'participants.csv',index=False)
            all_results.append(dict(fold=fold,outer=outer,results=results,selected=policy['selected']))
            save_json(out/'all_folds.json',all_results)
            print('RESULT',fold,{k:(v['metrics']['true_warning_events'],v['metrics']['total_false_alarms']) for k,v in results.items()},flush=True)
    aggregates = {key:reduce_results([r['results'][key] for r in all_results]) for key in all_results[0]['results']}
    save_json(out/'aggregate_results.json',aggregates)
    save_json(out/'execution_completed.json',dict(folds_attempted=4,completed=len(all_results),
        promoted=False,default_changed=False,reserved_used=False,delivered_latency_measured=False,
        fitting_performed=False,calibration_repeated=False))
    print(json.dumps(aggregates,indent=2),flush=True)


if __name__ == '__main__':
    main()
