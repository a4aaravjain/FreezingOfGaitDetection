"""Bounded local-context study after fixing physical-session continuity."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd,joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
from fog_pipeline import sha256,save_json
from development_data import load_development,EXPORT,EXPECTED_SHA256
from session_assessment import make_session_data,score_sessions
from long_history_experiments import summaries
from causal_motion_context import CausalMotionContext
from regularity_experiment import GROUPS,partitions,select_threshold,alarm_trace,reduce_results
from nested_rescue_experiment import CFG


def prepared(frame,ids):
    part=make_session_data(frame,ids,CFG)
    part['motion']=summaries(part['data'].x)
    return part


def contextual_features(part,floor):
    result=np.empty((len(part['meta']),73),np.float32)
    for _,g in part['meta'].groupby('chunk',sort=False):
        context=CausalMotionContext(floor)
        for i in g.index:result[i]=context.step(part['motion'][i])
    return result


def main():
    root=Path(__file__).resolve().parent;out=root/'runs/session_context_20261111_01'
    out.mkdir(exist_ok=False)
    names=['context_session_experiment.py','causal_motion_context.py','test_causal_motion_context.py',
        'session_assessment.py','test_session_assessment.py','development_data.py','regularity_experiment.py',
        'nested_rescue_experiment.py','assessment_core.py','fog_pipeline.py','long_history_experiments.py','pretraining_experiments.py']
    save_json(out/'preregistered_plan.json',dict(
        study='20-second past gait reference versus identical raw-summary control, on corrected continuous physical sessions',
        context='Historical development follow-on after a pipeline defect; not independent validation or replication of a published method.',
        motivation='Within-person gait changes may transfer better than absolute gait level; test explicitly instead of inferring from prior patient-dependent research.',
        research='https://www.mdpi.com/1424-8220/19/18/3898',
        development_export=str(EXPORT),data_sha256=EXPECTED_SHA256,groups=GROUPS,
        fit_calibration='Same fixed four folds: last four non-outer people calibrate; remaining people fit. No outer-person adaptation/labels.',
        features='raw24 control versus raw24+current-minus-past-median24+past-IQR-normalized24+history-fraction1; only previous80 prediction summaries. No future, label gating or task-ID features. Reset only at real time/sensor gaps.',
        floor='max(.1*fit-participant feature SD, .001); fitted only on supervised fitting windows',
        classifier=dict(type='HGB',max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2=5,early_stopping=False,seed=42),
        weighting='Natural prevalence; active FoG is a non-precursor negative; exclude unknown follow-up.',
        policy='Use exactly the preceding fixed11-threshold calibration grid; <1 unsuccessful/hour, zero unknown starts, <=1s pulse. No safe useful calibration detection means disabled. No selection across arms.',
        target='At least90% original onsets, no increased burdens, >=2 added episodes across>=2 people, preserve previous warnings/leads, native parity; no promotion without every existing requirement including delivered lead.',
        trials=8,source_sha256={n:sha256(root/n) for n in names}))
    (out/'source').mkdir()
    for n in names:shutil.copy2(root/n,out/'source'/n)
    frame=load_development(range(1,19));all_results=[]
    with threadpool_limits(limits=2):
        for fold,outer in enumerate(GROUPS,1):
            folder=out/f'fold_{fold}';folder.mkdir()
            fitids,calids=partitions(outer)
            save_json(folder/'partitions.json',dict(fit=fitids,calibration=calids,outer=outer))
            print('FOLD',fold,flush=True)
            fit,cal=prepared(frame,fitids),prepared(frame,calids)
            mask=fit['meta'].known_followup&((fit['meta'].target>=0)|fit['meta'].active_fog)
            y=fit['meta'].target.to_numpy().copy();y[fit['meta'].active_fog]=0
            if set(y[mask])!={0,1}:raise ValueError('Fit lacks classes; no repartition or retry')
            floor=np.maximum(.1*fit['motion'][mask].std(0),.001)
            fx,cx=contextual_features(fit,floor),contextual_features(cal,floor)
            models,thresholds={},{}
            for key,x,v in [('raw',fit['motion'],cal['motion']),('context',fx,cx)]:
                model=HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,
                    l2_regularization=5,early_stopping=False,random_state=42).fit(x[mask],y[mask])
                joblib.dump(model,folder/(key+'.joblib'));restored=joblib.load(folder/(key+'.joblib'))
                p=model.predict_proba(v)[:,1];np.testing.assert_array_equal(p,restored.predict_proba(v)[:,1])
                threshold,grid=select_threshold(cal,p)
                pd.DataFrame(grid).to_csv(folder/(key+'_calibration_grid.csv'),index=False)
                models[key],thresholds[key]=restored,threshold
                print(key,'threshold',threshold,flush=True)
            save_json(folder/'FROZEN_POLICY.json',dict(outer=outer,thresholds=thresholds,floor=floor.tolist(),
                model_hashes={k:sha256(folder/(k+'.joblib')) for k in models},reload_parity=True))
            test=prepared(frame,outer);tx=contextual_features(test,floor)
            results,people={},[]
            for key,x in [('raw',test['motion']),('context',tx)]:
                p=models[key].predict_proba(x)[:,1];trace=alarm_trace(test,p,thresholds[key])
                trace.to_csv(folder/(key+'_predictions.csv'),index=False)
                result=score_sessions(test['records'],trace,test['inventory'],test['incomplete'],CFG)
                results[key]=result;save_json(folder/(key+'_results.json'),result)
                for subject in outer:
                    records=[r for r in test['records'] if r.subject==f'FS{subject:02}']
                    inventory=[r for r in test['inventory'] if r['subject']==f'FS{subject:02}']
                    m=score_sessions(records,trace[trace.subject==subject],inventory,False,CFG)['metrics']
                    people.append(dict(subject=subject,model=key,**m))
            pd.DataFrame(people).to_csv(folder/'participants.csv',index=False)
            all_results.append(dict(fold=fold,outer=outer,results=results))
            save_json(out/'all_folds.json',all_results)
            print('RESULT',fold,{k:(r['metrics']['true_warning_events'],r['metrics']['total_false_alarms']) for k,r in results.items()},flush=True)
    save_json(out/'aggregate_results.json',{k:reduce_results([r['results'][k] for r in all_results]) for k in ['raw','context']})
    save_json(out/'execution_completed.json',dict(folds=4,fits=8,reserved_participants_excluded=True,
        physical_session_protocol=True,promoted=False,model_weights_default_changed=False))


if __name__=='__main__':main()
