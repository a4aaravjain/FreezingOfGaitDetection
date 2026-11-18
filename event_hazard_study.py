"""Checkpointed, finite champion/challenger study. No outer-driven tuning."""
from pathlib import Path
import json,shutil,traceback,gc
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits
from event_hazard_audit import ROOT,OUT,PRIOR,save,sha
from event_hazard_data import load_export,load_part,join,normalized
from event_hazard_models import fit,predict,probabilities,EPOCHS,PRESETS
from event_hazard_control import calibrate,choose_temperature,replay
from published_cascade_core import scored

SOURCES=['event_hazard_audit.py','event_hazard_data.py','event_hazard_models.py','event_hazard_control.py','event_hazard_study.py','test_event_hazard.py']


def freeze():
    old=json.loads((PRIOR/'preregistered_plan.json').read_text());path=OUT/'preregistered_plan.json'
    if path.exists():
        p=json.loads(path.read_text());assert all(sha(ROOT/n)==h for n,h in p['source_sha256'].items());return p
    p=dict(study='Event-level causal hazard champion/challenger; final verifier may reject all v2 alerts',folds=old['folds'],
        source_sha256={n:sha(ROOT/n) for n in SOURCES+list(old['source_sha256'])},
        export=old['data'],export_sha256=old['data_sha256'],champion_registry_sha256=sha(PRIOR/'artifact_registry.json'),
        fixed_denominators=dict(episodes=72,known_hours=1.1738981481481512,native_windows=17380),
        candidates='Exact previous per-fold/per-crossfit Level1 score caches and capacity policies; no new upstream fitting, threshold relaxation or OR path. Raw2-5s opportunity67/72, direct pulses65/72; diagnostic oracle is never a live rule.',
        models=dict(families=['tcn','gru','refinement'],history_seconds=20,ticks=80,native_sample_rate=60,stride_seconds=.25,input='Every native15sample block including directional6channels and2magnitudes; frozen450 causal score/shape features,48 energy changes and40 periodicity summaries. Raw-only prefix blocks before the initial3s score window; zero padding never crosses a physical sensor chunk.',
            tcn='24channel six residual left-causal dilations1,2,4,8,16,32',gru='24unit unidirectional GRU',refinement='two16channel left-causal stages, five dilations1,2,4,8,16;24dim embedding',
            heads='4 conditional sigmoid hazards; survival products form probabilities for[0,2),[2,3),[3,4),[4,5],none<=5. CurrentFoG primary labels masked; unknown non-event tails remain unknown.',
            epochs=EPOCHS,checkpoints=[EPOCHS//2,EPOCHS],seed=42,presets=PRESETS),
        bounded_matrix=['tcn event preset0/1','gru event preset0/1','refinement event preset0/1',
            'inner-best event model + auxiliary context','inner-best event model + multisensor teacher distillation',
            'inner-best event model + predetermined wearer calibration','inner-best without event losses','inner-best without hard-topk penalty'],
        ablations='no_context/no_distillation/no_personalization reuse the same inner-best event control for paired comparison with the respective enhancement. Enhancements are independent additions, not a confounded cumulative stack. no_event/no_topk are separately fitted. no_topk removes the additional loss, retaining its mining/sampling procedure.',
        total_fits='11 students/fold (6 family/preset fits+3 enhancements+2 fitted ablations),44 total; one additional24channel-input causal teacher/fold,4 total. Eight fixed epochs; no seed searches.',
        teacher='TCN architecture, selected event-loss preset,context heads,8fixed epochs, fitting participants only. All24 IMU channels and missingness masks, fit-only finite normalization. Never run teacher on C1/C2/outer; no outside data/weights. Temperature2 KL(.5) plus normalized embedding MSE(.05). Deployment student onlyright ankle.',
        context='walking/activity1;turning6/7;standing3withoutFoG as stopping proxy (intention unannotated);activeFoG;past<=5s recovery;trailing autocorrelation<.25 as periodicity proxy. Labels enter losses only, never inference inputs.',
        event_objectives='Hazard NLL plus per-onset candidate-bag maximum, subject/episode-balanced sampling, fitting-only top20peaks/known-negative-hour mining refreshed every2epochs, same-person ranking, late/active/post penalties, soft candidate-start sparsity relative to cached causal previous-tick risk. The latter is an optimization surrogate; complete physical starts determine final calibration and evaluation.',
        normalization='Population mean/std on fitting native-block features only; no whole outer-record stats. Wearer arm uses exactly first20chronological seconds of first session selected without labels; robust median/MAD with .25populationstd floor; no calibrated alerts during setup, no replacement interval, every original onset retained. Missing/split sensor history resets; wearer setup is not restarted or replaced.',
        controller='One-second fixed pulse,3second refractory; optional current+previous-tick confirmation; soft60second-decaying past-alert burden beta0/.5. Thresholds.025,.05,.1,.2,.35,.5,.65,.8,.9,.97,.995,1.1. No hour-long cooldown, label gate or baseline OR. Controller consumes actual previous-final-alert age; no fabricated neural feedback from event labels.',
        selection='C1 selects temperature[.75,1,1.5,2] by known hazard NLL; C2 selects complete-start policy. Checkpoint/preset/family/enhancement ranking prioritizes safe nonzero C2 episode detections, then participantmacro; C1 valid-hazard AP and NLL break all-disabled ties. Only base/context/distilled/personalized arms eligible as final per-fold challenger; ablations diagnostic. All arm decisions frozen before any new outer model predictions.',
        promotion='All mandatory user criteria jointly:>15/72,macro>.12943,extra warned original episodes in>=2people,unsuccessful/knownhour<1,zero unknown,identical denominators,1second pulses,nominal2-5s and compute/scheduling-adjusted>=2s output lead,max desktopcompute<250ms,reload/native/causality/leakage tests. Disabled systems cannot pass. Retain champion on any failure.',
        limitations='Historically reused development data, few episodes/people and short exposure; oracle capacity does not demonstrate feature separability. Standing/periodicity are declared proxies. Transport/actuator delay unmeasured; no claim of dependable real-world cueing.',
        no_extra_trials_after_outer_results=True,reserved_participants_excluded=True,locked_assessment_authorized=False)
    save(path,p);snapshot=OUT/'source';snapshot.mkdir()
    for n in p['source_sha256']:shutil.copy2(ROOT/n,snapshot/n)
    save(OUT/'subject_splits.json',old['folds']);return p


def cached_data(frame,spec,role,folder):
    path=folder/(role+'_data.joblib')
    manifest=folder/(role+'_data_cache.json')
    if path.exists():
        assert manifest.exists() and sha(path)==json.loads(manifest.read_text())['sha256'],'Incomplete/changed feature cache'
        return joblib.load(path)
    source=PRIOR/f"fold_{spec['fold']}"
    if role=='train':
        parts=[]
        for k,held in enumerate(spec['upstream_crossfit_held']):
            stage=source/f'oof_{k}';policy=json.loads((stage/'level1_policy.json').read_text())['policy']
            parts.append(load_part(frame,held,stage,'held',policy,include_teacher=True))
        data=join(parts)
    else:
        ids=spec[{'c1':'score_calibration','c2':'policy_calibration','outer':'outer'}[role]]
        policy=json.loads((source/'FROZEN_CASCADE.json').read_text())['level1']
        data=load_part(frame,ids,source if role=='outer' else source/'upstream_final',role,policy,include_teacher=False)
    joblib.dump(data,path,compress=0);save(manifest,dict(sha256=sha(path),bytes=path.stat().st_size))
    save(folder/(role+'_data_provenance.json'),dict(source=data['provenance'],ids=sorted(set(data['part']['meta'].subject.astype(int))),role=role))
    return data


def rank(record):
    m=record['calibration'];return (int(m['detected']>0 and m['rate']<1 and m['unknown']==0),m['detected'],m['macro'] or 0,record['c1_average_precision'],-record['c1_nll'],-record['epoch'])


def infer(folder,record,data):
    import tensorflow as tf
    with np.load(folder/'normalization.npz',allow_pickle=False) as z:norm=dict(mean=z['mean'],std=z['std'])
    x,ready,setup=normalized(data,norm,personal=record['personal'])
    model=tf.keras.models.load_model(folder/f"epoch_{record['epoch']}.keras",compile=False);outputs=predict(model,data,x,ready=ready)
    del model;gc.collect();pmf=probabilities(outputs['logits'],record['temperature']);risk=pmf[:,1:4].sum(1);risk[~ready]=0
    return outputs,pmf,risk,ready,setup


def trial(train,c1,c2,folder,family,preset=0,context=False,kd=None,personal=False,event=True,topk=True):
    folder=Path(folder);fit(train,folder,family,preset,context=context,kd=kd,personal=personal,event=event,topk=topk)
    completed=folder/'CALIBRATED.json'
    if completed.exists():return json.loads(completed.read_text())['selected']
    import tensorflow as tf
    with np.load(folder/'normalization.npz',allow_pickle=False) as z:norm=dict(mean=z['mean'],std=z['std'])
    x1,ready1,setup1=normalized(c1,norm,personal=personal);x2,ready2,setup2=normalized(c2,norm,personal=personal);choices=[]
    for epoch in [EPOCHS//2,EPOCHS]:
        model=tf.keras.models.load_model(folder/f'epoch_{epoch}.keras',compile=False)
        p1=predict(model,c1,x1,ready=ready1);p2=predict(model,c2,x2,ready=ready2);del model;gc.collect()
        temperature=choose_temperature(p1['logits'],c1,ready1);a=probabilities(p1['logits'],temperature);b=probabilities(p2['logits'],temperature)
        risk=b[:,1:4].sum(1);risk[~ready2]=0;policy,metrics,grid=calibrate(c2,risk,ready2)
        mask=(c1['y']>=0)&ready1;binary=np.isin(c1['y'][mask],[1,2,3]);ap=float(average_precision_score(binary,a[mask,1:4].sum(1))) if binary.any() else 0.
        nll=float(-np.log(np.maximum(a[np.flatnonzero(mask),c1['y'][mask]],1e-8)).mean()) if mask.any() else 1e9
        choice=dict(family=family,preset=preset,epoch=epoch,temperature=temperature,policy=policy,calibration=metrics,c1_average_precision=ap,c1_nll=nll,
            context=context,distilled=kd is not None,personal=personal,event=event,topk=topk,model_dir=str(folder))
        choices.append(choice);save(folder/f'epoch_{epoch}_policy_grid.json',grid)
        for name,data,outputs,pmf,ready in [('c1',c1,p1,a,ready1),('c2',c2,p2,b,ready2)]:
            ledger=data['part']['meta'][['subject','segment','chunk','time']].copy();ledger['candidate']=data['candidate'];ledger['hazard_available']=ready;ledger['target_bin']=data['y']
            for j in range(5):ledger[f'probability_{j}']=np.where(ready,pmf[:,j],np.nan)
            ledger.to_csv(folder/f'epoch_{epoch}_{name}_hazards.csv',index=False)
    selected=max(choices,key=rank);save(completed,dict(selected=selected,all_checkpoints=choices,setup_calibration_c1=setup1,setup_calibration_c2=setup2));return selected


def evaluate_fold(frame,spec):
    folder=OUT/f"fold_{spec['fold']}";folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():return json.loads((folder/'results.json').read_text())
    print('HAZARD FOLD',spec['fold'],'START',flush=True);save(folder/'partitions.json',spec)
    train=cached_data(frame,spec,'train',folder);c1=cached_data(frame,spec,'c1',folder);c2=cached_data(frame,spec,'c2',folder)
    actual=set(train['part']['meta'].subject);assert actual==set(spec['fit']) and not actual&(set(spec['outer'])|set(spec['score_calibration'])|set(spec['policy_calibration']))
    arms={};failures={}
    def attempt(name,**kwargs):
        try:arms[name]=trial(train,c1,c2,folder/name,**kwargs)
        except Exception as e:failures[name]=repr(e);save(folder/(name+'_FAILURE.json'),dict(error=repr(e),traceback=traceback.format_exc()));traceback.print_exc()
    for family in ['tcn','gru','refinement']:
        for preset in [0,1]:attempt(f'{family}_event_p{preset}',family=family,preset=preset)
    if not arms:raise RuntimeError('All prespecified base verifier fits failed; retain failures, no outer-based substitute')
    bestkey=max(arms,key=lambda k:rank(arms[k]));best=arms[bestkey];family,preset=best['family'],best['preset']
    save(folder/'INNER_BEST_EVENT.json',dict(name=bestkey,record=best))
    attempt('best_context',family=family,preset=preset,context=True)
    teacher=None
    try:
        teacher_folder=folder/'multisensor_teacher';fit(train,teacher_folder,'tcn',preset,context=True,teacher=True)
        import tensorflow as tf
        with np.load(teacher_folder/'normalization.npz',allow_pickle=False) as z:norm=dict(mean=z['mean'],std=z['std'])
        tx,_,_=normalized(train,norm,teacher=True);model=tf.keras.models.load_model(teacher_folder/f'epoch_{EPOCHS}.keras',compile=False)
        teacher=predict(model,train,tx);np.savez_compressed(teacher_folder/'fitting_teacher_targets.npz',**teacher);del model,tx;gc.collect()
        attempt('best_distilled',family=family,preset=preset,kd=teacher)
    except Exception as e:failures['multisensor_teacher']=repr(e);save(folder/'multisensor_teacher_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()))
    del teacher;gc.collect()
    attempt('best_personalized',family=family,preset=preset,personal=True)
    attempt('without_event_objectives',family=family,preset=preset,event=False)
    attempt('without_hard_topk_penalty',family=family,preset=preset,topk=False)
    eligible={k:v for k,v in arms.items() if k not in ['without_event_objectives','without_hard_topk_penalty']}
    winner=max(eligible,key=lambda k:rank(eligible[k]));aliases=dict(best_event=bestkey,no_context=bestkey,no_distillation=bestkey,no_personalized_calibration=bestkey,challenger=winner)
    frozen=dict(arms=arms,aliases=aliases,selected=winner,failures=failures,model_hashes={str(p.relative_to(folder)):sha(p) for p in folder.rglob('*') if p.suffix in ['.keras','.npz']})
    save(folder/'FROZEN_CHALLENGER.json',frozen)
    # The first new outer-model predictions occur only after the above decisions.
    outer=cached_data(frame,spec,'outer',folder);results={}
    old=pd.read_csv(PRIOR/f"fold_{spec['fold']}"/'matched_v2_predictions.csv',float_precision='round_trip')
    np.testing.assert_allclose(old.time,outer['part']['meta'].time,rtol=0,atol=1e-8);np.testing.assert_array_equal(old.chunk,outer['part']['meta'].chunk)
    np.testing.assert_allclose(old.duration,outer['part']['meta'].duration,rtol=0,atol=1e-10)
    old.to_csv(folder/'champion_predictions.csv',index=False);results['champion']=scored(outer['part'],old)
    for name,record in arms.items():
        _,pmf,risk,ready,setup=infer(Path(record['model_dir']),record,outer);trace=replay(outer,risk,record['policy'],ready)
        trace.to_csv(folder/(name+'_predictions.csv'),index=False);results[name]=scored(outer['part'],trace)
        ledger=outer['part']['meta'][['subject','segment','chunk','time']].copy();ledger['hazard_available']=ready;ledger['target_bin']=outer['y'];ledger['candidate']=outer['candidate']
        for j in range(5):ledger[f'probability_{j}']=np.where(ready,pmf[:,j],np.nan)
        ledger.to_csv(folder/(name+'_hazards.csv'),index=False);save(folder/(name+'_wearer_setup.json'),setup)
    for alias,target in aliases.items():
        results[alias]=results[target]
        shutil.copy2(folder/(target+'_predictions.csv'),folder/(alias+'_predictions.csv'));shutil.copy2(folder/(target+'_hazards.csv'),folder/(alias+'_hazards.csv'))
    people=[]
    for name,result in results.items():
        save(folder/(name+'_results.json'),result);pd.DataFrame(result['events']).to_csv(folder/(name+'_episodes.csv'),index=False)
        trace=pd.read_csv(folder/(name+'_predictions.csv'),float_precision='round_trip')
        for subject in spec['outer']:
            part=outer['part'];sub=dict(meta=part['meta'][part['meta'].subject==subject],records=[r for r in part['records'] if r.subject==f'FS{subject:02}'],inventory=[r for r in part['inventory'] if r['subject']==f'FS{subject:02}'],incomplete=False)
            people.append(dict(model=name,subject=subject,**scored(sub,trace[trace.subject==subject])['metrics']))
    pd.DataFrame(people).to_csv(folder/'participants.csv',index=False);save(folder/'results.json',results)
    print('HAZARD FOLD',spec['fold'],'COMPLETE',results['challenger']['metrics']['true_warning_events'],results['challenger']['metrics']['total_false_alarms'],flush=True)
    return results


def main():
    plan=freeze();frame=load_export();allfolds=[]
    with threadpool_limits(limits=2):
        for spec in plan['folds']:
            try:result=evaluate_fold(frame,spec);allfolds.append(dict(fold=spec['fold'],outer=spec['outer'],results=result))
            except Exception as e:
                folder=OUT/f"fold_{spec['fold']}";folder.mkdir(exist_ok=True);save(folder/'FOLD_FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc()))
                allfolds.append(dict(fold=spec['fold'],outer=spec['outer'],failure=repr(e)));traceback.print_exc()
            save(OUT/'all_folds.json',allfolds)
    save(OUT/'training_completed.json',dict(attempted=4,completed=sum('results' in f for f in allfolds),new_outer_driven_experiments=False,locked_assessment_executed=False))

if __name__=='__main__':main()
