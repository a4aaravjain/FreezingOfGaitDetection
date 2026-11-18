"""Complete checkpointed nested development experiment; no locked assessment."""
from pathlib import Path
import os,json,shutil,gc,traceback
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd,joblib
from threadpoolctl import threadpool_limits
from fog_pipeline import save_json,sha256
from session_assessment import prepare_sessions
from nested_rescue_experiment import baseline_trace
from published_cascade_core import (ROOT,HEADS,load_export,part_from_frame,select_level1,candidate_mask,
    candidate_metrics,feature_rows,pulse_trace,select_final_policy,scored,known_seconds)
from published_upstream import fit_upstream,predict_saved
from published_verifiers import fit_verifier,verifier_predict,fit_platt,calibrated

OUT=ROOT/'runs/published_cascade_20261111_01'
GROUPS=[[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14],[15,16,17,18]]
FAMILIES=['logistic','hgb','tcn']
ABLATIONS=['no_fogstar','no_stanford','no_resnet','no_motion','no_history']
SOURCES=['run_published_cascade.py','published_cascade_core.py','published_neural.py','published_upstream.py','published_verifiers.py',
    'session_assessment.py','nested_rescue_experiment.py','assessment_core.py','fog_pipeline.py','fogstar_adapter.py',
    'long_history_experiments.py','pretraining_experiments.py','gravity_gait.py','regularity_features.py',
    'fast_temporal_bank.py','waveform_bank.py','waveform_ensemble.py','rescue_cascade.py','test_published_cascade.py']


def balanced_people(ids,counts):
    groups=[[],[]];totals=[0,0]
    for subject in sorted(ids,key=lambda s:(-counts[s],s)):
        choices=[i for i in range(2) if len(groups[i])<int(np.ceil(len(ids)/2))]
        i=min(choices,key=lambda k:(totals[k],len(groups[k]),k));groups[i].append(subject);totals[i]+=counts[subject]
    return [sorted(g) for g in groups]


def freeze(frame):
    path=OUT/'preregistered_plan.json'
    if path.exists():
        plan=json.loads(path.read_text());assert all(sha256(ROOT/n)==h for n,h in plan['source_sha256'].items());return plan
    counts={}
    for subject in range(1,19):
        records,_,_,bad=prepare_sessions(frame[frame.subjectID==subject],[subject]);assert not bad
        counts[subject]=sum(len(r.onsets) for r in records)
    folds=[]
    for fold,outer in enumerate(GROUPS,1):
        remaining=sorted(set(range(1,19))-set(outer));cal=remaining[-4:];fit=remaining[:-4]
        c1,c2=balanced_people(cal,counts)
        folds.append(dict(fold=fold,outer=outer,fit=fit,score_calibration=c1,policy_calibration=c2,
            upstream_crossfit_held=balanced_people(fit,counts)))
    plan=dict(study='Published-architecture, cross-fitted same-tick rescue on corrected physical sessions',
        data='development_only_20261111_02.csv',data_sha256=sha256(ROOT/'development_only_20261111_02.csv'),folds=folds,
        fixed_target='onset2-5s after current native endpoint; exclude active FoG/unknown tails in ordinary binary fits; mined ResNet recipes deliberately include known active-FoG negative examples',
        preprocessing='right ankle6channels,60Hz,3s trailing window,.25s stride; no task resets; no future/zero-phase filtering or test statistics. FoGSTAR/Stanford crop to trailing2s inside the shared input. All valid windows replayed.',
        baseline_resolution='Original v2 trained people1-14, so cannot be an unseen-person baseline there. Paired nested baseline is the exact frozen v2 training recipe refitted only on each upstream fit group, with unchanged thresholds/alarms. Original saved weights are unchanged and separately compared only on historical people15-18.',
        upstream=['matched_v2_dense32','matched_v2_waveform384','matched_v2_waveform1024',
            'fogstar_causal_CNN','stanford_causal_CNN','resnet_core','resnet_balanced_focal','resnet_mined',
            'resnet_augmented','resnet_masked_pretrained'],
        training_strategy_comparisons='ResNet core vs subject/episode-balanced focal; then +hard-negative mining; then +rotation/noise/scale; then +fit-only masked pretraining. Fixed seed42.12 supervised epochs, mined variants2 extra hard-negative epochs; masked pretrain4 epochs. All recipe results retained. No seed search, external pretrained weights or early stopping.',
        upstream_crossfit='Two participant-stratified inner folds using only fitting-person annotation counts. Each scored fitting person excluded from all upstream weights, normalization, pretraining and hard-negative selection. Each upstream model has its own Level1 cutoff learned on the disjoint policy-calibration people, also excluding its held people.',
        level1='Max of10scores, permissive fixed probabilities/inner-calibration quantiles; causal prefix/spacing capacity gate at25% or33.3%, selected using policy-calibration people only. Never uses outer score quantiles. Prefer>=95% eligible opportunity recall, report all-original recall and irreducible unavailable windows separately. Train verifiers even if the recall target fails. Include equal-capacity risk-blind opportunity control to expose a trivial selector.',
        verifiers=FAMILIES,verifier_features='10 scores, agreement, past.5/1/2/3s slopes/curvature, run duration, periodicity/energy/FI/entropy/autocorrelation, waveform features, past changes, native reset validity, time since previous candidate.450features plus13past/current vectors for causal TCN. No subject/task/active labels as features.',
        verifier_training='Cross-fitted candidate rows only; subject-balanced natural-prevalence weights. Known late/active/postFoG/stopping/turning/background negatives retained. Unknown remains unknown. LogisticC.05, HGB100iter7leavesmin40L2=5, causal2-stage16channelTCN12epochs. No backward refinement.',
        calibration='Each fitted verifier has raw and optional Platt-score arms. Platt uses separate score-calibration participants; thresholds/model-family choice use policy-calibration participants. Any single-class failure retained. No substitution of in-sample upstream predictions.',
        final_threshold_grid=[.01,.025,.05,.1,.2,.35,.5,.7,.9,.95,.99,1.1],
        final_protocol='Same-tick requests produce full1s pulses,3s rescue refractory; never mutate baseline. Final physical OR; skip new rescue requests during baseline alert or within3s of previous baseline start. Actual OR rising edges, overlaps and possible extended pulses are scored and strict baseline physical-start preservation is checked. No claim of preservation merely from logical OR.',
        comparisons='Nested matchedv2; each published/recipe head alone at fixed.5 and calibration-safe threshold; Level1direct; each verifier raw/Platt fixed.5 and safe standalone/combined; best calibrated final OR; three external-component and motion/history ablations refitting the calibration-selected family; original-v2 operational comparison on people15-18 only.',
        ablations=ABLATIONS,known_exposure='Clip each replay interval to last annotation time minus5s; report both all monitored and known-followup hours, unsuccessful starts per known hour. No rate dilution by unknown tails.',
        promotion='Every actual matched baseline warning start preserved; >=2 added original episodes across>=2 people, improved participant macro, final unsuccessful rate<1/known monitored hour, >=2s actual delivered lead, no increased unknown starts,1s pulses,reload/native parity, causality/split/reset/leakage tests. No promotion without measured transport/actuator latency.',
        freeze_policy='All folds/comparisons/strategies frozen before model fitting; no additional trials after outer outcomes. Fold/source/checkpoints resume safely; retain failures and report incomplete status honestly.',
        source_sha256={n:sha256(ROOT/n) for n in SOURCES})
    save_json(path,plan);(OUT/'source').mkdir(exist_ok=False)
    for n in SOURCES:shutil.copy2(ROOT/n,OUT/'source'/n)
    return plan


def join_parts(parts):
    return dict(meta=pd.concat([p['meta'] for p in parts],ignore_index=True),records=[r for p in parts for r in p['records']],
        inventory=[r for p in parts for r in p['inventory']],incomplete=any(p['incomplete'] for p in parts))


def arrays_for_ledger(part,p,shape,policy,ablation='full'):
    return feature_rows(part,p,shape,policy,ablation)


def summarize_policy(grid,threshold):
    row=next(r for r in grid if r['threshold']==threshold)
    return row['gain'] if row['safe'] else 0


def evaluate_fold(frame,spec):
    fold=spec['fold'];folder=OUT/f'fold_{fold}';folder.mkdir(exist_ok=True)
    if (folder/'results.json').exists():return json.loads((folder/'results.json').read_text())
    outer=spec['outer'];fitids=spec['fit'];c1ids=spec['score_calibration'];c2ids=spec['policy_calibration'];cal=c1ids+c2ids
    save_json(folder/'partitions.json',spec)
    print('FOLD',fold,spec,flush=True)
    c1,c2=part_from_frame(frame,c1ids),part_from_frame(frame,c2ids)
    oof=[];oof_x=[];oof_seq=[];oof_c=[];oof_score=[];oof_shape=[];oof_policies=[]
    for k,held in enumerate(spec['upstream_crossfit_held']):
        ids=sorted(set(fitids)-set(held));train=part_from_frame(frame,ids);target=part_from_frame(frame,held)
        outputs=fit_upstream(train,dict(held=target,c1=c1,c2=c2),folder/f'oof_{k}',held,outer,cal)
        hp,hs=outputs['held'];cp,cs=outputs['c2'];policy,grid=select_level1(c2,cp)
        save_json(folder/f'oof_{k}/level1_policy.json',dict(policy=policy,grid=grid,threshold_fit_subjects=c2ids))
        x,seq,candidate=arrays_for_ledger(target,hp,hs,policy)
        oof.append(target);oof_x.append(x);oof_seq.append(seq);oof_c.append(candidate);oof_score.append(hp);oof_shape.append(hs);oof_policies.append(policy)
        del train,outputs;gc.collect()
    train=part_from_frame(frame,fitids)
    outputs=fit_upstream(train,dict(c1=c1,c2=c2),folder/'upstream_final',[],outer,cal)
    p1,s1=outputs['c1'];p2,s2=outputs['c2'];l1,grid=select_level1(c2,p2)
    save_json(folder/'level1_policy.json',dict(policy=l1,grid=grid,threshold_fit_subjects=c2ids))
    trainpart=join_parts(oof);tx=np.concatenate(oof_x);tseq=np.concatenate(oof_seq);tc=np.concatenate(oof_c)
    ledger=trainpart['meta'].copy();ledger['candidate']=tc
    for j,h in enumerate(HEADS):ledger[h]=np.concatenate(oof_score)[:,j]
    ledger.to_csv(folder/'cross_fitted_candidate_ledger.csv',index=False)
    x1,q1,cand1=arrays_for_ledger(c1,p1,s1,l1);x2,q2,cand2=arrays_for_ledger(c2,p2,s2,l1);base2=baseline_trace(c2,p2[:,:3])
    policies={};calranks={};failures={}
    for family in FAMILIES:
        modeldir=folder/'verifiers'/family
        try:
            fit_verifier(family,tx,tseq,tc,trainpart['meta'],modeldir)
            raw1=verifier_predict(family,modeldir,x1,q1);raw2=verifier_predict(family,modeldir,x2,q2)
            platt=fit_platt(raw1,c1['meta'],cand1,modeldir/'platt.joblib')
            for mode in ['raw','platt']:
                if mode=='platt' and platt is None:continue
                p=raw2 if mode=='raw' else calibrated(platt,raw2);key=family+'_'+mode
                alone,gridalone=select_final_policy(c2,p,cand2)
                combined,gridcombined=select_final_policy(c2,p,cand2,base2)
                save_json(modeldir/(mode+'_thresholds.json'),dict(alone=gridalone,combined=gridcombined))
                policies[key]=dict(family=family,mode=mode,alone=alone,combined=combined,model_dir=str(modeldir.relative_to(folder)))
                calranks[key]=summarize_policy(gridcombined,combined)
        except Exception as error:
            failures[family]=repr(error);save_json(folder/('verifier_'+family+'_FAILURE.json'),dict(error=repr(error),traceback=traceback.format_exc()))
    selected=max(policies,key=lambda k:(calranks[k],policies[k]['family']=='logistic',policies[k]['mode']=='raw')) if policies else None
    selected_family=policies[selected]['family'] if selected else 'logistic'
    ablation_policies={}
    for ablation in ABLATIONS:
        try:
            train_arrays=[arrays_for_ledger(p,s,h,l,ablation) for p,s,h,l in zip(oof,oof_score,oof_shape,oof_policies)]
            ax=np.concatenate([a[0] for a in train_arrays]);aq=np.concatenate([a[1] for a in train_arrays]);ac=np.concatenate([a[2] for a in train_arrays])
            modeldir=folder/'ablations'/ablation
            fit_verifier(selected_family,ax,aq,ac,trainpart['meta'],modeldir)
            ax2,aq2,ac2=arrays_for_ledger(c2,p2,s2,l1,ablation)
            ap=verifier_predict(selected_family,modeldir,ax2,aq2)
            threshold,agrid=select_final_policy(c2,ap,ac2,base2)
            save_json(modeldir/'thresholds.json',agrid)
            ablation_policies[ablation]=dict(family=selected_family,combined=threshold,model_dir=str(modeldir.relative_to(folder)))
            del train_arrays,ax,aq,ac;gc.collect()
        except Exception as error:
            failures[ablation]=repr(error);save_json(folder/(ablation+'_FAILURE.json'),dict(error=repr(error),traceback=traceback.format_exc()))
    # Standalone component operating points are calibrated before outer prediction.
    alone_thresholds={}
    for j,h in enumerate(HEADS):
        t,g=select_final_policy(c2,p2[:,j],np.ones(len(p2),bool))
        alone_thresholds[h]=t;save_json(folder/(h+'_alone_thresholds.json'),g)
    frozen=dict(level1=l1,verifiers=policies,selected=selected,ablations=ablation_policies,component_thresholds=alone_thresholds,failures=failures,
        model_hashes={str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.suffix in ['.keras','.joblib','.npz']})
    save_json(folder/'FROZEN_CASCADE.json',frozen)
    del train,tx,tseq,tc,x1,q1,x2,q2,outputs,oof_x,oof_seq;gc.collect()
    # First use of this fold's outer feature arrays/outcomes after all decisions are frozen.
    test=part_from_frame(frame,outer);p,shape=predict_saved(folder/'upstream_final',test);base=baseline_trace(test,p[:,:3])
    outerledger=test['meta'].copy()
    for j,h in enumerate(HEADS):outerledger[h]=p[:,j]
    outerledger.to_csv(folder/'outer_score_ledger.csv',index=False);np.savez_compressed(folder/'outer_shape_features.npz',shape=shape)
    x,q,candidate=arrays_for_ledger(test,p,shape,l1)
    save_json(folder/'outer_level1_metrics.json',candidate_metrics(test,candidate))
    blind=candidate_mask(test,np.ones_like(p),dict(threshold=.5,every=l1['every']))
    save_json(folder/'risk_blind_capacity_control.json',candidate_metrics(test,blind))
    traces={'matched_v2':base,'level1_direct':pulse_trace(test,np.ones(len(p)),.5,candidate)}
    for j,h in enumerate(HEADS):
        traces[h+'_alone_calibrated']=pulse_trace(test,p[:,j],alone_thresholds[h])
        if j>=3:traces[h+'_alone_fixed05']=pulse_trace(test,p[:,j],.5)
    for key,policy in policies.items():
        modeldir=folder/policy['model_dir'];vp=verifier_predict(policy['family'],modeldir,x,q)
        if policy['mode']=='platt':vp=calibrated(joblib.load(modeldir/'platt.joblib'),vp)
        traces[key+'_alone']=pulse_trace(test,vp,policy['alone'],candidate)
        traces[key+'_fixed05']=pulse_trace(test,vp,.5,candidate)
        traces[key+'_combined']=pulse_trace(test,vp,policy['combined'],candidate,base)
        pd.DataFrame(dict(time=test['meta'].time,candidate=candidate,probability=vp)).to_csv(folder/(key+'_verifier_ledger.csv'),index=False)
        if key==selected:traces['baseline_OR_best_rescue']=traces[key+'_combined']
    if selected is None:traces['baseline_OR_best_rescue']=pulse_trace(test,np.zeros(len(p)),1.1,candidate,base)
    for ablation,policy in ablation_policies.items():
        ax,aq,ac=arrays_for_ledger(test,p,shape,l1,ablation)
        ap=verifier_predict(policy['family'],folder/policy['model_dir'],ax,aq)
        traces[ablation]=pulse_trace(test,ap,policy['combined'],ac,base)
    # The existing, historically trained v2 weights are compared only on their non-fit people15-18.
    if outer==[15,16,17,18]:
        original=pd.read_csv(ROOT/'runs/continuous_session_20261111_02/continuous_decision_scores.csv')
        np.testing.assert_allclose(original.time,test['meta'].time,rtol=0,atol=1e-8)
        obase=test['meta'].copy();obase['alarm_start']=original.alarm_start.to_numpy();obase['alarm_active']=original.alarm_active.to_numpy()
        traces['original_saved_v2']=obase
        if selected:
            policy=policies[selected];vp=verifier_predict(policy['family'],folder/policy['model_dir'],x,q)
            if policy['mode']=='platt':vp=calibrated(joblib.load(folder/policy['model_dir']/'platt.joblib'),vp)
            traces['original_saved_v2_OR_rescue']=pulse_trace(test,vp,policy['combined'],candidate,obase)
    results={};personrows=[]
    for name,trace in traces.items():
        result=scored(test,trace);result['physical_baseline_starts_preserved']=bool(np.all(~base.alarm_start.to_numpy(bool)|trace.alarm_start.to_numpy(bool))) if name not in ['original_saved_v2','original_saved_v2_OR_rescue'] else None
        trace.to_csv(folder/(name+'_predictions.csv'),index=False);save_json(folder/(name+'_results.json'),result);results[name]=result
        for subject in outer:
            sub=dict(meta=test['meta'][test['meta'].subject==subject],records=[r for r in test['records'] if r.subject==f'FS{subject:02}'],
                inventory=[r for r in test['inventory'] if r['subject']==f'FS{subject:02}'],incomplete=False)
            m=scored(sub,trace[trace.subject==subject])['metrics'];personrows.append(dict(model=name,subject=subject,**m))
    pd.DataFrame(personrows).to_csv(folder/'participants.csv',index=False)
    save_json(folder/'results.json',results)
    print('FOLD COMPLETE',fold,{n:(v['metrics']['true_warning_events'],v['metrics']['total_false_alarms']) for n,v in results.items() if n in ['matched_v2','baseline_OR_best_rescue','level1_direct']},flush=True)
    return results


def main():
    frame=load_export();plan=freeze(frame);allresults=[]
    with threadpool_limits(limits=2):
        for spec in plan['folds']:
            try:
                result=evaluate_fold(frame,spec);allresults.append(dict(fold=spec['fold'],outer=spec['outer'],results=result))
            except Exception as error:
                folder=OUT/f"fold_{spec['fold']}";folder.mkdir(exist_ok=True)
                save_json(folder/'FOLD_FAILURE.json',dict(error=repr(error),traceback=traceback.format_exc()))
                allresults.append(dict(fold=spec['fold'],outer=spec['outer'],failure=repr(error)))
                traceback.print_exc()
            save_json(OUT/'all_folds.json',allresults)
    save_json(OUT/'execution_completed.json',dict(attempted=4,completed=sum('results' in x for x in allresults),reserved_participants_excluded=True,
        locked_assessment_executed=False,promoted=False))


if __name__=='__main__':main()
