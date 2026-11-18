"""Prespecified nested participant development; reads ONLY verified development export."""
from pathlib import Path
from dataclasses import replace,asdict
import os,json,shutil,hashlib,time,gc
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,pandas as pd,joblib
from scipy.special import expit
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from fog_pipeline import Config,build_windows,Alarm,save_json,sha256
from fogstar_adapter import segments_from_frame
from assessment_core import prepare,score
from gravity_gait import gravity_signals
from long_history_experiments import summaries
from waveform_bank import TemporalBank
from fast_temporal_bank import FastTemporalBank
from waveform_ensemble import PulseUnion
from rescue_cascade import RescueFeatures,PhysicalOR,poisson_interval,physical_preservation
from cascade_verifier_training import fit_pair,ledger_features
from candidate_gate import compare_candidates
ROOT=Path(__file__).resolve().parent
HEADS=['base','waveform','extended'];BASE_THRESHOLDS=[.675,.775,.875]
CFG=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)

def make_data(frame,ids):
 frame=frame[frame.subjectID.isin(ids)].copy();records,chunks,inventory,incomplete=prepare(frame,ids)
 if incomplete:raise ValueError('Native timestamp gap: record fold incomplete without revision')
 segments=segments_from_frame(frame,ankle='ankleR',include_gyro=True);d=build_windows(segments,CFG);meta=d.meta.copy();meta['chunk']=meta.segment;meta['segment']=meta.segment.map(lambda s:':'.join(s.split(':')[:3]));meta['subject']=meta.subject.map(lambda x:int(x[2:]));lookup={r.key:r for r in records};ends={s.key:s.time[-1]+1/60 for s in segments}
 known=[];labels=[];dur=[]
 for row in meta.itertuples():
  rec=lookup[row.segment];future=row.time+5<=rec.time[-1]+1e-9;known.append(future);onsets=rec.onsets[rec.onsets>row.time];delta=onsets[0]-row.time if len(onsets) else np.inf;labels.append(-1 if row.active_fog or not future else int(2-1e-9<=delta<=5+1e-9));dur.append(min(.25,max(0,ends[row.chunk]-row.time)))
 meta['target']=labels;meta['known_followup']=known;meta['duration']=dur
 return dict(data=d,meta=meta,records=records,inventory=inventory,incomplete=incomplete,motion=summaries(d.x),signals=gravity_signals(d.x))

def make_ledger(part,p,model_id):
 f=part['meta'].copy();f['segment_original']=f.segment;f['segment']=f.chunk;f['upstream_id']=model_id
 for j,h in enumerate(HEADS):f[h+'_probability']=p[:,j]
 for j in range(24):f['motion_'+str(j)]=part['motion'][:,j]
 return f.reset_index(drop=True)

def train_heads(part,output):
 import tensorflow as tf
 output.mkdir(exist_ok=False);labels=part['meta'].target.to_numpy();mask=labels>=0;y=labels[mask]
 if set(y)!={0,1}:raise ValueError('Upstream fit lacks classes')
 weight=np.where(y==1,min(sum(y==0)/max(1,sum(y==1)),10),1.);weight/=weight.mean();motion=part['motion'];mean=motion.mean(0);std=np.maximum(motion.std(0),1e-3)
 bank=FastTemporalBank.from_bank(TemporalBank(count=1024,seed=42).fit(part['signals']));z=bank.transform(part['signals']);xsmall=np.concatenate([z[:,:768],motion],1).astype(float);xlarge=np.concatenate([z,motion],1).astype(float)
 tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism();L=tf.keras.layers;dense=tf.keras.Sequential([L.Input((24,)),L.Dense(32,activation='relu'),L.Dropout(.3),L.Dense(1,activation='sigmoid')]);dense.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='binary_crossentropy');dense.fit(((motion-mean)/std)[mask],y,sample_weight=weight,epochs=24,batch_size=128,verbose=0,callbacks=[tf.keras.callbacks.CSVLogger(str(output/'dense_training.csv')),tf.keras.callbacks.TerminateOnNaN()]);dense.save(output/'dense.keras');w1,b1,w2,b2=dense.get_weights();np.savez(output/'dense_numpy.npz',w1=w1,b1=b1,w2=w2,b2=b2,mean=mean,std=std)
 dp=expit(np.maximum(((motion-mean)/std)@w1+b1,0)@w2+b2).ravel();np.testing.assert_allclose(dp,np.asarray(dense((motion-mean)/std,training=False)).ravel(),rtol=1e-5,atol=1e-6)
 models=[]
 for name,x in [('small',xsmall),('extended',xlarge)]:
  model=make_pipeline(StandardScaler(),RidgeClassifier(alpha=1,solver='cholesky'));model.fit(x[mask],y,ridgeclassifier__sample_weight=weight);joblib.dump(model,output/(name+'.joblib'));restored=joblib.load(output/(name+'.joblib'));np.testing.assert_allclose(model.decision_function(x[:20]),restored.decision_function(x[:20]));models.append(restored)
 save_json(output/'bank.json',dict(seed=42,scale=bank.scale.tolist(),kernels=[dict(length=l,dilation=d,weights=w.tolist(),channels=c.tolist(),bias=b) for l,d,w,c,b in bank.kernels]));save_json(output/'fit.json',dict(training_subjects=sorted(set(part['meta'].subject)),epochs=24,normalization='fit participants only',reload_parity=True))
 del dense,xsmall,xlarge,z;gc.collect();return dict(bank=bank,weights=(w1,b1,w2,b2),mean=mean,std=std,models=models)

def predict_heads(fit,part):
 w1,b1,w2,b2=fit['weights'];m=part['motion'];dense=expit(np.maximum(((m-fit['mean'])/fit['std'])@w1+b1,0)@w2+b2).ravel();z=fit['bank'].transform(part['signals']);small=np.concatenate([z[:,:768],m],1).astype(float);large=np.concatenate([z,m],1).astype(float)
 return np.column_stack([dense,expit(3*fit['models'][0].decision_function(small)),expit(3*fit['models'][1].decision_function(large))])

def baseline_trace(part,p):
 f=part['meta'].copy();f['alarm_start']=False;f['alarm_active']=False
 for _,g in f.groupby('chunk',sort=False):
  alarms=[Alarm(t,CFG) for t in BASE_THRESHOLDS];union=PulseUnion(1,3)
  for i in g.index:
   states=[a.step(f.loc[i,'time'],float(p[i,j])) for j,a in enumerate(alarms)];active,start=union.step(f.loc[i,'time'],[x[2] for x in states]);f.loc[i,'alarm_active']=active;f.loc[i,'alarm_start']=start
 return f

def scored(part,f):return score(part['records'],f,part['inventory'],part['incomplete'],CFG)
def candidate_stats(part,positive):
 f=part['meta'];known=f.known_followup;denom=float(f.loc[known,'duration'].sum());occupancy=float(np.dot(positive[known],f.loc[known,'duration'])/denom) if denom else 1.;events=[]
 for r in part['records']:
  g=f[f.segment==r.key]
  for onset in r.onsets:
   ids=g.index[(g.time>=onset-5-1e-9)&(g.time<=onset-2+1e-9)&~g.active_fog];events.append(bool(any(positive[ids])))
 starts=sum(int(np.sum(positive[g.index]&np.r_[True,~positive[g.index][:-1]])) for _,g in f.groupby('chunk',sort=False));hours=float(f.duration.sum()/3600)
 return dict(opportunity_recall=sum(events)/len(events) if events else None,candidate_positive_occupancy=occupancy,episode_opportunities=sum(events),episodes=len(events),candidate_positive_ticks_per_hour=float(sum(positive)/hours) if hours else None,candidate_runs_per_hour=starts/hours if hours else None,known_followup_hours=denom/3600)

def rescue_trace(part,base,positive,prob,threshold,standalone=False):
 f=base.copy();f['baseline_start']=base.alarm_start;f['baseline_active']=base.alarm_active;f['candidate']=positive;f['verifier_probability']=prob;f['rescue_start']=False;f['rescue_active']=False
 for _,g in f.groupby('chunk',sort=False):
  alarm=Alarm(threshold,replace(CFG,ema_alpha=1,persistence=1));output=PhysicalOR()
  for i in g.index:
   _,active,start=alarm.step(f.loc[i,'time'],float(prob[i]) if positive[i] else 0.);f.loc[i,'rescue_start']=start;f.loc[i,'rescue_active']=active;f.loc[i,'alarm_active'],f.loc[i,'alarm_start']=output.step(False if standalone else bool(base.loc[i,'alarm_active']),active)
 return f

def select_policy(part,base,positive,p):
 previous=scored(part,base);rows=[];best=None
 for threshold in [.5,.7,.9,.95,.99,1.1]:
  f=rescue_trace(part,base,positive,p,threshold);s=scored(part,f);m=s['metrics'];old=previous['metrics'];oldmatched={(e['segment'],e['onset']):e['warning_seconds'] for e in previous['events'] if e['warning_seconds'] is not None};newmatched={(e['segment'],e['onset']):e['warning_seconds'] for e in s['events'] if e['warning_seconds'] is not None};preserved=physical_preservation(base,f);rate=m['false_alarms_per_hour'];safe=preserved and rate is not None and rate<1 and m['censored_alarm_starts']<=old['censored_alarm_starts'] and m['maximum_alarm_duration_seconds']<=1+1e-6 and m['false_alarm_seconds']<=old['false_alarm_seconds']+1e-6 and all(k in newmatched and newmatched[k]>=v-1e-6 for k,v in oldmatched.items())
  gain=m['true_warning_events']-old['true_warning_events'];rows.append(dict(threshold=threshold,safe=safe,extra_episodes=gain,physical_preservation=preserved,**m))
  if safe and gain>0:
   rank=(gain,-m['censored_alarm_starts'],threshold)
   if best is None or rank>best[0]:best=(rank,threshold)
 return (best[1] if best else 1.1),rows

def combine_parts(parts):
 from fog_pipeline import Windows
 d=parts[0].copy();d['meta']=pd.concat([p['meta'] for p in parts],ignore_index=True);d['motion']=np.concatenate([p['motion'] for p in parts]);return d

def main():
 out=ROOT/'runs/nested_rescue_20261111_02';out.mkdir(exist_ok=False);source=ROOT/'development_only_20261111_01.csv';manifest=json.loads(source.with_suffix('.manifest.json').read_text());assert sha256(source)==manifest['output_sha256'];frame=pd.read_csv(source);assert set(frame.subjectID)==set(range(1,19))
 groups=[[1,2,3,4],[5,6,7,8],[9,10,11,12,13,14],[15,16,17,18]]
 plan=dict(parent_plan_sha256=sha256(ROOT/'runs/rescue_cascade_20261111_01/preregistered_plan.json'),development_export_sha256=sha256(source),groups=groups,inner_crossfit='two disjoint halves of fit subjects; each held participant excluded from its upstream fit; all outer and calibration people excluded',upstream='Matched v2 architecture refits: Dense32 24 fixed epochs; 1024 shared waveform kernels; small/extended ridge alpha1 Cholesky float64; original v2 thresholds unchanged. Original saved v2 is never refit or promoted.',seed=42,scoring='assessment_core full annotation inventory including sensor gaps',family_selection='inner calibration only',source_sha256={n:sha256(ROOT/n) for n in ['nested_rescue_experiment.py','rescue_cascade.py','cascade_verifier_training.py','assessment_core.py','fog_pipeline.py','fogstar_adapter.py','waveform_bank.py','fast_temporal_bank.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py']})
 save_json(out/'preregistered_execution.json',plan);(out/'source').mkdir()
 for n in plan['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
 allresults=[]
 with threadpool_limits(limits=2):
  for fold,outer in enumerate(groups,1):
   folder=out/f'fold_{fold}';folder.mkdir();remaining=sorted(set(range(1,19))-set(outer));cal=remaining[-4:];fitids=remaining[:-4];half=len(fitids)//2;heldgroups=[fitids[:half],fitids[half:]];save_json(folder/'partitions.json',dict(outer=outer,calibration=cal,fit=fitids,crossfit_held=heldgroups));print('FOLD',fold,'fit',fitids,'cal',cal,'outer',outer,flush=True)
   try:
    ledgers=[];provenance={}
    for k,held in enumerate(heldgroups):
     train=sorted(set(fitids)-set(held));trainpart=make_data(frame,train);heldpart=make_data(frame,held);upstream=train_heads(trainpart,folder/f'oof_{k}');p=predict_heads(upstream,heldpart);identity=f'oof_{k}';ledgers.append(make_ledger(heldpart,p,identity));provenance[identity]={h:dict(training_subjects=train) for h in HEADS};del trainpart,heldpart,upstream;gc.collect()
    trainledger=pd.concat(ledgers,ignore_index=True);trainledger.to_csv(folder/'out_of_fold_candidates_source.csv',index=False);save_json(folder/'provenance.json',provenance)
    trainpart=make_data(frame,fitids);calpart=make_data(frame,cal);upstream=train_heads(trainpart,folder/'upstream_final');cp=predict_heads(upstream,calpart);calbase=baseline_trace(calpart,cp);callog=make_ledger(calpart,cp,'final');provfinal={'final':{h:dict(training_subjects=fitids) for h in HEADS}};l1grid=[]
    for t in [.15,.30,.45]:l1grid.append(dict(threshold=t,**candidate_stats(calpart,cp.max(1)>=t)))
    save_json(folder/'level1_candidates.json',l1grid);eligible=[x for x in l1grid if x['candidate_positive_occupancy']<=.25];chosen=max(eligible,key=lambda x:(x['opportunity_recall'] or 0,-x['candidate_positive_occupancy'],x['threshold'])) if eligible else dict(threshold=1.1,opportunity_recall=0);l1=chosen['threshold'];models={};policies={};calranks={}
    for ablation in ['full','no_score_derivatives','no_motion_transition']:
     try:pair=fit_pair(trainledger,provenance,outer,cal,l1,folder/ablation,ablation)
     except ValueError as error:
      save_json(folder/(ablation+'_failure.json'),dict(error=str(error)));pair={}
     x,c,y=ledger_features(callog,provfinal,outer,[],l1)
     if ablation=='no_score_derivatives':x[:,3:12]=0
     if ablation=='no_motion_transition':x[:,41:]=0
     for family,model in pair.items():
      name=ablation+'_'+family;models[name]=model;p=model.predict_proba(x)[:,1];t,grid=select_policy(calpart,calbase,c,p);policies[name]=t;save_json(folder/(name+'_thresholds.json'),grid);calranks[name]=max([g['extra_episodes'] for g in grid if g['threshold']==t],default=0)
    selected=max(['logistic','hgb'],key=lambda family:(calranks.get('full_'+family,0),family=='logistic'));save_json(folder/'FROZEN_CASCADE.json',dict(level1_threshold=l1,level1_target_met=(chosen['opportunity_recall'] or 0)>=.95,policies=policies,selected_family=selected,upstream_training_subjects=fitids,outer=outer,model_sha256={str(p.relative_to(folder)):sha256(p) for p in folder.rglob('*') if p.suffix in ['.joblib','.keras','.npz']}))
    # Outer labels/outcomes are used only after fitting and policies have been frozen.
    outerpart=make_data(frame,outer);op=predict_heads(upstream,outerpart);ob=baseline_trace(outerpart,op);outerledger=make_ledger(outerpart,op,'final');outerledger.to_csv(folder/'outer_scores.csv',index=False);candidate=op.max(1)>=l1;stats=candidate_stats(outerpart,candidate);save_json(folder/'outer_level1.json',stats);traces={'matched_v2':ob,'level1_direct':rescue_trace(outerpart,ob,candidate,np.ones(len(op)),.5,True)}
    for ablation in ['full','no_score_derivatives','no_motion_transition']:
     x,c,y=ledger_features(outerledger,provfinal,outer,[],l1)
     if ablation=='no_score_derivatives':x[:,3:12]=0
     if ablation=='no_motion_transition':x[:,41:]=0
     families=['logistic','hgb'] if ablation=='full' else [selected]
     for family in families:
      key=ablation+'_'+family;model=models.get(key);p=model.predict_proba(x)[:,1] if model is not None else np.zeros(len(x));threshold=policies.get(key,1.1)
      if model is not None:np.testing.assert_allclose(p,np.array([model.predict_proba(row[None])[0,1] for row in x]),rtol=1e-5,atol=1e-6)
      combined=rescue_trace(outerpart,ob,c,p,threshold);name=family+'_rescue' if ablation=='full' else ablation;traces[name]=rescue_trace(outerpart,ob,c,p,threshold,True)
      if ablation=='full' and family==selected:traces['baseline_OR_selected_rescue']=combined
      if ablation!='full':traces['baseline_OR_'+ablation]=combined
    results={}
    for name,f in traces.items():
     result=scored(outerpart,f);result['physical_baseline_preserved']=physical_preservation(ob,f) if 'baseline_start' in f else name=='matched_v2';m=result['metrics'];result['false_rate95_poisson']=poisson_interval(m['total_false_alarms'],m['monitored_seconds']/3600);f.to_csv(folder/(name+'_predictions.csv'),index=False);save_json(folder/(name+'_results.json'),result)
     people=[]
     for person in outer:
      subject=f'FS{person:02}';records=[r for r in outerpart['records'] if r.subject==subject];ii=[i for i in outerpart['inventory'] if i['subject']==subject];ff=f[f.subject==person];resultp=score(records,ff,ii,False,CFG);people.append(dict(subject=person,**resultp['metrics']))
     pd.DataFrame(people).to_csv(folder/(name+'_participants.csv'),index=False);result['participants']=people;results[name]=result
    save_json(folder/'results.json',results);allresults.append(dict(fold=fold,outer=outer,results=results));save_json(out/'all_folds.json',allresults);print('FINISHED',fold,{n:v['metrics']['true_warning_events'] for n,v in results.items()},flush=True)
    del trainpart,calpart,outerpart,upstream,models;gc.collect()
   except Exception as error:
    save_json(folder/'FAILURE.json',dict(type=type(error).__name__,error=str(error),preserved=True));allresults.append(dict(fold=fold,outer=outer,failure=str(error)));save_json(out/'all_folds.json',allresults);print('FOLD FAILURE',fold,repr(error),flush=True)
 save_json(out/'execution_completed.json',dict(folds_attempted=4,successful_folds=sum('results' in r for r in allresults),reserved_used=False,default_candidate_changed=False))
if __name__=='__main__':main()
