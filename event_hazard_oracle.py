"""Diagnostic label oracle and frozen-score curves; never a deployment rule."""
import json
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import milp,Bounds,LinearConstraint
from sklearn.metrics import roc_auc_score,average_precision_score
from published_cascade_core import load_export,part_from_frame,candidate_mask,HEADS,scored
from waveform_ensemble import PulseUnion
from event_hazard_data import hazard_bin
from event_hazard_audit import PRIOR,OUT,save


def array_trace(part,request):
    f=part['meta'].copy();active=np.zeros(len(f),bool);starts=active.copy()
    for _,g in f.groupby('chunk',sort=False):
        pulse=PulseUnion(1,3)
        for i in g.index:active[i],starts[i]=pulse.step(float(f.loc[i,'time']),[bool(request[i])])
    f['alarm_active'],f['alarm_start']=active,starts
    return f


def label_oracle(part,candidate):
    m=part['meta'];chosen=np.zeros(len(m),bool);pairs=[]
    for r in part['records']:
        mask=(m.segment==r.key)&candidate&~m.active_fog
        for onset in r.onsets:
            for i in m.index[mask&(m.time>=onset-5-1e-9)&(m.time<=onset-2+1e-9)]:
                t=float(m.loc[i,'time']);j=np.searchsorted(r.onsets,t,side='right');state=r.labels[np.searchsorted(r.time,t,side='right')-1]
                if j<len(r.onsets) and r.onsets[j]==onset and state==1:pairs.append((int(i),r.key,float(onset)))
    if not pairs:return chosen,dict(status='no opportunities',bound=0)
    constraints=[]
    for key in sorted({(r,o) for _,r,o in pairs}):constraints.append([float((r,o)==key) for _,r,o in pairs])
    for chunk in m.chunk.unique():
        ids=[i for i,_,_ in pairs if m.loc[i,'chunk']==chunk]
        for t in sorted({float(m.loc[i,'time']) for i in ids}):
            constraints.append([float(m.loc[i,'chunk']==chunk and t-1e-8<=m.loc[i,'time']<t+3-1e-8) for i,_,_ in pairs])
    matrix=np.array(constraints)
    result=milp(-np.ones(len(pairs)),integrality=np.ones(len(pairs)),bounds=Bounds(0,1),
        constraints=LinearConstraint(matrix,0,1),options=dict(time_limit=60))
    if result.x is None:raise RuntimeError('Diagnostic MILP has no incumbent')
    for on,(i,_,_) in zip(result.x,pairs):
        if on>.5:chosen[i]=True
    return chosen,dict(status=int(result.status),message=result.message,objective=float(-result.fun),dual_bound=float(-result.mip_dual_bound),mip_gap=float(result.mip_gap))


def main():
    f=load_export();plan=json.loads((PRIOR/'preregistered_plan.json').read_text());opportunities=[];uncovered=[];distributions=[];ranking=[];categories=[];curves=[];oracles=[]
    for spec in plan['folds']:
        folder=PRIOR/f"fold_{spec['fold']}";part=part_from_frame(f,spec['outer']);m=part['meta']
        scores=pd.read_csv(folder/'outer_score_ledger.csv',float_precision='round_trip');p=scores[HEADS].to_numpy()
        frozen=json.loads((folder/'FROZEN_CASCADE.json').read_text());candidate=candidate_mask(part,p,frozen['level1'])
        records={r.key:r for r in part['records']};y=[];event=[]
        for row in m.itertuples():
            upcoming=records[row.segment].onsets[records[row.segment].onsets>row.time];delta=float(upcoming[0]-row.time) if len(upcoming) else np.inf
            b=hazard_bin(delta,row.known_followup,row.active_fog);y.append(1 if b in [1,2,3] else 0 if row.known_followup else -1)
            event.append((row.segment,float(upcoming[0])) if b in [1,2,3] else None)
        y=np.array(y);known=candidate&(y>=0);risk=p.max(1)
        for r in part['records']:
            for onset in r.onsets:
                eligible=(m.segment==r.key)&~m.active_fog&(m.time>=onset-5-1e-8)&(m.time<=onset-2+1e-8)
                n=int(sum(candidate&eligible));opportunities.append(dict(fold=spec['fold'],subject=r.subject,segment=r.key,onset=float(onset),candidate_count=n))
                if not n:uncovered.append(dict(fold=spec['fold'],segment=r.key,onset=float(onset),valid_inference_windows=int(sum(eligible)),reason='no valid sensor opportunity' if not any(eligible) else 'frozen candidate gate missed all valid opportunities'))
        for subject,g in m.groupby('subject',sort=True):
            for h in HEADS:
                for value,label in [(1,'true'),(0,'false')]:
                    v=scores.loc[g.index[known[g.index]&(y[g.index]==value)],h].to_numpy()
                    distributions.append(dict(fold=spec['fold'],subject=int(subject),head=h,kind=label,n=len(v),
                        **{f'q{q}':float(np.quantile(v,q/100)) if len(v) else None for q in [0,10,25,50,75,90,100]}))
        for record,g in m.groupby('segment',sort=False):
            ix=g.index[known[g.index]];labels=y[ix];values=risk[ix]
            ranking.append(dict(fold=spec['fold'],segment=record,n_true=int(sum(labels==1)),n_false=int(sum(labels==0)),
                pairwise_auc=float(roc_auc_score(labels,values)) if len(set(labels))==2 else None,
                average_precision=float(average_precision_score(labels,values)) if sum(labels==1) else None))
        for category,g in m.loc[candidate&(y==0)].groupby('hard_category'):
            categories.append(dict(fold=spec['fold'],category=category,candidates=len(g),known_negative_seconds=float(m.loc[(y==0)&(m.hard_category==category),'duration'].sum())))
        request,optimization=label_oracle(part,candidate);trace=array_trace(part,request);result=scored(part,trace)
        trace.to_csv(OUT/f"oracle_fold_{spec['fold']}_label_only.csv",index=False)
        oracles.append(dict(fold=spec['fold'],optimization=optimization,result=result))
        for t in sorted(set([0.,.1,.25,.5,.75,.9,.95,.99,1.1]+[float(np.quantile(risk[candidate],q)) for q in np.linspace(0,1,65)])):
            result=scored(part,array_trace(part,candidate&(risk>=t)))['metrics']
            curves.append(dict(fold=spec['fold'],threshold=t,detected=result['true_warning_events'],false=result['total_false_alarms'],
                known_hours=result['known_monitored_seconds']/3600,rate=result['unsuccessful_per_known_hour'],unknown=result['censored_alarm_starts']))
        print('Oracle diagnostic fold',spec['fold'],'done',flush=True)
    pd.DataFrame(opportunities).to_csv(OUT/'oracle_episode_opportunities.csv',index=False)
    pd.DataFrame(uncovered).to_csv(OUT/'uncovered_candidate_episodes.csv',index=False)
    pd.DataFrame(distributions).to_csv(OUT/'candidate_score_distributions.csv',index=False)
    pd.DataFrame(ranking).to_csv(OUT/'within_record_candidate_rankings.csv',index=False)
    pd.DataFrame(categories).to_csv(OUT/'false_candidate_categories.csv',index=False)
    pd.DataFrame(curves).to_csv(OUT/'diagnostic_frozen_score_threshold_curves.csv',index=False)
    maximum=sum(r['result']['metrics']['true_warning_events'] for r in oracles)
    assert sum(r['result']['metrics']['total_false_alarms']+r['result']['metrics']['censored_alarm_starts'] for r in oracles)==0,'Oracle must reproduce the unchanged official next-onset matching rule'
    pd.DataFrame([dict(retained_detections=k,minimum_label_oracle_unsuccessful_starts=0,minimum_rate=0.) for k in range(maximum+1)]).to_csv(OUT/'label_oracle_zero_false_frontier.csv',index=False)
    save(OUT/'oracle_feasibility.json',dict(folds=oracles,original_onsets=len(opportunities),candidate_covered=sum(r['candidate_count']>0 for r in opportunities),
        maximum_label_oracle_credited=sum(r['result']['metrics']['true_warning_events'] for r in oracles),
        min_unsuccessful_oracle=sum(r['result']['metrics']['total_false_alarms'] for r in oracles),
        caveat='Label oracle uses future truth and cannot be deployed. Its existence is a capacity bound, not proof that present features separate events or that a learned model generalizes. Threshold curves use fixed existing max scores, never select a challenger policy. Risk-blind capacity control previously covered71/72 versus67/72 frozen model candidates;65/72 was direct-pulse coverage.'))
    print('Oracle feasibility complete; not a deployment rule.',flush=True)

if __name__=='__main__':main()
