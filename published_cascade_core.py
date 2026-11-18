"""Pure causal cascade features, capacity gate, pulse merge and session accounting."""
from collections import deque
from dataclasses import replace
from pathlib import Path
import numpy as np,pandas as pd
from fog_pipeline import sha256,Alarm
from session_assessment import make_session_data,score_sessions
from nested_rescue_experiment import CFG,baseline_trace
from long_history_experiments import summaries
from gravity_gait import gravity_signals,gravity_features
from regularity_features import regularity_features
from waveform_ensemble import PulseUnion
from rescue_cascade import PhysicalOR

ROOT=Path(__file__).resolve().parent
EXPORT=ROOT/'development_only_20261111_02.csv'
DIGEST='2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
HEADS=['base','waveform','extended','fogstar','stanford','resnet_core','resnet_balanced','resnet_mined','resnet_augmented','resnet_pretrained']


def load_export():
    if sha256(EXPORT)!=DIGEST:raise ValueError('Approved export checksum mismatch before parsing')
    f=pd.read_csv(EXPORT)
    if set(f.subjectID)!=set(range(1,19)):raise ValueError('Unexpected export scope')
    return f


def part_from_frame(frame,ids):
    p=make_session_data(frame,ids,CFG);m=p['meta']
    p['motion']=summaries(p['data'].x);p['signals']=gravity_signals(p['data'].x)
    records={r.key:r for r in p['records']};activities={}
    for key,g in frame[frame.subjectID.isin(ids)].groupby(['subjectID','sessionID'],sort=True):
        activities[f'FS{int(key[0]):02}:{key[1]}:run0']=(g.timestamp.to_numpy(),g.activity.to_numpy())
    category=[];episodes=[]
    for row in m.itertuples():
        r=records[row.segment];onsets=r.onsets[r.onsets>row.time];lead=onsets[0]-row.time if len(onsets) else np.inf
        t,activity=activities[row.segment];j=np.searchsorted(t,row.time,side='right')-1
        ends=r.time[1:][(r.labels[1:]==1)&(r.labels[:-1]==2)];past=ends[ends<=row.time]
        c=('active_fog' if row.active_fog else 'late_precursor' if lead<2 else 'post_fog' if len(past) and row.time-past[-1]<=5 else 'stopping' if activity[j] in [2,3,4,5] else 'turning' if activity[j] in [6,7] else 'other')
        category.append(c);episodes.append(row.segment+'@'+format(float(onsets[0]),'.9f') if row.target==1 else '')
    m['hard_category']=category;m['episode']=episodes
    return p


def known_seconds(part):
    ends={r.key:r.time[-1] for r in part['records']}
    return float(sum(max(0.,min(row.duration,ends[row.segment]-5-row.time)) for row in part['meta'].itertuples()))


def scored(part,trace):
    result=score_sessions(part['records'],trace,part['inventory'],part['incomplete'],CFG)
    seconds=known_seconds(part);result['metrics']['known_monitored_seconds']=seconds
    result['metrics']['unsuccessful_per_known_hour']=result['metrics']['total_false_alarms']*3600/seconds if seconds else None
    return result


class CandidateGate:
    """Causal capacity ceiling, not a selection using evaluation-participant statistics.

    Accepted candidates have >=every ticks spacing and prefix count<=floor(N/every).
    Whole-record known-followup is a prefix, so cap<=1/every on known exposure too.
    Every=3 or4 bounds occupancy at33.3% or25%, below the40% ceiling.
    """
    def __init__(self,threshold,every):self.threshold=threshold;self.every=every;self.reset()
    def reset(self):self.ticks=0;self.accepted=0;self.last=-100000
    def step(self,p):
        self.ticks+=1
        yes=bool(np.max(p)>=self.threshold and self.accepted<self.ticks//self.every and self.ticks-self.last>=self.every)
        if yes:self.accepted+=1;self.last=self.ticks
        return yes


def candidate_mask(part,p,policy):
    out=np.zeros(len(p),bool)
    for _,g in part['meta'].groupby('chunk',sort=False):
        gate=CandidateGate(**policy)
        for i in g.index:out[i]=gate.step(p[i])
    return out


def candidate_metrics(part,candidate):
    f=part['meta'];ends={r.key:r.time[-1] for r in part['records']}
    weights=np.array([max(0,min(r.duration,ends[r.segment]-5-r.time)) for r in f.itertuples()])
    events=[];eligible=[]
    for record in part['records']:
        g=f[f.segment==record.key]
        for onset in record.onsets:
            idx=g.index[(g.time>=onset-5-1e-9)&(g.time<=onset-2+1e-9)&~g.active_fog]
            eligible.append(bool(len(idx)));events.append(bool(np.any(candidate[idx])))
    return dict(original_episodes=len(events),eligible_episodes=sum(eligible),candidate_episodes=sum(events),
        episode_opportunity_recall=sum(events)/len(events) if events else None,
        eligible_opportunity_recall=sum(events)/sum(eligible) if sum(eligible) else None,
        candidate_positive_occupancy=float(np.dot(candidate,weights)/weights.sum()) if weights.sum() else None,
        known_monitor_hours=float(weights.sum()/3600),candidate_count=int(sum(candidate)))


def select_level1(part,p):
    maxima=p.max(1);grid=[]
    thresholds=sorted(set([.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99,1.1]+[float(np.quantile(maxima,q)) for q in [.6,.7,.75,.8,.85,.9,.95,.975]]))
    for every in [4,3]:
        for threshold in thresholds:
            policy=dict(threshold=threshold,every=every);candidate=candidate_mask(part,p,policy)
            grid.append(dict(**policy,**candidate_metrics(part,candidate)))
    safe=[g for g in grid if g['candidate_positive_occupancy'] is not None and g['candidate_positive_occupancy']<=.4+1e-9]
    target=[g for g in safe if (g['eligible_opportunity_recall'] or 0)>=.95]
    pool=target or safe
    best=max(pool,key=lambda g:((g['episode_opportunity_recall'] or 0),-g['candidate_positive_occupancy'],g['threshold']))
    return {k:best[k] for k in ['threshold','every']},grid


class CascadeFeatures:
    def __init__(self,policy):self.policy=policy;self.reset()
    def reset(self):self.past=deque(maxlen=13);self.shape_past=deque(maxlen=13);self.run=np.zeros(4);self.n=0;self.previous_candidate=-1000
    def step(self,p,shape,time,candidate):
        p=np.asarray(p,float);shape=np.asarray(shape,float);self.n+=1
        if p.shape!=(10,) or shape.shape!=(248,) or not np.isfinite(p).all() or not np.isfinite(shape).all():raise ValueError('Invalid causal feature input')
        history=list(self.past)+[p];motions=list(self.shape_past)+[shape]
        def prior(lag):return history[max(0,len(history)-1-lag)]
        slopes=np.concatenate([p-prior(lag) for lag in [2,4,8,12]])
        curve=np.concatenate([p-2*prior(lag//2)+prior(lag) for lag in [2,4,8,12]])
        levels=np.array([.15,.3,.5,self.policy['threshold']]);self.run=np.where(p.max()>=levels,self.run+1,0)
        agreement=np.array([p.min(),p.max(),p.mean(),p.std(),(p>=.25).mean(),(p>=.5).mean()])
        changes=np.concatenate([shape[:24]-motions[max(0,len(motions)-1-lag)][:24] for lag in [2,4,8,12]])
        age=min(30.,time-self.previous_candidate) if self.previous_candidate>-999 else 30.
        x=np.r_[p,agreement,slopes,curve,np.minimum(self.run*.25,30),age,float(candidate),shape,changes,
            min(len(self.past)/12,1),min(self.n*.25,30),float(self.n==1),1.].astype(np.float32)
        if candidate:self.previous_candidate=time
        self.past.append(p.copy());self.shape_past.append(shape.copy())
        return x


def feature_rows(part,p,shape,policy,ablation='full'):
    p=p.copy();shape=shape.copy()
    if ablation=='no_fogstar':p[:,3]=0
    if ablation=='no_stanford':p[:,4]=0
    if ablation=='no_resnet':p[:,5:]=0
    candidates=candidate_mask(part,p,policy);x=np.empty((len(p),450),np.float32);sequences=np.zeros((len(p),13,450),np.float32)
    for _,g in part['meta'].groupby('chunk',sort=False):
        state=CascadeFeatures(policy);past=deque(maxlen=13)
        for i in g.index:
            value=state.step(p[i],shape[i],part['meta'].loc[i,'time'],candidates[i])
            if ablation=='no_motion':value[102:446]=0
            if ablation=='no_history':value[16:102]=0;value[350:446]=0
            x[i]=value;past.append(value.copy());sequences[i,-len(past):]=np.array(past)
            if ablation=='no_history':sequences[i,:-1]=0
    return x,sequences,candidates


def pulse_trace(part,prob,threshold,candidate=None,base=None):
    f=part['meta'].copy();starts=np.zeros(len(f),bool);active=starts.copy();rescue_start=starts.copy();rescue_active=starts.copy()
    if candidate is None:candidate=np.ones(len(f),bool)
    for _,g in f.groupby('chunk',sort=False):
        pulse=PulseUnion(1,3);physical=PhysicalOR();lastbase=-np.inf
        for i in g.index:
            t=f.loc[i,'time']
            if base is not None and base.loc[i,'alarm_start']:lastbase=t
            allowed=base is None or (not base.loc[i,'alarm_active'] and t-lastbase>=3-1e-9)
            request=bool(candidate[i] and prob[i]>=threshold and allowed)
            ra,rs=pulse.step(t,[request]);rescue_active[i]=ra;rescue_start[i]=rs
            if base is None:active[i],starts[i]=ra,rs
            else:active[i],starts[i]=physical.step(bool(base.loc[i,'alarm_active']),ra)
    f['alarm_active'],f['alarm_start']=active,starts;f['rescue_active'],f['rescue_start']=rescue_active,rescue_start
    if base is not None:f['baseline_active'],f['baseline_start']=base.alarm_active,base.alarm_start
    return f


def select_final_policy(part,p,candidate,base=None):
    rows=[];best=None;previous=scored(part,base) if base is not None else None
    for threshold in [.01,.025,.05,.1,.2,.35,.5,.7,.9,.95,.99,1.1]:
        trace=pulse_trace(part,p,threshold,candidate,base);result=scored(part,trace);m=result['metrics']
        safe=(m['unsuccessful_per_known_hour'] is not None and m['unsuccessful_per_known_hour']<1 and m['maximum_alarm_duration_seconds']<=1+1e-6)
        if previous is not None:
            old=previous['metrics'];a={(e['segment'],e['onset']):e['warning_seconds'] for e in previous['events'] if e['warning_seconds'] is not None};b={(e['segment'],e['onset']):e['warning_seconds'] for e in result['events'] if e['warning_seconds'] is not None}
            preserved=bool(np.all(~base.alarm_start.to_numpy(bool)|trace.alarm_start.to_numpy(bool)))
            safe&=preserved and m['censored_alarm_starts']<=old['censored_alarm_starts'] and all(k in b and b[k]>=v-1e-6 for k,v in a.items())
            gain=m['true_warning_events']-old['true_warning_events']
        else:safe&=m['censored_alarm_starts']==0;gain=m['true_warning_events'];preserved=True
        rows.append(dict(threshold=threshold,safe=bool(safe),gain=gain,preserved=preserved,**m))
        rank=(gain,-m['censored_alarm_starts'],threshold)
        if safe and gain>0 and (best is None or rank>best[0]):best=(rank,threshold)
    return best[1] if best else 1.1,rows
