"""Past-only multi-rate history; labels and teacher channels remain separate."""
from pathlib import Path
from collections import deque
import json
import numpy as np,pandas as pd
from published_cascade_core import ROOT,HEADS,load_export,part_from_frame,CascadeFeatures,CandidateGate
from session_assessment import prepare_sessions
from event_hazard_audit import PRIOR,OUT,save,sha

HISTORY=80
ALL_CHANNELS=[f'{site}_{sensor}_{axis}' for site in ['ankleL','ankleR','back','wrist'] for sensor in ['acc','gyro'] for axis in 'xyz']


def hazard_bin(delta,known,active=False):
    if np.isnan(delta) or delta<0:raise ValueError('Invalid future-onset time')
    if active:return -1
    if 0<=delta<2-1e-8:return 0
    if 2-1e-8<=delta<3-1e-8:return 1
    if 3-1e-8<=delta<4-1e-8:return 2
    if 4-1e-8<=delta<=5+1e-8:return 3
    return 4 if known else -1


def extended_raw(block):
    block=np.asarray(block,np.float32);values=[]
    for j in range(0,block.shape[1],6):
        a=block[:,j:j+6]
        values.extend([a,np.linalg.norm(a[:,:3],axis=1)[:,None],np.linalg.norm(a[:,3:],axis=1)[:,None]])
    return np.concatenate(values,1).ravel()


def periodicity(raw):
    out=[]
    for seconds in [2,5,10,20]:
        a=raw[-min(len(raw),seconds*60):]
        for start in [0,3]:
            x=np.linalg.norm(a[:,start:start+3],axis=1);x=x-x.mean()
            power=abs(np.fft.rfft(x))**2;freq=np.fft.rfftfreq(len(x),1/60)
            lo=power[(freq>=.5)&(freq<3)].sum();hi=power[(freq>=3)&(freq<=8)].sum()
            ac=np.correlate(x,x,mode='full')[len(x)-1:];ac/=max(float(ac[0]),1e-8)
            gait=ac[16:min(121,len(ac))];q=power/max(float(power.sum()),1e-8)
            out.extend([np.log1p(hi/(lo+1e-8)),float(gait.max()) if len(gait) else 0.,
                float(freq[np.argmax(power)]),float(-(q*np.log(q+1e-8)).sum()/np.log(max(2,len(q)))),np.log1p(float(np.mean(x*x)))])
    return np.array(out,np.float32)


class EventFeatures:
    def __init__(self,policy):self.policy=policy;self.reset()
    def reset(self):self.raw=deque(maxlen=1200);self.base=CascadeFeatures(self.policy)
    def step(self,block,p=None,shape=None,candidate=False):
        block=np.asarray(block,np.float32)
        if block.shape!=(15,6) or not np.isfinite(block).all():raise ValueError('Student feature input must be 15 finite right-ankle six-axis samples')
        self.raw.extend(block.copy());raw=np.array(self.raw,np.float32)
        magnitudes=np.c_[raw,np.linalg.norm(raw[:,:3],axis=1),np.linalg.norm(raw[:,3:],axis=1)]
        energy=np.array([np.log1p(np.mean(magnitudes[-min(len(raw),int(s*60)):]**2,axis=0)) for s in [.5,1,2,5,10,20]],np.float32)
        current_energy=np.log1p(np.mean(magnitudes[-15:]**2,axis=0))
        changes=(energy-current_energy).ravel()
        base=self.base.step(p,shape,self.time,candidate) if p is not None else np.zeros(450,np.float32)
        return np.r_[extended_raw(block),base,changes,periodicity(raw),float(p is not None)].astype(np.float32)
    def at(self,t,block,p=None,shape=None,candidate=False):self.time=float(t);return self.step(block,p,shape,candidate)


def load_part(frame,ids,stage,role,policy,include_teacher=True):
    part=part_from_frame(frame,ids);meta=part['meta'];stage=Path(stage)
    scorefile=stage/(role+'_scores.csv') if role!='outer' else stage/'outer_score_ledger.csv'
    featurefile=stage/(role+'_features.npz') if role!='outer' else stage/'outer_shape_features.npz'
    scores=pd.read_csv(scorefile,float_precision='round_trip');np.testing.assert_allclose(scores.time,meta.time,rtol=0,atol=1e-8)
    np.testing.assert_array_equal(scores.chunk,meta.chunk)
    p=scores[HEADS].to_numpy()
    with np.load(featurefile,allow_pickle=False) as f:shape=f['shape'].copy()
    _,chunks,_,bad=prepare_sessions(frame[frame.subjectID.isin(ids)],ids)
    if bad:raise ValueError('Unexpected physical session timestamp gap')
    groups={(int(a),b):g for (a,b),g in frame[frame.subjectID.isin(ids)].groupby(['subjectID','sessionID'],sort=True)}
    setup={}
    for subject in ids:
        first=next((g for (s,_),g in groups.items() if s==subject))
        setup[subject]=dict(start=float(first.timestamp.iloc[0]),end=float(first.timestamp.iloc[0]+20),
            session=str(first.sessionID.iloc[0]),completed_recording=bool(first.timestamp.iloc[-1]>=first.timestamp.iloc[0]+20-1e-8))
    xs=[];teachers=[];starts=[];times=[];owners=[];sessions=[];rowindex=np.zeros(len(meta),int);candidate=np.zeros(len(meta),bool)
    offset=0
    for chunk in chunks:
        subject=int(chunk['record'].split(':')[0][2:]);session=chunk['record'].split(':')[1]
        g=next(g for (s,v),g in groups.items() if s==subject and str(v)==session)
        begin=int(np.searchsorted(g.timestamp.to_numpy(),chunk['time'][0]));n=len(chunk['time']);allraw=g[ALL_CHANNELS].to_numpy(np.float32)[begin:begin+n] if include_teacher else None
        np.testing.assert_array_equal(g.timestamp.to_numpy()[begin:begin+n],chunk['time'])
        locs=meta.index[meta.chunk==chunk['id']].to_numpy();count=len(chunk['time'])//15
        state=EventFeatures(policy);gate=CandidateGate(**policy)
        for k in range(count):
            t=float(chunk['time'][15*k+14]);block=chunk['x'][15*k:15*k+15];index=k-11
            if index>=0:
                i=locs[index];assert abs(t-meta.loc[i,'time'])<1e-8
                cand=gate.step(p[i]);x=state.at(t,block,p[i],shape[i],cand);rowindex[i]=offset+k;candidate[i]=cand
            else:x=state.at(t,block)
            if include_teacher:
                teacherraw=allraw[15*k:15*k+15];mask=np.isfinite(teacherraw).reshape(15,4,6).mean(axis=(0,2))
                teacher=np.r_[extended_raw(teacherraw),x[120:],mask].astype(np.float32)
            else:teacher=np.empty(0,np.float32)
            xs.append(x);teachers.append(teacher);starts.append(offset);times.append(t);owners.append(subject);sessions.append(session)
        assert len(locs)==max(0,count-11)
        offset+=count
    records={r.key:r for r in part['records']};bins=[];context=[];next_onset=[]
    for row in meta.itertuples():
        record=records[row.segment];future=record.onsets[record.onsets>row.time];delta=float(future[0]-row.time) if len(future) else np.inf
        bins.append(hazard_bin(delta,row.known_followup,row.active_fog))
        next_onset.append(float(future[0]) if len(future) else np.inf)
        session=row.segment.split(':')[1];g=next(g for (s,v),g in groups.items() if s==row.subject and str(v)==session)
        j=np.searchsorted(g.timestamp.to_numpy(),row.time,side='right')-1;activity=int(g.activity.iloc[j])
        ends=record.time[1:][(record.labels[:-1]==2)&(record.labels[1:]==1)];post=bool(np.any((ends<=row.time)&(row.time-ends<=5)))
        # Standing is an annotation-supported proxy; intention to stop is not annotated.
        context.append([activity==1,activity in [6,7],activity==3 and not row.active_fog,row.active_fog,post and not row.active_fog,
            float(xs[rowindex[row.Index]][120+450+48+1])<.25])
    small={k:part[k] for k in ['meta','records','inventory','incomplete']}
    return dict(x=np.array(xs,np.float32),teacher=np.array(teachers,np.float32),starts=np.array(starts,int),block_time=np.array(times),
        block_owner=np.array(owners,int),block_session=np.array(sessions),row_index=rowindex,candidate=candidate,
        y=np.array(bins,int),next_onset=np.array(next_onset),context=np.array(context,np.float32),part=small,setup=setup,policy=policy,
        provenance=dict(score_file=str(scorefile),score_sha256=sha(scorefile),feature_file=str(featurefile),feature_sha256=sha(featurefile),ids=list(ids),teacher_channels=ALL_CHANNELS if include_teacher else []))


def join(parts):
    result={k:[] for k in ['x','teacher','starts','block_time','block_owner','block_session','row_index','candidate','y','next_onset','context']};offset=0;setups={}
    for p in parts:
        for key in result:result[key].append(p[key]+offset if key in ['starts','row_index'] else p[key])
        offset+=len(p['x']);setups.update(p['setup'])
    result={k:np.concatenate(v) for k,v in result.items()}
    result.update(part=dict(meta=pd.concat([p['part']['meta'] for p in parts],ignore_index=True),records=sum([p['part']['records'] for p in parts],[]),
        inventory=sum([p['part']['inventory'] for p in parts],[]),incomplete=any(p['part']['incomplete'] for p in parts)),setup=setups,
        provenance=[p['provenance'] for p in parts])
    return result


def normalizer(data,teacher=False):
    x=data['teacher' if teacher else 'x'];mean=np.nanmean(x,axis=0);std=np.maximum(np.nanstd(x,axis=0),1e-3)
    mean=np.nan_to_num(mean);std=np.nan_to_num(std,nan=1.)
    return dict(mean=mean,std=std)


def normalized(data,norm,teacher=False,personal=False):
    raw=data['teacher' if teacher else 'x'];x=np.zeros_like(raw);ready=np.ones(len(data['part']['meta']),bool);calibration=[]
    for subject in np.unique(data['block_owner']):
        owner=data['block_owner']==subject;mean,std=norm['mean'],norm['std']
        if personal:
            s=data['setup'][int(subject)];setup=owner&(data['block_session']==s['session'])&(data['block_time']<=s['end']+1e-8)
            completed=s['completed_recording'] and int(setup.sum())>=4
            if completed:
                mean=np.nanmedian(raw[setup],axis=0);std=np.maximum(1.4826*np.nanmedian(abs(raw[setup]-mean),axis=0),.25*norm['std'])
                mean=np.where(np.isfinite(mean),mean,norm['mean']);std=np.maximum(np.nan_to_num(std,nan=1),1e-3)
            mask=data['part']['meta'].subject.to_numpy()==subject
            time=data['part']['meta'].time.to_numpy();session=data['part']['meta'].segment.str.split(':').str[1].to_numpy()
            ready[mask]=completed&((session[mask]!=s['session'])|(time[mask]>=s['end']-1e-8))
            calibration.append(dict(subject=int(subject),**s,samples=int(setup.sum()),ready=bool(completed)))
        x[owner]=np.clip(np.nan_to_num((raw[owner]-mean)/std,nan=0,posinf=0,neginf=0),-8,8)
    return x,ready,calibration


def sequences(data,x,rows):
    positions=data['row_index'][rows,None]-np.arange(HISTORY-1,-1,-1)[None,:]
    valid=positions>=data['starts'][data['row_index'][rows],None]
    return x[np.maximum(positions,0)]*valid[:,:,None]
