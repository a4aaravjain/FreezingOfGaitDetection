"""Final alert controller; may veto every upstream warning. No label inputs."""
import numpy as np,pandas as pd
from event_hazard_models import probabilities
from published_cascade_core import scored,known_seconds


class EventController:
    def __init__(self,threshold=1.1,confirm=False,beta=0.):
        self.threshold=float(threshold);self.confirm=bool(confirm);self.beta=float(beta);self.reset()
    def reset(self):self.last=-np.inf;self.previous=False;self.burden=0.;self.baseline=0.;self.time=None
    def step(self,t,probability,candidate,ready=True):
        if not np.isfinite(t) or not np.isfinite(probability):self.reset();raise ValueError('Invalid controller input')
        if self.time is not None and t<=self.time:self.reset();raise ValueError('Non-increasing controller time')
        dt=.25 if self.time is None else t-self.time;self.time=t;self.burden*=np.exp(-dt/60)
        risk=float(probability)/(1+self.beta*self.burden)
        high=bool(ready and risk>=self.threshold)
        start=bool(candidate and high and (not self.confirm or self.previous) and t>=self.last+3)
        self.previous=high;self.baseline=.975*self.baseline+.025*float(probability)
        if start:self.last=t;self.burden+=1
        return bool(t<self.last+1-1e-9),start,risk


def replay(data,risk,policy,ready=None):
    m=data['part']['meta'];active=np.zeros(len(m),bool);starts=active.copy();adjusted=np.zeros(len(m))
    if ready is None:ready=np.ones(len(m),bool)
    for _,g in m.groupby('chunk',sort=False):
        controller=EventController(**policy)
        for i in g.index:active[i],starts[i],adjusted[i]=controller.step(float(m.loc[i,'time']),float(risk[i]),bool(data['candidate'][i]),bool(ready[i]))
    f=m.copy();f['alarm_active']=active;f['alarm_start']=starts;f['risk']=risk;f['adjusted_risk']=adjusted;f['candidate']=data['candidate'];f['setup_ready']=ready
    return f


def fast_metrics(data,trace):
    records={r.key:r for r in data['part']['records']};matched=set();false=unknown=0
    for row in trace.loc[trace.alarm_start].itertuples():
        r=records[row.segment];j=np.searchsorted(r.onsets,row.time,side='right');state=r.labels[np.searchsorted(r.time,row.time,side='right')-1]
        if j<len(r.onsets) and 2-1e-9<=r.onsets[j]-row.time<=5+1e-9 and state==1 and (r.key,j) not in matched:matched.add((r.key,j))
        elif j<len(r.onsets) and r.onsets[j]-row.time<=5+1e-9 and state==1:false+=1
        elif state==1 and row.time+5>r.time[-1]+1e-9:unknown+=1
        else:false+=1
    hours=known_seconds(data['part'])/3600;by_person={}
    for r in records.values():
        p=by_person.setdefault(r.subject,[0,0]);p[0]+=sum((r.key,j) in matched for j in range(len(r.onsets)));p[1]+=len(r.onsets)
    macro=float(np.mean([v[0]/v[1] for v in by_person.values() if v[1]])) if any(v[1] for v in by_person.values()) else None
    return dict(detected=len(matched),unsuccessful=false,unknown=unknown,known_hours=hours,rate=false/hours if hours else None,macro=macro)


def choose_temperature(logits,data,ready=None):
    y=data['y'];valid=y>=0
    if ready is not None:valid&=ready
    if not valid.any():return 1.
    return min([.75,1.,1.5,2.],key=lambda t:float(-np.log(np.maximum(probabilities(logits,t)[np.flatnonzero(valid),y[valid]],1e-8)).mean()))


def calibrate(data,risk,ready):
    grid=[];best=None
    for threshold in [.025,.05,.1,.2,.35,.5,.65,.8,.9,.97,.995,1.1]:
        for confirm in [False,True]:
            for beta in [0.,.5]:
                policy=dict(threshold=threshold,confirm=confirm,beta=beta);metrics=fast_metrics(data,replay(data,risk,policy,ready))
                safe=metrics['rate'] is not None and metrics['rate']<1 and metrics['unknown']==0
                row=dict(**policy,**metrics,safe=safe);grid.append(row)
                rank=(metrics['detected'],metrics['macro'] or 0,threshold,-int(confirm),-beta)
                if safe and metrics['detected']>0 and (best is None or rank>best[0]):best=(rank,policy,metrics)
    if best is None:
        policy=dict(threshold=1.1,confirm=False,beta=0.);metrics=fast_metrics(data,replay(data,risk,policy,ready))
    else:_,policy,metrics=best
    full=scored(data['part'],replay(data,risk,policy,ready))['metrics']
    assert (full['true_warning_events'],full['total_false_alarms'],full['censored_alarm_starts'])==(metrics['detected'],metrics['unsuccessful'],metrics['unknown'])
    return policy,metrics,grid
