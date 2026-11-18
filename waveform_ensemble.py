"""Research stream combining dense summaries and fixed waveform-pattern evidence."""
from collections import deque
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.special import expit
from fog_pipeline import Alarm,Config
from summary_stream import SummaryStream
from waveform_bank import TemporalBank
from gravity_gait import gravity_signals
from long_history_experiments import summaries

class PulseUnion:
    def __init__(self,pulse_seconds=1.,cooldown_seconds=3.):
        if pulse_seconds<=0 or cooldown_seconds<pulse_seconds:raise ValueError('Invalid pulse/cooldown')
        self.pulse=pulse_seconds;self.cooldown=cooldown_seconds;self.reset()
    def reset(self):self.last=-np.inf
    def step(self,time,starts):
        start=bool(any(starts) and time>=self.last+self.cooldown)
        if start:self.last=time
        return bool(time<self.last+self.pulse-1e-9),start

def load_bank(path):
    d=json.loads(Path(path).read_text());bank=TemporalBank(count=len(d['kernels']),seed=d['seed']);bank.scale=np.array(d['scale'],np.float32)
    bank.kernels=[(k['length'],k['dilation'],np.array(k['weights'],np.float32),np.array(k['channels'],np.float32),float(k['bias'])) for k in d['kernels']]
    return bank

class WaveformEnsemble:
    def __init__(self,run,ankle='ankleR'):
        run=Path(run);m=json.loads((run/'manifest.json').read_text())
        if ankle!='ankleR' or m['ankle']!='ankleR':raise ValueError('Requires right-ankle input')
        for name,digest in m['artifact_sha256'].items():
            if hashlib.sha256((run/name).read_bytes()).hexdigest()!=digest:raise ValueError('Artifact checksum mismatch: '+name)
        self.cfg=Config(**m['config']);self.base=SummaryStream(run/'base',ankle=ankle);self.bank=load_bank(run/'bank.json')
        if self.base.cfg!=self.cfg:raise ValueError('Incompatible stream configuration')
        with np.load(run/'ridge_weights.npz',allow_pickle=False) as d:
            self.mean,self.scale,self.coef,self.intercept=[d[k].copy() for k in ['mean','scale','coef','intercept']]
        if m.get('feature_engine','reference')=='grouped':
            from fast_temporal_bank import FastTemporalBank
            self.bank=FastTemporalBank.from_bank(self.bank)
        elif m.get('feature_engine','reference')!='reference':raise ValueError('Unknown feature engine')
        self.primary_kernel_count=m.get('primary_kernel_count',self.bank.count)
        if not 0<self.primary_kernel_count<=self.bank.count:raise ValueError('Invalid primary kernel count')
        self.extended=None;self.extended_alarm=None
        if m.get('extended_head'):
            with np.load(run/'extended_weights.npz',allow_pickle=False) as d:
                self.extended={k:d[k].copy() for k in ['mean','scale','coef','intercept']}
            self.extended_alarm=Alarm(m['extended_head']['threshold'],self.cfg)
        self.aux=Alarm(m['waveform_threshold'],self.cfg);self.union=PulseUnion(self.cfg.alert_pulse_seconds,self.cfg.cooldown_seconds);self.reset()
    def reset(self):
        self.base.reset();self.aux.reset();self.union.reset();self.last=None
        if self.extended_alarm is not None:self.extended_alarm.reset()
    def step(self,time,sample):
        sample=np.asarray(sample,np.float32)
        if sample.shape!=(6,) or not np.isfinite(sample).all() or not np.isfinite(time):
            self.reset();raise ValueError('Expected finite timestamp and six right-ankle channels')
        if self.last is not None:
            dt=time-self.last
            if dt<=0 or abs(dt-1/60)>.0011:
                self.reset()
                if dt<=0:raise ValueError('Non-increasing timestamp; reset')
        self.last=time;base=self.base.step(time,sample)
        if base is None:return None
        raw=np.array(self.base.buffer,np.float32)[None]
        bank_features=self.bank.transform(gravity_signals(raw));summary=summaries(raw)
        f=np.concatenate([bank_features[:,:2*self.primary_kernel_count],summary],axis=1)
        # Match sklearn StandardScaler's float32 transform before the ridge margin.
        z=f.copy();z-=self.mean;z/=self.scale
        probability=float(expit(3*(z@self.coef.T+self.intercept)).ravel()[0])
        smoothed,active,start=self.aux.step(time,probability)
        starts=[base['alarm_start'],start];extra={}
        if self.extended is not None:
            ef=np.concatenate([bank_features,summary],axis=1).astype(np.float64)
            parameters=self.extended;ez=(ef-parameters['mean'])/parameters['scale']
            ep=float(expit(3*(ez@parameters['coef'].T+parameters['intercept'])).ravel()[0])
            es,ea,estart=self.extended_alarm.step(time,ep);starts.append(estart)
            extra=dict(extended_probability=ep,extended_smoothed=es,extended_start=estart)
        output,start_union=self.union.step(time,starts)
        return dict(time=time,base_probability=base['probability'],waveform_probability=probability,base_smoothed=base['smoothed'],waveform_smoothed=smoothed,base_start=base['alarm_start'],waveform_start=start,alarm_active=output,alarm_start=start_union,**extra)
