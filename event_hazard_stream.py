"""Native six-axis student. No annotations, teacher sensors or identities accepted."""
from pathlib import Path
from collections import deque
import os,json,time
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,tensorflow as tf
from scipy.special import expit
from event_hazard_audit import PRIOR,sha
from event_hazard_data import EventFeatures,HISTORY
from event_hazard_models import probabilities
from event_hazard_control import EventController
from published_cascade_core import CandidateGate
from published_upstream import load_baseline
from published_neural import NEURAL
from long_history_experiments import summaries
from gravity_gait import gravity_signals,gravity_features
from regularity_features import regularity_features


class FrozenScoreEngine:
    def __init__(self,folder,fused=False,warm=True):
        folder=Path(folder);self.fit=load_baseline(folder/'matched_v2')
        norm=json.loads((folder/'normalization.json').read_text());self.mean=np.array(norm['mean'],np.float32);self.std=np.array(norm['std'],np.float32)
        self.models=[tf.keras.models.load_model(folder/(name+'.keras'),compile=False) for name in NEURAL]
        self.fused=bool(fused)
        if self.fused:
            self.forward=tf.function(lambda x:tf.concat([m(x,training=False) for m in self.models],axis=1),input_signature=[tf.TensorSpec((1,180,6),tf.float32)])
            self.forward(tf.zeros((1,180,6),tf.float32))
        else:
            self.functions=[tf.function(lambda x,model=m:model(x,training=False),input_signature=[tf.TensorSpec((1,180,6),tf.float32)]) for m in self.models]
            for fn in self.functions:fn(tf.zeros((1,180,6),tf.float32))
    def step(self,raw):
        raw=np.asarray(raw,np.float32)[None];motion=summaries(raw);wave=self.fit['bank'].transform(gravity_signals(raw))
        w1,b1,w2,b2=self.fit['weights'];base=expit(np.maximum(((motion-self.fit['mean'])/self.fit['std'])@w1+b1,0)@w2+b2).ravel()[0]
        small=np.concatenate([wave[:,:768],motion],1).astype(float);large=np.concatenate([wave,motion],1).astype(float)
        scores=[float(base),float(expit(3*self.fit['models'][0].decision_function(small))[0]),float(expit(3*self.fit['models'][1].decision_function(large))[0])]
        x=((raw-self.mean)/self.std).astype(np.float32)
        values=np.asarray(self.forward(x)).ravel() if self.fused else np.array([np.asarray(fn(x)).ravel()[0] for fn in self.functions])
        scores.extend(map(float,values));shape=np.concatenate([motion,gravity_features(raw),regularity_features(raw),wave[:,:64]],1)[0]
        return np.array(scores),shape


class EventHazardStream:
    def __init__(self,folder,fused=False):
        tic=time.perf_counter();folder=Path(folder);self.folder=folder
        self.frozen=json.loads((folder/'FROZEN_CHALLENGER.json').read_text())
        for name,h in self.frozen['model_hashes'].items():
            if sha(folder/name)!=h:raise ValueError('Changed frozen challenger artifact: '+name)
        number=json.loads((folder/'partitions.json').read_text())['fold'];old=PRIOR/f'fold_{number}'
        prior=json.loads((old/'FROZEN_CASCADE.json').read_text())
        for name,h in prior['model_hashes'].items():
            if name.startswith('upstream_final') and sha(old/name)!=h:raise ValueError('Changed frozen upstream artifact')
        self.level1=prior['level1'];self.engine=FrozenScoreEngine(old/'upstream_final',fused=fused)
        self.record=self.frozen['arms'][self.frozen['selected']];modeldir=Path(self.record['model_dir'])
        with np.load(modeldir/'normalization.npz',allow_pickle=False) as z:self.population_mean,self.population_std=z['mean'].copy(),z['std'].copy()
        model=tf.keras.models.load_model(modeldir/f"epoch_{self.record['epoch']}.keras",compile=False)
        self.forward=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec((1,HISTORY,659),tf.float32)])
        self.forward(tf.zeros((1,HISTORY,659),tf.float32));self.reset_wearer()
        if warm:
            # Setup uses synthetic zeros only; no participant statistics survive.
            for tick in range(1200):self.step(tick/60,np.zeros(6,np.float32))
            self.reset_wearer()
        self.setup_seconds=time.perf_counter()-tic
    def reset_wearer(self):
        self.setup_start=None;self.setup_closed=False;self.setup_rows=[];self.setup_valid=not self.record['personal']
        self.mean=self.population_mean.copy();self.std=self.population_std.copy();self.reset_history()
    def reset_history(self):
        self.buffer=deque(maxlen=180);self.block=[];self.counter=0;self.last=None
        self.gate=CandidateGate(**self.level1);self.features=EventFeatures(self.level1);self.history=deque(maxlen=HISTORY)
        self.controller=EventController(**self.record['policy']);self.last_features=None;self.last_scores=None
    def reset_session(self):
        # A short/inconvenient first session cannot be replaced by a new setup.
        if self.record['personal'] and not self.setup_closed:self._close_setup(incomplete=True)
        self.reset_history()
    def _close_setup(self,incomplete=False):
        self.setup_closed=True;self.setup_valid=bool(not incomplete and len(self.setup_rows)>=4)
        if self.setup_valid:
            raw=np.array(self.setup_rows,np.float32);mean=np.median(raw,axis=0);std=np.maximum(1.4826*np.median(abs(raw-mean),axis=0),.25*self.population_std)
            self.mean=np.where(np.isfinite(mean),mean,self.population_mean);self.std=np.maximum(np.nan_to_num(std,nan=1),1e-3)
        self.setup_rows=[]
    def step(self,t,sample,arrival_wall=None,on_alert=None):
        entered=time.perf_counter();dispatch_ms=max(0.,(entered-arrival_wall)*1000) if arrival_wall is not None else None
        sample=np.asarray(sample,np.float32)
        if not np.isfinite(t) or sample.shape!=(6,) or not np.isfinite(sample).all():
            if np.isfinite(t) and self.setup_start is None:self.setup_start=float(t)
            self.reset_history();raise ValueError('Invalid native right-ankle sample')
        if self.setup_start is None:self.setup_start=float(t)
        if self.record['personal'] and not self.setup_closed and t>=self.setup_start+20-1e-8:self._close_setup()
        if self.last is not None and (t<=self.last or abs(t-self.last-1/60)>1e-5):
            reverse=t<=self.last;self.reset_history()
            if reverse:raise ValueError('Non-increasing timestamp')
        self.last=float(t);self.buffer.append(sample.copy());self.block.append(sample.copy());self.counter+=1
        if len(self.block)<15:return None
        block=np.array(self.block,np.float32);self.block=[]
        if self.counter<180:p=shape=None;candidate=False
        else:p,shape=self.engine.step(np.array(self.buffer,np.float32));candidate=self.gate.step(p)
        x=self.features.at(t,block,p,shape,candidate);self.history.append(x.copy())
        if self.record['personal'] and not self.setup_closed and t<=self.setup_start+20+1e-8:self.setup_rows.append(x.copy())
        if self.counter<180:return None
        ready=not self.record['personal'] or (self.setup_closed and self.setup_valid)
        pmf=None;risk=0.
        if ready:
            seq=np.zeros((HISTORY,659),np.float32);values=np.array(self.history,np.float32)
            seq[-len(values):]=np.clip(np.nan_to_num((values-self.mean)/self.std,nan=0,posinf=0,neginf=0),-8,8)
            outputs=self.forward(seq[None]);pmf=probabilities(np.asarray(outputs['logits']),self.record['temperature'])[0];risk=float(pmf[1:4].sum())
        active,start,adjusted=self.controller.step(float(t),risk,candidate,ready)
        compute_ms=(time.perf_counter()-entered)*1000
        result=dict(time=float(t),alarm_active=active,alarm_start=start,candidate=candidate,hazard_available=ready,probabilities=pmf,
            risk=risk,adjusted_risk=adjusted,compute_ms=compute_ms,dispatch_delay_ms=dispatch_ms,
            effective_output_ms=compute_ms+(dispatch_ms or 0.),pulse_seconds=1.)
        self.last_features=x;self.last_scores=p
        if start and on_alert is not None:on_alert(dict(result))
        return result
