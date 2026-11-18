"""Native right-ankle prototype for the frozen selected published-code cascade."""
from pathlib import Path
from collections import deque
import json,time,os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
import numpy as np,joblib,tensorflow as tf
from scipy.special import expit,logit
from fog_pipeline import Alarm,sha256
from nested_rescue_experiment import CFG
from waveform_ensemble import PulseUnion
from rescue_cascade import PhysicalOR
from published_upstream import load_baseline
from published_neural import NEURAL
from published_cascade_core import CandidateGate,CascadeFeatures
from long_history_experiments import summaries
from gravity_gait import gravity_signals,gravity_features
from regularity_features import regularity_features


def compiled(model,shape):
    fn=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec(shape,tf.float32)])
    fn(tf.zeros([1]+list(shape[1:]),tf.float32))
    return fn


class NativeVerifier:
    def __init__(self,family,folder):
        folder=Path(folder);self.family=family
        if family=='tcn':
            self.model=compiled(tf.keras.models.load_model(folder/'model.keras',compile=False),(None,13,450))
            with np.load(folder/'normalization.npz',allow_pickle=False) as z:self.mean,self.std=z['mean'],z['std']
        else:self.model=joblib.load(folder/'model.joblib')
        self.platt=joblib.load(folder/'platt.joblib') if (folder/'platt.joblib').exists() else None
    def step(self,x,q,mode='raw'):
        # Retain the estimator's dtype so threshold comparisons and Platt's
        # clipped-logit transform have the same arithmetic as batch replay.
        p=np.asarray(self.model(((q-self.mean)/self.std)[None].astype(np.float32))).ravel()[0] if self.family=='tcn' else self.model.predict_proba(x[None])[0,1]
        if mode=='platt':p=self.platt.predict_proba(logit(np.clip(np.asarray([p]),1e-6,1-1e-6))[:,None])[0,1]
        return p


class PublishedCascadeStream:
    def __init__(self,folder):
        tic=time.perf_counter();folder=Path(folder);self.folder=folder
        self.frozen=json.loads((folder/'FROZEN_CASCADE.json').read_text())
        for name,digest in self.frozen['model_hashes'].items():
            if sha256(folder/name)!=digest:raise ValueError('Frozen model mismatch: '+name)
        upstream=folder/'upstream_final';self.fit=load_baseline(upstream/'matched_v2')
        normal=json.loads((upstream/'normalization.json').read_text());self.mean=np.array(normal['mean'],np.float32);self.std=np.array(normal['std'],np.float32)
        self.networks=[compiled(tf.keras.models.load_model(upstream/(key+'.keras'),compile=False),(None,180,6)) for key in NEURAL]
        selected=self.frozen['selected'];self.policy=self.frozen['verifiers'][selected] if selected else None
        self.verifier=NativeVerifier(self.policy['family'],folder/self.policy['model_dir']) if self.policy else None
        self.reset();self.setup_seconds=time.perf_counter()-tic
    def reset(self):
        self.buffer=deque(maxlen=180);self.ticks=0;self.last=None
        self.heads=[Alarm(t,CFG) for t in [.675,.775,.875]];self.basepulse=PulseUnion(1,3);self.rescuepulse=PulseUnion(1,3);self.physical=PhysicalOR()
        self.gate=CandidateGate(**self.frozen['level1']);self.features=CascadeFeatures(self.frozen['level1']);self.feature_history=deque(maxlen=13)
        self.lastbase=-np.inf;self.last_x=None;self.last_q=None;self.last_scores=None
        self.rescue_errors=0;self.last_rescue_error=None
    def step(self,t,sample,on_baseline=None):
        tick_started=time.perf_counter()
        sample=np.asarray(sample,np.float32)
        if sample.shape!=(6,) or not np.isfinite(sample).all() or not np.isfinite(t):self.reset();raise ValueError('Invalid native right-ankle sample')
        if self.last is not None and (t<=self.last or abs(t-self.last-1/60)>.0011):
            reverse=t<=self.last;self.reset()
            if reverse:raise ValueError('Non-increasing timestamp')
        self.last=t;self.buffer.append(sample.copy());self.ticks+=1
        if self.ticks<180 or (self.ticks-180)%15:return None
        raw=np.asarray(self.buffer,np.float32)[None];m=summaries(raw);z=self.fit['bank'].transform(gravity_signals(raw))
        w1,b1,w2,b2=self.fit['weights'];dp=float(expit(np.maximum(((m-self.fit['mean'])/self.fit['std'])@w1+b1,0)@w2+b2).ravel()[0])
        small=np.concatenate([z[:,:768],m],1).astype(float);large=np.concatenate([z,m],1).astype(float)
        p=[dp,float(expit(3*self.fit['models'][0].decision_function(small))[0]),float(expit(3*self.fit['models'][1].decision_function(large))[0])]
        normalized=((raw-self.mean)/self.std).astype(np.float32)
        starts=[alarm.step(t,float(value))[2] for alarm,value in zip(self.heads,p[:3])]
        ba,bs=self.basepulse.step(t,starts)
        if bs:self.lastbase=t
        baseline_ms=(time.perf_counter()-tick_started)*1000
        # A caller can deliver baseline source state immediately, before any
        # neural/rescue work. This is an independent callback, not a delayed
        # warning returned only after verification. Real cue delivery is unmeasured.
        if on_baseline is not None:on_baseline(dict(time=t,alarm_active=ba,alarm_start=bs,compute_ms=baseline_ms))
        try:
            p.extend(float(np.asarray(fn(normalized)).ravel()[0]) for fn in self.networks);p=np.array(p)
            shape=np.concatenate([m,gravity_features(raw),regularity_features(raw),z[:,:64]],1)[0]
            candidate=self.gate.step(p);x=self.features.step(p,shape,t,candidate);self.feature_history.append(x.copy())
            q=np.zeros((13,450),np.float32);q[-len(self.feature_history):]=np.array(self.feature_history)
            vp=self.verifier.step(x,q,self.policy['mode']) if candidate and self.verifier else 0.
            if not np.isfinite(vp):raise ValueError('Nonfinite rescue score')
        except Exception as error:
            # A rescue failure never resets, suppresses or waits on baseline.
            self.rescue_errors+=1;self.last_rescue_error=type(error).__name__
            self.gate.reset();self.features.reset();self.feature_history.clear()
            p=np.r_[p[:3],np.zeros(7)];candidate=False;vp=0.;x=np.zeros(450,np.float32);q=np.zeros((13,450),np.float32)
        allowed=not ba and t-self.lastbase>=3-1e-9
        threshold=self.policy['combined'] if self.policy else 1.1
        ra,rs=self.rescuepulse.step(t,[bool(candidate and vp>=threshold and allowed)])
        active,start=self.physical.step(ba,ra)
        self.last_x,self.last_q,self.last_scores=x,q,p
        return dict(time=t,probabilities=p,candidate=candidate,verifier_probability=vp,
            baseline_compute_ms=baseline_ms,
            rescue_errors=self.rescue_errors,
            baseline_active=ba,baseline_start=bs,rescue_active=ra,rescue_start=rs,alarm_active=active,alarm_start=start)
