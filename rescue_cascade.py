"""Independent baseline channel and same-tick rescue. Labels are not inference inputs."""
from collections import deque
from dataclasses import replace
import numpy as np
from fog_pipeline import Alarm
from long_history_experiments import summaries

class RescueFeatures:
 def __init__(self,level1_threshold):self.threshold=level1_threshold;self.reset()
 def reset(self):self.history=deque(maxlen=4);self.previous_delta=np.zeros(3);self.run=0
 def step(self,scores,motion):
  p=np.asarray(scores,float);m=np.asarray(motion,float)
  if p.shape!=(3,) or m.shape!=(24,) or not np.isfinite(p).all() or not np.isfinite(m).all() or (p<0).any() or (p>1).any():raise ValueError('Invalid score/motion input')
  candidate=bool(p.max()>=self.threshold);self.run=self.run+1 if candidate else 0
  previous=self.history[-1][0] if self.history else p
  oldp,oldm=self.history[0] if len(self.history)==4 else (p,m)
  delta=p-previous
  result=np.r_[p,delta,p-oldp,delta-self.previous_delta,self.run*.25,[p.min(),p.mean(),p.max(),p.std()],m,m-oldm]
  self.history.append((p.copy(),m.copy()));self.previous_delta=delta.copy();return candidate,result

class PhysicalOR:
 def reset(self):self.previous=False
 def __init__(self):self.reset()
 def step(self,baseline_active,rescue_active):
  active=bool(baseline_active or rescue_active);start=active and not self.previous;self.previous=active;return active,start

class RescueCascade:
 def __init__(self,baseline,verifier,level1_threshold,verifier_threshold):
  self.baseline=baseline;self.verifier=verifier;self.features=RescueFeatures(level1_threshold)
  self.rescue=Alarm(verifier_threshold,replace(baseline.cfg,ema_alpha=1,persistence=1,alert_pulse_seconds=1));self.output=PhysicalOR();self.reset()
 def reset(self):self.baseline.reset();self.features.reset();self.rescue.reset();self.output.reset();self.last=None;self.rescue_errors=0
 def step(self,time,sample):
  sample=np.asarray(sample,np.float32)
  if sample.shape!=(6,) or not np.isfinite(sample).all() or not np.isfinite(time):self.reset();raise ValueError('Invalid native sample')
  if self.last is not None and (time<=self.last or abs(time-self.last-1/60)>.0011):
   reversed_time=time<=self.last;self.reset()
   if reversed_time:raise ValueError('Non-increasing time')
  self.last=time;baseline=self.baseline.step(time,sample)
  if baseline is None:return None
  try:
   motion=summaries(np.asarray(self.baseline.base.buffer,np.float32)[None])[0]
   scores=[baseline[k+'_probability'] for k in ['base','waveform','extended']]
   candidate,features=self.features.step(scores,motion)
   p=float(self.verifier.predict_proba(features[None])[0,1]) if candidate else 0.
   _,rescue_active,rescue_start=self.rescue.step(time,p)
  except Exception:
   # Rescue failure cannot alter the independent baseline result.
   self.rescue_errors+=1;self.rescue.reset();candidate=False;p=0.;rescue_active=rescue_start=False
  active,start=self.output.step(baseline['alarm_active'],rescue_active)
  return dict(time=time,baseline=baseline,candidate=candidate,verifier_probability=p,rescue_active=rescue_active,rescue_start=rescue_start,alarm_active=active,alarm_start=start,rescue_errors=self.rescue_errors)

def validate_provenance(provenance,held_subject,outer_subjects,calibration_subjects=()):
 for head in ['base','waveform','extended']:
  fit=set(provenance[head]['training_subjects'])
  if not fit or fit & ({held_subject}|set(outer_subjects)|set(calibration_subjects)):raise ValueError('Upstream participant leakage')

def candidate_label(target,active_fog,known_followup):
 if not known_followup:return None
 if active_fog:return 0
 return int(target) if target in [0,1] else None

def poisson_interval(false_starts,hours):
 if hours<=0:return [None,None]
 from scipy.stats import chi2
 lower=0. if false_starts==0 else .5*chi2.ppf(.025,2*false_starts)/hours
 upper=.5*chi2.ppf(.975,2*(false_starts+1))/hours
 return [float(lower),float(upper)]


def physical_preservation(baseline,combined):
    # This checks actual output edges, not merely a logical OR of requests.
    if len(baseline)!=len(combined):return False
    for key in ['segment','time']:
        if not np.array_equal(baseline[key].to_numpy(),combined[key].to_numpy()):return False
    if not np.array_equal(baseline.alarm_start.to_numpy(),combined.baseline_start.to_numpy()):return False
    if not np.array_equal(baseline.alarm_active.to_numpy(),combined.baseline_active.to_numpy()):return False
    return bool(np.all(~baseline.alarm_start.to_numpy(bool)|combined.alarm_start.to_numpy(bool)))
