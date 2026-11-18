import unittest,tempfile,json
from pathlib import Path
from types import SimpleNamespace
from collections import deque
from dataclasses import replace
import numpy as np,pandas as pd
from fog_pipeline import Config
from rescue_cascade import RescueFeatures,PhysicalOR,RescueCascade,candidate_label,validate_provenance,poisson_interval,physical_preservation
from cascade_verifier_training import ledger_features,fit_pair
class Verifier:
 def predict_proba(self,x):return np.tile([0.,1.],(len(x),1))
class Baseline:
 cfg=replace(Config(),sample_rate=60,alert_pulse_seconds=1)
 def reset(self):self.n=0;self.base=SimpleNamespace(buffer=deque(maxlen=180))
 def __init__(self):self.reset()
 def step(self,t,x):
  self.n+=1;self.base.buffer.append(x.copy())
  if self.n<180 or (self.n-180)%15:return None
  return dict(time=t,base_probability=.8,waveform_probability=.3,extended_probability=.4,alarm_start=self.n==210,alarm_active=210<=self.n<270)
def ledger():
 rows=[]
 for i in range(200):
  y=i%4==0;p=.9 if y else .2
  rows.append(dict(subject=1,segment='synthetic',time=i*.25,upstream_id='a',target=int(y),active_fog=False,known_followup=True,**{h+'_probability':p for h in ['base','waveform','extended']},**{'motion_'+str(j):float(y)+j*.01 for j in range(24)}))
 return pd.DataFrame(rows),{'a':{h:dict(training_subjects=[2,3]) for h in ['base','waveform','extended']}}
class CascadeTests(unittest.TestCase):
 def test_prefix_causality(self):
  a=RescueFeatures(.3);b=RescueFeatures(.3);rng=np.random.default_rng(9)
  for i in range(12):
   p=rng.uniform(size=3);m=rng.normal(size=24);np.testing.assert_array_equal(a.step(p,m)[1],b.step(p,m)[1])
  before=a.history[0][0].copy();b.step(np.ones(3),np.zeros(24));np.testing.assert_array_equal(a.history[0][0],before)
 def test_unknown_and_active_candidate_labels(self):
  self.assertIsNone(candidate_label(0,False,False));self.assertIsNone(candidate_label(-1,True,False));self.assertEqual(candidate_label(-1,True,True),0)
 def test_outer_or_held_leakage_rejected(self):
  _,p=ledger()
  with self.assertRaises(ValueError):validate_provenance(p['a'],1,[2])
  with self.assertRaises(ValueError):validate_provenance(p['a'],2,[9])
  validate_provenance(p['a'],1,[9])
 def test_labels_not_numeric_inputs(self):
  f,p=ledger();x,_,_=ledger_features(f,p,[9],[8],.15);f.target=1-f.target;f.active_fog=True;y,_,_=ledger_features(f,p,[9],[8],.15);np.testing.assert_array_equal(x,y)
 def test_same_tick_and_independent_baseline(self):
  ref=Baseline();cascade=RescueCascade(Baseline(),Verifier(),.15,.5);outputs=[]
  for i in range(300):
   t=i/60;sample=np.zeros(6);a=ref.step(t,sample);b=cascade.step(t,sample)
   self.assertEqual(a,None if b is None else b['baseline'])
   if b:outputs.append(b);self.assertEqual(b['time'],t);self.assertGreaterEqual(b['alarm_active'],b['baseline']['alarm_active'])
  self.assertTrue(outputs[0]['rescue_start'])
 def test_invalid_time_and_sample_reset_both_levels(self):
  for kind in ['gap','repeat','reverse','nan']:
   c=RescueCascade(Baseline(),Verifier(),.15,.5)
   for i in range(200):c.step(i/60,np.zeros(6))
   if kind=='gap':self.assertIsNone(c.step(10,np.zeros(6)))
   else:
    t=199/60 if kind=='repeat' else 0 if kind=='reverse' else 201/60
    with self.assertRaises(ValueError):c.step(t,[np.nan]*6 if kind=='nan' else np.zeros(6))
   self.assertEqual(len(c.features.history),0);self.assertIsNone(c.rescue.ema)
 def test_physical_or_collision_is_visible_not_baseline_suppression(self):
  p=PhysicalOR();self.assertEqual(p.step(False,True),(True,True));self.assertEqual(p.step(True,True),(True,False))
 def test_collision_blocks_physical_preservation(self):
  a=pd.DataFrame(dict(segment=['s','s'],time=[0,.25],alarm_start=[False,True],alarm_active=[False,True]));b=a.copy();b['baseline_start']=a.alarm_start;b['baseline_active']=a.alarm_active;b['alarm_start']=[True,False]
  self.assertFalse(physical_preservation(a,b))
 def test_poisson_zero_rate_upper_bound(self):self.assertGreater(poisson_interval(0,.18156)[1],20)
 def test_families_reload_on_synthetic_candidates(self):
  f,p=ledger()
  with tempfile.TemporaryDirectory() as d:
   models=fit_pair(f,p,[9],[8],.15,Path(d)/'fit');self.assertEqual(set(models),{'logistic','hgb'})
   x,_,_=ledger_features(f,p,[9],[8],.15)
   for m in models.values():self.assertEqual(m.predict_proba(x).shape,(len(f),2))
if __name__=='__main__':unittest.main()
