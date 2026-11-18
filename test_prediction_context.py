import unittest
from types import SimpleNamespace
import numpy as np,pandas as pd
from trend_warning import ScoreTrend
from context_waveform import context_windows
class TrendTests(unittest.TestCase):
 def test_uses_only_prior_scores(self):
  a=ScoreTrend(.5);b=ScoreTrend(.5)
  prefix=[.1,.2,.3,.4,.5]
  self.assertEqual([a.step(v) for v in prefix],[b.step(v) for v in prefix]);self.assertAlmostEqual(a.step(.6),.8)
 def test_reset_removes_prior_segment(self):
  a=ScoreTrend(.5)
  for v in [.1,.2,.3,.4]:a.step(v)
  a.reset();self.assertEqual(a.step(.9),.9)
 def test_bounded_and_rejects_invalid(self):
  a=ScoreTrend(2.)
  for v in [0,0,0,0]:a.step(v)
  self.assertEqual(a.step(1.),1.)
  for v in [np.nan,-.1,1.1]:
   with self.assertRaises(ValueError):a.step(v)
class ContextTests(unittest.TestCase):
 def data(self):
  raw=np.repeat(np.arange(600,dtype=np.float32)[:,None],6,axis=1)
  return SimpleNamespace(x=np.zeros((3,180,6),np.float32),y=np.zeros(3),meta=pd.DataFrame(dict(segment=['a']*3,end_row=[179,359,539])),segments=[SimpleNamespace(key='a',xyz_g=raw)])
 def test_padding_fraction_and_latest_pair(self):
  x,f=context_windows(self.data(),6)
  self.assertEqual(f[0,0],.5);self.assertEqual(f[1,0],1.)
  np.testing.assert_array_equal(x[0,:90],0)
  self.assertEqual(x[1,-1,0],358.5);self.assertEqual(x[2,-1,0],538.5)
 def test_future_samples_do_not_affect_earlier_windows(self):
  d=self.data();a,_=context_windows(d,6);d.segments[0].xyz_g[360:]=9999;b,_=context_windows(d,6)
  np.testing.assert_array_equal(a[:2],b[:2])
 def test_invalid_context_rejected(self):
  with self.assertRaises(ValueError):context_windows(self.data(),5)
if __name__=='__main__':unittest.main(verbosity=2)
