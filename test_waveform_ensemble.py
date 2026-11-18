import unittest
import numpy as np
from waveform_ensemble import PulseUnion,WaveformEnsemble
class UnionTests(unittest.TestCase):
 def test_simultaneous_sources_make_one_pulse(self):
  p=PulseUnion();self.assertEqual(p.step(0,[True,True]),(True,True));self.assertEqual(p.step(.25,[True,False]),(True,False));self.assertEqual(p.step(1,[False,False]),(False,False))
 def test_refractory_prevents_retrigger_without_delayed_queue(self):
  p=PulseUnion();p.step(0,[True]);self.assertEqual(p.step(2.75,[True]),(False,False));self.assertEqual(p.step(3,[False]),(False,False));self.assertEqual(p.step(3.25,[True]),(True,True))
 def test_reset_does_not_carry_pulse_across_segments(self):
  p=PulseUnion();p.step(10,[True]);p.reset();self.assertEqual(p.step(10.25,[False]),(False,False));self.assertEqual(p.step(10.5,[True]),(True,True))
 def test_both_sources_can_initiate(self):
  for starts in [[True,False],[False,True]]:self.assertEqual(PulseUnion().step(0,starts),(True,True))
 def test_invalid_pulse_policy_rejected(self):
  for pulse,cooldown in [(0,3),(4,3),(-1,3)]:
   with self.assertRaises(ValueError):PulseUnion(pulse,cooldown)
if __name__=='__main__':unittest.main(verbosity=2)
