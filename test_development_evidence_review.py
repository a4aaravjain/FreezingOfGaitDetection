import unittest
from development_evidence_review import diagnose,longest_run
class ReviewTests(unittest.TestCase):
 def test_low_score(self):self.assertEqual(diagnose([.2],[.1],[False],.7),'score_below_threshold')
 def test_smoothing(self):self.assertEqual(diagnose([.9],[.6],[False],.7),'smoothing_candidate')
 def test_persistence(self):self.assertEqual(diagnose([.8,.8],[.6,.8],[False,False],.7),'persistence_candidate')
 def test_state(self):self.assertEqual(diagnose([.8,.8],[.8,.8],[False,False],.7),'state_or_cooldown_candidate')
 def test_start(self):self.assertEqual(diagnose([.8],[.8],[True],.7),'component_start_present')
 def test_empty(self):self.assertEqual(diagnose([],[],[],.7),'no_eligible_window')
 def test_consecutive(self):self.assertEqual(longest_run([True,True,False,True]),2)
if __name__=='__main__':unittest.main()
