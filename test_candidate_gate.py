import unittest
from candidate_gate import compare_candidates
class GateTests(unittest.TestCase):
 def setUp(self):
  self.metrics=dict(total_false_alarms=0,censored_alarm_starts=1,false_alarm_seconds=0,maximum_alarm_duration_seconds=1,total_fog_episodes=16,monitored_seconds=600)
  self.events=[dict(segment='a',onset=10,warning_seconds=3)]
 def check(self,m,e):return compare_candidates(self.metrics,self.events,m,e)['eligible_for_streaming_verification']
 def test_accepts_additional_episode_with_preserved_warning(self):self.assertTrue(self.check(self.metrics,self.events+[dict(segment='b',onset=20,warning_seconds=2.5)]))
 def test_rejects_unknown_alert_increase(self):self.assertFalse(self.check(dict(self.metrics,censored_alarm_starts=2),[dict(segment='a',onset=10,warning_seconds=4)]))
 def test_rejects_previous_warning_later_despite_new_success(self):self.assertFalse(self.check(self.metrics,[dict(segment='a',onset=10,warning_seconds=2.5),dict(segment='b',onset=20,warning_seconds=3)]))
 def test_rejects_changed_exposure(self):self.assertFalse(self.check(dict(self.metrics,monitored_seconds=500),[dict(segment='a',onset=10,warning_seconds=4)]))
 def test_rejects_identical_candidate(self):self.assertFalse(self.check(self.metrics,self.events))
 def test_accepts_earlier_warning(self):self.assertTrue(self.check(self.metrics,[dict(segment='a',onset=10,warning_seconds=3.5)]))
if __name__=='__main__':unittest.main()
