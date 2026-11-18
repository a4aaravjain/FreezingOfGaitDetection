import unittest,numpy as np
from test_locked_assessment import fixture
from nested_rescue_experiment import make_data,baseline_trace,scored,candidate_stats,rescue_trace
from rescue_cascade import physical_preservation
class NestedTests(unittest.TestCase):
 def test_original_onset_survives_sensor_gap_and_labels_stay_known(self):
  f=fixture();f.loc[590:620,'ankleR_acc_x']=np.nan;d=make_data(f,[1]);self.assertEqual(sum(i['original_onsets'] for i in d['inventory']),1);self.assertTrue(d['meta'].known_followup.any());self.assertTrue(d['meta'].target.eq(1).any())
 def test_disabled_rescue_exactly_preserves_baseline(self):
  d=make_data(fixture(),[1]);p=np.zeros((len(d['meta']),3));base=baseline_trace(d,p);rescue=rescue_trace(d,base,np.ones(len(p),bool),np.ones(len(p)),1.1);self.assertTrue(physical_preservation(base,rescue));self.assertEqual(scored(d,rescue)['metrics']['total_fog_episodes'],1)
 def test_all_positive_candidates_have_full_occupancy(self):
  d=make_data(fixture(),[1]);s=candidate_stats(d,np.ones(len(d['meta']),bool));self.assertAlmostEqual(s['candidate_positive_occupancy'],1);self.assertEqual(s['opportunity_recall'],1)
if __name__=='__main__':unittest.main()
