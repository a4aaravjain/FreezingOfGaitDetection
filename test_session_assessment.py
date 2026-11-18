import unittest
import numpy as np
from test_locked_assessment import fixture,SilentStream
from session_assessment import prepare_sessions,make_session_data,score_sessions
from assessment_core import replay,score,CHANNELS
from nested_rescue_experiment import CFG


class SessionTests(unittest.TestCase):
    def test_task_boundary_does_not_break_native_history(self):
        f=fixture();f.loc[360:,'taskID']=2
        records,chunks,inventory,bad=prepare_sessions(f,[1])
        self.assertEqual(len(records),1);self.assertEqual(len(chunks),1)
        self.assertFalse(bad)
        data=make_session_data(f,[1],CFG)
        self.assertTrue(np.any((data['meta'].time>=6)&(data['meta'].time<7)))
        np.testing.assert_array_equal(records[0].onsets,[10])

    def test_onset_exactly_at_task_boundary_is_retained(self):
        f=fixture();f.loc[600:,'taskID']=2
        records,_,inventory,_=prepare_sessions(f,[1])
        self.assertEqual(sum(i['original_onsets'] for i in inventory),1)
        np.testing.assert_array_equal(records[0].onsets,[10])

    def test_real_timestamp_gap_is_never_joined(self):
        f=fixture();f=f.drop(index=350).reset_index(drop=True)
        records,chunks,_,bad=prepare_sessions(f,[1])
        self.assertTrue(bad);self.assertEqual(len(records),2);self.assertEqual(len(chunks),2)
        self.assertGreater(chunks[1]['time'][0]-chunks[0]['time'][-1],1/60)

    def test_sensor_gap_still_resets_history_but_retains_onset(self):
        f=fixture();f.loc[590:620,CHANNELS]=np.nan
        records,chunks,inventory,bad=prepare_sessions(f,[1])
        self.assertEqual(len(records),1);self.assertEqual(len(chunks),2)
        self.assertEqual(score(records,replay(chunks,SilentStream()),inventory,bad,CFG)['metrics']['total_fog_episodes'],1)

    def test_followup_across_task_boundary_is_known(self):
        f=fixture();f.loc[480:,'taskID']=2
        data=make_session_data(f,[1],CFG)
        rows=data['meta'][(data['meta'].time>5)&(data['meta'].time<7)]
        self.assertTrue(rows.known_followup.all())
        self.assertTrue(rows.target.eq(1).all())

    def test_nonincreasing_session_time_rejected(self):
        f=fixture();f.loc[360:,'timestamp']-=6
        with self.assertRaises(ValueError):prepare_sessions(f,[1])

    def test_prediction_assigned_to_wrong_session_rejected(self):
        import pandas as pd
        f=fixture();r,_,i,bad=prepare_sessions(f,[1])
        p=pd.DataFrame(dict(segment=[r[0].key],time=[100.]))
        with self.assertRaises(ValueError):score_sessions(r,p,i,bad,CFG)

    def test_task_names_do_not_affect_features_or_decisions(self):
        f=fixture();a=make_session_data(f,[1],CFG)
        f['taskID']=np.arange(len(f))%4
        b=make_session_data(f,[1],CFG)
        np.testing.assert_array_equal(a['data'].x,b['data'].x)
        np.testing.assert_array_equal(a['meta'].time,b['meta'].time)
        np.testing.assert_array_equal(a['meta'].target,b['meta'].target)


if __name__=='__main__':unittest.main()
