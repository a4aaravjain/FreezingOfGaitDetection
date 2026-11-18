import unittest
from unittest.mock import patch
import numpy as np
from regularity_experiment import partitions, GROUPS, select_threshold, alarm_trace
from nested_rescue_experiment import make_data, scored
from test_locked_assessment import fixture
from regularity_stream import RegularityStream
from fog_pipeline import Alarm
from nested_rescue_experiment import CFG


class ExperimentTests(unittest.TestCase):
    def stream_fixture(self):
        class ConstantModel:
            def predict_proba(self,x):
                return np.tile([.2,.8],(len(x),1))
        stream = RegularityStream.__new__(RegularityStream)
        stream.model,stream.feature,stream.alarm = ConstantModel(),'entropy',Alarm(.5,CFG)
        stream.reset()
        return stream

    def test_missing_time_requires_fresh_full_history(self):
        stream = self.stream_fixture()
        for i in range(180):
            result = stream.step(i/60,np.zeros(6))
        self.assertIsNotNone(result)
        self.assertIsNone(stream.step(4,np.zeros(6)))
        for i in range(1,179):
            self.assertIsNone(stream.step(4+i/60,np.zeros(6)))
        self.assertIsNotNone(stream.step(4+179/60,np.zeros(6)))

    def test_invalid_or_reversed_time_resets_history(self):
        stream = self.stream_fixture()
        stream.step(1,np.zeros(6))
        with self.assertRaises(ValueError):
            stream.step(1,np.zeros(6))
        self.assertEqual(len(stream.buffer),0)
        with self.assertRaises(ValueError):
            stream.step(2,np.full(6,np.nan))
        self.assertEqual(len(stream.buffer),0)

    def test_caller_cannot_modify_buffered_past(self):
        stream = self.stream_fixture()
        sample = np.zeros(6,np.float32)
        stream.step(1,sample)
        sample[:] = 100
        np.testing.assert_array_equal(stream.buffer[0],0)

    def test_all_participants_evaluated_once_and_fit_cal_disjoint(self):
        self.assertEqual(sorted(sum(GROUPS, [])),list(range(1,19)))
        for outer in GROUPS:
            fit, cal = partitions(outer)
            self.assertFalse(set(fit)&set(cal) or set(fit)&set(outer) or set(cal)&set(outer))
            self.assertEqual(set(fit+cal+outer),set(range(1,19)))

    def test_selection_disables_when_only_unsafe_detections_exist(self):
        part = make_data(fixture(),[1])
        # Persistent high score starts too early and remains high: zero safe useful detections.
        threshold, _ = select_threshold(part,np.ones(len(part['meta'])))
        self.assertEqual(threshold,1.1)

    def test_late_warning_never_counts_as_two_second_prediction(self):
        part = make_data(fixture(),[1])
        p = np.zeros(len(part['meta']))
        p[part['meta'].time>=8.1] = 1
        result = scored(part,alarm_trace(part,p,.5))
        self.assertEqual(result['metrics']['true_warning_events'],0)

    def test_alarm_outputs_independent_of_annotations(self):
        part = make_data(fixture(),[1])
        p = np.linspace(0,1,len(part['meta']))
        a = alarm_trace(part,p,.5)
        part['meta']['target'] = 1
        part['meta']['active_fog'] = True
        part['meta']['known_followup'] = False
        b = alarm_trace(part,p,.5)
        np.testing.assert_array_equal(a.alarm_start,b.alarm_start)
        np.testing.assert_array_equal(a.alarm_active,b.alarm_active)


if __name__ == '__main__':
    unittest.main()
