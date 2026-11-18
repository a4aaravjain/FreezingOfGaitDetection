import unittest
import numpy as np
from privileged_distillation import training_only_extra,right_features
from distilled_risk import DistilledRisk
from test_locked_assessment import fixture
from nested_rescue_experiment import make_data


class DistillationTests(unittest.TestCase):
    def test_teacher_windows_match_right_timestamps_after_sensor_gap(self):
        import pandas as pd
        f = pd.concat([fixture(),fixture()],ignore_index=True)
        f.timestamp = np.arange(len(f))/60
        for column in list(f.columns):
            if column.startswith('ankleR_'):
                f[column] = np.sin(np.arange(len(f))/9)+np.arange(len(f))/1000
                f[column.replace('ankleR_','ankleL_')] = f[column]
                f[column.replace('ankleR_','back_')] = f[column]
        f.loc[200:230,[c for c in f if c.startswith('ankleR_')]] = np.nan
        part = make_data(f,[1])
        self.assertTrue(part['meta'].chunk.str.endswith(':231').any())
        extra = training_only_extra(f,part)
        x = right_features(part)
        np.testing.assert_allclose(extra[:,:120],x,atol=1e-6)
        np.testing.assert_allclose(extra[:,120:],x,atol=1e-6)

    def test_missing_teacher_window_does_not_remove_right_window(self):
        f = fixture()
        for column in list(f.columns):
            if column.startswith('ankleR_'):
                f[column.replace('ankleR_','ankleL_')] = np.nan
                f[column.replace('ankleR_','back_')] = f[column]
        part = make_data(f,[1])
        extra = training_only_extra(f,part)
        self.assertEqual(len(extra),len(part['meta']))
        self.assertTrue(np.isnan(extra[:,:120]).all())
        self.assertTrue(np.isfinite(extra[:,120:]).all())

    def test_regressor_probability_interface_is_bounded(self):
        class Fake:
            def predict(self,x):return np.array([-.2,.4,1.2])
        p = DistilledRisk(Fake()).predict_proba(np.zeros((3,120)))
        np.testing.assert_array_equal(p[:,1],[0,.4,1])
        np.testing.assert_array_equal(p.sum(1),1)


if __name__ == '__main__':
    unittest.main()
