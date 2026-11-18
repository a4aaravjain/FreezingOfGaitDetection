import unittest
from types import SimpleNamespace
import numpy as np
import pandas as pd
from covariance_non_neural import covariance_features
from balanced_non_neural import balanced_weights

class CovarianceTests(unittest.TestCase):
    def test_constant_signals_are_finite(self):
        f=covariance_features(np.ones((2,180,4)),np.ones(4))
        self.assertEqual(f.shape,(2,87))
        self.assertTrue(np.isfinite(f).all())
    def test_covariance_invariant_to_constant_offset(self):
        x=np.random.default_rng(42).normal(size=(2,180,4))
        a=covariance_features(x,np.ones(4));b=covariance_features(x+3,np.ones(4))
        np.testing.assert_allclose(a[:,:78],b[:,:78],atol=1e-6)
        np.testing.assert_allclose(b[:,78:82]-a[:,78:82],3,atol=1e-6)
    def test_other_windows_do_not_change_features(self):
        x=np.random.default_rng(42).normal(size=(2,180,4))
        a=covariance_features(x,np.ones(4));x[1]=100
        np.testing.assert_array_equal(a[0],covariance_features(x,np.ones(4))[0])
    def test_invalid_values_rejected(self):
        x=np.ones((1,180,4));x[0,0,0]=np.nan
        with self.assertRaises(ValueError):covariance_features(x,np.ones(4))
    def test_balance_people_classes_and_episodes(self):
        data=SimpleNamespace(y=np.array([0,0,1,1,1,0,1,-1]),
          meta=pd.DataFrame(dict(subject=['a']*5+['b']*3,segment=['a']*5+['b']*3,time=[0,1,7,8,17,0,7,11])),
          segments=[SimpleNamespace(key=k,onsets=np.array([10,20])) for k in ['a','b']])
        w=balanced_weights(data)
        self.assertAlmostEqual(w[:5].sum(),w[5:].sum())
        self.assertAlmostEqual(w[:2].sum(),w[2:5].sum())
        self.assertAlmostEqual(w[2:4].sum(),w[4])
        self.assertAlmostEqual(w.mean(),1)
if __name__=='__main__':unittest.main(verbosity=2)
