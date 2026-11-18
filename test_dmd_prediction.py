import unittest
import numpy as np
from dmd_prediction import dmd_features
class DMDTests(unittest.TestCase):
 def test_stationary_signal_is_finite_and_zero_after_centering(self):
  x=np.zeros((2,180,6),np.float32);x[:,:,2]=1
  np.testing.assert_array_equal(dmd_features(x),np.zeros((2,10)))
 def test_other_windows_cannot_change_current_features(self):
  rng=np.random.default_rng(42);x=rng.normal(size=(2,180,6)).astype(np.float32);before=dmd_features(x);x[1]*=100
  np.testing.assert_array_equal(before[0],dmd_features(x)[0])
 def test_dynamic_signal_differs_from_stationary(self):
  x=np.zeros((1,180,6),np.float32);t=np.arange(180)/60;x[0,:,2]=1+.1*np.sin(2*np.pi*2*t);x[0,:,0]=.1*np.cos(2*np.pi*2*t)
  f=dmd_features(x);self.assertTrue(np.isfinite(f).all());self.assertGreater(float(np.linalg.norm(f)),0)
if __name__=='__main__':unittest.main()
