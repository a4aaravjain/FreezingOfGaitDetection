import unittest
import numpy as np
from waveform_bank import TemporalBank

class BankTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(123).normal(size=(4,180,6)).astype(np.float32)
        self.bank=TemporalBank(count=8).fit(self.x)
    def test_batch_size_and_other_windows_do_not_affect_scores(self):
        a=self.bank.transform(self.x,batch=1)
        np.testing.assert_array_equal(a,self.bank.transform(self.x,batch=3))
        changed=self.x.copy();changed[2:]=100
        np.testing.assert_array_equal(a[:2],self.bank.transform(changed)[:2])
    def test_constant_offsets_are_removed(self):
        a=self.bank.transform(self.x);b=self.bank.transform(self.x+2)
        np.testing.assert_allclose(a,b,atol=2e-6)
    def test_temporal_order_changes_features(self):
        changed=self.x[:,np.random.default_rng(9).permutation(180)]
        self.assertGreater(float(np.max(abs(self.bank.transform(self.x)-self.bank.transform(changed)))),.1)
    def test_matches_scalar_valid_convolution(self):
        z=self.x/self.bank.scale;z-=z.mean(1,keepdims=True)
        f=self.bank.transform(self.x)
        for j,(length,dilation,w,c,bias) in enumerate(self.bank.kernels):
            v=[]
            for i in range(180-(length-1)*dilation):
                v.append(sum(w[k]*np.dot(z[0,i+k*dilation],c) for k in range(length))+bias)
            self.assertAlmostEqual(f[0,j*2],np.mean(np.array(v)>0),places=6)
            self.assertAlmostEqual(f[0,j*2+1],max(v),places=5)
    def test_fixed_seed_reproduces_features(self):
        np.testing.assert_array_equal(self.bank.transform(self.x),TemporalBank(count=8).fit(self.x).transform(self.x))
    def test_invalid_input_is_rejected(self):
        with self.assertRaises(ValueError):self.bank.transform(self.x[:,:179])
        x=self.x.copy();x[0,0,0]=np.nan
        with self.assertRaises(ValueError):self.bank.transform(x)
    def test_stationary_windows_are_finite(self):
        bank=TemporalBank(count=8).fit(np.zeros_like(self.x))
        self.assertTrue(np.isfinite(bank.transform(np.zeros_like(self.x))).all())
if __name__=='__main__':unittest.main(verbosity=2)
