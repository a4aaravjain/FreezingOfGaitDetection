import unittest
import numpy as np
from waveform_bank import TemporalBank
from fast_temporal_bank import FastTemporalBank
class FastBankTests(unittest.TestCase):
 def test_all_features_match_original_across_seeds_and_batches(self):
  x=np.random.default_rng(22).normal(size=(7,180,4)).astype(np.float32)
  for seed in [17,42,73]:
   bank=TemporalBank(count=384,seed=seed).fit(x);fast=FastTemporalBank.from_bank(bank)
   reference=bank.transform(x,batch=3)
   for batch in [1,3,128]:np.testing.assert_allclose(fast.transform(x,batch),reference,rtol=1e-6,atol=1e-6)
 def test_extended_bank_preserves_smaller_bank_prefix(self):
  x=np.random.default_rng(81).normal(size=(4,180,4)).astype(np.float32)
  small=FastTemporalBank.from_bank(TemporalBank(count=384,seed=42).fit(x));large=FastTemporalBank.from_bank(TemporalBank(count=1024,seed=42).fit(x))
  np.testing.assert_array_equal(small.transform(x),large.transform(x)[:,:768])
 def test_invalid_values_rejected(self):
  x=np.ones((1,180,4),np.float32);fast=FastTemporalBank.from_bank(TemporalBank(count=8).fit(x));x[0,0,0]=np.nan
  with self.assertRaises(ValueError):fast.transform(x)
 def test_stationary_values_and_constant_offsets(self):
  x=np.random.default_rng(9).normal(size=(2,180,4)).astype(np.float32)
  bank=TemporalBank(count=32).fit(x);fast=FastTemporalBank.from_bank(bank)
  np.testing.assert_allclose(fast.transform(x+2),bank.transform(x+2),rtol=1e-6,atol=1e-6)
  np.testing.assert_array_equal(fast.transform(np.zeros_like(x)),bank.transform(np.zeros_like(x)))
if __name__=='__main__':unittest.main(verbosity=2)
