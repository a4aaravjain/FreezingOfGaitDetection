import unittest
import numpy as np
from regularity_features import complexity, regularity_features


class RegularityTests(unittest.TestCase):
    def test_constant_finite_and_zero_complexity(self):
        x = np.ones((2, 180, 4))
        a = complexity(x)
        self.assertTrue(np.isfinite(a).all())
        np.testing.assert_array_equal(a[:, :20], 0)

    def test_periodic_more_regular_than_noise(self):
        rng = np.random.default_rng(42)
        t = np.arange(180)/60
        z = np.stack([np.sin(2*np.pi*2*t), rng.normal(size=180)], 1)[None]
        f = complexity(z)
        self.assertLess(f[0, 0], f[0, 1])  # sample entropy
        self.assertLess(f[0, 4], f[0, 5])  # spectral entropy
        self.assertGreater(f[0, 6], f[0, 7])  # autocorrelation

    def test_entropy_translation_and_scale_invariant(self):
        z = np.random.default_rng(3).normal(size=(2, 180, 3))
        np.testing.assert_allclose(complexity(z)[:, :15], complexity(5*z+20)[:, :15], atol=1e-9)

    def test_feature_does_not_depend_on_other_or_future_windows(self):
        raw = np.random.default_rng(9).normal(size=(3, 180, 6)).astype(np.float32)
        a = regularity_features(raw)
        np.testing.assert_array_equal(a[0], regularity_features(raw[:1])[0])
        raw[1:] *= 100
        np.testing.assert_array_equal(a[0], regularity_features(raw)[0])
        self.assertEqual(a.shape, (3, 96))

    def test_invalid_input_rejected(self):
        with self.assertRaises(ValueError):
            regularity_features(np.zeros((1, 181, 6)))
        with self.assertRaises(ValueError):
            regularity_features(np.full((1, 180, 6), np.nan))


if __name__ == '__main__':
    unittest.main()
