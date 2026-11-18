import unittest,numpy as np
from causal_motion_context import CausalMotionContext


class ContextTests(unittest.TestCase):
    def test_current_peak_cannot_change_its_own_reference(self):
        c=CausalMotionContext(np.ones(24))
        c.step(np.zeros(24))
        x=c.step(np.ones(24)*4)
        np.testing.assert_array_equal(x[24:48],4)
        np.testing.assert_array_equal(x[48:72],4)

    def test_future_changes_cannot_affect_already_generated_features(self):
        rng=np.random.default_rng(1);m=rng.normal(size=(100,24))
        c=CausalMotionContext(np.ones(24));a=np.array([c.step(x) for x in m])
        m[50:]*=100
        c=CausalMotionContext(np.ones(24));b=np.array([c.step(x) for x in m])
        np.testing.assert_array_equal(a[:50],b[:50])

    def test_reset_discards_previous_gait_reference(self):
        c=CausalMotionContext(np.ones(24))
        c.step(np.ones(24)*100);c.reset()
        np.testing.assert_array_equal(c.step(np.zeros(24)),0)

    def test_stationary_history_is_finite_and_caller_cannot_mutate_it(self):
        c=CausalMotionContext(np.ones(24)*.01);m=np.zeros(24)
        c.step(m);m[:]=100
        x=c.step(np.zeros(24))
        self.assertTrue(np.isfinite(x).all());np.testing.assert_array_equal(x[24:72],0)


if __name__=='__main__':unittest.main()
