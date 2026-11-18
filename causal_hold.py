"""Bounded causal sample holding. Never use a future reading to fill a gap."""
import numpy as np

class CausalHold:
    def __init__(self,sample_rate=60,max_hold_seconds=.1,channels=6):
        if sample_rate<=0 or max_hold_seconds<0 or channels<1:raise ValueError('Invalid hold settings')
        self.rate=sample_rate;self.limit=max_hold_seconds;self.channels=channels;self.reset()
    def reset(self):self.last_time=None;self.observed_time=None;self.observed=None
    def step(self,t,values):
        x=np.asarray(values,np.float32)
        if x.shape!=(self.channels,) or not np.isfinite(t):self.reset();raise ValueError('Invalid sample')
        if self.last_time is not None:
            dt=t-self.last_time
            if dt<=0:self.reset();raise ValueError('Non-increasing time')
            if abs(dt-1/self.rate)>.0011:self.reset()
        self.last_time=t
        if np.isfinite(x).all():
            self.observed=x.copy();self.observed_time=t
            return np.r_[x,np.float32(0)].astype(np.float32)
        if self.observed is not None and t-self.observed_time<=self.limit+1e-9:
            return np.r_[self.observed,np.float32(1)].astype(np.float32)
        return None
