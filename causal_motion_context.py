"""Past-only local gait normalization; no labels, task names or participant IDs."""
from collections import deque
import numpy as np


class CausalMotionContext:
    def __init__(self,floor,steps=80):
        self.floor=np.asarray(floor,float)
        if self.floor.shape!=(24,) or not np.isfinite(self.floor).all() or (self.floor<=0).any():
            raise ValueError('Invalid fit-only normalization floor')
        self.steps=steps;self.reset()
    def reset(self):self.past=deque(maxlen=self.steps)
    def step(self,motion):
        m=np.asarray(motion,float)
        if m.shape!=(24,) or not np.isfinite(m).all():raise ValueError('Invalid motion summary')
        if self.past:
            past=np.array(self.past)
            median=np.median(past,axis=0)
            q25,q75=np.quantile(past,[.25,.75],axis=0)
            delta=m-median
            z=np.clip(delta/np.maximum(q75-q25,self.floor),-10,10)
        else:delta=np.zeros(24);z=np.zeros(24)
        output=np.r_[m,delta,z,len(self.past)/self.steps].astype(np.float32)
        # Update only after deriving the current features from prior observations.
        self.past.append(m.copy())
        return output
