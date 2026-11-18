"""Frozen single-right-ankle context predictor at native 60 Hz."""
from pathlib import Path
from collections import deque
import json
import numpy as np,joblib
from fog_pipeline import Alarm,sha256
from nested_rescue_experiment import CFG
from causal_motion_context import CausalMotionContext
from long_history_experiments import summaries


class SessionContextStream:
    def __init__(self,folder,key='context'):
        folder=Path(folder)
        policy=json.loads((folder/'FROZEN_POLICY.json').read_text())
        plan=json.loads((folder.parent/'preregistered_plan.json').read_text())
        root=Path(__file__).resolve().parent
        for name in ['nested_rescue_experiment.py','fog_pipeline.py','causal_motion_context.py','long_history_experiments.py']:
            if sha256(root/name)!=plan['source_sha256'][name]:raise ValueError('Unfrozen runtime dependency')
        if sha256(folder/(key+'.joblib'))!=policy['model_hashes'][key]:raise ValueError('Frozen model mismatch')
        self.model=joblib.load(folder/(key+'.joblib'));self.key=key;self.cfg=CFG
        self.context=CausalMotionContext(policy['floor']);self.alarm=Alarm(policy['thresholds'][key],CFG)
        self.reset()
    def reset(self):
        self.buffer=deque(maxlen=180);self.tick=0;self.last=None
        self.context.reset();self.alarm.reset()
    def step(self,timestamp,sample):
        x=np.asarray(sample,np.float32)
        if x.shape!=(6,) or not np.isfinite(x).all() or not np.isfinite(timestamp):
            self.reset();raise ValueError('Invalid native timestamp/right-ankle input')
        if self.last is not None and (timestamp<=self.last or abs(timestamp-self.last-1/60)>.0011):
            reversed_time=timestamp<=self.last;self.reset()
            if reversed_time:raise ValueError('Non-increasing timestamp')
        self.last=timestamp;self.buffer.append(x.copy());self.tick+=1
        if self.tick<180 or (self.tick-180)%15:return None
        raw=np.asarray(self.buffer,np.float32)[None]
        features=summaries(raw)[0]
        if self.key=='context':features=self.context.step(features)
        p=float(self.model.predict_proba(features[None])[0,1])
        smoothed,active,start=self.alarm.step(timestamp,p)
        return dict(time=timestamp,probability=p,smoothed=smoothed,alarm_active=active,alarm_start=start)
