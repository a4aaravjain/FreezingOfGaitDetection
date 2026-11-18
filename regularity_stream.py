"""Native 60 Hz replay for the frozen development-only regularity experiment."""
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import joblib
from fog_pipeline import Alarm, Config, sha256
from long_history_experiments import summaries
from regularity_features import regularity_features


class RegularityStream:
    def __init__(self, folder, feature='entropy'):
        folder = Path(folder)
        policy = json.loads((folder/'FROZEN_POLICY.json').read_text())
        self.key = feature+'_'+policy['selected'][feature]
        if sha256(folder/(self.key+'.joblib')) != policy['hashes'][self.key]:
            raise ValueError('Frozen model checksum mismatch')
        self.model = joblib.load(folder/(self.key+'.joblib'))
        self.feature = feature
        plan = json.loads((folder.parent/'preregistered_plan.json').read_text())
        self.cfg = Config(**plan['alarm_config'])
        self.alarm = Alarm(policy['policies'][self.key],self.cfg)
        self.reset()

    def reset(self):
        self.buffer = deque(maxlen=180)
        self.tick = 0
        self.last = None
        self.alarm.reset()

    def step(self, timestamp, sample):
        sample = np.asarray(sample,np.float32)
        if sample.shape != (6,) or not np.isfinite(sample).all() or not np.isfinite(timestamp):
            self.reset()
            raise ValueError('Expected timestamp and six finite right-ankle channels')
        if self.last is not None and (timestamp <= self.last or abs(timestamp-self.last-1/60) > .0011):
            reversed_time = timestamp <= self.last
            self.reset()
            if reversed_time:
                raise ValueError('Non-increasing timestamp')
        self.last = timestamp
        self.buffer.append(sample.copy())
        self.tick += 1
        if self.tick < 180 or (self.tick-180) % 15:
            return None
        raw = np.asarray(self.buffer,np.float32)[None]
        x = summaries(raw)
        if self.feature == 'entropy':
            x = np.concatenate([x,regularity_features(raw)],1)
        p = float(self.model.predict_proba(x)[0,1])
        smoothed, active, start = self.alarm.step(timestamp,p)
        return dict(time=timestamp,probability=p,smoothed=smoothed,alarm_active=active,alarm_start=start)
