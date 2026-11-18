"""Research-only native 60 Hz right-ankle summary predictor, six sensor channels."""
from collections import deque
from pathlib import Path
import json
import hashlib
import numpy as np
from fog_pipeline import Alarm,Config
from long_history_experiments import summaries

class SummaryStream:
    def __init__(self, run, ankle="ankleR", engine="numpy"):
        run=Path(run)
        manifest=json.loads((run/'manifest.json').read_text())
        if ankle != 'ankleR' or manifest.get('ankle') != 'ankleR':
            raise ValueError('This stream requires a right-ankle model and right-ankle sensor input')
        self.cfg=Config(**manifest['config'])
        if self.cfg.sample_rate!=60 or self.cfg.history_rows!=180:
            raise ValueError('This feature implementation requires 60 Hz and 3 seconds')
        result=next(r for r in json.loads((run/'results.json').read_text()) if r['model']=='current_summary_mlp')
        norm=json.loads((run/'normalization.json').read_text())
        self.mean=np.array(norm['mean'],np.float32);self.std=np.array(norm['std'],np.float32)
        if engine == 'numpy':
            export=json.loads((run/'numpy_export.json').read_text())
            for filename,key in [('current_summary_mlp.keras','model_sha256'),('current_summary_mlp_weights.npz','weights_sha256')]:
                if hashlib.sha256((run/filename).read_bytes()).hexdigest()!=export[key]:
                    raise ValueError('Model/export checksum mismatch')
            with np.load(run/'current_summary_mlp_weights.npz',allow_pickle=False) as data:
                w1,b1,w2,b2=[data[k].copy() for k in ['w1','b1','w2','b2']]
            if w1.shape!=(24,32) or b1.shape!=(32,) or w2.shape!=(32,1) or b2.shape!=(1,):
                raise ValueError('Unsupported exported architecture')
            if not all(np.isfinite(a).all() for a in [w1,b1,w2,b2]):raise ValueError('Nonfinite weights')
            from scipy.special import expit
            self.predict=lambda x:expit(np.maximum(x@w1+b1,0)@w2+b2)
        elif engine == 'tensorflow':
            import tensorflow as tf
            model=tf.keras.models.load_model(run/'current_summary_mlp.keras')
            self.predict=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec([1,24],tf.float32)])
        else:raise ValueError('Unknown inference engine')
        self.predict(np.zeros((1,24),np.float32))
        self.alarm=Alarm(result['threshold'],self.cfg)
        self.reset()

    def reset(self):
        self.buffer=deque(maxlen=180);self.count=0;self.last=None;self.alarm.reset()

    def step(self,time,acc_g_gyro_degrees_per_second):
        x=np.asarray(acc_g_gyro_degrees_per_second,np.float32)
        if x.shape!=(6,) or not np.isfinite(x).all() or not np.isfinite(time):
            self.reset();raise ValueError('Expected finite timestamp and six acceleration/rotation values')
        if self.last is not None:
            dt=time-self.last
            if dt<=0 or abs(dt-1/60)>.0011:
                self.reset()
                if dt<=0:raise ValueError('Non-increasing timestamp; stream reset')
        self.last=time;self.buffer.append(x.copy());self.count+=1
        if self.count<180 or (self.count-180)%self.cfg.interval_rows:return None
        f=(summaries(np.asarray(self.buffer)[None])-self.mean)/self.std
        p=float(np.asarray(self.predict(f)).ravel()[0])
        smoothed,active,start=self.alarm.step(time,p)
        return dict(time=time,probability=p,smoothed=smoothed,alarm_active=active,alarm_start=start)
