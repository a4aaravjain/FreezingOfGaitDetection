"""Label-free research stream for the pooled model. Disabled policies stay disabled."""
from pathlib import Path
from dataclasses import replace
import json
import numpy as np
from fog_pipeline import Config,OnlinePredictor,sha256
from pooled_motion import invariant_motion

class MotionStream:
    def __init__(self,run,sample_rate=64):
        import tensorflow as tf
        run=Path(run);m=json.loads((run/'manifest.json').read_text())
        if sample_rate not in (60,64):raise ValueError('Use native 60 or 64 Hz input')
        if sha256(run/'model.keras')!=m['model_sha256']:raise ValueError('Model checksum mismatch')
        self.policy_status=m['policy_status']
        self.cfg=replace(Config(),sample_rate=sample_rate,history_seconds=6,horizon_seconds=5)
        self.model=tf.keras.models.load_model(run/'model.keras')
        params=m['normalization_parameters']
        self.mean=np.array(params['mean'],np.float32);self.std=np.array(params['std'],np.float32)
        @tf.function(input_signature=[tf.TensorSpec([None,120,3],tf.float32)])
        def graph(x):return self.model(x,training=False)
        self.graph=graph
        graph(tf.zeros((1,120,3),tf.float32))  # complete tracing before accepting samples
        self.online=OnlinePredictor(self.predict_windows,{'mean':[0,0,0],'std':[1,1,1]},m['threshold'],self.cfg)

    def predict_windows(self,x):
        import tensorflow as tf
        z=(invariant_motion(x,self.cfg.sample_rate)-self.mean)/self.std
        return self.graph(tf.convert_to_tensor(z,tf.float32))

    def step(self,time_seconds,acceleration_g):
        return self.online.step(time_seconds,acceleration_g)

    def reset(self):
        self.online.reset()
