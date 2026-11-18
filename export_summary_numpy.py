"""Export the small Dense-ReLU-Dense-Sigmoid network without TensorFlow at inference."""
from pathlib import Path
import json,hashlib
import numpy as np
import tensorflow as tf
r=Path('runs/long_history_round_1')
m=tf.keras.models.load_model(r/'current_summary_mlp.keras')
assert [type(l).__name__ for l in m.layers]==['Dense','Dropout','Dense']
assert m.layers[0].activation.__name__=='relu' and m.layers[2].activation.__name__=='sigmoid'
w1,b1,w2,b2=m.get_weights();assert w1.shape==(24,32) and w2.shape==(32,1)
np.savez(r/'current_summary_mlp_weights.npz',w1=w1,b1=b1,w2=w2,b2=b2)
(r/'numpy_export.json').write_text(json.dumps(dict(model_sha256=hashlib.sha256((r/'current_summary_mlp.keras').read_bytes()).hexdigest(),weights_sha256=hashlib.sha256((r/'current_summary_mlp_weights.npz').read_bytes()).hexdigest(),architecture='Dense32 relu, inference dropout identity, Dense1 sigmoid'),indent=2))
print('Exported verified architecture')
