"""Prespecified temporal-model development experiments; no test-set feedback."""
from pathlib import Path
from dataclasses import replace
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','4')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
from fog_pipeline import Config,save_json
from ankles import ROOT,run_development

def main():
    out=ROOT/'runs'/'temporal_round_1';out.mkdir(exist_ok=False)
    trials=[dict(name='tcn_6s_h3',architecture='tcn',history=6.,horizon=3.),
            dict(name='tcn_6s_h5',architecture='tcn',history=6.,horizon=5.),
            dict(name='gru_6s_h5',architecture='gru',history=6.,horizon=5.)]
    save_json(out/'planned_experiments.json',dict(trials=trials,epochs=24,seed=42,
        minimum_lead_seconds=2,train=['S01','S02','S03','S04','S05','S06'],validation=['S07','S08'],
        success='>=90% episodes with >=2 s lead, <=1 unsuccessful warning/hour; development only',
        horizon_note='2-5 s variants allow earlier useful warnings; they do not lower the 2 s minimum.'))
    for trial in trials:
        import tensorflow as tf
        tf.keras.backend.clear_session()
        cfg=replace(Config(),history_seconds=trial['history'],horizon_seconds=trial['horizon'],epochs=24)
        run_development(ROOT,out/trial['name'],cfg,architecture=trial['architecture'])
    print('TEMPORAL ROUND COMPLETE',out,flush=True)
if __name__=='__main__':main()
