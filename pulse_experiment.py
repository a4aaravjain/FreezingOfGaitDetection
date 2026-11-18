"""Fixed one-second notification pulse: matched start times, measured burden."""
from pathlib import Path
from dataclasses import replace,asdict
import json,shutil
import numpy as np,pandas as pd
from fog_pipeline import Config,build_windows,evaluate,save_json,sha256
from fogstar_adapter import load_fogstar
ROOT=Path(__file__).resolve().parent
out=ROOT/'runs/pulse_round_1';out.mkdir(exist_ok=False)
cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25)
d=build_windows(load_fogstar(ROOT/'sensor_data.csv',range(15,19),ankle='ankleR',include_gyro=True),cfg)
pulse=replace(cfg,alert_pulse_seconds=1)
names=['long_history_round_1','summary_repeat_17','summary_repeat_73','summary_repeat_101']
save_json(out/'manifest.json',dict(config=asdict(pulse),threshold=.675,runs=names,source_sha256={n:sha256(ROOT/n) for n in ['pulse_experiment.py','fog_pipeline.py','fogstar_adapter.py']}))
results=[]
for name in names:
 frame=pd.read_csv(ROOT/'runs'/name/'current_summary_mlp_predictions.csv');np.testing.assert_array_equal(frame.segment,d.meta.segment);np.testing.assert_allclose(frame.time,d.meta.time)
 p=frame.probability.to_numpy()
 before,br,be=evaluate(d,p,.675,cfg);after,ar,ae=evaluate(d,p,.675,pulse)
 np.testing.assert_array_equal(br.alarm_start,ar.alarm_start)
 np.testing.assert_allclose(br.smoothed,ar.smoothed)
 for k in ['true_warning_events','total_false_alarms','censored_alarm_starts','warning_times_seconds','monitored_seconds']:
  assert before['device'][k]==after['device'][k],k
 assert after['device']['maximum_alarm_duration_seconds']<=1+1e-6
 assert after['device']['total_alarm_seconds']<=before['device']['total_alarm_seconds']
 assert after['device']['false_alarm_seconds']<=before['device']['false_alarm_seconds']
 results.append(dict(run=name,before=before,after=after))
 ar.to_csv(out/(name+'_predictions.csv'),index=False)
 print(name,after['device']['true_warning_events'],'warnings; active seconds',before['device']['total_alarm_seconds'],'->',after['device']['total_alarm_seconds'],flush=True)
save_json(out/'results.json',results)
# Preserve a separate, reproducible research candidate without replacing old runs.
base=ROOT/'runs/right_ankle_timing_candidate';candidate=ROOT/'runs/right_ankle_pulse_candidate';candidate.mkdir(exist_ok=False)
for n in ['current_summary_mlp.keras','current_summary_mlp_weights.npz','numpy_export.json','normalization.json']:shutil.copy2(base/n,candidate/n)
manifest=json.loads((base/'manifest.json').read_text());manifest.update(config=asdict(pulse),parent_run=str(base),policy_status='research only: same warning starts, bounded output pulse; clinical cue effectiveness untested',source_sha256={n:sha256(ROOT/n) for n in ['pulse_experiment.py','summary_stream.py','fog_pipeline.py','long_history_experiments.py','pretraining_experiments.py','fogstar_adapter.py']})
save_json(candidate/'manifest.json',manifest)
save_json(candidate/'results.json',[dict(model='current_summary_mlp',threshold=.675,metrics=results[0]['after'],model_reload_verified=True)])
shutil.copy2(out/(names[0]+'_predictions.csv'),candidate/'current_summary_mlp_predictions.csv')
(candidate/'source').mkdir()
for n in manifest['source_sha256']:shutil.copy2(ROOT/n,candidate/'source'/n)
