from pathlib import Path
import json,shutil,joblib
import numpy as np,pandas as pd
from fog_pipeline import save_json,sha256
r=Path(__file__).resolve().parent;parent=r/'runs/right_ankle_waveform_candidate';fit=r/'runs/stable_extended_waveform_round_1'
results=json.loads((fit/'results.json').read_text());selected=next(x for x in results if x['model']=='cholesky');old=json.loads((parent/'results.json').read_text());a=old['combined'];b=selected['combined']
assert (fit/'solver_agreement.json').is_file()
assert b['true_warning_events']>=a['true_warning_events'] and b['total_false_alarms']<=a['total_false_alarms'] and b['censored_alarm_starts']<=a['censored_alarm_starts'] and b['total_alarm_seconds']<=a['total_alarm_seconds']+1e-6
before={(e['segment'],e['onset']):e['warning_seconds'] for e in old['events'] if e['warning_seconds'] is not None};after={(e['segment'],e['onset']):e['warning_seconds'] for e in selected['events'] if e['warning_seconds'] is not None}
assert set(before).issubset(after)
changes=[]
for key,lead in before.items():
 assert after[key]>=lead-1e-6
 if after[key]>lead+1e-6:changes.append(dict(segment=key[0],onset=key[1],before=lead,after=after[key]))
assert changes
out=r/'runs/right_ankle_waveform_v2';out.mkdir(exist_ok=False);shutil.copytree(parent/'base',out/'base');shutil.copy2(parent/'ridge_weights.npz',out/'ridge_weights.npz');shutil.copy2(fit/'bank.json',out/'bank.json')
model=joblib.load(fit/'cholesky.joblib');scaler=model.steps[0][1];ridge=model.steps[1][1]
np.savez(out/'extended_weights.npz',mean=scaler.mean_,scale=scaler.scale_,coef=ridge.coef_,intercept=ridge.intercept_)
f=pd.read_csv(fit/'cholesky_combined_predictions.csv');extra=pd.read_csv(fit/'cholesky_predictions.csv');np.testing.assert_array_equal(f.segment,extra.segment);np.testing.assert_allclose(f.time,extra.time)
f['extended_probability']=extra.probability;f['extended_smoothed']=extra.smoothed;f['extended_start']=extra.alarm_start;f.to_csv(out/'predictions.csv',index=False)
save_json(out/'results.json',dict(baseline=a,original_dense=old['baseline'],combined=b,events=selected['events'],earlier_warnings=changes,all_previous_successes_retained_no_later=True,interpretation='Reused development result; 7/16 unchanged, one warning earlier, unknown alerts and total alert time unchanged; verify raw streaming before promotion'))
artifacts=[p for p in out.rglob('*') if p.is_file() and p.name not in ['predictions.csv','results.json']]
source=['build_waveform_v2.py','waveform_ensemble.py','verify_waveform_ensemble.py','fast_temporal_bank.py','waveform_bank.py','summary_stream.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py','fog_pipeline.py','fogstar_adapter.py'];(out/'source').mkdir()
for n in source:shutil.copy2(r/n,out/'source'/n)
save_json(out/'manifest.json',dict(config=json.loads((parent/'manifest.json').read_text())['config'],ankle='ankleR',waveform_threshold=.775,base_threshold=.675,feature_engine='grouped',primary_kernel_count=384,extended_head=dict(kernel_count=1024,threshold=selected['threshold'],fit_precision='float64',solver='cholesky cross-checked against lsqr'),train=list(range(1,15)),development=list(range(15,19)),reserved=list(range(19,23)),status='research only; reused development recordings; not independently validated',artifact_sha256={str(p.relative_to(out)).replace(chr(92),'/'):sha256(p) for p in artifacts},source_sha256={n:sha256(r/n) for n in source},parent_runs=[str(parent),str(fit)],dataset_sha256=sha256(r/'sensor_data.csv')))
print('Candidate v2',b['true_warning_events'],'false',b['total_false_alarms'],'unknown',b['censored_alarm_starts'],'earlier',changes,flush=True)
