"""Saved development artifacts only. Never opens a raw dataset or a reserved cohort."""
from pathlib import Path
import argparse,json,hashlib,shutil,sys,platform,importlib.metadata
import numpy as np,pandas as pd
from fog_pipeline import Config,Alarm
from waveform_ensemble import PulseUnion
ROOT=Path(__file__).resolve().parent

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def longest_run(values):
 best=run=0
 for value in values:
  run=run+1 if value else 0;best=max(best,run)
 return best

def diagnose(raw,smoothed,starts,threshold,persistence=2):
 if len(raw)==0:return 'no_eligible_window'
 if any(starts):return 'component_start_present'
 if max(smoothed)>=threshold:
  return 'state_or_cooldown_candidate' if longest_run(np.asarray(smoothed)>=threshold)>=persistence else 'persistence_candidate'
 if max(raw)>=threshold:return 'smoothing_candidate'
 return 'score_below_threshold'

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
 run=ROOT/'runs/right_ankle_waveform_v2';m=json.loads((run/'manifest.json').read_text());result=json.loads((run/'results.json').read_text());frame=pd.read_csv(run/'predictions.csv');cfg=Config(**m['config'])
 assert set(frame.subject)=={'FS15','FS16','FS17','FS18'}
 checks={name:digest(run/name)==sha for name,sha in m['artifact_sha256'].items()}
 if not all(checks.values()):raise ValueError('Candidate artifact hash mismatch')
 heads=[('base',m['base_threshold']),('waveform',m['waveform_threshold']),('extended',m['extended_head']['threshold'])]
 for _,g in frame.groupby('segment',sort=False):
  alarms={h:Alarm(t,cfg) for h,t in heads};union=PulseUnion(cfg.alert_pulse_seconds,cfg.cooldown_seconds)
  for row in g.itertuples():
   starts=[]
   for h,t in heads:
    smooth,active,start=alarms[h].step(row.time,getattr(row,h+'_probability'));np.testing.assert_allclose(smooth,getattr(row,h+'_smoothed'),rtol=1e-5,atol=1e-6);assert start==getattr(row,h+'_start');starts.append(start)
   active,start=union.step(row.time,starts);assert active==row.alarm_active and start==row.alarm_start
 rows=[]
 for e in result['events']:
  g=frame[frame.segment==e['segment']];window=g[(g.time>=e['onset']-5-1e-9)&(g.time<=e['onset']-2+1e-9)&~g.active_fog];row=dict(subject=e['segment'].split(':')[0],segment=e['segment'],onset=e['onset'],detected=e['warning_seconds'] is not None,warning_seconds=e['warning_seconds'],eligible_windows=len(window))
  for h,t in heads:
   raw=window[h+'_probability'].to_numpy();smooth=window[h+'_smoothed'].to_numpy();starts=window[h+'_start'].to_numpy();row.update({h+'_raw_max':None if len(raw)==0 else float(max(raw)),h+'_smoothed_max':None if len(raw)==0 else float(max(smooth)),h+'_threshold':t,h+'_diagnosis':diagnose(raw,smooth,starts,t,cfg.persistence)})
  tags=[row[h+'_diagnosis'] for h,_ in heads]
  row['diagnostic_bucket']='detected' if row['detected'] else ('no_eligible_window' if not len(window) else ('all_heads_below_raw_threshold' if all(t=='score_below_threshold' for t in tags) else 'score_filter_or_state_requires_inspection'))
  rows.append(row)
 audit=pd.DataFrame(rows);audit.to_csv(out/'episodes.csv',index=False)
 participants=audit.groupby('subject').agg(episodes=('detected','size'),warned=('detected','sum')).reset_index()
 for person in sorted(set(frame.subject)-set(participants.subject)):participants.loc[len(participants)]=[person,0,0]
 participants['sensitivity']=participants.warned/participants.episodes.replace(0,np.nan);participants.to_csv(out/'participants.csv',index=False)
 assert len(audit)==16 and audit.detected.sum()==7
 claims=dict(windows=len(frame),episodes=len(audit),warned=int(audit.detected.sum()),missed=int((~audit.detected).sum()),miss_buckets=audit.loc[~audit.detected,'diagnostic_bucket'].value_counts().to_dict(),all_candidate_artifact_hashes_match=True,all_saved_alarm_states_reproduced=True,metrics_from_saved_result=result['combined'],causal_interpretation='Buckets localize saved-score failures; they do not establish biological causation or information insufficiency',raw_or_reserved_data_opened=False)
 (out/'summary.json').write_text(json.dumps(claims,indent=2),encoding='utf-8')
 (out/'candidate').mkdir();runtime=['waveform_ensemble.py','summary_stream.py','waveform_bank.py','fast_temporal_bank.py','gravity_gait.py','long_history_experiments.py','pretraining_experiments.py','fog_pipeline.py','fogstar_adapter.py']
 for name in ['bank.json','ridge_weights.npz','extended_weights.npz','manifest.json']:shutil.copy2(run/name,out/'candidate'/name)
 shutil.copytree(run/'base',out/'candidate/base');(out/'runtime').mkdir()
 for name in runtime:shutil.copy2(ROOT/name,out/'runtime'/name)
 files={str(p.relative_to(out)):digest(p) for folder in ['candidate','runtime'] for p in (out/folder).rglob('*') if p.is_file()}
 # Freeze without hashing/opening raw datasets, even for provenance.
 freeze=dict(date='2026-11-11',status='PREASSESSMENT_REVIEW_REQUIRED_NOT_AUTHORIZED',candidate='right_ankle_waveform_v2',files_sha256=files,review_source_sha256=digest(Path(__file__)),saved_inputs_sha256={n:digest(run/n) for n in ['manifest.json','results.json','predictions.csv','stream_verification.json','paired_stream_benchmark.json']},config=m['config'],thresholds={h:t for h,t in heads},channels=['ankleR_acc_'+a for a in 'xyz']+['ankleR_gyro_'+a for a in 'xyz'],units=['g']*3+['degrees/second']*3,release_scope='FoG-STAR 19-22 only after explicit final review; Stanford 6-7 excluded until native-128-Hz adapter is separately frozen',raw_data_access_performed=False,python=sys.version,platform=platform.platform(),packages={n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy','scikit-learn']},command='python development_evidence_review.py --output '+str(out))
 (out/'freeze_manifest.json').write_text(json.dumps(freeze,indent=2),encoding='utf-8');print(json.dumps(claims,indent=2))
if __name__=='__main__':main()
