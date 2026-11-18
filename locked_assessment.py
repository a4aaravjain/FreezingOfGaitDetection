"""Requires a human-approved, hash-frozen bundle; never auto-authorizes access."""
from pathlib import Path
import argparse,json,hashlib,datetime,sys

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def authorize(bundle,approval):
 bundle=Path(bundle).resolve();manifest=bundle/'freeze_manifest.json';m=json.loads(manifest.read_text());a=json.loads(Path(approval).read_text())
 if a.get('decision')!='APPROVE_ONE_LOCKED_ASSESSMENT' or a.get('bundle_sha256')!=sha(manifest) or a.get('subjects')!=[19,20,21,22] or not a.get('reviewer'):raise PermissionError('Explicit final review of exact bundle/scope required')
 if m.get('execution_ready') is not True:raise PermissionError('Bundle execution prerequisites not complete')
 for name,digest in m['files_sha256'].items():
  path=(bundle/name).resolve()
  if not path.is_relative_to(bundle) or sha(path)!=digest:raise ValueError('Frozen file mismatch: '+name)
 if Path(__file__).resolve()!=(bundle/'runtime/locked_assessment.py').resolve() or sha(Path(__file__))!=m['files_sha256']['runtime/locked_assessment.py']:raise ValueError('Execute the frozen runner copy')
 return m

def run(bundle,approval,data,output):
 bundle=Path(bundle).resolve();m=authorize(bundle,approval)
 # This exclusive marker is in the bundle, so changing output cannot permit retry.
 with (bundle/'ASSESSMENT_STARTED.json').open('x',encoding='utf-8') as f:json.dump(dict(bundle_sha256=sha(bundle/'freeze_manifest.json'),approval_sha256=sha(approval),subjects=[19,20,21,22],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),output=str(Path(output).resolve())),f,indent=2)
 out=Path(output)
 try:
  out.mkdir(parents=True,exist_ok=False)
  if sha(data)!=m['approved_partition_sha256']:raise ValueError('Approved partition checksum mismatch')
  import pandas as pd
  from assessment_core import prepare,replay,score
  from waveform_ensemble import WaveformEnsemble
  import assessment_core,waveform_ensemble,fog_pipeline
  for module in [assessment_core,waveform_ensemble,fog_pipeline]:
   if Path(module.__file__).resolve().parent!=(bundle/'runtime').resolve():raise ValueError('Unfrozen module imported')
  stream=WaveformEnsemble(bundle/'candidate');frame=pd.read_csv(data)
  records,chunks,inventory,incomplete=prepare(frame,[19,20,21,22]);pd.DataFrame(inventory).to_csv(out/'inventory.csv',index=False)
  predictions=replay(chunks,stream);predictions.to_csv(out/'predictions.csv',index=False);result=score(records,predictions,inventory,incomplete,stream.cfg)
  people=[]
  for subject in sorted({r.subject for r in records}):
   rr=[r for r in records if r.subject==subject];keys={r.key for r in rr};ff=predictions[predictions.segment.isin(keys)];ii=[i for i in inventory if i['subject']==subject];s=score(rr,ff,ii,incomplete,stream.cfg);people.append(dict(subject=subject,**s['metrics']))
  pd.DataFrame(people).to_csv(out/'participants.csv',index=False);values=[x['episode_sensitivity'] for x in people if x['episode_sensitivity'] is not None];result['participant_macro_sensitivity']=sum(values)/len(values) if values else None
  (out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
 except BaseException as error:
  # Preserve the marker even when output creation or acquisition fails.
  (bundle/'ASSESSMENT_ERROR.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error),status='INCOMPLETE_NO_AUTOMATIC_RETRY'),indent=2),encoding='utf-8');raise
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--bundle',type=Path,required=True);p.add_argument('--approval',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.bundle,a.approval,a.data,a.output)
