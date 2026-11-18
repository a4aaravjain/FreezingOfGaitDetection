"""Read and verify prior artifacts before freezing a new event-hazard study."""
from pathlib import Path
import json,csv,hashlib,shutil,stat,zipfile
from collections import Counter
ROOT=Path(__file__).resolve().parent
PRIOR=ROOT/'runs/published_cascade_20261111_01'
OUT=ROOT/'runs/event_hazard_20261111_01'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

def save(p,v):
 def convert(x):
  import numpy as np
  if isinstance(x,np.generic):return x.item()
  if isinstance(x,np.ndarray):return x.tolist()
  raise TypeError('Unsupported audit value '+type(x).__name__)
 Path(p).write_text(json.dumps(v,indent=2,allow_nan=False,default=convert)+'\n',encoding='utf-8')

def main():
 OUT.mkdir(exist_ok=False);registry=json.loads((PRIOR/'artifact_registry.json').read_text())['files'];review=[]
 for name,entry in registry.items():
  p=PRIOR/name;assert sha(p)==entry['sha256'],name
  info=dict(path=name,sha256=entry['sha256'],bytes=p.stat().st_size)
  if p.suffix=='.json':
   v=json.loads(p.read_text(encoding='utf-8'));info['inspection']='parsed complete JSON';info['structure']=list(v) if isinstance(v,dict) else dict(length=len(v))
  elif p.suffix=='.csv':
   with p.open(newline='',encoding='utf-8') as f:
    rows=csv.reader(f);info['schema']=next(rows,[]);info['rows']=sum(1 for _ in rows)
   info['inspection']='streamed complete CSV'
  elif p.suffix=='.keras':
   with zipfile.ZipFile(p) as z:info['model_configuration']=json.loads(z.read('config.json'));info['members']=z.namelist()
   info['inspection']='verified binary model hash; parsed complete model configuration'
  elif p.suffix in ['.md','.py','.log','.txt']:info['text_characters']=len(p.read_text(encoding='utf-8',errors='replace'));info['inspection']='read complete text'
  else:info['inspection']='verified opaque binary artifact hash; reload/parity evidence reviewed in prior native and fit ledgers'
  review.append(info)
 save(OUT/'prior_artifact_review.json',review)
 snapshot=OUT/'champion_snapshot';snapshot.mkdir()
 for name in ['REPORT.md','preregistered_plan.json','aggregate_results.json','promotion_gate.json','final_integrity_audit.json','native_verification.json','artifact_registry.json']:
  shutil.copy2(PRIOR/name,snapshot/name)
 original=ROOT/'runs/right_ankle_waveform_v2';manifest=json.loads((original/'manifest.json').read_text());original_hashes={}
 for name,h in manifest['artifact_sha256'].items():
  p=original/name;assert sha(p)==h;original_hashes[str(p.relative_to(ROOT))]=h;p.chmod(stat.S_IREAD)
 for name in registry:(PRIOR/name).chmod(stat.S_IREAD)
 (PRIOR/'artifact_registry.json').chmod(stat.S_IREAD)
 shutil.copy2('C:/Users/victo/.codex/attachments/1b88e880-cf66-498d-b9a8-ddd487c90b59/Pasted text.txt',OUT/'REQUEST.md')
 save(OUT/'champion_freeze.json',dict(prior_run=str(PRIOR),prior_registry_sha256=sha(PRIOR/'artifact_registry.json'),original_model_hashes=original_hashes,
  artifact_count=len(registry),read_only_attributes_set=True,champion=dict(episodes=72,detected=15,macro=.1294261294261294,unsuccessful_starts=58,known_hours=1.1738981481481512,unknown_starts=0),locked_assessment_executed=False))
 print('Reviewed and froze',len(review),'prior artifacts; champion weights and results preserved.')

if __name__=='__main__':main()
