from pathlib import Path
import json,urllib.request,hashlib,concurrent.futures
root=Path('external_downloads/stanford');root.mkdir(exist_ok=True)
t=json.loads(Path('external_downloads/stanford_tree.json').read_text());revision=t['sha']
entries=[a for a in t['tree'] if a['type']=='blob' and (a['path'].startswith('data/raw/imus6_subjects7/') or a['path'] in ['README.md','LICENSE','code/datapreprocessing.py'])]
def get(a):
 p=root/a['path'];p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  b=urllib.request.urlopen('https://raw.githubusercontent.com/stanfordnmbl/imu-fog-detection/'+revision+'/'+a['path'],timeout=90).read();p.write_bytes(b)
 b=p.read_bytes();actual=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
 if actual!=a['sha']:raise ValueError('Checksum mismatch '+str(p))
 return dict(path=a['path'],bytes=len(b),git_blob_sha1=actual,sha256=hashlib.sha256(b).hexdigest())
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
 records=[]
 for rec in pool.map(get,entries):
  records.append(rec);print('Verified',rec['path'],flush=True)
(root/'download_manifest.json').write_text(json.dumps(dict(revision=revision,files=records),indent=2))
print('TOTAL',len(records),sum(a['bytes'] for a in records),flush=True)
