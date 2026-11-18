"""Download code/docs for inspection only; no external imports, data or weights."""
from pathlib import Path
import urllib.request,json,hashlib,concurrent.futures,stat
from fog_pipeline import save_json
ROOT=Path(__file__).resolve().parent
REPOS=['Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor','stanfordnmbl/imu-fog-detection',
       'CristianTuretta/FoG-STAR','janbrederecke/fog','saiketa/GaitFM','yabufarha/ms-tcn']


def fetch(url):
    request=urllib.request.Request(url,headers={'User-Agent':'FoG-Development-Source-Audit','Accept':'application/vnd.github+json'})
    with urllib.request.urlopen(request,timeout=60) as response:return response.read()


def audit(repo,base):
    out=base/repo.replace('/','__');out.mkdir()
    report=dict(url='https://github.com/'+repo,executed=False,data_downloaded=False,weights_downloaded=False,files=[])
    try:
        metadata=json.loads(fetch('https://api.github.com/repos/'+repo))
        commit=json.loads(fetch('https://api.github.com/repos/'+repo+'/commits/'+metadata['default_branch']))['sha']
        tree=json.loads(fetch('https://api.github.com/repos/'+repo+'/git/trees/'+commit+'?recursive=1'))['tree']
        report.update(commit=commit,license=metadata.get('license'),default_branch=metadata['default_branch'])
        paths=[x['path'] for x in tree if x['type']=='blob']
        docs=[p for p in paths if p.lower() in ['readme.md','license','license.md','license.txt','copying','notice','requirements.txt']]
        code=[p for p in paths if p.endswith('.py') and not any(part.lower() in ['data','datasets','logs','results','checkpoints','wandb'] for part in Path(p).parts[:-1])]
        code=sorted(code,key=lambda p:(not any(k in p.lower() for k in ['model','network','resnet','train','pretrain','augment','cnn','tcn']),len(p),p))[:24]
        extra=['model.json'] if repo.startswith('Lu1g1n0/') and 'model.json' in paths else []
        for name in docs+code+extra:
            payload=fetch('https://raw.githubusercontent.com/'+repo+'/'+commit+'/'+name)
            if len(payload)>2_000_000:raise ValueError('Unexpectedly large source: '+name)
            dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(payload)
            report['files'].append(dict(path=name,sha256=hashlib.sha256(payload).hexdigest(),bytes=len(payload)))
        report['status']='Downloaded for read-only inspection; not admitted for copying/execution'
    except Exception as error:report.update(status='audit_incomplete',error=repr(error))
    save_json(out/'AUDIT.json',report)
    for p in out.rglob('*'):
        if p.is_file():p.chmod(stat.S_IREAD)
    return report


def main():
    base=ROOT/'external_sources/published_20261111_01';base.mkdir(parents=True,exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        reports=list(pool.map(lambda repo:audit(repo,base),REPOS))
    save_json(base/'INDEX.json',reports)
    for r in reports:print(r['url'],r.get('commit'),r.get('license'),r['status'],[f['path'] for f in r['files']],flush=True)


if __name__=='__main__':main()
