"""Fetch only reference code/docs; never execute third-party preprocessing."""
from pathlib import Path
import concurrent.futures
from audit_published_sources import audit
from event_hazard_audit import OUT,save,ROOT

REPOS=['havakv/pycox','locuslab/TCN','WaqasSultani/AnomalyDetectionCVPR2018','Ism-ail11/XTinyHAR']
def main():
 base=ROOT/'external_sources/event_hazard_20261111_01';base.mkdir(exist_ok=False)
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:reports=list(pool.map(lambda repo:audit(repo,base),REPOS))
 save(base/'INDEX.json',reports);save(OUT/'external_repository_audit.json',reports)
 for r in reports:print(r['url'],r.get('commit'),r.get('license'),r['status'],[f['path'] for f in r['files']],flush=True)
if __name__=='__main__':main()
