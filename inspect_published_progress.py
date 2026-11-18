"""Read-only progress view; never reads sensor values or outcome metrics."""
from pathlib import Path
import json
from datetime import datetime

r=Path(__file__).resolve().parent/'runs/published_cascade_20261111_01'
fits=list(r.glob('fold_*/*/*_DONE.json'))
print(datetime.now().isoformat(timespec='seconds'),'upstream neural checkpoints',len(fits),'/84')
log=r/'execution.log'
phases=[s for s in log.read_text(errors='replace').splitlines() if s.startswith(('FOLD','UPSTREAM'))]
if phases:print(phases[-1])
for name in ['execution_completed.json','pipeline_status.json','PIPELINE_FAILURE.json']:
 p=r/name
 if p.exists():print(name,p.read_text())
for name in ['fold1_directory_completion.log','native_replay.log','final_tests.log','report_generation.log']:
 p=r/name
 if p.exists():print(name,datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec='seconds'),p.stat().st_size)
