"""Check task metadata against physical timestamp continuity, approved export only."""
from pathlib import Path
import json,shutil
import numpy as np,pandas as pd
from fog_pipeline import sha256,save_json


def main():
    root=Path(__file__).resolve().parent
    out=root/'runs/session_boundary_audit_20261111_01'
    out.mkdir(exist_ok=False)
    source=root/'development_only_20261111_01.csv'
    assert sha256(source)=='2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
    save_json(out/'plan.json',dict(source_sha256=sha256(source),script_sha256=sha256(Path(__file__)),
        rule='Audit only. No concatenation, model fitting or scoring changes. A task-ID change is a real reset only if time or sensor continuity is broken. Never bridge an unobserved gap.'))
    shutil.copy2(Path(__file__),out/'source.py')
    f=pd.read_csv(source,usecols=['timestamp','subjectID','sessionID','taskID','fog'])
    assert set(f.subjectID)==set(range(1,19))
    rows=[]
    for (subject,session),g in f.groupby(['subjectID','sessionID'],sort=True):
        t,y,task=g.timestamp.to_numpy(),g.fog.to_numpy(),g.taskID.to_numpy()
        for i in np.flatnonzero(task[1:]!=task[:-1])+1:
            rows.append(dict(subject=int(subject),session=int(session),before_task=int(task[i-1]),after_task=int(task[i]),
                previous_time=float(t[i-1]),next_time=float(t[i]),delta=float(t[i]-t[i-1]),
                contiguous=bool(abs(t[i]-t[i-1]-1/60)<1e-5),fog_before=int(y[i-1]),fog_after=int(y[i])))
    b=pd.DataFrame(rows);b.to_csv(out/'boundaries.csv',index=False)
    result=dict(boundaries=len(b),native_contiguous=int(b.contiguous.sum()),
        positive_gaps=int((b.delta>1/60+1e-5).sum()),nonincreasing=int((b.delta<=0).sum()),
        possible_newly_observable_onsets=int((b.contiguous&(b.fog_before==0)&(b.fog_after==1)).sum()))
    save_json(out/'summary.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
