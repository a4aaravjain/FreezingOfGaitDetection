"""Prespecified history-availability audit, approved export only; no classifier."""
from pathlib import Path
import json,shutil
import pandas as pd,numpy as np
from fog_pipeline import sha256,save_json


def main():
    root=Path(__file__).resolve().parent
    out=root/'runs/short_history_availability_20261111_01'
    out.mkdir(exist_ok=False)
    source=root/'development_only_20261111_01.csv'
    assert sha256(source)=='2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
    save_json(out/'plan.json',dict(histories=[.5,1.5,3],ankle='ankleR',minimum_lead=2,horizon=5,
        source_sha256=sha256(source),script_sha256=sha256(Path(__file__)),
        meaning='Availability only; shorter histories are not assumed to improve discrimination.'))
    shutil.copy2(Path(__file__),out/'source.py')
    frame=pd.read_csv(source)
    assert set(frame.subjectID)==set(range(1,19))
    rows=[]
    for ids,g in frame.groupby(['subjectID','sessionID','taskID'],sort=True):
        t,y=g.timestamp.to_numpy(),g.fog.to_numpy()
        assert np.all(abs(np.diff(t)-1/60)<1e-5)
        valid=np.isfinite(g[['ankleR_'+s+'_'+a for s in ['acc','gyro'] for a in 'xyz']]).all(1).to_numpy()
        onsets=t[1:][(y[1:]==1)&(y[:-1]==0)]
        for history in [.5,1.5,3]:
            count=0;ends=[]
            for i,v in enumerate(valid):
                count=count+1 if v else 0
                if count>=round(history*60) and (count-round(history*60))%15==0:ends.append(i)
            ends=np.array(ends,int)
            for onset in onsets:
                times=t[ends[(t[ends]>=onset-5-1e-9)&(t[ends]<=onset-2+1e-9)&(y[ends]==0)]]
                rows.append(dict(subject=int(ids[0]),session=int(ids[1]),task=int(ids[2]),onset=float(onset),
                    history=history,eligible_ticks=len(times),consecutive_pair=bool(np.any(np.isclose(np.diff(times),.25,atol=1e-7)))))
    events=pd.DataFrame(rows);events.to_csv(out/'episodes.csv',index=False)
    summary=[dict(history=float(h),onsets=len(g),any_window=int((g.eligible_ticks>0).sum()),
        two_consecutive_windows=int(g.consecutive_pair.sum())) for h,g in events.groupby('history')]
    save_json(out/'summary.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
