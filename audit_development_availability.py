"""Availability audit on the approved export only; never an accuracy estimate."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
from causal_hold import CausalHold
from fog_pipeline import sha256,save_json


def decisions(valid):
    count = 0
    ends = []
    for i,v in enumerate(valid):
        count = count+1 if v else 0
        if count>=180 and (count-180)%15==0:
            ends.append(i)
    return np.asarray(ends,int)


def main():
    root = Path(__file__).resolve().parent
    out = root/'runs/sensor_availability_20261111_01'
    out.mkdir(exist_ok=False)
    source = root/'development_only_20261111_01.csv'
    assert sha256(source)=='2925d016fa4fe7993b4770d9e5f7927b13b0a0f097571f14c93207078121c08c'
    save_json(out/'audit_plan.json',dict(source_hash=sha256(source),allowed=list(range(1,19)),
        hold_bounds=[0,.05,.1],purpose='Availability of two-to-five-second precursor windows, not prediction performance',
        hypothesis='Short causal gap handling may prevent unnecessary three-second warmup resets. Side availability is not side accuracy.',
        research='https://www.nature.com/articles/s41597-026-06645-1',
        source_code_hash=sha256(root/'audit_development_availability.py')))
    shutil.copy2(root/'audit_development_availability.py',out/'source.py')
    frame = pd.read_csv(source)
    assert set(frame.subjectID)==set(range(1,19))
    episode_rows,record_rows = [],[]
    for ids,g in frame.groupby(['subjectID','sessionID','taskID'],sort=True):
        t,y = g.timestamp.to_numpy(),g.fog.to_numpy()
        assert np.all(abs(np.diff(t)-1/60)<1e-5)
        onsets = t[1:][(y[1:]==1)&(y[:-1]==0)]
        for side in ['ankleL','ankleR']:
            raw = g[[side+'_'+s+'_'+a for s in ['acc','gyro'] for a in 'xyz']].to_numpy(np.float32)
            for limit in [0,.05,.1]:
                holder = CausalHold(60,limit,6)
                valid = np.isfinite(raw).all(1) if not limit else np.array([holder.step(float(time),x) is not None for time,x in zip(t,raw)])
                ends = decisions(valid)
                record_rows.append(dict(subject=int(ids[0]),session=int(ids[1]),task=int(ids[2]),side=side,hold=limit,
                    total_rows=len(raw),missing_rows=int((~np.isfinite(raw).all(1)).sum()),
                    usable_rows=int(valid.sum()),prediction_ticks=len(ends)))
                for onset in onsets:
                    mask = (t[ends]>=onset-5-1e-9)&(t[ends]<=onset-2+1e-9)&(y[ends]==0)
                    times = t[ends[mask]]
                    episode_rows.append(dict(subject=int(ids[0]),session=int(ids[1]),task=int(ids[2]),
                        onset=float(onset),side=side,hold=limit,eligible_ticks=len(times),
                        consecutive_pair=bool(np.any(np.isclose(np.diff(times),.25,atol=1e-7))),
                        earliest_nominal_opportunity=float(onset-times[0]) if len(times) else None))
    episodes = pd.DataFrame(episode_rows)
    records = pd.DataFrame(record_rows)
    episodes.to_csv(out/'episodes.csv',index=False)
    records.to_csv(out/'recordings.csv',index=False)
    rows = []
    for (side,limit),group in episodes.groupby(['side','hold']):
        rr = records[(records.side==side)&(records.hold==limit)]
        rows.append(dict(side=side,hold=float(limit),onsets=len(group),
            with_any_opportunity=int((group.eligible_ticks>0).sum()),
            with_two_consecutive_opportunities=int(group.consecutive_pair.sum()),
            total_prediction_ticks=int(rr.prediction_ticks.sum())))
    save_json(out/'summary.json',rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':
    main()
