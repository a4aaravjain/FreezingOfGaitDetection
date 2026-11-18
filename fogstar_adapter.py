"""FoG-STAR native 60 Hz adapter. Explicit ankle, no interpolation/resampling."""
from pathlib import Path
import numpy as np
import pandas as pd
from fog_pipeline import Segment

def segments_from_frame(frame,ankle='ankleL',include_gyro=False,max_hold_seconds=0):
    if ankle not in ('ankleL','ankleR','both'):raise ValueError('Choose ankleL, ankleR or both')
    segments=[]
    for ids,g in frame.groupby(['subjectID','sessionID','taskID'],sort=True):
        t=g.timestamp.to_numpy(float)
        columns=[side+'_'+sensor+'_'+a for side in (['ankleL','ankleR'] if ankle=='both' else [ankle]) for sensor in (['acc','gyro'] if include_gyro else ['acc']) for a in 'xyz']
        xyz=g[columns].to_numpy(np.float32)
        y=g.fog.to_numpy()
        if not np.isfinite(t).all() or not np.isin(y,[0,1]).all() or np.any(np.diff(t)<=0):
            raise ValueError('Invalid FoG-STAR timestamps/labels')
        dt=np.diff(t)
        if len(dt) and abs(np.median(dt)-1/60)>1e-6:raise ValueError('Expected seconds at 60 Hz')
        if max_hold_seconds:
            from causal_hold import CausalHold
            holder=CausalHold(60,max_hold_seconds,xyz.shape[1])
            held=[holder.step(ti,row) for ti,row in zip(t,xyz)]
            xyz=np.array([row if row is not None else np.full(xyz.shape[1]+1,np.nan) for row in held],np.float32)
        valid=np.isfinite(xyz).all(1)
        breaks=np.r_[True,(~valid[:-1])|(~valid[1:])|(np.abs(dt-1/60)>1e-5)]
        starts=np.flatnonzero(valid&breaks)
        subject='FS'+str(int(ids[0])).zfill(2)
        for start in starts:
            stop=start+1
            while stop<len(t) and valid[stop] and not breaks[stop]:stop+=1
            ts=t[start:stop];ys=y[start:stop].astype(np.int8)+1
            onset=ts[1:][(ys[1:]==2)&(ys[:-1]==1)]
            key=f'{subject}:{ids[1]}:{ids[2]}:{ankle}:{start}'
            segments.append(Segment(key,subject,ts,xyz[start:stop],ys,onset,bool(ys[0]==2)))
    return segments

def load_fogstar(path,subjects,ankle='ankleL',include_gyro=False,max_hold_seconds=0):
    frame=pd.read_csv(path)
    selected=frame[frame.subjectID.isin(subjects)]
    if set(selected.subjectID.unique())!=set(subjects):raise ValueError('Missing subjects')
    return segments_from_frame(selected,ankle,include_gyro,max_hold_seconds)
