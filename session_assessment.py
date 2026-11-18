"""Physical-session preparation: task metadata never resets a continuous sensor."""
import numpy as np
from fog_pipeline import Segment,build_windows
from assessment_core import CHANNELS


def score_sessions(records,frame,inventory,incomplete,cfg):
    """Reject misassigned session keys or out-of-record times before scoring."""
    from assessment_core import score
    lookup={r.key:r for r in records}
    for key,g in frame.groupby('segment',sort=False):
        if key not in lookup:
            raise ValueError('Prediction refers to an unknown physical session')
        times=g.time.to_numpy(float);r=lookup[key]
        if not np.isfinite(times).all() or (np.diff(times)<=0).any() or (times<r.time[0]-1e-8).any() or (times>r.time[-1]+1e-8).any():
            raise ValueError('Prediction time is outside its physical session or not increasing')
    return score(records,frame,inventory,incomplete,cfg)


def prepare_sessions(frame,subjects):
    required=['subjectID','sessionID','taskID','timestamp','fog']+CHANNELS
    if any(c not in frame for c in required) or set(frame.subjectID.unique())!=set(subjects):
        raise ValueError('Unexpected participant scope/schema')
    records,chunks,inventory=[],[],[]
    incomplete=False
    for ids,g in frame.groupby(['subjectID','sessionID'],sort=True):
        t=g.timestamp.to_numpy(float);y=g.fog.to_numpy();x=g[CHANNELS].to_numpy(np.float32)
        if not len(t) or not np.isfinite(t).all() or (np.diff(t)<=0).any() or not np.isin(y,[0,1]).all():
            raise ValueError('Invalid physical session; never reorder time to hide it')
        contiguous=np.abs(np.diff(t)-1/60)<=1e-5
        if len(t)>1 and abs(np.median(np.diff(t))-1/60)>1e-6:
            raise ValueError('Unsupported native rate')
        if not contiguous.all():incomplete=True
        cuts=np.r_[0,np.flatnonzero(~contiguous)+1,len(t)]
        for start,stop in zip(cuts[:-1],cuts[1:]):
            st,sy,sx=t[start:stop],y[start:stop],x[start:stop]
            key=f'FS{int(ids[0]):02}:{ids[1]}:run{start}'
            onsets=st[1:][(sy[1:]==1)&(sy[:-1]==0)]
            records.append(Segment(key,f'FS{int(ids[0]):02}',st,sx,sy.astype(np.int8)+1,onsets,bool(sy[0]==1)))
            valid=np.isfinite(sx).all(1)
            breaks=np.r_[True,(~valid[:-1])|(~valid[1:])]
            for begin in np.flatnonzero(valid&breaks):
                end=begin+1
                while end<len(st) and valid[end]:end+=1
                chunks.append(dict(record=key,id=key+':chunk:'+str(begin),time=st[begin:end],x=sx[begin:end],labels=sy[begin:end]))
            inventory.append(dict(record=key,subject=f'FS{int(ids[0]):02}',original_onsets=len(onsets),
                rows=len(st),valid_sensor_rows=int(valid.sum()),timestamp_gaps=0,
                record_seconds=float(st[-1]-st[0]+1/60),left_censored=bool(sy[0]==1)))
    return records,chunks,inventory,incomplete


def make_session_data(frame,ids,cfg):
    """Native windows and targets use full physical annotation follow-up, including gaps."""
    frame=frame[frame.subjectID.isin(ids)].copy()
    records,chunks,inventory,incomplete=prepare_sessions(frame,ids)
    segments=[]
    for c in chunks:
        y=c['labels'];t=c['time']
        segments.append(Segment(c['id'],c['record'].split(':')[0],t,c['x'],y.astype(np.int8)+1,
            t[1:][(y[1:]==1)&(y[:-1]==0)],bool(y[0]==1)))
    data=build_windows(segments,cfg)
    meta=data.meta.copy();meta['chunk']=meta.segment
    lookup={c['id']:c for c in chunks};records_by_key={r.key:r for r in records}
    meta['segment']=meta.chunk.map(lambda k:lookup[k]['record'])
    meta['subject']=meta.subject.map(lambda s:int(s[2:]))
    targets,known,duration=[],[],[]
    for row in meta.itertuples():
        record=records_by_key[row.segment]
        future=row.time+cfg.horizon_seconds<=record.time[-1]+1e-9
        onset=record.onsets[record.onsets>row.time]
        delta=onset[0]-row.time if len(onset) else np.inf
        known.append(future)
        targets.append(-1 if row.active_fog or not future else int(cfg.minimum_lead_seconds-1e-9<=delta<=cfg.horizon_seconds+1e-9))
        duration.append(min(cfg.interval_seconds,lookup[row.chunk]['time'][-1]+1/60-row.time))
    meta['target'],meta['known_followup'],meta['duration']=targets,known,duration
    return dict(data=data,meta=meta,records=records,inventory=inventory,incomplete=incomplete)
