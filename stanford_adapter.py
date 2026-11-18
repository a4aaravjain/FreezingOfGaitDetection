"""Native 128 Hz Stanford raw ankle adapter. No filtering/interpolation."""
from pathlib import Path
import re
import numpy as np,pandas as pd
from fog_pipeline import Segment,sha256,save_json
ROOT=Path(__file__).resolve().parent

def frame_segment(frame,key,side='r'):
    t=frame.time.to_numpy(float);labels=frame.freeze_label.to_numpy()
    ids=frame.subject_ID.unique()
    if len(ids)!=1 or not np.isfinite(t).all() or not np.isin(labels,[0,1]).all():raise ValueError('Invalid subject/time/labels')
    if np.any(abs(np.diff(t)-1/128)>1.1e-6):raise ValueError('Noncontiguous native 128 Hz time')
    columns=['imu_ankle_'+side+'_'+sensor+axis for sensor in 'ag' for axis in 'xyz']
    x=frame[columns].to_numpy(np.float32)
    if not np.isfinite(x).all():raise ValueError('Missing ankle samples')
    x[:,:3]/=9.80665;x[:,3:]*=180/np.pi
    y=labels.astype(np.int8)+1
    return Segment(key,'ST'+str(int(ids[0])).zfill(2),t,x,y,t[1:][(y[1:]==2)&(y[:-1]==1)],bool(y[0]==2))

def load_stanford(subjects):
    raw=ROOT/'external_downloads/stanford/data/raw/imus6_subjects7'
    cache=ROOT/'external_downloads/stanford/cache_right';cache.mkdir(exist_ok=True)
    segments=[]
    for p in sorted(raw.glob('*.xlsx')):
        subject=int(re.match(r'pt(\d+)_',p.name)[1])
        if subject not in subjects:continue
        cp=cache/(p.stem+'.npz');digest=sha256(p)
        if cp.exists():
            a=np.load(cp,allow_pickle=False)
            if str(a['source_sha256'])!=digest:raise ValueError('Stale cache')
            segment=Segment(p.stem,'ST'+str(subject).zfill(2),a['time'],a['x'],a['labels'],a['onsets'],bool(a['left']))
        else:
            segment=frame_segment(pd.read_excel(p,engine='calamine'),p.stem)
            if segment.subject!='ST'+str(subject).zfill(2):raise ValueError('Filename and subject disagree')
            np.savez_compressed(cp,time=segment.time,x=segment.xyz_g,labels=segment.labels,onsets=segment.onsets,left=segment.left_censored_fog,source_sha256=digest)
        segments.append(segment)
    if {s.subject for s in segments}!={'ST'+str(i).zfill(2) for i in subjects}:raise ValueError('Missing subjects')
    return segments

if __name__=='__main__':
    # Reserved subjects 6/7 are deliberately not parsed or summarized.
    segments=load_stanford(range(1,6))
    records=[dict(recording=s.key,subject=s.subject,rows=len(s.time),seconds=float(s.time[-1]-s.time[0]),onsets=len(s.onsets),left_censored=s.left_censored_fog) for s in segments]
    save_json(ROOT/'external_downloads/stanford/development_inventory.json',records)
    print('Development recordings',len(records),'onsets',sum(r['onsets'] for r in records),'minutes',sum(r['seconds'] for r in records)/60)
