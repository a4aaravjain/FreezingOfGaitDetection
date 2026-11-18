"""Audit FoG-STAR without training or pooling it with Daphnet."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent

def audit():
    p=ROOT/'sensor_data.csv'; digest=hashlib.md5(p.read_bytes()).hexdigest()
    if digest!='952a37ab147da35e6d4e7a1e9bac44cb':raise ValueError('FoG-STAR release checksum mismatch')
    frame=pd.read_csv(p);reports=[]
    for ids,g in frame.groupby(['subjectID','sessionID','taskID'],sort=True):
        t=g.timestamp.to_numpy(float);y=g.fog.to_numpy()
        if not np.isin(y,[0,1]).all() or not np.isfinite(t).all() or np.any(np.diff(t)<=0):raise ValueError('Invalid timestamps or labels')
        # Local values are seconds: validate rather than following the incorrect ms header.
        if abs(np.median(np.diff(t))-1/60)>1e-6:raise ValueError('Unexpected sampling interval')
        onsets=t[1:][(y[1:]==1)&(y[:-1]==0)]
        row=dict(subject=int(ids[0]),session=int(ids[1]),task=int(ids[2]),rows=len(g),
                 observed_onsets=len(onsets),starts_in_fog=bool(y[0]==1))
        for ankle in ['ankleL','ankleR']:
            xyz=g[[ankle+'_acc_'+axis for axis in 'xyz']].to_numpy(float)
            valid=np.isfinite(xyz).all(1)
            # Complete 3s history ending 2-3s before onset, independently for each ankle.
            breaks=np.r_[True,(~valid[:-1])|(~valid[1:])|(np.abs(np.diff(t)-1/60)>1e-5)]
            starts=np.flatnonzero(valid&breaks); spans=[]
            for start in starts:
                stop=start+1
                while stop<len(t) and valid[stop] and not breaks[stop]:stop+=1
                spans.append((start,stop))
            eligible=0
            for onset in onsets:
                if any(np.any((t[start+179:stop]>=onset-3-1e-9)&(t[start+179:stop]<=onset-2+1e-9)&(y[start+179:stop]==0)) for start,stop in spans if stop-start>=180):eligible+=1
            row[ankle+'_missing_rows']=int((~valid).sum());row[ankle+'_segments']=len(spans)
            row[ankle+'_onsets_with_complete_prior_history']=eligible
        reports.append(row)
    out=ROOT/'runs'/'external_provenance';out.mkdir(exist_ok=True)
    pd.DataFrame(reports).to_csv(out/'fogstar_recordings.csv',index=False)
    report=dict(source='https://zenodo.org/records/17838806',license='CC-BY-4.0',
       local_file=str(p),md5=digest,checksum_matches_publisher=True,rows=len(frame),
       subjects=int(frame.subjectID.nunique()),recordings=len(reports),
       observed_onsets=sum(r['observed_onsets'] for r in reports),
       local_timestamp_unit='seconds (verified 1/60 increments; publication README says ms)',
       sampling_rate=60,accelerometer_unit='g',training_performed=False,
       admission='Candidate for separate native-60Hz development; do not mix axes or interpolate future samples.',
       timing_limit='Video annotations/reconstructed sensor timing do not establish end-to-end hardware lead time.',
       left_ankle_eligible_onsets=sum(r['ankleL_onsets_with_complete_prior_history'] for r in reports),
       right_ankle_eligible_onsets=sum(r['ankleR_onsets_with_complete_prior_history'] for r in reports))
    (out/'fogstar_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
if __name__=='__main__':audit()
