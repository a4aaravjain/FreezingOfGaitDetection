"""Assessment operations on an already-authorized frame; no file reads here."""
import numpy as np,pandas as pd
from fog_pipeline import Segment,Config,event_metrics
CHANNELS=['ankleR_'+sensor+'_'+axis for sensor in ['acc','gyro'] for axis in 'xyz']

def prepare(frame,subjects):
 required=['subjectID','sessionID','taskID','timestamp','fog']+CHANNELS
 if any(c not in frame for c in required):raise ValueError('Missing schema columns')
 if set(frame.subjectID.unique())!=set(subjects):raise ValueError('Partition must contain exactly the approved people')
 records=[];inventory=[];chunks=[];incomplete=False
 for ids,g in frame.groupby(['subjectID','sessionID','taskID'],sort=True):
  t=g.timestamp.to_numpy(float);labels=g.fog.to_numpy();x=g[CHANNELS].to_numpy(np.float32)
  if not len(t) or not np.isfinite(t).all() or np.any(np.diff(t)<=0) or not np.isin(labels,[0,1]).all():raise ValueError('Invalid time/annotation; preserve failed assessment')
  dt=np.diff(t);contiguous=np.abs(dt-1/60)<=1e-5
  if len(dt) and abs(np.median(dt)-1/60)>1e-6:raise ValueError('Unsupported native sample rate')
  if not contiguous.all():incomplete=True
  valid=np.isfinite(x).all(1);key=f'FS{int(ids[0]):02}:{ids[1]}:{ids[2]}'
  onset=t[1:][(labels[1:]==1)&(labels[:-1]==0)]
  records.append(Segment(key,f'FS{int(ids[0]):02}',t,x,labels.astype(np.int8)+1,onset,bool(labels[0]==1)))
  breaks=np.r_[True,(~valid[:-1])|(~valid[1:])|(~contiguous)]
  for start in np.flatnonzero(valid&breaks):
   stop=start+1
   while stop<len(t) and valid[stop] and not breaks[stop]:stop+=1
   chunks.append(dict(record=key,id=key+':chunk:'+str(start),time=t[start:stop],x=x[start:stop],labels=labels[start:stop]))
  inventory.append(dict(record=key,subject=f'FS{int(ids[0]):02}',original_onsets=len(onset),rows=len(t),valid_sensor_rows=int(valid.sum()),timestamp_gaps=int((~contiguous).sum()),record_seconds=float(t[-1]-t[0]+1/60),left_censored=bool(labels[0]==1)))
 return records,chunks,inventory,incomplete

def replay(chunks,stream):
 rows=[]
 for c in chunks:
  stream.reset();local=[]
  for time,sample,label in zip(c['time'],c['x'],c['labels']):
   result=stream.step(float(time),sample)
   if result is not None:local.append(dict(result,segment=c['record'],chunk=c['id'],active_fog=bool(label)))
  for i,row in enumerate(local):
   end=min(row['time']+.25,float(c['time'][-1]+1/60))
   row['duration']=max(0.,end-row['time']);rows.append(row)
 return pd.DataFrame(rows,columns=list(dict.fromkeys(['segment','chunk','time','active_fog','alarm_start','alarm_active','duration']+[k for row in rows for k in row])))

def score(records,frame,inventory,incomplete,cfg):
 # Whole-record annotation timelines retain onsets hidden by missing sensors.
 metrics,events=event_metrics(type('Data',(),{'segments':records})(),frame,cfg)
 durations=[]
 for _,g in frame.groupby('chunk',sort=False):
  length=0.
  for row in g.itertuples():
   if row.alarm_active:length+=row.duration
   elif length:durations.append(length);length=0.
  if length:durations.append(length)
 metrics['maximum_alarm_duration_seconds']=max(durations,default=0.)
 metrics['average_alarm_duration_seconds']=float(np.mean(durations)) if durations else 0.
 metrics['valid_sensor_seconds']=sum(i['valid_sensor_rows']/60 for i in inventory)
 metrics['total_recording_seconds']=sum(i['record_seconds'] for i in inventory)
 metrics['original_annotated_onsets']=sum(i['original_onsets'] for i in inventory)
 metrics['observed_contiguous_sensor_onsets']=int(sum(sum((r.labels[1:]==2)&(r.labels[:-1]==1)&np.isfinite(r.xyz_g[1:]).all(1)&np.isfinite(r.xyz_g[:-1]).all(1)&(abs(np.diff(r.time)-1/60)<=1e-5)) for r in records))
 failures=[]
 if metrics['episode_sensitivity'] is None or metrics['episode_sensitivity']<.9:failures.append('sensitivity')
 if metrics['false_alarms_per_hour'] is None or metrics['false_alarms_per_hour']>1:failures.append('unsuccessful_starts_per_hour')
 if not metrics['true_warning_events']:failures.append('no_successful_warning')
 if metrics['maximum_alarm_duration_seconds']>1+1/60:failures.append('pulse_duration')
 if metrics['alarm_active_fraction']>.15:failures.append('alert_occupancy')
 if metrics['false_alarm_active_fraction']>.1:failures.append('false_occupancy')
 return dict(metrics=metrics,events=events,known_target_failures=failures,assessment_status='INCONCLUSIVE_INCOMPLETE_TIMELINE' if incomplete else ('FAILURE' if failures else 'POINT_TARGETS_MET'),alert_burden_status='INCONCLUSIVE_UNKNOWN_STARTS' if metrics['censored_alarm_starts'] else 'SCORED_ONLY_SHORT_EXPOSURE',clinical_validation=False)
