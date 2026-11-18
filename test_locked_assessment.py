import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
import locked_assessment as locked
from assessment_core import prepare,replay,score,CHANNELS
from fog_pipeline import Config,event_metrics
from dataclasses import replace
class SilentStream:
 def reset(self):self.count=0
 def step(self,t,x):
  self.count+=1
  if self.count<180 or (self.count-180)%15:return None
  return dict(time=t,alarm_start=False,alarm_active=False)
def fixture():
 n=720;f=pd.DataFrame(dict(subjectID=[1]*n,sessionID=[1]*n,taskID=[1]*n,timestamp=np.arange(n)/60,fog=[0]*600+[1]*120))
 for c in CHANNELS:f[c]=1. if c=='ankleR_acc_z' else 0.
 return f
class InventoryTests(unittest.TestCase):
 def setUp(self):self.cfg=replace(Config(),sample_rate=60,horizon_seconds=5,alert_pulse_seconds=1)
 def test_missing_sensors_do_not_erase_onset(self):
  f=fixture();f.loc[590:620,CHANNELS]=np.nan;r,c,i,bad=prepare(f,[1]);self.assertEqual(i[0]['original_onsets'],1);p=replay(c,SilentStream());v=score(r,p,i,bad,self.cfg);self.assertEqual(v['metrics']['total_fog_episodes'],1);self.assertEqual(v['metrics']['observed_contiguous_sensor_onsets'],0);self.assertEqual(len(c),2)
 def test_no_valid_input_still_has_original_episode(self):
  f=fixture();f[CHANNELS]=np.nan;r,c,i,bad=prepare(f,[1]);v=score(r,replay(c,SilentStream()),i,bad,self.cfg);self.assertEqual(v['metrics']['total_fog_episodes'],1);self.assertEqual(v['metrics']['monitored_seconds'],0);self.assertIn('no_successful_warning',v['known_target_failures'])
 def test_labels_do_not_change_generated_alerts(self):
  class TimeStream(SilentStream):
   def step(self,t,x):
    value=super().step(t,x)
    if value is not None:value.update(alarm_start=bool(6.9<t<7.1),alarm_active=bool(6.9<t<7.9))
    return value
  f=fixture();_,a,_,_=prepare(f,[1]);f['fog']=1;_,b,_,_=prepare(f,[1]);pa=replay(a,TimeStream());pb=replay(b,TimeStream())
  pd.testing.assert_frame_equal(pa[['time','alarm_start','alarm_active']],pb[['time','alarm_start','alarm_active']])
 def test_no_fog_person_preserved(self):
  f=fixture();f['fog']=0;r,c,i,bad=prepare(f,[1]);v=score(r,replay(c,SilentStream()),i,bad,self.cfg);self.assertIsNone(v['metrics']['episode_sensitivity']);self.assertGreater(v['metrics']['monitored_seconds'],0)
 def test_full_contiguous_replay_matches_existing_event_counts(self):
  r,c,i,bad=prepare(fixture(),[1]);p=replay(c,SilentStream());old,_=event_metrics(type('D',(),dict(segments=r))(),p,self.cfg);new=score(r,p,i,bad,self.cfg)['metrics']
  for key in ['true_warning_events','total_false_alarms','censored_alarm_starts','total_fog_episodes','eligible_episodes','monitored_seconds']:self.assertEqual(old[key],new[key])
 def test_alert_matching_agrees_for_true_late_and_unknown_starts(self):
  class TimedStream(SilentStream):
   def step(self,t,x):
    value=super().step(t,x);index=round(t*60);starts=[434,524,674]
    if value is not None:value.update(alarm_start=index in starts,alarm_active=any(k<=index<k+60 for k in starts))
    return value
  f=fixture();f.loc[660:,'fog']=0;r,c,i,bad=prepare(f,[1]);p=replay(c,TimedStream());old,_=event_metrics(type('D',(),dict(segments=r))(),p,self.cfg);new=score(r,p,i,bad,self.cfg)['metrics']
  self.assertEqual(new['true_warning_events'],1);self.assertEqual(new['total_false_alarms'],1);self.assertEqual(new['censored_alarm_starts'],1)
  for key in ['true_warning_events','total_false_alarms','censored_alarm_starts','monitored_seconds','total_alarm_seconds','false_alarm_seconds']:self.assertAlmostEqual(old[key],new[key])
 def test_warning_can_match_annotation_during_sensor_gap(self):
  class WarningStream(SilentStream):
   def step(self,t,x):
    value=super().step(t,x)
    if value is not None:value.update(alarm_start=round(t*60)==434,alarm_active=434<=round(t*60)<494)
    return value
  f=fixture();f.loc[590:620,CHANNELS]=np.nan;r,c,i,bad=prepare(f,[1]);v=score(r,replay(c,WarningStream()),i,bad,self.cfg)
  self.assertEqual(v['metrics']['total_fog_episodes'],1);self.assertEqual(v['metrics']['true_warning_events'],1)
 def test_missing_time_is_inconclusive_not_silently_removed(self):
  f=fixture().drop(index=300).reset_index(drop=True);r,c,i,bad=prepare(f,[1]);self.assertTrue(bad);self.assertEqual(score(r,replay(c,SilentStream()),i,bad,self.cfg)['assessment_status'],'INCONCLUSIVE_INCOMPLETE_TIMELINE')
 def test_wrong_rate_and_people_rejected(self):
  with self.assertRaises(ValueError):prepare(fixture(),[2])
  f=fixture();f.timestamp=np.arange(len(f))/128
  with self.assertRaises(ValueError):prepare(f,[1])
 def test_separate_chunks_do_not_join_pulse_lengths(self):
  r,c,i,bad=prepare(fixture(),[1]);p=pd.DataFrame([dict(segment=r[0].key,chunk='a',time=3.,active_fog=False,alarm_start=True,alarm_active=True,duration=.25),dict(segment=r[0].key,chunk='b',time=7.,active_fog=False,alarm_start=True,alarm_active=True,duration=.25)])
  self.assertEqual(score(r,p,i,bad,self.cfg)['metrics']['maximum_alarm_duration_seconds'],.25)
class GuardTests(unittest.TestCase):
 def setup_bundle(self,root):
  b=Path(root);(b/'runtime').mkdir();source=Path(locked.__file__).read_bytes();(b/'runtime/locked_assessment.py').write_bytes(source);data=b/'synthetic.csv';data.write_text('synthetic test bytes only')
  m=dict(execution_ready=True,files_sha256={'runtime/locked_assessment.py':locked.sha(b/'runtime/locked_assessment.py')},approved_partition_sha256='deliberately_wrong');(b/'freeze_manifest.json').write_text(json.dumps(m));a=dict(decision='APPROVE_ONE_LOCKED_ASSESSMENT',bundle_sha256=locked.sha(b/'freeze_manifest.json'),subjects=[19,20,21,22],reviewer='synthetic test fixture');(b/'approval.json').write_text(json.dumps(a));return b,data
 def test_denied_review_does_not_open_dataset_or_mark_started(self):
  with tempfile.TemporaryDirectory() as root:
   b,data=self.setup_bundle(root);a=json.loads((b/'approval.json').read_text());a['decision']='PENDING';(b/'approval.json').write_text(json.dumps(a))
   with patch('locked_assessment.sha',side_effect=AssertionError('Must not read data hashes before denial')):
    with self.assertRaises(PermissionError):locked.run(b,b/'approval.json',data,b/'out')
   self.assertFalse((b/'ASSESSMENT_STARTED.json').exists())
 def test_failure_keeps_marker_and_retry_is_blocked(self):
  with tempfile.TemporaryDirectory() as root:
   b,data=self.setup_bundle(root)
   with patch.object(locked,'__file__',str(b/'runtime/locked_assessment.py')):
    with self.assertRaises(ValueError):locked.run(b,b/'approval.json',data,b/'out')
   self.assertTrue((b/'ASSESSMENT_STARTED.json').exists());self.assertTrue((b/'ASSESSMENT_ERROR.json').exists())
   with patch.object(locked,'__file__',str(b/'runtime/locked_assessment.py')):
    with self.assertRaises(FileExistsError):locked.run(b,b/'approval.json',data,b/'different_output')
 def test_changed_frozen_source_blocks_before_marker(self):
  with tempfile.TemporaryDirectory() as root:
   b,data=self.setup_bundle(root);(b/'runtime/locked_assessment.py').write_text('tampered')
   with self.assertRaises(ValueError):locked.run(b,b/'approval.json',data,b/'out')
   self.assertFalse((b/'ASSESSMENT_STARTED.json').exists())
if __name__=='__main__':unittest.main()
