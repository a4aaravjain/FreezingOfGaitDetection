"""Adversarial scientific checks; run with python test_pipeline.py."""
import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
import warnings
import numpy as np
import pandas as pd
from fog_pipeline import *

def synthetic(seconds=30,onsets=(10,20),duration=1):
    t=np.arange(seconds*64)/64
    y=np.ones(len(t),np.int8)
    for o in onsets:
        y[(t>=o)&(t<o+duration)]=2
    x=np.column_stack([np.sin(2*np.pi*t),np.cos(t),np.ones(len(t))]).astype(np.float32)
    return Segment("synthetic:0","S01",t,x,y,np.array(onsets,dtype=float),False)

class PipelineTests(unittest.TestCase):
    def test_target_boundaries_and_censoring(self):
        s=synthetic()
        ends=np.array([6*64,7*64,10*64-1,10*64,11*64,17*64,28*64])
        self.assertEqual(targets(s,ends).tolist(),[0,1,1,-1,0,1,-1])

    def test_no_fog_is_negative_until_censored_tail(self):
        s=synthetic(onsets=())
        d=build_windows([s])
        self.assertEqual(set(d.y),{-1,0})
        self.assertTrue((d.y[d.meta.time<27]==0).all())

    def test_windows_end_at_prediction_time(self):
        s=synthetic()
        d=build_windows([s])
        for i in [0,1,len(d.y)-1]:
            e=int(d.meta.iloc[i].end_row)
            np.testing.assert_array_equal(d.x[i],s.xyz_g[e-191:e+1])
            self.assertEqual(d.meta.iloc[i].time,s.time[e])
        self.assertEqual(d.x.shape[1:],(192,3))

    def test_future_sensors_do_not_change_past_features(self):
        s=synthetic()
        before=build_windows([s])
        s.xyz_g[s.time>12]=1e5
        after=build_windows([s])
        np.testing.assert_array_equal(before.x[before.meta.time<=12],after.x[after.meta.time<=12])

    def test_normalization_parameters_are_fixed(self):
        params=fit_normalizer([synthetic()])
        x=np.ones((4,192,3),np.float32)
        original=normalize(x,params)
        x[3]=10000
        np.testing.assert_array_equal(original[:3],normalize(x,params)[:3])

    def test_label_zero_and_timestamp_gap_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"S01R01.txt"
            a=np.zeros((1300,11)); a[:,0]=np.floor(np.arange(len(a))*1000/64)
            a[:,1:10]=1000; a[:,10]=1
            a[500:600,10]=0
            a[600:640,10]=2  # left-censored, not a known onset
            a[1000:,0]+=1000 # acquisition gap
            np.savetxt(path,a,fmt="%g")
            s=read_recording(path)
            self.assertEqual([len(x.time) for x in s],[500,400,300])
            self.assertEqual(sum(len(x.onsets) for x in s),0)
            self.assertEqual(sum(x.left_censored_fog for x in s),1)
            d=build_windows(s)
            self.assertEqual(d.meta.segment.nunique(),3)
            self.assertTrue((d.meta.end_row>=191).all())
            self.assertTrue((d.y[d.meta.time>14]<1).all())
            a[100,0]=a[99,0];np.savetxt(path,a,fmt="%g")
            with self.assertRaises(ValueError): read_recording(path)

    def test_bad_labels_and_nonfinite_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"S01R01.txt"
            a=np.ones((256,11));a[:,0]=np.floor(np.arange(256)*1000/64)
            for value in [3,np.nan]:
                a[5,10]=value;np.savetxt(path,a,fmt="%g")
                with self.assertRaises(ValueError):read_recording(path)

    def test_subject_and_threshold_leakage_guard(self):
        with self.assertRaises(ValueError):verify_split(["S01"],["S01"])
        with self.assertRaises(ValueError):verify_split(["S01"],["S02"],["S02"])
        d=build_windows([synthetic()])
        with self.assertRaises(ValueError):select_threshold(d,np.zeros(len(d.y)),role="test")

    def test_causal_alarm_prefix_invariance(self):
        d=build_windows([synthetic()])
        p=np.linspace(0,1,len(d.y))
        r=replay(d,p,.5)
        p[30:]=1-p[30:]
        r2=replay(d,p,.5)
        np.testing.assert_array_equal(r.smoothed[:30],r2.smoothed[:30])
        np.testing.assert_array_equal(r.alarm_start[:30],r2.alarm_start[:30])

    def test_active_fog_predictions_are_retained(self):
        d=build_windows([synthetic()])
        r=replay(d,np.ones(len(d.y)),.5)
        self.assertTrue(r.loc[r.active_fog,"alarm_active"].all())
        self.assertTrue((d.y[d.meta.active_fog]==-1).all())

    def test_continuous_alarm_does_not_detect_future_events(self):
        d=build_windows([synthetic(seconds=60)])
        m,_,_=evaluate(d,np.ones(len(d.y)),.5)
        self.assertEqual(m["device"]["episodes_predicted_before_onset"],0)
        self.assertEqual(m["device"]["total_false_alarms"],1)
        self.assertGreater(m["device"]["alarm_active_fraction"],.95)
        self.assertGreater(m["device"]["false_alarm_seconds"],45)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            threshold,_=select_threshold(d,np.ones(len(d.y)))
        self.assertEqual(threshold,1.1)

    def test_no_alarm_has_undefined_warning_time(self):
        d=build_windows([synthetic()])
        m,_,_=evaluate(d,np.zeros(len(d.y)),.5)
        self.assertEqual(m["device"]["total_false_alarms"],0)
        self.assertIsNone(m["device"]["average_warning_seconds"])

    def test_true_warning_start_and_duplicate_penalty(self):
        c=replace(Config(),ema_alpha=1,persistence=1,cooldown_seconds=0)
        d=build_windows([synthetic()])
        p=np.zeros(len(d.y))
        for target in [8,9]:
            p[np.argmin(abs(d.meta.time-target))]=1
        m,_,events=evaluate(d,p,.5,c)
        self.assertEqual(m["device"]["episodes_predicted_before_onset"],1)
        self.assertEqual(m["device"]["true_warning_events"],1)
        self.assertEqual(m["device"]["total_false_alarms"],1)
        self.assertAlmostEqual(events[0]["warning_seconds"],2.015625)

    def test_one_warning_cannot_match_two_onsets(self):
        c=replace(Config(),ema_alpha=1,persistence=1)
        d=build_windows([synthetic(onsets=(10,12),duration=.25)])
        p=np.zeros(len(d.y));p[np.argmin(abs(d.meta.time-9.5))]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertEqual(m["device"]["episodes_predicted_before_onset"],1)

    def test_post_fog_next_episode_is_positive(self):
        s=synthetic(onsets=(10,12),duration=.25)
        self.assertEqual(targets(s,np.array([11*64])).tolist(),[1])

    def test_no_event_and_short_segment_denominators(self):
        d=build_windows([synthetic(onsets=()),synthetic(seconds=2,onsets=(1,))])
        # Unique key needed for separate recording segments.
        d.segments[1].key="short:0"
        m,_,_=evaluate(d,np.zeros(len(d.y)),.5)
        self.assertEqual(m["device"]["total_fog_episodes"],1)
        self.assertEqual(m["device"]["eligible_episodes"],0)
        self.assertEqual(m["device"]["episode_sensitivity"],0)

    def test_online_offline_exact_state_parity(self):
        s=synthetic(seconds=15)
        params=fit_normalizer([s])
        predict=lambda x:1/(1+np.exp(-x[:,-1,0]))
        d=build_windows([s])
        p=predict(normalize(d.x,params))
        r=replay(d,p,.55)
        stream=OnlinePredictor(predict,params,.55)
        outputs=[o for t,x in zip(s.time,s.xyz_g) if (o:=stream.step(t,x)) is not None]
        for key in ["probability","smoothed","alarm_active","alarm_start"]:
            np.testing.assert_allclose([o[key] for o in outputs],r[key],rtol=1e-6)
        stream.step(50,[0,0,1])
        self.assertEqual(stream.count,1)
        self.assertFalse(stream.alarm.active)

    def test_online_nonmonotonic_resets(self):
        stream=OnlinePredictor(lambda x:[.5],dict(mean=[0]*3,std=[1]*3),.5)
        stream.step(1,[0,0,1])
        with self.assertRaises(ValueError):stream.step(1,[0,0,1])
        self.assertEqual(stream.count,0)

    def test_censored_alarm_is_not_a_false_start(self):
        c=replace(Config(),ema_alpha=1,persistence=1)
        d=build_windows([synthetic(onsets=())])
        p=np.zeros(len(d.y));p[d.meta.time>28]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertEqual(m["device"]["total_false_alarms"],0)
        self.assertEqual(m["device"]["censored_alarm_starts"],1)
        self.assertGreater(m["device"]["unscorable_alarm_seconds"],1)
        self.assertEqual(m["device"]["false_alarm_seconds"],0)

    def test_replay_rejects_invalid_probabilities(self):
        d=build_windows([synthetic()])
        for value in [np.nan,np.inf,-.1,1.1]:
            with self.assertRaises(ValueError):
                replay(d,np.full(len(d.y),value),.5)

    def test_gap_in_prediction_stream_is_rejected(self):
        d=build_windows([synthetic()])
        d.meta.loc[4,"time"]+=.1
        with self.assertRaises(ValueError):replay(d,np.zeros(len(d.y)),.5)

    def test_development_rejects_historical_subjects(self):
        from ankles import run_development
        with self.assertRaises(ValueError):
            run_development(Path("."),Path("unused"),Config(),["S01"],["S09"])

    def test_final_rejects_exposed_subjects_and_smoke(self):
        from ankles import final_evaluation
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            for mode in ["smoke","development"]:
                save_json(p/"manifest.json",dict(mode=mode,policy_status="feasible",
                                                historically_exposed_subjects=["S09","S10"]))
                with self.assertRaises(ValueError):
                    final_evaluation(p,p,["S09"],True)
            self.assertFalse((p/"FINAL_EVALUATION_STARTED.json").exists())

if __name__=="__main__":
    unittest.main(verbosity=2)

