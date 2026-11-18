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
        self.assertEqual(targets(s,ends).tolist(),[0,1,0,-1,0,1,-1])

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
        p=np.zeros(len(d.y));p[np.argmin(abs(d.meta.time-7.5))]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertEqual(m["device"]["episodes_predicted_before_onset"],1)

    def test_post_fog_next_episode_is_positive(self):
        s=synthetic(onsets=(10,14),duration=.25)
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

    def test_duration_guard_rejects_long_low_duty_alarm(self):
        c=replace(Config(),ema_alpha=1,persistence=1,cooldown_seconds=0)
        d=build_windows([synthetic(seconds=300,onsets=(10,))])
        probability=np.zeros(len(d.y))
        probability[(d.meta.time>=7.75)&(d.meta.time<28)]=1
        m,_,_=evaluate(d,probability,.5,c)
        self.assertLess(m["device"]["alarm_active_fraction"],.15)
        self.assertGreater(m["device"]["maximum_alarm_duration_seconds"],19)
        self.assertEqual(policy_rejection_reasons(m,c),["continuous_alarm_too_long"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            threshold,rows=select_threshold(d,probability,c)
        self.assertEqual(threshold,1.1)
        self.assertTrue(all("continuous_alarm_too_long" in r["rejection_reasons"] for r in rows))

    def test_duration_guard_does_not_truncate_reported_time(self):
        c=replace(Config(),ema_alpha=1,persistence=1)
        d=build_windows([synthetic(seconds=100,onsets=(10,))])
        p=np.zeros(len(d.y));p[(d.meta.time>=7.75)&(d.meta.time<18)]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertAlmostEqual(m["device"]["total_alarm_seconds"],10.25)
        self.assertIn("continuous_alarm_too_long",policy_rejection_reasons(m,c))

    def test_policy_reasons_include_raw_rate(self):
        c=replace(Config(),ema_alpha=1,persistence=1)
        d=build_windows([synthetic(seconds=300,onsets=(10,))])
        p=np.zeros(len(d.y));p[(d.meta.time>=7.75)&(d.meta.time<9)]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertEqual(policy_rejection_reasons(m,c),[])
        m["window"]["positive_fraction"]=.99
        self.assertEqual(policy_rejection_reasons(m,c),["raw_positive_rate_over_budget"])

    def test_invalid_duration_budget(self):
        for value in [0,-1,np.nan]:
            with self.assertRaises(ValueError):
                replace(Config(),max_alarm_duration_seconds=value).validate()

    def test_strict_two_second_boundary(self):
        s=synthetic()
        self.assertEqual(targets(s,np.array([8*64-1,8*64,8*64+1,9*64])).tolist(),[1,1,0,0])

    def test_late_warning_is_unsuccessful_and_false(self):
        c=replace(Config(),ema_alpha=1,persistence=1)
        d=build_windows([synthetic()]); p=np.zeros(len(d.y))
        p[np.argmin(abs(d.meta.time-9))]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertEqual(m['device']['late_warning_events'],1)
        self.assertEqual(m['device']['total_false_alarms'],1)
        self.assertEqual(m['device']['true_warning_events'],0)
        self.assertFalse(m['device']['meets_research_target'])

    def test_invalid_lead(self):
        for lead in [-1,3,4]:
            with self.assertRaises(ValueError): replace(Config(),minimum_lead_seconds=lead).validate()

    def test_fogstar_missing_rows_split_without_stitching(self):
        from fogstar_adapter import segments_from_frame
        n=600;f=pd.DataFrame(dict(timestamp=np.arange(n)/60,subjectID=1,sessionID=1,taskID=1,fog=0))
        for a in 'xyz':f['ankleL_acc_'+a]=1.
        f.loc[250:259,'ankleL_acc_x']=np.nan
        f.loc[260:280,'fog']=1
        segments=segments_from_frame(f)
        self.assertEqual([len(s.time) for s in segments],[250,340])
        self.assertEqual(sum(len(s.onsets) for s in segments),0)
        self.assertTrue(segments[1].left_censored_fog)
        cfg=replace(Config(),sample_rate=60);cfg.validate()
        d=build_windows(segments,cfg)
        self.assertEqual(d.x.shape[1],180)
        self.assertTrue((d.meta.end_row>=179).all())

    def test_features_batch_and_future_independence(self):
        from strict_baselines import features
        d=build_windows([synthetic()])
        f=features(d.x)
        np.testing.assert_allclose(f[3:4],features(d.x[3:4]),atol=1e-6)
        changed=d.x.copy();changed[4:]=100
        np.testing.assert_array_equal(f[:4],features(changed)[:4])

    def test_native_60hz_stream_parity(self):
        c=replace(Config(),sample_rate=60)
        t=np.arange(600)/60;s=Segment('FS01:0','FS01',t,np.ones((600,3),np.float32),np.ones(600,np.int8),np.array([]),False)
        d=build_windows([s],c);params=fit_normalizer([s]);predict=lambda x:np.full(len(x),.7)
        r=replay(d,predict(d.x),.5,c);stream=OnlinePredictor(predict,params,.5,c)
        output=[o for ti,x in zip(t,s.xyz_g) if (o:=stream.step(ti,x)) is not None]
        np.testing.assert_array_equal([o['alarm_start'] for o in output],r.alarm_start)

    def test_known_duplicate_near_tail_is_not_censored(self):
        c=replace(Config(),ema_alpha=1,persistence=1,cooldown_seconds=0)
        d=build_windows([synthetic(seconds=10.5,onsets=(10,))]);p=np.zeros(len(d.y))
        for when in [7.25,7.75]:p[np.argmin(abs(d.meta.time-when))]=1
        m,_,_=evaluate(d,p,.5,c)
        self.assertEqual(m['device']['true_warning_events'],1)
        self.assertEqual(m['device']['total_false_alarms'],1)
        self.assertEqual(m['device']['censored_alarm_starts'],0)

    def test_extended_horizon_keeps_two_second_minimum(self):
        c=replace(Config(),history_seconds=6,horizon_seconds=5)
        c.validate();s=synthetic()
        self.assertEqual(targets(s,np.array([5*64,7*64,8*64,9*64]),c).tolist(),[1,1,1,0])
        d=build_windows([s],c)
        self.assertEqual(d.x.shape[1:],(384,3))
        self.assertTrue((d.meta.time>=s.time[383]).all())

    def test_invariant_motion_rotation_and_batch_independence(self):
        from pooled_motion import invariant_motion
        rng=np.random.default_rng(7);x=rng.normal(size=(3,384,3)).astype(np.float32)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        a=invariant_motion(x,64);b=invariant_motion(x@q,64)
        self.assertEqual(a.shape,(3,120,3))
        np.testing.assert_allclose(a,b,atol=2e-6)
        np.testing.assert_allclose(a[0:1],invariant_motion(x[0:1],64),atol=1e-6)
        self.assertTrue(np.isfinite(invariant_motion(np.ones((2,360,3)),60)).all())

    def test_non_neural_features_preserve_order_and_batch_independence(self):
        from non_neural_experiments import temporal_features
        rng=np.random.default_rng(19);x=rng.normal(size=(2,384,3)).astype(np.float32)
        x[:,192:]*=3
        a=temporal_features(x)
        np.testing.assert_allclose(a[:1],temporal_features(x[:1]),atol=1e-6)
        b=temporal_features(np.concatenate([x[:,192:],x[:,:192]],axis=1))
        self.assertFalse(np.allclose(a,b))
        self.assertTrue(np.isfinite(a).all())
        with self.assertRaises(ValueError):temporal_features(x[:,:192])

    def test_gyroscope_missing_sample_resets_joint_history(self):
        from fogstar_adapter import segments_from_frame
        n=400;f=pd.DataFrame(dict(timestamp=np.arange(n)/60,subjectID=1,sessionID=1,taskID=1,fog=0))
        for sensor in ['acc','gyro']:
            for a in 'xyz':f['ankleL_'+sensor+'_'+a]=1.
        f.loc[200,'ankleL_gyro_y']=np.nan
        acceleration=segments_from_frame(f)
        joint=segments_from_frame(f,include_gyro=True)
        self.assertEqual(len(acceleration),1)
        self.assertEqual([len(s.time) for s in joint],[200,199])
        self.assertEqual(joint[0].xyz_g.shape[1],6)

    def test_hold_uses_only_last_observation_and_expires(self):
        from causal_hold import CausalHold
        h=CausalHold(60,.1,3)
        self.assertIsNone(h.step(0,[np.nan]*3))
        np.testing.assert_array_equal(h.step(1/60,[1,2,3]),[1,2,3,0])
        for i in range(2,8):np.testing.assert_array_equal(h.step(i/60,[np.nan]*3),[1,2,3,1])
        self.assertIsNone(h.step(8/60,[np.nan]*3))
        np.testing.assert_array_equal(h.step(9/60,[4,5,6]),[4,5,6,0])
        self.assertIsNone(h.step(1,[np.nan]*3))

    def test_hold_does_not_backfill_initial_missing_samples(self):
        from fogstar_adapter import segments_from_frame
        n=400;f=pd.DataFrame(dict(timestamp=np.arange(n)/60,subjectID=1,sessionID=1,taskID=1,fog=0))
        for a in 'xyz':f['ankleL_acc_'+a]=1.
        f.loc[0:2,'ankleL_acc_x']=np.nan;f.loc[200:201,'ankleL_acc_y']=np.nan
        s=segments_from_frame(f,max_hold_seconds=.1)
        self.assertEqual(len(s),1);self.assertAlmostEqual(s[0].time[0],3/60)
        self.assertEqual(s[0].xyz_g.shape,(397,4))
        self.assertEqual(s[0].xyz_g[:,-1].sum(),2)

    def test_state_supervision_separates_useful_late_and_active(self):
        from state_forecast import state_labels
        cfg=replace(Config(),horizon_seconds=5)
        seg=synthetic();times=np.array([6.,8.,9.,10.25,29.])
        ends=(times*64).astype(int)
        meta=pd.DataFrame(dict(segment=[seg.key]*len(times),time=times,active_fog=seg.labels[ends]==2))
        data=Windows(np.empty((len(times),0,3)),targets(seg,ends,cfg),meta,[seg])
        self.assertEqual(state_labels(data,cfg).tolist(),[1,1,2,3,-1])

    def test_auxiliary_features_keep_units_and_sampling_duration(self):
        from pretraining_experiments import motion_features
        a=np.ones((2,180,6),np.float32);b=np.ones((2,384,6),np.float32)
        a[:,:,3:]=10;b[:,:,3:]=10
        np.testing.assert_allclose(motion_features(a,60),motion_features(b,128),atol=1e-6)
        rng=np.random.default_rng(12);q,_=np.linalg.qr(rng.normal(size=(3,3)))
        rotated=np.concatenate([a[:,:,:3]@q,a[:,:,3:]@q],axis=2)
        np.testing.assert_allclose(motion_features(a,60),motion_features(rotated,60),atol=1e-5)
        with self.assertRaises(ValueError):motion_features(a,128)

    def test_long_history_is_causal_and_resets_at_segments(self):
        from long_history_experiments import histories
        f=np.arange(10,dtype=np.float32)[:,None]
        meta=pd.DataFrame(dict(segment=['a']*5+['b']*5))
        a=histories(f,meta,steps=3,stride=2)
        np.testing.assert_array_equal(a[4,:,0],[0,2,4])
        np.testing.assert_array_equal(a[5,:,1],[0,0,1])
        self.assertEqual(a[5,-1,0],5)
        changed=f.copy();changed[3:]=999
        np.testing.assert_array_equal(a[:3],histories(changed,meta,steps=3,stride=2)[:3])

    def test_bilateral_columns_are_explicit_and_missing_side_splits(self):
        from fogstar_adapter import segments_from_frame
        f=pd.DataFrame(dict(subjectID=[1]*10,sessionID=[1]*10,taskID=[1]*10,timestamp=np.arange(10)/60,fog=[0]*10))
        for side,value in [('ankleL',2.),('ankleR',7.)]:
            for sensor in ['acc','gyro']:
                for axis in 'xyz':f[side+'_'+sensor+'_'+axis]=value
        a=segments_from_frame(f,ankle='both',include_gyro=True)
        np.testing.assert_array_equal(a[0].xyz_g[0],[2]*6+[7]*6)
        f.loc[4,'ankleL_gyro_x']=np.nan
        a=segments_from_frame(f,ankle='both',include_gyro=True)
        self.assertEqual([len(z.time) for z in a],[4,5])

    def test_stanford_units_side_and_native_time(self):
        from stanford_adapter import frame_segment
        f=pd.DataFrame(dict(subject_ID=[1]*512,time=np.arange(512)/128,freeze_label=[0]*400+[1]*112))
        for side in ['l','r']:
            for sensor,value in [('a',9.80665),('g',np.pi)]:
                for axis in 'xyz':f['imu_ankle_'+side+'_'+sensor+axis]=value*(1 if side=='r' else 2)
        a=frame_segment(f,'test','r');b=frame_segment(f,'test','l')
        np.testing.assert_allclose(a.xyz_g[0],[1]*3+[180]*3,rtol=1e-6)
        np.testing.assert_allclose(b.xyz_g[0],[2]*3+[360]*3,rtol=1e-6)
        self.assertEqual(a.onsets[0],400/128)
        cfg=replace(Config(),sample_rate=128)
        w=build_windows([a],cfg)
        self.assertEqual(w.x.shape[1],384)
        f.loc[10,'time']+=.01
        with self.assertRaises(ValueError):frame_segment(f,'bad')

    def test_multiscale_rotation_invariance(self):
        from ankle_multiscale import multiscale
        rng=np.random.default_rng(2);x=rng.normal(size=(2,180,6)).astype(np.float32)
        q,_=np.linalg.qr(rng.normal(size=(3,3)))
        rotated=np.concatenate([x[:,:,:3]@q,x[:,:,3:]@q],axis=2)
        np.testing.assert_allclose(multiscale(x),multiscale(rotated),rtol=2e-4,atol=2e-4)

    def test_summary_stream_rejects_wrong_ankle_before_loading(self):
        from summary_stream import SummaryStream
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            (folder/'manifest.json').write_text('{"ankle":"ankleL"}')
            with self.assertRaisesRegex(ValueError,'right-ankle'):SummaryStream(folder)
            (folder/'manifest.json').write_text('{"ankle":"ankleR"}')
            with self.assertRaisesRegex(ValueError,'right-ankle'):SummaryStream(folder,ankle='ankleL')

    def test_pulse_caps_output_without_retriggering_persistent_risk(self):
        cfg=replace(Config(),ema_alpha=1,persistence=1,alert_pulse_seconds=1)
        alarm=Alarm(.5,cfg)
        outputs=[alarm.step(i*.25,.9) for i in range(40)]
        self.assertEqual(sum(a[2] for a in outputs),1)
        self.assertEqual(sum(a[1] for a in outputs),4)
        self.assertTrue(alarm.active) # latched risk prevents repeat alerts
        alarm.reset()
        self.assertTrue(alarm.step(0,.9)[2])

    def test_pulse_preserves_all_starts_and_cooldown(self):
        cfg=replace(Config(),ema_alpha=.5,persistence=2)
        continuous=Alarm(.6,cfg);pulsed=Alarm(.6,replace(cfg,alert_pulse_seconds=1))
        scores=np.tile(np.r_[np.full(30,.9),np.full(20,.1)],4)
        for i,p in enumerate(scores):
            a=continuous.step(i*.25,p);b=pulsed.step(i*.25,p)
            self.assertEqual(a[0],b[0]);self.assertEqual(a[2],b[2])
            self.assertLessEqual(b[1],a[1])
        with self.assertRaises(ValueError):replace(cfg,alert_pulse_seconds=-1).validate()

    def test_episode_bags_keep_positive_history_before_onset(self):
        from episode_calibration import episode_bags
        data=build_windows([synthetic()],replace(Config(),horizon_seconds=5))
        features=np.zeros((len(data.y),24),np.float32)
        features[:,0]=np.arange(len(data.y))+1
        bags,valid,labels=episode_bags(data,features)
        self.assertEqual(int(labels.sum()),2)
        for bag,mask,label in zip(bags,valid,labels):
            ids=bag[mask.astype(bool),0].astype(int)-1
            if label:
                self.assertTrue(np.all(data.y[ids]==1))
                self.assertTrue(np.all(np.diff(ids)==1))
                times=data.meta.iloc[ids].time.to_numpy()
                onset=next(o for o in data.segments[0].onsets if o>times[0])
                self.assertTrue(np.all(onset-times>=2-1e-9))
                self.assertTrue(np.all(onset-times<=5+1e-9))
            else:
                self.assertTrue(np.all((data.y[ids]==0)|data.meta.iloc[ids].active_fog.to_numpy()))

    def test_local_contrast_uses_only_prior_same_segment_samples(self):
        from local_contrast import local_contrast
        x=np.arange(24,dtype=np.float32).reshape(12,2)
        meta=pd.DataFrame(dict(segment=['a']*6+['b']*6))
        before=local_contrast(x,meta,history=4)
        changed=x.copy();changed[5:]=999
        np.testing.assert_array_equal(before[:5],local_contrast(changed,meta,history=4)[:5])
        np.testing.assert_array_equal(before[6],np.zeros(5))
        self.assertEqual(before[5,-1],1)
        current_changed=x.copy();current_changed[4]+=3
        after=local_contrast(current_changed,meta,history=4)
        np.testing.assert_allclose(after[4,:2]-before[4,:2],3)

    def test_personal_setup_is_entire_first_recording_without_label_selection(self):
        from earlier_recording_adaptation import chronological_split
        base=synthetic()
        first=replace(base,key='FS15:1:1:ankleR:0')
        first_gap=replace(base,key='FS15:1:1:ankleR:100',time=base.time+40)
        later=replace(base,key='FS15:1:3:ankleR:0',time=base.time+100)
        setup,assessment=chronological_split([later,first_gap,first])
        self.assertEqual({s.key for s in setup},{first.key,first_gap.key})
        self.assertEqual([s.key for s in assessment],[later.key])
        changed=replace(first,onsets=np.array([]))
        setup2,_=chronological_split([later,first_gap,changed])
        self.assertEqual({s.key for s in setup2},{s.key for s in setup})

    def test_personal_setup_rejects_overlapping_recordings(self):
        from earlier_recording_adaptation import chronological_split
        first=replace(synthetic(),key='FS15:1:1:ankleR:0')
        overlapping=replace(first,key='FS15:1:3:ankleR:0',time=first.time+10)
        with self.assertRaises(ValueError):chronological_split([first,overlapping])

if __name__=="__main__":
    unittest.main(verbosity=2)

