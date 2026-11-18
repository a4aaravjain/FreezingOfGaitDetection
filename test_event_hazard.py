import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from event_hazard_data import EventFeatures,hazard_bin,load_part,normalizer,normalized,sequences,HISTORY,ALL_CHANNELS
from event_hazard_models import make_model,probabilities,fit,predict
from event_hazard_control import EventController,replay,fast_metrics
from published_cascade_core import part_from_frame,scored,HEADS


def fixture():
    n=2400;t=np.arange(n)/60;f=pd.DataFrame(dict(subjectID=1,sessionID=1,taskID=np.where(t<20,1,2),timestamp=t,fog=((t>=30)&(t<31)).astype(int),activity=np.where(t<10,3,1)))
    for j,c in enumerate(ALL_CHANNELS):f[c]=np.sin(2*np.pi*t*(1.2 if 'acc' in c else 1.7)+j*.1)+(1 if c.endswith('acc_z') else 0)
    return f


def data_fixture(folder,frame=None):
    f=fixture() if frame is None else frame;p=part_from_frame(f,sorted(f.subjectID.unique()));m=p['meta'].copy()
    for j,h in enumerate(HEADS):m[h]=np.minimum(.9,.2+np.arange(len(m))*.002+j*.005)
    m.to_csv(Path(folder)/'c1_scores.csv',index=False);np.savez(Path(folder)/'c1_features.npz',shape=np.zeros((len(m),248),np.float32))
    return load_part(f,sorted(f.subjectID.unique()),folder,'c1',dict(threshold=.1,every=3))


class HazardTests(unittest.TestCase):
    def test_exact_boundaries_and_unknowns(self):
        for t,b in [(0,0),(1,0),(2,1),(3,2),(4,3),(5,3),(5.01,4)]:self.assertEqual(hazard_bin(t,True),b)
        self.assertEqual(hazard_bin(np.inf,False),-1);self.assertEqual(hazard_bin(3,False),2)
        self.assertEqual(hazard_bin(3,True,True),-1)
        with self.assertRaises(ValueError):hazard_bin(np.nan,True)
        p=probabilities(np.zeros((2,4)));np.testing.assert_allclose(p.sum(1),1)

    def test_features_are_left_causal(self):
        rng=np.random.default_rng(2);blocks=rng.normal(size=(30,15,6)).astype(np.float32);scores=rng.random((30,10));shape=rng.normal(size=(30,248))
        def run(a):
            state=EventFeatures(dict(threshold=.2,every=3));return np.array([state.at(i*.25,b,p,s,i%3==2) for i,(b,p,s) in enumerate(zip(a,scores,shape))])
        before=run(blocks);blocks[20:]*=50;np.testing.assert_array_equal(before[:20],run(blocks)[:20]);self.assertEqual(before.shape[1],659)

    def test_all_sequence_families_are_left_causal(self):
        x=np.random.default_rng(3).normal(size=(1,HISTORY,12)).astype(np.float32)
        for family in ['tcn','gru','refinement']:
            _,temporal=make_model(family,12);a=np.asarray(temporal(x,training=False));changed=x.copy();changed[:,50:]*=100
            np.testing.assert_allclose(a[:,:50],np.asarray(temporal(changed,training=False))[:,:50],rtol=1e-5,atol=1e-6)

    def test_teacher_and_annotation_mutation_cannot_change_student_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            original=fixture();a=data_fixture(tmp,original);changed=original.copy()
            for c in ALL_CHANNELS:
                if not c.startswith('ankleR_'):changed[c]*=100
            changed['activity']=7;changed['fog']=1
            # Reuse fixed prediction caches, so neither labels nor teacher channels can affect features.
            b=load_part(changed,[1],tmp,'c1',dict(threshold=.1,every=3))
            np.testing.assert_array_equal(a['x'],b['x']);self.assertFalse(np.array_equal(a['teacher'],b['teacher']))
            self.assertEqual(a['teacher'].shape[1],1023)

    def test_setup_is_fixed_past_only_and_keeps_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=data_fixture(tmp);norm=normalizer(a);x,ready,ledger=normalized(a,norm,personal=True)
            self.assertTrue(all(~ready[a['part']['meta'].time<20]));self.assertTrue(ready[-1]);self.assertEqual(ledger[0]['start'],0)
            changed={**a,'x':a['x'].copy()};changed['x'][a['block_time']>25]+=100
            y,_,_=normalized(changed,norm,personal=True)
            np.testing.assert_array_equal(x[a['block_time']<=25],y[a['block_time']<=25])
            self.assertEqual(sum(len(r.onsets) for r in a['part']['records']),1)

    def test_sequences_never_cross_physical_sessions(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=fixture();second=f.copy();second['sessionID']=2;second['timestamp']+=100
            a=data_fixture(tmp,pd.concat([f,second],ignore_index=True));norm=normalizer(a);x,_,_=normalized(a,norm)
            row=np.flatnonzero(a['part']['meta'].segment.str.contains(':2:'))[0];before=sequences(a,x,np.array([row]))
            x[:a['starts'][a['row_index'][row]]]+=500
            np.testing.assert_array_equal(before,sequences(a,x,np.array([row])))

    def test_final_starts_determine_metrics_and_actual_start_lead(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=data_fixture(tmp);m=a['part']['meta'];risk=np.zeros(len(m));risk[(m.time>=25)&(m.time<=28)]=.9
            trace=replay(a,risk,dict(threshold=.5,confirm=True,beta=.5));fast=fast_metrics(a,trace);full=scored(a['part'],trace)['metrics']
            self.assertEqual((fast['detected'],fast['unsuccessful'],fast['unknown']),(full['true_warning_events'],full['total_false_alarms'],full['censored_alarm_starts']))
            self.assertLess(int(trace.alarm_start.sum()),int(sum(risk>.5)))
            self.assertTrue(all(2-1e-9<=v<=5+1e-9 for v in full['warning_times_seconds']))
            controller=EventController(.5);outputs=[controller.step(i*.25,.9 if i==0 else 0,True) for i in range(8)]
            self.assertEqual(sum(v[0] for v in outputs),4)

    def test_all_recipes_fit_and_reload_on_synthetic_data(self):
        import event_hazard_models as module
        with tempfile.TemporaryDirectory() as tmp:
            data=data_fixture(tmp);teacher=None
            with patch.object(module,'EPOCHS',2):
                for family in ['tcn','gru','refinement']:
                    folder=Path(tmp)/family
                    info=fit(data,folder,family,context=True,kd=teacher,personal=family=='refinement',topk=family!='gru')
                    self.assertTrue(info['reload_parity']);self.assertEqual(info['fit_subjects'],[1])
                    import tensorflow as tf
                    model=tf.keras.models.load_model(folder/'epoch_2.keras',compile=False);norm=normalizer(data);x,_,_=normalized(data,norm,personal=family=='refinement')
                    teacher=predict(model,data,x);self.assertTrue(np.isfinite(teacher['logits']).all())


if __name__=='__main__':unittest.main()
