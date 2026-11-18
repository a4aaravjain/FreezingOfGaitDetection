import unittest,tempfile,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,pandas as pd
from published_cascade_core import CandidateGate,CascadeFeatures,pulse_trace,part_from_frame,scored,feature_rows
from test_locked_assessment import fixture as native_fixture


def fixture():
    f=native_fixture();f['activity']=1
    return f


class CascadeTests(unittest.TestCase):
    def test_candidate_capacity_for_every_prefix_under_distribution_shift(self):
        for every in [3,4]:
            for p in [np.ones(100),np.r_[np.zeros(20),np.ones(80)]]:
                gate=CandidateGate(.2,every);accepted=[]
                for n,value in enumerate(p,1):
                    accepted.append(gate.step([value]));self.assertLessEqual(sum(accepted),n//every)
                positions=np.flatnonzero(accepted)
                self.assertTrue(np.all(np.diff(positions)>=every))

    def test_feature_future_perturbation_cannot_change_past(self):
        rng=np.random.default_rng(2);p=rng.random((30,10));shape=rng.normal(size=(30,248));policy=dict(threshold=.2,every=3)
        def replay(a):
            state=CascadeFeatures(policy);return np.array([state.step(v,s,i*.25,bool(i%3==2)) for i,(v,s) in enumerate(zip(a,shape))])
        a=replay(p);p[20:]=1;np.testing.assert_array_equal(a[:20],replay(p)[:20]);self.assertEqual(a.shape,(30,450))

    def test_rescue_pulse_stays_one_second_when_score_falls(self):
        part=part_from_frame(fixture(),[1]);p=np.zeros(len(part['meta']));p[0]=1
        f=pulse_trace(part,p,.5)
        self.assertEqual(int(f.alarm_start.sum()),1)
        self.assertAlmostEqual(float(f.loc[f.alarm_active,'duration'].sum()),1)

    def test_logical_or_can_hide_a_baseline_edge_and_must_be_detected(self):
        part=part_from_frame(fixture(),[1]);n=len(part['meta']);base=part['meta'].copy();base['alarm_start']=False;base['alarm_active']=False
        base.loc[2,'alarm_start']=True;base.loc[2:5,'alarm_active']=True
        p=np.zeros(n);p[0]=1;f=pulse_trace(part,p,.5,np.ones(n,bool),base)
        self.assertTrue(base.loc[2,'alarm_start']);self.assertFalse(f.loc[2,'alarm_start'])
        self.assertFalse(np.all(~base.alarm_start.to_numpy(bool)|f.alarm_start.to_numpy(bool)))

    def test_history_ablation_removes_implicit_tcn_history(self):
        part=part_from_frame(fixture(),[1]);n=len(part['meta']);p=np.ones((n,10))*.5;s=np.ones((n,248))
        _,q,_=feature_rows(part,p,s,dict(threshold=.2,every=3),'no_history')
        np.testing.assert_array_equal(q[:,:-1],0)


class ExportAuditTests(unittest.TestCase):
    def test_new_export_scope_and_manifest(self):
        import csv
        from fog_pipeline import sha256
        r=Path(__file__).resolve().parent;p=r/'development_only_20261111_02.csv';m=json.loads(p.with_suffix('.manifest.json').read_text())
        self.assertEqual(sha256(p),m['output_sha256']);self.assertIsInstance(m['source_content_hash'],str)
        count=0;people=set()
        with p.open(newline='') as f:
            for row in csv.DictReader(f):
                subject=int(row['subjectID']);self.assertIn(subject,range(1,19));people.add(subject);count+=1
        self.assertEqual(count,m['permitted_rows']);self.assertEqual(people,set(range(1,19)))

    def test_input_hash_does_not_decode_reserved_fields(self):
        import hashlib,export_development_only as module
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);raw=b'x,subjectID,sessionID,taskID\n1,1,1,1\n"POISON,19,1,1\n'
            (p/'input').write_bytes(raw);seen=[];original=module.parse_allowed
            def parser(row,columns):seen.append(row);return original(row,columns)
            with patch.object(module,'parse_allowed',side_effect=parser):report=module.export(p/'input',p/'out',True)
            self.assertEqual(report['source_content_hash'],hashlib.sha256(raw).hexdigest());self.assertEqual(len(seen),1)


class NeuralSmokeTests(unittest.TestCase):
    def test_all_training_recipes_run_reload_and_keep_scope(self):
        import published_neural as neural
        n=32;rng=np.random.default_rng(1)
        meta=pd.DataFrame(dict(target=[1,0,0,0]*8,active_fog=[False]*n,known_followup=[True]*n,
            subject=[1]*16+[2]*16,episode=['a','', '', '']*4+['b','','','']*4,
            hard_category=['other']*n,time=np.arange(n)*.25))
        part=dict(data=type('D',(),dict(x=rng.normal(size=(n,180,6)).astype(np.float32)))(),meta=meta)
        with tempfile.TemporaryDirectory() as d,patch.object(neural,'SUPERVISED_EPOCHS',1),patch.object(neural,'PRETRAIN_EPOCHS',1):
            for key in neural.NEURAL:
                model=neural.train_one(key,part,dict(mean=np.zeros(6,np.float32),std=np.ones(6,np.float32)),Path(d))
                p=neural.predict(model,part['data'].x[:2]);self.assertTrue(np.isfinite(p).all());self.assertTrue(((p>=0)&(p<=1)).all())
                self.assertEqual(json.loads((Path(d)/(key+'_DONE.json')).read_text())['training_subjects'],[1,2])

    def test_causal_refinement_does_not_read_later_sequence_positions(self):
        import published_neural as neural
        _,model=neural.verifier_tcn(450);x=np.random.default_rng(5).normal(size=(1,13,450)).astype(np.float32)
        a=np.asarray(model(x,training=False));x[:,8:]*=100;b=np.asarray(model(x,training=False))
        np.testing.assert_allclose(a[:,:8],b[:,:8],atol=1e-6)


if __name__=='__main__':unittest.main()
