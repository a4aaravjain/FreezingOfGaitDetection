"""Technical recovery must preserve fitted weights and exact training membership."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from fog_pipeline import sha256
from published_bookkeeping_recovery import recovered_fit,safe_save


class BookkeepingRecoveryTests(unittest.TestCase):
    def test_native_json_scalars(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'record.json'
            safe_save(path,dict(subjects=[np.int64(1)],count=np.int64(2),valid=np.bool_(True)))
            self.assertEqual(json.loads(path.read_text()),dict(subjects=[1],count=2,valid=True))

    def test_fitted_weight_reuse_requires_exact_training_membership(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp);folder=out/'fold_1'/'verifiers'/'logistic';folder.mkdir(parents=True)
            x=np.array([[0.],[1.],[2.],[3.]])
            model=make_pipeline(StandardScaler(),LogisticRegression()).fit(x,[0,0,1,1])
            joblib.dump(model,folder/'model.joblib');before=sha256(folder/'model.joblib')
            meta=pd.DataFrame(dict(subject=[1,1,2,2],target=[0,0,1,1],active_fog=False,known_followup=True,time=np.arange(4)))
            candidate=np.ones(4,bool)
            pd.DataFrame(dict(subject=meta.subject,time=meta.time,candidate=candidate,known=meta.known_followup,used=candidate,label=meta.target)).to_csv(folder/'training_candidate_ledger.csv',index=False)
            records=[]
            def resume(family,x,seq,candidate,meta,path):
                self.assertTrue((path/'FIT.json').exists())
            recovered_fit(resume,'logistic',x,None,candidate,meta,folder,out,records)
            self.assertEqual(sha256(folder/'model.joblib'),before)
            self.assertEqual(records[0]['action'],'reused_saved_fitted_weights')
            self.assertEqual(json.loads((folder/'FIT.json').read_text())['subjects'],[1,2])
            (folder/'FIT.json').unlink();candidate[0]=False
            def fresh_fit(family,x,seq,candidate,meta,path):
                self.assertFalse(path.exists())
            recovered_fit(fresh_fit,'logistic',x,None,candidate,meta,folder,out,records)
            self.assertEqual(records[-1]['action'],'preserved_initial_invalid_attempt')
            self.assertEqual(sha256(Path(records[-1]['archive'])/'model.joblib'),before)


if __name__=='__main__':unittest.main()
