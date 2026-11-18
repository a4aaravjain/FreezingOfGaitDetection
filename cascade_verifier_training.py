"""Strict ledger consumer for nested candidate learning; never creates in-sample upstream scores."""
import json,time
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from rescue_cascade import RescueFeatures,candidate_label,validate_provenance

HEADS=['base','waveform','extended']
def ledger_features(frame,provenance,outer,calibration,l1):
 required=['subject','segment','time','upstream_id','target','active_fog','known_followup']+[h+'_probability' for h in HEADS]+['motion_'+str(i) for i in range(24)]
 if any(k not in frame for k in required):raise ValueError('Missing full nested score/motion ledger; do not substitute in-sample scores')
 features=[];candidates=[];labels=[]
 for _,g in frame.groupby('segment',sort=False):
  state=RescueFeatures(l1);last=None
  for row in g.to_dict('records'):
   validate_provenance(provenance[row['upstream_id']],int(row['subject']),outer,calibration)
   if last is not None and (row['time']<=last or abs(row['time']-last-.25)>.002):raise ValueError('Split/reset score histories at gaps')
   last=row['time'];candidate,x=state.step([row[h+'_probability'] for h in HEADS],[row['motion_'+str(i)] for i in range(24)])
   features.append((row['time'],x));candidates.append(candidate);labels.append(candidate_label(row['target'],row['active_fog'],row['known_followup']))
 # Require sorted, segment-contiguous RangeIndex so arrays align with metadata.
 ordered=pd.concat([g for _,g in frame.groupby('segment',sort=False)]).index.to_numpy()
 if not np.array_equal(ordered,np.arange(len(frame))):raise ValueError('Ledger rows must be contiguous and RangeIndexed')
 return np.stack([v for _,v in features]),np.asarray(candidates),np.array([-1 if y is None else y for y in labels])

def fit_pair(frame,provenance,outer,calibration,l1,output,ablation='full'):
 x,candidate,y=ledger_features(frame,provenance,outer,calibration,l1);mask=candidate&(y>=0)
 if set(y[mask])!={0,1}:raise ValueError('Candidate training lacks both known classes; record failure')
 if ablation=='no_score_derivatives':x[:,3:12]=0
 elif ablation=='no_motion_transition':x[:,41:65]=0
 elif ablation!='full':raise ValueError('Unknown preregistered ablation')
 output.mkdir(exist_ok=False);pd.DataFrame(dict(subject=frame.subject,candidate=candidate,label=y,used_for_training=mask)).to_csv(output/'training_ledger.csv',index=False)
 models={}
 for name,model in [('logistic',make_pipeline(StandardScaler(),LogisticRegression(C=.1,max_iter=1500,random_state=42))),('hgb',HistGradientBoostingClassifier(max_iter=100,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=5,early_stopping=False,random_state=42))]:
  start=time.perf_counter();model.fit(x[mask],y[mask]);p=model.predict_proba(x)[:,1];joblib.dump(model,output/(name+'.joblib'));loaded=joblib.load(output/(name+'.joblib'));np.testing.assert_allclose(p,loaded.predict_proba(x)[:,1]);models[name]=loaded
  (output/(name+'_fit.json')).write_text(json.dumps(dict(training_seconds=time.perf_counter()-start,known_candidates=int(mask.sum()),positive=int((y[mask]==1).sum()),reload_parity=True,outer_excluded=sorted(outer)),indent=2))
 return models
