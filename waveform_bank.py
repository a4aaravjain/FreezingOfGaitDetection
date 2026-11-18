"""Small ROCKET-inspired fixed temporal bank; not the official ROCKET/MiniRocket implementation."""
from pathlib import Path
from dataclasses import replace, asdict
import argparse, json, shutil
import numpy as np
import pandas as pd
import joblib
from scipy.special import expit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import RidgeClassifier, LogisticRegression
from threadpoolctl import threadpool_limits
from fog_pipeline import Config, build_windows, select_threshold, evaluate, save_json, sha256, verify_split
from fogstar_adapter import load_fogstar
from gravity_gait import gravity_signals
from long_history_experiments import summaries
ROOT=Path(__file__).resolve().parent

class TemporalBank:
    def __init__(self, count=384, seed=42):
        self.count=count;self.seed=seed
    def fit(self, x):
        x=np.asarray(x,np.float32)
        if x.ndim!=3 or x.shape[1]!=180 or not np.isfinite(x).all():raise ValueError('Expected finite trailing 180-sample windows')
        self.scale=np.maximum(x.std(axis=(0,1)),1e-3)
        rng=np.random.default_rng(self.seed);self.kernels=[]
        for i in range(self.count):
            length=int(rng.choice([7,9,11]));dilation=int(rng.choice([1,2,4,8,16]))
            weights=rng.normal(size=length).astype(np.float32);weights-=weights.mean();weights/=np.linalg.norm(weights)
            channels=rng.normal(size=x.shape[2]).astype(np.float32);channels/=np.linalg.norm(channels)
            self.kernels.append((length,dilation,weights,channels,float(rng.uniform(-1,1))))
        return self
    def transform(self,x,batch=128):
        x=np.asarray(x,np.float32)
        if x.ndim!=3 or x.shape[1:]!=(180,len(self.scale)) or not np.isfinite(x).all():raise ValueError('Invalid window shape/values')
        result=np.empty((len(x),2*self.count),np.float32)
        for start in range(0,len(x),batch):
            z=x[start:start+batch]/self.scale
            z=z-z.mean(1,keepdims=True)
            for j,(length,dilation,w,c,bias) in enumerate(self.kernels):
                projected=np.einsum('btc,c->bt',z,c,optimize=False)
                n=180-(length-1)*dilation;v=np.zeros((len(z),n),np.float32)
                for k in range(length):v+=w[k]*projected[:,k*dilation:k*dilation+n]
                v+=bias
                result[start:start+len(z),2*j]=(v>0).mean(1)
                result[start:start+len(z),2*j+1]=v.max(1)
        if not np.isfinite(result).all():raise ValueError('Nonfinite features')
        return result

def native_signals(raw):
    raw=np.asarray(raw,np.float32)
    return np.concatenate([raw[:,:,:3],np.sign(raw[:,:,3:])*np.log1p(abs(raw[:,:,3:]))],axis=2)

def shape_signals(raw):
    z=gravity_signals(raw)
    return (z-z.mean(1,keepdims=True))/np.maximum(z.std(1,keepdims=True),.05)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--development',action='store_true');parser.add_argument('--output',default='waveform_bank_separate_round_1');parser.add_argument('--kernels',type=int,default=384);parser.add_argument('--representations',nargs='+',choices=['native','gravity','shape'],default=['native','gravity']);args=parser.parse_args()
    out=ROOT/'runs'/args.output;out.mkdir(exist_ok=False)
    cfg=replace(Config(),sample_rate=60,horizon_seconds=5,ema_alpha=.25,alert_pulse_seconds=1)
    train=list(range(1,15)) if args.development else list(range(1,11));cal=list(range(15,19)) if args.development else list(range(11,15));assessment=list(range(15,19))
    verify_split(train,cal,[] if args.development else assessment)
    manifest=dict(config=asdict(cfg),train=train,calibration=cal,assessment=assessment,development_reuse=args.development,reserved=list(range(19,23)),ankle='ankleR',kernel_count=args.kernels,representations=args.representations,seed=42,reference='https://arxiv.org/abs/1910.13051',method='Custom small multivariate random temporal bank, NOT official ROCKET implementation',limitations='Assessment people historically examined; not pristine final test. In development mode assessment duplicates calibration.',source_sha256={n:sha256(ROOT/n) for n in ['waveform_bank.py','gravity_gait.py','fog_pipeline.py','fogstar_adapter.py','long_history_experiments.py','pretraining_experiments.py']},dataset_sha256=sha256(ROOT/'sensor_data.csv'))
    save_json(out/'manifest.json',manifest);(out/'source').mkdir()
    for n in manifest['source_sha256']:shutil.copy2(ROOT/n,out/'source'/n)
    td=build_windows(load_fogstar(ROOT/'sensor_data.csv',train,ankle='ankleR',include_gyro=True),cfg)
    cd=build_windows(load_fogstar(ROOT/'sensor_data.csv',cal,ankle='ankleR',include_gyro=True),cfg)
    ad=cd if args.development else build_windows(load_fogstar(ROOT/'sensor_data.csv',assessment,ankle='ankleR',include_gyro=True),cfg)
    datasets=[td,cd,ad];summary=[summaries(d.x) for d in datasets];results=[]
    mask=td.y>=0;y=td.y[mask];weight=np.where(y==1,min((y==0).sum()/max(1,(y==1).sum()),10),1.);weight/=weight.mean()
    with threadpool_limits(limits=2):
      for representation in args.representations:
        fn={'native':native_signals,'gravity':gravity_signals,'shape':shape_signals}[representation]
        signals=[fn(d.x) for d in datasets];bank=TemporalBank(count=args.kernels).fit(signals[0]);joblib.dump(bank,out/(representation+'_bank.joblib'))
        features=[np.concatenate([bank.transform(z),s],axis=1) for z,s in zip(signals,summary)]
        print(representation,'features',features[0].shape,flush=True)
        for name,base in [('ridge_1',RidgeClassifier(alpha=1)),('ridge_1000',RidgeClassifier(alpha=1000)),('logistic',LogisticRegression(C=.1,max_iter=2000,random_state=42))]:
          key=representation+'_'+name;model=make_pipeline(StandardScaler(),base)
          model.fit(features[0][mask],y,**{list(model.named_steps)[-1]+'__sample_weight':weight})
          def scores(m,z):return expit(3*m.decision_function(z)) if name.startswith('ridge') else m.predict_proba(z)[:,1]
          p=scores(model,features[1]);joblib.dump(model,out/(key+'.joblib'));restored=joblib.load(out/(key+'.joblib'));np.testing.assert_allclose(p,scores(restored,features[1]))
          threshold,grid=select_threshold(cd,p,cfg);cm,cr,_=evaluate(cd,p,threshold,cfg)
          save_json(out/(key+'_frozen_policy.json'),dict(threshold=threshold,config=asdict(cfg),model_sha256=sha256(out/(key+'.joblib')),bank_sha256=sha256(out/(representation+'_bank.joblib')),score_note='Ridge sigmoid margin is not calibrated probability'))
          ap=p if args.development else scores(restored,features[2]);am,ar,events=evaluate(ad,ap,threshold,cfg)
          pd.DataFrame(grid).to_csv(out/(key+'_calibration_candidates.csv'),index=False);cr.to_csv(out/(key+'_calibration_predictions.csv'),index=False);ar.to_csv(out/(key+'_assessment_predictions.csv'),index=False)
          result=dict(model=key,threshold=threshold,calibration=cm,assessment=am,events=events,model_reload_verified=True);results.append(result);save_json(out/'results.json',results)
          print('RESULT',key,'cal',cm['device']['true_warning_events'],'assessment',am['device']['true_warning_events'],'false',am['device']['total_false_alarms'],flush=True)
if __name__=='__main__':main()
