"""Compact left-causal hazard models and prespecified event-learning objectives.

Independent Keras implementations of TCN/GRU/refinement mathematics. Attribution
and original-code restrictions are documented in the run's source audit.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2');os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
from pathlib import Path
import json,gc
import numpy as np,pandas as pd,tensorflow as tf
from event_hazard_data import HISTORY,normalizer,normalized,sequences
from event_hazard_audit import save,sha

EPOCHS=8
PRESETS=[dict(mil=2.,topk=2.,rank=.25,late=.5,active=.5,sparse=.05),dict(mil=2.,topk=4.,rank=.5,late=1.,active=1.,sparse=.1)]


def make_model(family,dim):
    L=tf.keras.layers;inp=L.Input((HISTORY,dim));z=L.Dense(24,activation='relu')(inp)
    if family=='gru':z=L.GRU(24,return_sequences=True)(z)
    elif family=='tcn':
        for dilation in [1,2,4,8,16,32]:
            h=L.LayerNormalization(axis=-1)(z);h=L.Conv1D(24,3,padding='causal',dilation_rate=dilation,activation='relu')(h)
            h=L.Dropout(.1)(h);h=L.Conv1D(24,1)(h);z=L.Add()([z,h])
    elif family=='refinement':
        for stage in range(2):
            z=L.Conv1D(16,1)(z)
            for dilation in [1,2,4,8,16]:
                h=L.Conv1D(16,3,padding='causal',dilation_rate=dilation,activation='relu')(z)
                h=L.Dropout(.1)(h);h=L.Conv1D(16,1)(h);z=L.Add()([z,h])
            if stage==0:z=L.Conv1D(4,1,activation='sigmoid')(z)
        z=L.Dense(24,activation='relu')(z)
    else:raise ValueError(family)
    temporal=tf.keras.Model(inp,z)
    last=L.Flatten()(L.Cropping1D((HISTORY-1,0))(z));embedding=L.Dense(24,name='embedding')(last)
    return tf.keras.Model(inp,dict(logits=L.Dense(4,name='logits')(embedding),context=L.Dense(6,name='context')(embedding),embedding=embedding)),temporal


def probabilities(logits,temperature=1.):
    h=1/(1+np.exp(-np.clip(np.asarray(logits)/temperature,-40,40)))
    survival=np.cumprod(1-h,axis=-1)
    return np.concatenate([h*np.concatenate([np.ones((len(h),1)),survival[:,:-1]],1),survival[:,-1:]],1)


def tf_pmf(logits,temperature=1.):
    h=tf.sigmoid(logits/temperature);s=tf.math.cumprod(1-h,axis=-1)
    return tf.concat([h*tf.concat([tf.ones_like(s[:,:1]),s[:,:-1]],1),s[:,-1:]],1)


def predict(model,data,x,batch=128,ready=None):
    values={k:np.zeros((len(data['y']),d),np.float32) for k,d in [('logits',4),('context',6),('embedding',24)]}
    permitted=np.flatnonzero(ready) if ready is not None else np.arange(len(data['y']))
    fn=tf.function(lambda v:model(v,training=False),input_signature=[tf.TensorSpec((None,HISTORY,x.shape[1]),tf.float32)])
    for begin in range(0,len(permitted),batch):
        rows=permitted[begin:begin+batch];out=fn(sequences(data,x,rows))
        for k in values:values[k][rows]=np.asarray(out[k])
    return values


def event_bags(data,ready):
    m=data['part']['meta'];bags=[]
    for record in data['part']['records']:
        for onset in record.onsets:
            rows=m.index[(m.segment==record.key)&(m.time>=onset-5-1e-8)&(m.time<=onset-2+1e-8)&~m.active_fog&data['candidate']&ready&np.isin(data['y'],[1,2,3])&np.isclose(data['next_onset'],onset,atol=1e-8,rtol=0)].to_numpy()
            if len(rows):bags.append(dict(rows=rows,subject=int(m.loc[rows[0],'subject']),segment=record.key,onset=float(onset)))
    return bags


def fit(data,folder,family,preset=0,context=False,teacher=False,kd=None,personal=False,event=True,topk=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'FIT.json').exists():
        info=json.loads((folder/'FIT.json').read_text());assert all(sha(folder/n)==h for n,h in info['hashes'].items());return info
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
    norm=normalizer(data,teacher);x,ready,calibration=normalized(data,norm,teacher,personal)
    np.savez(folder/'normalization.npz',**norm);save(folder/'setup_calibration.json',calibration)
    m=data['part']['meta'];owner=m.subject.to_numpy();active=m.active_fog.to_numpy();known=m.known_followup.to_numpy();y=data['y']
    allowed=ready&((y>=0)|(active&known));available=np.flatnonzero(allowed)
    if not len(available):raise ValueError('No permitted fitting examples after fixed setup')
    subjects=np.unique(owner[available]);groups={s:available[owner[available]==s] for s in subjects}
    negative=allowed&known&~np.isin(y,[1,2,3]);neg=np.flatnonzero(negative)
    positive=np.isin(y,[1,2,3]);bags=event_bags(data,ready);rng=np.random.default_rng(42)
    bag_owners=sorted({b['subject'] for b in bags});by_owner={s:[b for b in bags if b['subject']==s] for s in bag_owners}
    # Subjects are already sampled uniformly; inverse-count weighting here
    # would apply a second, unintended participant correction.
    weights=np.ones(len(y),np.float32)
    classes=np.bincount(y[(y>=0)&allowed],minlength=5);classweight=np.minimum(classes.sum()/np.maximum(5*classes,1),12).astype(np.float32)
    ledger=m[['subject','segment','chunk','time','known_followup','active_fog','hard_category']].copy()
    ledger['target_bin']=y;ledger['candidate']=data['candidate'];ledger['ready']=ready;ledger['used']=allowed
    ledger.to_csv(folder/'fitting_ledger.csv',index=False)
    save(folder/'bag_ledger.json',[dict(subject=b['subject'],segment=b['segment'],onset=b['onset'],rows=b['rows'].tolist()) for b in bags])
    settings=dict(family=family,preset=preset,event=event,hard_topk=topk,context=context,teacher=teacher,distillation=kd is not None,personal=personal,
        epochs=EPOCHS,history_ticks=HISTORY,history_seconds=20,batch_size=64,seed=42,input_dim=x.shape[1],loss_weights=PRESETS[preset],context_weight=.2,distillation_weight=.5,representation_weight=.05,
        class_weight=classweight.tolist(),fit_subjects=[int(s) for s in subjects],context_label_limitations='Standing without FoG is a stopping proxy; intention is unannotated. Periodicity disruption is a trailing-signal proxy, not clinical ground truth.',
        final_alert_age='Used by final causal controller refractory/budget state, never fabricated from future event labels.',
        sparse_surrogate='Soft current candidate-start mass relative to cached past-tick model risk; refreshed using fitting data every two epochs. Complete actual alert starts determine inner policy selection.')
    save(folder/'configuration.json',settings);model,_=make_model(family,x.shape[1]);optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1)
    costs=PRESETS[preset];cached=np.zeros(len(y),np.float32);mined=neg.copy();history=[]
    if kd is None:kd_prob=np.ones((len(y),5),np.float32)/5;kd_embed=np.zeros((len(y),24),np.float32)
    else:kd_prob=probabilities(kd['logits'],2).astype(np.float32);kd_embed=kd['embedding'].astype(np.float32)
    previous=np.arange(len(y));same=np.r_[False,(m.chunk.to_numpy()[1:]==m.chunk.to_numpy()[:-1])];previous[same]-=1
    @tf.function(reduce_retracing=True)
    def update(a,target,weight,clsweight,ctx,negmask,late,actpost,cand,prev,bmask,rankmask,teacherp,teacherz):
        with tf.GradientTape() as tape:
            out=model(a,training=True);pmf=tf.clip_by_value(tf_pmf(out['logits']),1e-6,1);risk=tf.reduce_sum(pmf[:,1:4],axis=1)
            valid=tf.cast(target>=0,tf.float32);indices=tf.stack([tf.range(tf.shape(target)[0]),tf.maximum(target,0)],1)
            ce=tf.reduce_sum(-tf.math.log(tf.gather_nd(pmf,indices))*weight*clsweight*valid)/tf.maximum(tf.reduce_sum(weight*valid),1.)
            peaks=tf.reduce_max(tf.reshape(risk[:24],(4,6))*bmask,axis=1);bagvalid=tf.reduce_max(bmask,axis=1)
            mil=tf.reduce_sum(-tf.math.log(tf.maximum(peaks,1e-6))*bagvalid)/tf.maximum(tf.reduce_sum(bagvalid),1.)
            hard=tf.reduce_sum(-tf.math.log(tf.maximum(1-risk,1e-6))*negmask)/tf.maximum(tf.reduce_sum(negmask),1.)
            poslog=tf.math.log(tf.maximum(peaks,1e-6)/tf.maximum(1-peaks,1e-6));neglog=tf.math.log(risk[24:40]/tf.maximum(1-risk[24:40],1e-6))
            ranking=tf.reduce_sum(tf.nn.softplus(1+neglog[None,:]-poslog[:,None])*rankmask)/tf.maximum(tf.reduce_sum(rankmask),1.)
            latecost=tf.reduce_sum(risk*late)/tf.maximum(tf.reduce_sum(late),1.)
            activecost=tf.reduce_sum(risk*actpost)/tf.maximum(tf.reduce_sum(actpost),1.)
            sparse=tf.reduce_mean(risk*(1-prev)*cand)
            auxiliary=tf.reduce_sum(tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=ctx,logits=out['context']),axis=1)*weight)/tf.maximum(tf.reduce_sum(weight),1.)
            distill=tf.reduce_sum(tf.reduce_sum(teacherp*(tf.math.log(tf.maximum(teacherp,1e-6))-tf.math.log(tf.maximum(tf_pmf(out['logits'],2),1e-6))),axis=1)*weight)/tf.maximum(tf.reduce_sum(weight),1.)*4
            rep=tf.reduce_sum(tf.reduce_mean((tf.math.l2_normalize(out['embedding'],axis=1)-tf.math.l2_normalize(teacherz,axis=1))**2,axis=1)*weight)/tf.maximum(tf.reduce_sum(weight),1.)
            total=ce
            if event:total+=costs['mil']*mil+(costs['topk']*hard if topk else 0)+costs['rank']*ranking+costs['late']*latecost+costs['active']*activecost+costs['sparse']*sparse
            if context:total+=.2*auxiliary
            if kd is not None:total+=.5*distill+.05*rep
        gradients=tape.gradient(total,model.trainable_variables)
        optimizer.apply_gradients([(g,v) for g,v in zip(gradients,model.trainable_variables) if g is not None])
        return tf.stack([total,ce,mil,hard,ranking,latecost,activecost,sparse,auxiliary,distill])
    for epoch in range(1,EPOCHS+1):
        if epoch>1 and (epoch-1)%2==0:
            outputs=predict(model,data,x);cached=probabilities(outputs['logits'])[:,1:4].sum(1).astype(np.float32);mined=[];mine_rows=[]
            for (s,record),g in m.loc[negative&data['candidate']].groupby(['subject','segment'],sort=True):
                ids=g.index.to_numpy();hours=float(m.loc[(owner==s)&negative,'duration'].sum()/3600);count=max(1,int(np.ceil(20*hours)))
                ids=ids[np.argsort(cached[ids],kind='stable')[-count:]];mined.extend(ids)
                mine_rows.extend(dict(epoch=epoch,subject=int(s),segment=record,row=int(i),risk=float(cached[i])) for i in ids)
            mined=np.array(mined,int) if len(mined) else neg
            pd.DataFrame(mine_rows).to_csv(folder/f'hard_peaks_epoch_{epoch}.csv',index=False)
        values=[]
        for _ in range(int(np.ceil(len(available)/64))):
            selected=[];bagmask=np.zeros((4,6),np.float32);picked=[]
            for b in range(4):
                owner_bags=by_owner[int(rng.choice(bag_owners))] if bags else []
                bag=owner_bags[int(rng.integers(len(owner_bags)))] if owner_bags else None;picked.append(bag)
                ids=rng.choice(bag['rows'],size=min(6,len(bag['rows'])),replace=False) if bag else np.array([],int)
                bagmask[b,:len(ids)]=1;selected.extend(list(ids)+[int(available[0])]*(6-len(ids)))
            for j in range(16):
                bag=picked[j%4];pool=mined if event and j<8 else neg
                if bag is not None:
                    samepool=pool[(owner[pool]==bag['subject'])&(m.segment.to_numpy()[pool]==bag['segment'])]
                    if not len(samepool):samepool=pool[owner[pool]==bag['subject']]
                    if len(samepool):pool=samepool
                selected.append(int(rng.choice(pool if len(pool) else available)))
            for _ in range(24):selected.append(int(rng.choice(groups[rng.choice(subjects)])))
            ids=np.array(selected,int);rankmask=np.zeros((4,16),np.float32)
            for b,bag in enumerate(picked):
                if bag is not None:rankmask[b]=(owner[ids[24:40]]==bag['subject'])&negative[ids[24:40]]
            cw=classweight[np.maximum(y[ids],0)];actpost=known[ids]&(active[ids]|(data['context'][ids,4]>0))&~positive[ids]
            negmask=np.zeros(64,np.float32);negmask[24:40]=negative[ids[24:40]]
            sampleweight=weights[ids].copy();sampleweight[:24]*=bagmask.ravel();real=sampleweight>0
            result=update(sequences(data,x,ids),y[ids].astype(np.int32),sampleweight,cw,data['context'][ids],negmask,
                ((y[ids]==0)&real).astype(np.float32),(actpost&real).astype(np.float32),(data['candidate'][ids]&real).astype(np.float32),cached[previous[ids]],bagmask,rankmask,kd_prob[ids],kd_embed[ids])
            numbers=np.asarray(result);assert np.isfinite(numbers).all();values.append(numbers)
        mean=np.mean(values,0);history.append(dict(epoch=epoch,**dict(zip(['total','hazard_ce','episode_mil','hard_topk','ranking','late_penalty','active_post_penalty','start_sparsity','auxiliary','distillation'],map(float,mean)))))
        pd.DataFrame(history).to_csv(folder/'training.csv',index=False)
        if epoch in [EPOCHS//2,EPOCHS]:model.save(folder/f'epoch_{epoch}.keras')
        print('EVENT FIT',folder.name,'epoch',epoch,'of',EPOCHS,flush=True)
    test=sequences(data,x,available[:16]);before=model(test,training=False);loaded=tf.keras.models.load_model(folder/f'epoch_{EPOCHS}.keras',compile=False)
    for k in before:np.testing.assert_allclose(np.asarray(before[k]),np.asarray(loaded(test,training=False)[k]),rtol=1e-5,atol=1e-6)
    info=dict(configuration=settings,fit_subjects=[int(s) for s in subjects],known_fitting_rows=int(allowed.sum()),episode_bags=len(bags),reload_parity=True,
        hashes={p.name:sha(p) for p in folder.iterdir() if p.is_file() and p.name!='FIT.json'})
    save(folder/'FIT.json',info);del model,loaded;gc.collect();return info
