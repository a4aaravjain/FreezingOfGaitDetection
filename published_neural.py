"""Causal architectural adaptations. See run licenses/source audit for attribution.

Stanford CNN: O'Day et al.2022, BSD3 Stanford NMBL.
Residual topology: janbrederecke/fog MIT; compact width/depth for CPU.
FoG-STAR topology is independently implemented from the published description;
no unlicensed source or pretrained weights are imported.
"""
import os
os.environ.setdefault('TF_NUM_INTRAOP_THREADS','2')
os.environ.setdefault('TF_NUM_INTEROP_THREADS','2')
from pathlib import Path
import json,gc
import numpy as np
import tensorflow as tf
from fog_pipeline import save_json,sha256

NEURAL=['fogstar','stanford','resnet_core','resnet_balanced','resnet_mined','resnet_augmented','resnet_pretrained']
SUPERVISED_EPOCHS=12
PRETRAIN_EPOCHS=4


def encoder(key):
    L=tf.keras.layers;x=L.Input((180,6));z=x
    if key=='fogstar':
        z=L.Cropping1D((60,0))(z)
        for filters,kernel,pool in [(16,9,0),(16,7,4),(24,5,3),(24,5,0)]:
            z=L.Conv1D(filters,kernel,padding='causal',activation='relu',kernel_regularizer=tf.keras.regularizers.l2(.001))(z)
            if pool:z=L.MaxPooling1D(pool)(z);z=L.Dropout(.4)(z)
        z=L.GlobalAveragePooling1D()(z);z=L.Dense(48,activation='relu')(z);z=L.Dropout(.4)(z)
    elif key=='stanford':
        z=L.Cropping1D((60,0))(z)
        for _ in range(2):
            z=L.Conv1D(16,17,padding='causal',activation='relu')(z);z=L.Dropout(.5)(z)
        z=L.MaxPooling1D(2)(z);z=L.Flatten()(z);z=L.Dense(10,activation='relu')(z)
    else:
        z=L.Conv1D(16,17,padding='causal')(z)
        for channels,dilation in [(16,1),(24,2),(32,4)]:
            skip=L.Conv1D(channels,1)(z) if z.shape[-1]!=channels else z
            h=L.LayerNormalization(axis=-1)(z);h=L.LeakyReLU(negative_slope=.1)(h)
            h=L.Conv1D(channels,17,padding='causal',dilation_rate=dilation)(h)
            h=L.Dropout(.2)(h);h=L.LayerNormalization(axis=-1)(h);h=L.LeakyReLU(negative_slope=.1)(h)
            h=L.Conv1D(channels,17,padding='causal',dilation_rate=dilation)(h)
            z=L.Add()([skip,h])
        z=L.GlobalAveragePooling1D()(z)
    return tf.keras.Model(x,z,name='causal_'+key+'_encoder')


def model_for(key):
    enc=encoder(key)
    model=tf.keras.Model(enc.input,tf.keras.layers.Dense(1,activation='sigmoid')(enc.output))
    return model,enc


def predict(model,x,batch=256):
    return np.concatenate([np.asarray(model(a,training=False)).ravel() for a in [x[i:i+batch] for i in range(0,len(x),batch)]])


def augment(raw,rng):
    x=raw.copy();axis=rng.normal(size=(len(x),3));axis/=np.maximum(np.linalg.norm(axis,axis=1,keepdims=True),1e-9)
    theta=rng.uniform(-np.pi/18,np.pi/18,size=(len(x),1,1));u=axis[:,None,:]
    for start in [0,3]:
        a=x[:,:,start:start+3]
        x[:,:,start:start+3]=a*np.cos(theta)+np.cross(u,a)*np.sin(theta)+u*(a*u).sum(2,keepdims=True)*(1-np.cos(theta))
    scale=rng.uniform(.95,1.05,size=(len(x),1,1));x*=scale
    x[:,:,:3]+=rng.normal(0,.01,x[:,:,:3].shape)
    x[:,:,3:]+=rng.normal(0,.5,x[:,:,3:].shape)
    return x


def balanced_indices(meta,labels,mask,rng,mined=None):
    allowed=np.flatnonzero(mask);positive=allowed[labels[allowed]==1];negative=allowed[labels[allowed]==0]
    subjects=meta.subject.to_numpy();episodes=meta.episode.to_numpy()
    pospeople=np.unique(subjects[positive]);negpeople=np.unique(subjects[negative])
    if not len(pospeople) or not len(negpeople):raise ValueError('Balanced fit lacks classes')
    posgroups={s:{e:positive[(subjects[positive]==s)&(episodes[positive]==e)] for e in np.unique(episodes[positive][subjects[positive]==s])} for s in pospeople}
    neggroups={s:negative[subjects[negative]==s] for s in negpeople}
    result=[]
    for _ in range(int(np.ceil(len(allowed)/128))):
        batch=[]
        for _ in range(32):
            s=rng.choice(pospeople);keys=list(posgroups[s]);batch.append(int(rng.choice(posgroups[s][keys[int(rng.integers(len(keys)))]])))
        for j in range(96):
            if mined is not None and len(mined) and j<48:batch.append(int(rng.choice(mined)))
            else:batch.append(int(rng.choice(neggroups[rng.choice(negpeople)])))
        rng.shuffle(batch);result.extend(batch)
    return np.asarray(result,int)


def train_one(key,part,normalization,folder):
    folder=Path(folder);folder.mkdir(exist_ok=True);checkpoint=folder/(key+'.keras');done=folder/(key+'_DONE.json')
    if done.exists():
        info=json.loads(done.read_text());assert sha256(checkpoint)==info['model_sha256'];return tf.keras.models.load_model(checkpoint,compile=False)
    tf.keras.backend.clear_session();tf.keras.utils.set_random_seed(42);tf.config.experimental.enable_op_determinism()
    model,enc=model_for(key);rng=np.random.default_rng(42)
    raw=part['data'].x;meta=part['meta'];labels=meta.target.to_numpy().copy();labels[meta.active_fog]=0
    basic=meta.target.ge(0).to_numpy();known=meta.known_followup.to_numpy();enhanced=key not in ['fogstar','stanford','resnet_core']
    if set(labels[basic])!={0,1}:raise ValueError('Neural fit lacks classes')
    mean,std=normalization['mean'],normalization['std'];normalized=((raw-mean)/std).astype(np.float32)
    if key=='resnet_pretrained':
        decoder=tf.keras.layers.Reshape((180,6))(tf.keras.layers.Dense(1080)(enc.output))
        auto=tf.keras.Model(enc.input,decoder);auto.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss='mse')
        history=[]
        for epoch in range(PRETRAIN_EPOCHS):
            people=meta.subject.to_numpy();unique=np.unique(people);groups={s:np.flatnonzero(people==s) for s in unique}
            ids=np.array([rng.choice(groups[rng.choice(unique)]) for _ in range(len(raw))])
            x=normalized[ids].copy();mask=rng.random((len(x),180,1))<.75;x=np.where(mask,0,x)
            h=auto.fit(x,normalized[ids],epochs=1,batch_size=128,shuffle=False,verbose=0)
            loss=float(h.history['loss'][0]);assert np.isfinite(loss);history.append(dict(epoch=epoch+1,loss=loss))
        import pandas as pd
        pd.DataFrame(history).to_csv(folder/(key+'_masked_pretraining.csv'),index=False)
        enc.save_weights(folder/(key+'_pretrained.weights.h5'));del auto
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2) if enhanced else 'binary_crossentropy'
    model.compile(optimizer=tf.keras.optimizers.Adam(.0005,clipnorm=1),loss=loss)
    history=[];mined=None
    epochs=SUPERVISED_EPOCHS+(2 if key in ['resnet_mined','resnet_augmented','resnet_pretrained'] else 0)
    for epoch in range(epochs):
        mask=basic.copy()
        if epoch>=SUPERVISED_EPOCHS:
            mask=known&((meta.target.to_numpy()>=0)|meta.active_fog.to_numpy())
            if mined is None:
                p=predict(model,normalized);negative=mask&(labels==0);mined=[];selected=np.zeros(len(meta),bool)
                for _,g in meta.assign(negative=negative).groupby(['subject','hard_category'],sort=True):
                    ids=g.index[g.negative].to_numpy();n=max(1,int(np.ceil(.2*len(ids))))
                    chosen=ids[np.argsort(p[ids],kind='stable')[-n:]] if len(ids) else ids
                    selected[chosen]=True;mined.extend(chosen)
                mined=np.asarray(mined,int)
                import pandas as pd
                pd.DataFrame(dict(subject=meta.subject,time=meta.time,category=meta.hard_category,known_negative=negative,score=p,mined=selected)).to_csv(folder/(key+'_hard_negative_ledger.csv'),index=False)
        if enhanced:ids=balanced_indices(meta,labels,mask,rng,mined if epoch>=SUPERVISED_EPOCHS else None);weight=np.ones(len(ids),np.float32)
        else:
            ids=np.flatnonzero(mask);rng.shuffle(ids);y=labels[ids];weight=np.where(y==1,min(sum(y==0)/max(1,sum(y==1)),10),1.).astype(np.float32);weight/=weight.mean()
        x=augment(raw[ids],rng) if key in ['resnet_augmented','resnet_pretrained'] else raw[ids]
        x=((x-mean)/std).astype(np.float32)
        h=model.fit(x,labels[ids],sample_weight=weight,epochs=1,batch_size=128,shuffle=False,verbose=0)
        value=float(h.history['loss'][0])
        if not np.isfinite(value):raise ValueError('Nonfinite training loss')
        history.append(dict(epoch=epoch+1,loss=value,rows=len(ids),positives=int(labels[ids].sum()),hard_negative_stage=epoch>=SUPERVISED_EPOCHS))
        import pandas as pd
        pd.DataFrame(history).to_csv(folder/(key+'_training.csv'),index=False)
    model.save(checkpoint);restored=tf.keras.models.load_model(checkpoint,compile=False)
    np.testing.assert_allclose(predict(model,normalized[:16]),predict(restored,normalized[:16]),rtol=1e-5,atol=1e-6)
    save_json(done,dict(model_sha256=sha256(checkpoint),epochs=epochs,pretrain_epochs=PRETRAIN_EPOCHS if key=='resnet_pretrained' else 0,training_subjects=sorted(set(meta.subject)),reload_parity=True))
    del model,enc,normalized;gc.collect();return restored


def verifier_tcn(dim):
    """Original compact Keras adaptation of dilated refinement; entirely causal."""
    L=tf.keras.layers;x=L.Input((13,dim));z=L.Conv1D(16,1)(x)
    for stage in range(2):
        if stage:z=L.Conv1D(16,1)(z)
        for dilation in [1,2,4]:
            h=L.Conv1D(16,3,padding='causal',dilation_rate=dilation,activation='relu')(z)
            h=L.Conv1D(16,1)(h);h=L.Dropout(.2)(h);z=L.Add()([z,h])
        z=L.Conv1D(1,1,activation='sigmoid')(z)
    sequence=tf.keras.Model(x,z)
    last=L.Cropping1D((12,0))(z);last=L.Flatten()(last)
    return tf.keras.Model(x,last),sequence
