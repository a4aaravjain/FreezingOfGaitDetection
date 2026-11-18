"""Group identical kernel lengths/dilations without changing their arithmetic order."""
import numpy as np
from waveform_bank import TemporalBank

class FastTemporalBank(TemporalBank):
    @classmethod
    def from_bank(cls,bank):
        obj=cls(count=bank.count,seed=bank.seed);obj.scale=bank.scale.copy();obj.kernels=bank.kernels
        groups={}
        for i,(length,dilation,w,c,bias) in enumerate(obj.kernels):groups.setdefault((length,dilation),[]).append(i)
        obj.groups=[]
        for (length,dilation),ids in groups.items():
            obj.groups.append((length,dilation,np.array(ids),np.stack([obj.kernels[i][2] for i in ids]),np.stack([obj.kernels[i][3] for i in ids]),np.array([obj.kernels[i][4] for i in ids],np.float32)))
        return obj
    def transform(self,x,batch=128):
        x=np.asarray(x,np.float32)
        if x.ndim!=3 or x.shape[1:]!=(180,len(self.scale)) or not np.isfinite(x).all():raise ValueError('Invalid window shape/values')
        result=np.empty((len(x),2*self.count),np.float32)
        for start in range(0,len(x),batch):
            z=x[start:start+batch]/self.scale;z-=z.mean(1,keepdims=True)
            for length,dilation,ids,w,c,bias in self.groups:
                projected=np.einsum('btc,kc->bkt',z,c,optimize=False)
                n=180-(length-1)*dilation;v=np.zeros((len(z),len(ids),n),np.float32)
                for k in range(length):v+=w[None,:,k,None]*projected[:,:,k*dilation:k*dilation+n]
                v+=bias[None,:,None]
                result[start:start+len(z),ids*2]=(v>0).mean(2)
                result[start:start+len(z),ids*2+1]=v.max(2)
        if not np.isfinite(result).all():raise ValueError('Nonfinite features')
        return result
