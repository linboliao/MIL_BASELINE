"""Frozen source95 rule and patient-group bootstrap primitives."""
from pathlib import Path
import hashlib,math,csv
import numpy as np

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def writecsv(p,rows):
 with p.open('x',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def scores(y,p,t):
 z=p>=t
 tp=int(np.sum(z&(y==1)));fn=int(np.sum((~z)&(y==1)))
 tn=int(np.sum((~z)&(y==0)));fp=int(np.sum(z&(y==0)))
 return dict(bacc=.5*(tp/(tp+fn)+tn/(tn+fp)),
  sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),accuracy=(tp+tn)/len(y),
  tn=tn,fp=fp,fn=fn,tp=tp)
def patient_weights(df,rng,n):
 names=np.array(sorted(df.patient_group_id.astype(str).unique()))
 gids=np.searchsorted(names,df.patient_group_id.astype(str).to_numpy())
 grp=df.groupby('patient_group_id').label.max().reindex(names).to_numpy().astype(bool)
 P=np.flatnonzero(grp);NEG=np.flatnonzero(~grp)
 assert len(P)>0 and len(NEG)>0
 W=np.zeros((n,len(names)),np.int16)
 W[:,P]=rng.multinomial(len(P),np.ones(len(P))/len(P),size=n)
 W[:,NEG]=rng.multinomial(len(NEG),np.ones(len(NEG))/len(NEG),size=n)
 assert np.all(W.sum(1)==len(names))
 return names,gids,W,grp
def evaluate_draw(y,p,thrs,W,gids):
 w=W[:,gids]
 decision=p[None,:]>=thrs[:,None]
 pos=(y==1)[None,:];neg=~pos
 totalpos=(w*pos).sum(1);totalneg=(w*neg).sum(1)
 tp=(w*decision*pos).sum(1);tn=(w*(~decision)*neg).sum(1)
 with np.errstate(invalid='raise',divide='raise'):
  sens=tp/totalpos;spec=tn/totalneg
  return dict(bacc=.5*(sens+spec),sensitivity=sens,specificity=spec,
              accuracy=(tp+tn)/(totalpos+totalneg))
def get_thr(labels,probs):
 ps=np.sort(probs[labels==1])[::-1];assert len(ps)
 return float(ps[max(1,int(math.ceil(.95*len(ps))))-1])
def weighted_thr(cal,score,W,gids):
 y=cal.label.to_numpy(np.int64)
 pos_ix=np.flatnonzero(y==1)
 sorted_ix=pos_ix[np.argsort(-score[pos_ix],kind='stable')]
 weights=W[:,gids[sorted_ix]]
 k=np.maximum(1,np.ceil(.95*weights.sum(axis=1)).astype(int))
 idx=np.argmax(np.cumsum(weights,axis=1)>=k[:,None],axis=1)
 return score[sorted_ix][idx].astype(float)
