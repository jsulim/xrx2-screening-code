"""Plate-effect diagnostic and unsupervised plate-centering ablation."""
from pathlib import Path
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score

OUT=Path(os.environ['RXRX2_OUTPUT_ROOT']); META=Path(os.environ['RXRX2_METADATA']); EMB=Path(os.environ['RXRX2_EMBEDDINGS'])
G={'type_I_II_interferons':['IFN-alpha-A','IFNB-1','IFNE','IFN-omega','IFN-gamma'],'il1_tnf_superfamily':['IL-1-alpha','IL-1-beta','TNF-alpha','TNF-beta','IL-18','IL-33','IL-36-gamma','TL-1A'],'chemokines_cxc_cc':['ENA-78','GCP-2','GRO-alpha','GRO-B','GRO-gamma','IL-8','IP-10','I-TAC','CCL-1','CCL14a','CCL21','CCL28','CCL-4','MCP-1','MCP-2','MCP-3'],'angiogenic_vascular_regulatory':['Angiopoietin-1','Angiopoietin-2','Angiopoietin-4','VEGF-165','VEGFC','VEGF-D','EG-VEGF','PLGF-4']}
lab={t:c for c,ts in G.items() for t in ts}; ids={c:i for i,c in enumerate(G)}
m=pd.read_csv(META); e=pd.read_csv(EMB); d=m[m.treatment.isin(lab)].merge(e,on='site_id',validate='one_to_one'); d['y']=d.treatment.map(lab).map(ids); f=[c for c in d if c.startswith('feature_')]
# Plate centering uses only the unlabeled distribution of each incoming plate.
X=d[f].to_numpy(float); plate=d.groupby(['experiment','plate'])[f].transform('mean').to_numpy(float); Xc=X-plate
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42); y=d.y.to_numpy(); groups=d.treatment.to_numpy()
rows=[]
for name,z in [('raw_embedding',X),('plate_centered',Xc)]:
 p=np.empty(len(d),int)
 for tr,te in cv.split(z,y,groups):
  clf=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); clf.fit(z[tr],y[tr]); p[te]=clf.predict(z[te])
 rows.append({'representation':name,'macro_f1':f1_score(y,p,average='macro'),'balanced_accuracy':balanced_accuracy_score(y,p)})
pd.DataFrame(rows).to_csv(OUT/'plate_centering_ablation.csv',index=False)
# Condition-level distance: compare each well to same condition peers, split by same/different plate.
keys=['well_id','treatment','treatment_conc','experiment','plate']; w=d.groupby(keys,as_index=False)[f].mean(); rec=[]
for _,q in w.groupby(['treatment','treatment_conc','experiment']):
 a=q[f].to_numpy(float); a=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12); s=a@a.T
 for i in range(len(q)):
  for j in range(i+1,len(q)):
   rec.append({'pair_type':'same_plate' if q.iloc[i].plate==q.iloc[j].plate else 'different_plate','cosine_similarity':s[i,j]})
pairs=pd.DataFrame(rec); pair_summary=pairs.groupby('pair_type').agg(mean_cosine=('cosine_similarity','mean'),sd_cosine=('cosine_similarity','std'),pairs=('cosine_similarity','size')).reset_index(); pair_summary.to_csv(OUT/'within_condition_plate_similarity.csv',index=False)
fig,ax=plt.subplots(1,2,figsize=(9,4)); pd.DataFrame(rows).set_index('representation')[['macro_f1','balanced_accuracy']].plot.bar(ax=ax[0],ylim=(0,1),rot=0,title='Plate-centering ablation'); ax[1].bar(pair_summary.pair_type,pair_summary.mean_cosine,yerr=pair_summary.sd_cosine,capsize=4); ax[1].set(title='Same-condition embedding similarity',ylabel='Cosine similarity'); fig.tight_layout(); fig.savefig(OUT/'fig_plate_effect.png',dpi=300)
print(pd.DataFrame(rows).to_string(index=False)); print(pair_summary.to_string(index=False))
