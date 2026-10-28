"""Treatment-held-out replicate aggregation curve for RxRx2 embeddings."""
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

SEED=42; rng=np.random.default_rng(SEED)
OUT=Path(os.environ['RXRX2_OUTPUT_ROOT']); META=Path(os.environ['RXRX2_METADATA']); EMB=Path(os.environ['RXRX2_EMBEDDINGS'])
G={'type_I_II_interferons':['IFN-alpha-A','IFNB-1','IFNE','IFN-omega','IFN-gamma'],'il1_tnf_superfamily':['IL-1-alpha','IL-1-beta','TNF-alpha','TNF-beta','IL-18','IL-33','IL-36-gamma','TL-1A'],'chemokines_cxc_cc':['ENA-78','GCP-2','GRO-alpha','GRO-B','GRO-gamma','IL-8','IP-10','I-TAC','CCL-1','CCL14a','CCL21','CCL28','CCL-4','MCP-1','MCP-2','MCP-3'],'angiogenic_vascular_regulatory':['Angiopoietin-1','Angiopoietin-2','Angiopoietin-4','VEGF-165','VEGFC','VEGF-D','EG-VEGF','PLGF-4']}
label={t:c for c,ts in G.items() for t in ts}; classes=list(G); ident={c:i for i,c in enumerate(classes)}
meta=pd.read_csv(META); emb=pd.read_csv(EMB); site=meta[meta.treatment.isin(label)].merge(emb,on='site_id',validate='one_to_one'); site['y']=site.treatment.map(label).map(ident); f=[c for c in site if c.startswith('feature_')]
# Four fields/site become one well embedding. A condition has repeated wells.
keys=['well_id','treatment','treatment_conc','experiment','plate','y']; well=site.groupby(keys,as_index=False)[f].mean()
cond_keys=['treatment','treatment_conc','experiment','y']
groups=list(well.groupby(cond_keys,sort=False)); cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)

def make_conditions(k, seed):
 r=np.random.default_rng(seed); rows=[]
 for key,frame in groups:
  n=min(k,len(frame)); picked=frame.iloc[r.choice(len(frame),size=n,replace=False)]
  x=picked[f].to_numpy(float).mean(0); rows.append([*key,n,*x])
 return pd.DataFrame(rows,columns=[*cond_keys,'n_wells_used',*f])

all_metrics=[]
for k in [1,2,3,6]:
 for rep in range(10):
  d=make_conditions(k,SEED+100*k+rep); X=d[f].to_numpy(np.float32); y=d.y.to_numpy(); tg=d.treatment.to_numpy(); pred=np.empty(len(d),int)
  for tr,te in cv.split(X,y,tg):
   m=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED))]); m.fit(X[tr],y[tr]); pred[te]=m.predict(X[te])
  # Rank concentrations within each treatment, not by incomparable absolute doses.
  rank=d.groupby('treatment').treatment_conc.rank(method='dense',pct=True); bin=pd.cut(rank,[0,.34,.67,1.01],labels=['low','mid','high'])
  all_metrics.append({'replicates_used':k,'repeat':rep,'stratum':'all','macro_f1':f1_score(y,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(y,pred)})
  for s in ['low','mid','high']:
   z=bin==s; all_metrics.append({'replicates_used':k,'repeat':rep,'stratum':s,'macro_f1':f1_score(y[z],pred[z],average='macro'),'balanced_accuracy':balanced_accuracy_score(y[z],pred[z])})
metrics=pd.DataFrame(all_metrics); summary=metrics.groupby(['replicates_used','stratum']).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),ba_mean=('balanced_accuracy','mean'),ba_sd=('balanced_accuracy','std')).reset_index(); metrics.to_csv(OUT/'replicate_aggregation_repeats.csv',index=False); summary.to_csv(OUT/'replicate_aggregation_summary.csv',index=False)
fig,ax=plt.subplots(figsize=(7,4))
for s,g in summary.groupby('stratum'):
 ax.errorbar(g.replicates_used,g.ba_mean,yerr=g.ba_sd,marker='o',capsize=3,label=s)
ax.set(xlabel='Well replicates averaged per condition',ylabel='Balanced accuracy',title='Replicate aggregation under treatment-held-out evaluation',ylim=(0,1)); ax.legend(title='Concentration rank'); fig.tight_layout(); fig.savefig(OUT/'fig_replicate_aggregation.png',dpi=300)
print(summary.to_string(index=False))
