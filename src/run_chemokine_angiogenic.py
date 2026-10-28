"""Binary diagnostic: chemokine vs angiogenic phenotype overlap."""
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
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

OUT=Path(os.environ['RXRX2_OUTPUT_ROOT']); META=Path(os.environ['RXRX2_METADATA']); EMB=Path(os.environ['RXRX2_EMBEDDINGS'])
chem=['ENA-78','GCP-2','GRO-alpha','GRO-B','GRO-gamma','IL-8','IP-10','I-TAC','CCL-1','CCL14a','CCL21','CCL28','CCL-4','MCP-1','MCP-2','MCP-3']
ang=['Angiopoietin-1','Angiopoietin-2','Angiopoietin-4','VEGF-165','VEGFC','VEGF-D','EG-VEGF','PLGF-4']
meta=pd.read_csv(META); emb=pd.read_csv(EMB); d=meta[meta.treatment.isin(chem+ang)].merge(emb,on='site_id',validate='one_to_one'); d['y']=d.treatment.isin(ang).astype(int); f=[c for c in d if c.startswith('feature_')]
def score(frame,unit):
 X=frame[f].to_numpy(np.float32); y=frame.y.to_numpy(); groups=frame.treatment.to_numpy(); cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42); p=np.zeros(len(frame))
 for tr,te in cv.split(X,y,groups):
  m=Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=42))]); m.fit(X[tr],y[tr]); p[te]=m.predict_proba(X[te])[:,1]
 pred=(p>=.5).astype(int); return {'unit':unit,'n_samples':len(frame),'macro_f1':f1_score(y,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(y,pred),'roc_auc':roc_auc_score(y,p)}
site_result=score(d,'site')
keys=['well_id','treatment','treatment_conc','experiment','plate','y']; well=d.groupby(keys,as_index=False)[f].mean(); cond=well.groupby(['treatment','treatment_conc','experiment','y'],as_index=False)[f].mean(); condition_result=score(cond,'condition_mean_all_wells')
out=pd.DataFrame([site_result,condition_result]); out.to_csv(OUT/'chemokine_vs_angiogenic_binary.csv',index=False)
fig,ax=plt.subplots(figsize=(6,4)); out.set_index('unit')[['macro_f1','balanced_accuracy','roc_auc']].plot.bar(ax=ax,ylim=(0,1),rot=0,title='Chemokine vs angiogenic treatment-held-out diagnostic'); fig.tight_layout(); fig.savefig(OUT/'fig_chemokine_angiogenic_binary.png',dpi=300)
print(out.to_string(index=False))
