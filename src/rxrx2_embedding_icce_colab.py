"""CPU-only ICCE experiment pipeline for RxRx2 released embeddings.

Before running in Colab, upload metadata.csv and embeddings.csv to:
MyDrive/SUDAL/rxrx2/data/
"""
from pathlib import Path
import os, json, time, pickle, random
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, balanced_accuracy_score, brier_score_loss
from sklearn.utils import resample
from xgboost import XGBClassifier

SEED=42; random.seed(SEED); np.random.seed(SEED)
ROOT=Path(os.environ.get('RXRX2_DATA_ROOT','/content/drive/MyDrive/SUDAL/rxrx2/data'))
OUT=Path(os.environ.get('RXRX2_OUTPUT_ROOT','/content/drive/MyDrive/SUDAL/rxrx2/results_embedding_icce')); OUT.mkdir(parents=True,exist_ok=True)
META=Path(os.environ.get('RXRX2_METADATA',str(ROOT/'metadata.csv')))
EMB=Path(os.environ.get('RXRX2_EMBEDDINGS',str(ROOT/'embeddings.csv')))
assert META.exists() and EMB.exists(), f'Missing input CSV: metadata={META}, embeddings={EMB}'

GROUPS={
 'type_I_II_interferons':['IFN-alpha-A','IFNB-1','IFNE','IFN-omega','IFN-gamma'],
 'il1_tnf_superfamily':['IL-1-alpha','IL-1-beta','TNF-alpha','TNF-beta','IL-18','IL-33','IL-36-gamma','TL-1A'],
 'chemokines_cxc_cc':['ENA-78','GCP-2','GRO-alpha','GRO-B','GRO-gamma','IL-8','IP-10','I-TAC','CCL-1','CCL14a','CCL21','CCL28','CCL-4','MCP-1','MCP-2','MCP-3'],
 'angiogenic_vascular_regulatory':['Angiopoietin-1','Angiopoietin-2','Angiopoietin-4','VEGF-165','VEGFC','VEGF-D','EG-VEGF','PLGF-4']}
CLASSES=list(GROUPS); LABEL={t:c for c,ts in GROUPS.items() for t in ts}; ID={c:i for i,c in enumerate(CLASSES)}

meta=pd.read_csv(META); emb=pd.read_csv(EMB)
df=meta[meta.treatment.isin(LABEL)].copy(); df['label']=df.treatment.map(LABEL); df['y']=df.label.map(ID)
df=df.merge(emb,on='site_id',validate='one_to_one'); feats=[c for c in df if c.startswith('feature_')]
assert len(feats)==128, len(feats)
df.to_csv(OUT/'analysis_manifest.csv',index=False)
print('Sites:',len(df),'treatments:',df.treatment.nunique(),'features:',len(feats))

def models():
 return {
  'Logistic':Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',C=1.0,random_state=SEED))]),
  'XGBoost':XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.06,subsample=.85,colsample_bytree=.85,objective='multi:softprob',eval_metric='mlogloss',n_jobs=4,random_state=SEED),
  'SmallMLP':Pipeline([('scale',StandardScaler()),('clf',MLPClassifier(hidden_layer_sizes=(64,32),alpha=1e-3,max_iter=250,early_stopping=True,random_state=SEED))])}

def balance(x,y):
 d=pd.DataFrame(x); d['_y']=y; n=d._y.value_counts().max()
 return pd.concat([resample(d[d._y==c],replace=True,n_samples=n,random_state=SEED) for c in range(4)]).sample(frac=1,random_state=SEED)

def ece(y,p,bins=10):
 conf=p.max(1); pred=p.argmax(1); out=0.
 for lo,hi in zip(np.linspace(0,1,bins,endpoint=False),np.linspace(.1,1,bins)):
  m=(conf>=lo)&(conf<hi if hi<1 else conf<=hi)
  if m.any(): out+=m.mean()*abs((pred[m]==y[m]).mean()-conf[m].mean())
 return out

cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
X=df[feats].to_numpy(np.float32); y=df.y.to_numpy(); groups=df.treatment.to_numpy()
all_oof=[]
for name,model in models().items():
 print('Running',name)
 for fold,(tr,te) in enumerate(cv.split(X,y,groups),1):
  fit=model
  if name=='SmallMLP':
   b=balance(X[tr],y[tr]); fit.fit(b.drop(columns='_y').to_numpy(),b._y.to_numpy())
  else: fit.fit(X[tr],y[tr])
  p=fit.predict_proba(X[te]); z=df.iloc[te][['site_id','treatment','experiment','label','y']].copy(); z['model']=name; z['fold']=fold
  for k in range(4): z[f'p{k}']=p[:,k]
  all_oof.append(z)
oof=pd.concat(all_oof,ignore_index=True); oof.to_csv(OUT/'treatment_heldout_oof_predictions.csv',index=False)

rows=[]
for name,g in oof.groupby('model'):
 p=g[[f'p{k}' for k in range(4)]].to_numpy(); yy=g.y.to_numpy(); pred=p.argmax(1)
 rows.append({'model':name,'macro_f1':f1_score(yy,pred,average='macro'),'balanced_accuracy':balanced_accuracy_score(yy,pred),'ece':ece(yy,p),'brier':np.mean([brier_score_loss((yy==k).astype(int),p[:,k]) for k in range(4)])})
summary=pd.DataFrame(rows).sort_values('macro_f1',ascending=False); summary.to_csv(OUT/'model_summary.csv',index=False); print(summary)
best=summary.iloc[0].model

# Fixed confidence thresholds: selective/defer prediction (no test-set threshold tuning).
g=oof[oof.model==best]; p=g[[f'p{k}' for k in range(4)]].to_numpy(); yy=g.y.to_numpy(); pred=p.argmax(1); conf=p.max(1); sel=[]
for tau in np.arange(.25,.91,.05):
 m=conf>=tau; sel.append({'threshold':tau,'coverage':m.mean(),'accepted_macro_f1':f1_score(yy[m],pred[m],average='macro') if m.any() else np.nan,'accepted_error':1-(pred[m]==yy[m]).mean() if m.any() else np.nan})
selective=pd.DataFrame(sel); selective.to_csv(OUT/'selective_prediction_curve.csv',index=False)

# Dimension ablation with the stable lightweight Logistic baseline.
ab=[]
for d in [16,32,64,128]:
 ps=[]; ys=[]
 for tr,te in cv.split(X,y,groups):
  pipe=Pipeline([('scale',StandardScaler()),('pca',PCA(n_components=d,random_state=SEED)),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED))])
  pipe.fit(X[tr],y[tr]); ps.extend(pipe.predict(X[te])); ys.extend(y[te])
 ab.append({'dimensions':d,'macro_f1':f1_score(ys,ps,average='macro'),'balanced_accuracy':balanced_accuracy_score(ys,ps)})
ab=pd.DataFrame(ab); ab.to_csv(OUT/'dimension_ablation.csv',index=False)

# HUVEC shift: train HUVEC-1, test HUVEC-2 on the same held-out treatment folds.
shift=[]
for name,model in models().items():
 pp=[]; yy=[]
 for tr,te in cv.split(X,y,groups):
  tr=tr[df.iloc[tr].experiment.to_numpy()=='HUVEC-1']; te=te[df.iloc[te].experiment.to_numpy()=='HUVEC-2']
  if len(tr)==0 or len(te)==0: continue
  fit=model
  if name=='SmallMLP':
   b=balance(X[tr],y[tr]); fit.fit(b.drop(columns='_y').to_numpy(),b._y.to_numpy())
  else: fit.fit(X[tr],y[tr])
  pp.extend(fit.predict(X[te])); yy.extend(y[te])
 shift.append({'model':name,'HUVEC1_to_HUVEC2_macro_f1':f1_score(yy,pp,average='macro'),'HUVEC1_to_HUVEC2_balanced_accuracy':balanced_accuracy_score(yy,pp)})
pd.DataFrame(shift).to_csv(OUT/'huvec_shift.csv',index=False)

# CPU latency/model size, fitted on all selected sites.
lat=[]
for name,model in models().items():
 if name=='SmallMLP':
  b=balance(X,y); model.fit(b.drop(columns='_y').to_numpy(),b._y.to_numpy())
 else: model.fit(X,y)
 t=time.perf_counter();
 for _ in range(200): model.predict_proba(X[:1])
 ms=(time.perf_counter()-t)*1000/200
 lat.append({'model':name,'cpu_ms_per_site':ms,'serialized_kb':len(pickle.dumps(model))/1024})
pd.DataFrame(lat).to_csv(OUT/'edge_cost.csv',index=False)

fig,ax=plt.subplots(1,3,figsize=(15,4)); summary.set_index('model')[['macro_f1','balanced_accuracy']].plot.bar(ax=ax[0],ylim=(0,1),title='Treatment-held-out performance'); selective.plot(x='coverage',y='accepted_error',marker='o',ax=ax[1],title=f'Selective risk ({best})'); ab.plot(x='dimensions',y='macro_f1',marker='o',ax=ax[2],title='Dimension ablation'); fig.tight_layout(); fig.savefig(OUT/'fig_embedding_results.png',dpi=300)
with open(OUT/'run_metadata.json','w') as f: json.dump({'groups':GROUPS,'excluded':['IL-28A','IL-28B','IL-29'],'seed':SEED,'best_model':best},f,indent=2)
print('Done. Outputs:',OUT)
