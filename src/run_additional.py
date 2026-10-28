from pathlib import Path
import os, json, time
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from xgboost import XGBClassifier

SEED=42; rng=np.random.default_rng(SEED)
ROOT=Path(os.environ['RXRX2_DATA_ROOT']); OUT=Path(os.environ['RXRX2_OUTPUT_ROOT'])
META=Path(os.environ['RXRX2_METADATA']); EMB=Path(os.environ['RXRX2_EMBEDDINGS'])
GROUPS={'type_I_II_interferons':['IFN-alpha-A','IFNB-1','IFNE','IFN-omega','IFN-gamma'],'il1_tnf_superfamily':['IL-1-alpha','IL-1-beta','TNF-alpha','TNF-beta','IL-18','IL-33','IL-36-gamma','TL-1A'],'chemokines_cxc_cc':['ENA-78','GCP-2','GRO-alpha','GRO-B','GRO-gamma','IL-8','IP-10','I-TAC','CCL-1','CCL14a','CCL21','CCL28','CCL-4','MCP-1','MCP-2','MCP-3'],'angiogenic_vascular_regulatory':['Angiopoietin-1','Angiopoietin-2','Angiopoietin-4','VEGF-165','VEGFC','VEGFD','EG-VEGF','PLGF-4']}
# Preserve exact dataset spelling VEGF-D.
GROUPS['angiogenic_vascular_regulatory'][5]='VEGF-D'
classes=list(GROUPS); label={t:c for c,ts in GROUPS.items() for t in ts}; ident={c:i for i,c in enumerate(classes)}
meta=pd.read_csv(META); emb=pd.read_csv(EMB); df=meta[meta.treatment.isin(label)].copy(); df['y']=df.treatment.map(label).map(ident); df=df.merge(emb,on='site_id',validate='one_to_one'); f=[c for c in df if c.startswith('feature_')]; X=df[f].to_numpy(np.float32); y=df.y.to_numpy(); g=df.treatment.to_numpy()

def xgb(): return XGBClassifier(n_estimators=250,max_depth=4,learning_rate=.06,subsample=.85,colsample_bytree=.85,objective='multi:softprob',eval_metric='mlogloss',n_jobs=4,random_state=SEED)
def logit(): return Pipeline([('scale',StandardScaler()),('clf',LogisticRegression(max_iter=3000,class_weight='balanced',random_state=SEED))])

# 1. Pure HUVEC shift diagnostic: same treatments are allowed across source/target.
shift=[]
for name,build in [('Logistic',logit),('XGBoost',xgb)]:
 tr=np.where(df.experiment.to_numpy()=='HUVEC-1')[0]; te=np.where(df.experiment.to_numpy()=='HUVEC-2')[0]
 m=build(); m.fit(X[tr],y[tr]); p=m.predict(X[te]); shift.append({'model':name,'setting':'HUVEC1 train -> HUVEC2 same-treatment diagnostic','macro_f1':f1_score(y[te],p,average='macro'),'balanced_accuracy':balanced_accuracy_score(y[te],p)})
pd.DataFrame(shift).to_csv(OUT/'pure_huvec_shift.csv',index=False)

# 2. Treatment-cluster bootstrap CI from the already held-out OOF predictions.
oof=pd.read_csv(OUT/'treatment_heldout_oof_predictions.csv'); boot=[]
for name,z in oof.groupby('model'):
 ts=z.treatment.unique()
 for _ in range(1000):
  pick=rng.choice(ts,size=len(ts),replace=True); b=pd.concat([z[z.treatment==t] for t in pick],ignore_index=True); p=b[[f'p{i}' for i in range(4)]].to_numpy().argmax(1)
  boot.append({'model':name,'macro_f1':f1_score(b.y,p,average='macro'),'balanced_accuracy':balanced_accuracy_score(b.y,p)})
boot=pd.DataFrame(boot); ci=boot.groupby('model').agg(macro_f1_mean=('macro_f1','mean'),macro_f1_lo=('macro_f1',lambda x:np.quantile(x,.025)),macro_f1_hi=('macro_f1',lambda x:np.quantile(x,.975)),ba_mean=('balanced_accuracy','mean'),ba_lo=('balanced_accuracy',lambda x:np.quantile(x,.025)),ba_hi=('balanced_accuracy',lambda x:np.quantile(x,.975))).reset_index(); ci.to_csv(OUT/'treatment_bootstrap_ci.csv',index=False)

# 3. Nested reliability/defer test. Calibration uses one entire treatment per class;
# the outer test fold remains unseen. Policy: choose lowest threshold with calibration error <= .40.
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED); nested=[]
for fold,(outer_tr,outer_te) in enumerate(cv.split(X,y,g),1):
 trdf=df.iloc[outer_tr]; cal_t=[]
 for c in range(4): cal_t.append(sorted(trdf[trdf.y==c].treatment.unique())[0])
 calmask=np.isin(g[outer_tr],cal_t); fitidx=outer_tr[~calmask]; calidx=outer_tr[calmask]
 m=xgb(); m.fit(X[fitidx],y[fitidx]); cp=m.predict_proba(X[calidx]); cc=cp.max(1); cpr=cp.argmax(1)
 choices=[]
 for tau in np.arange(.40,.81,.05):
  a=cc>=tau; choices.append((tau,a.mean(),1-(cpr[a]==y[calidx][a]).mean() if a.any() else 1))
 valid=[q for q in choices if q[2]<=.40]; tau=(valid[0] if valid else choices[-1])[0]
 tp=m.predict_proba(X[outer_te]); conf=tp.max(1); pred=tp.argmax(1); a=conf>=tau
 nested.append({'fold':fold,'threshold':tau,'coverage':a.mean(),'accepted_error':1-(pred[a]==y[outer_te][a]).mean() if a.any() else np.nan,'accepted_macro_f1':f1_score(y[outer_te][a],pred[a],average='macro') if a.any() else np.nan,'all_error':1-(pred==y[outer_te]).mean()})
nested=pd.DataFrame(nested); nested.to_csv(OUT/'nested_defer_evaluation.csv',index=False)

# 4. XGBoost confusion and concentration strata from treatment-held-out OOF predictions.
z=oof[oof.model=='XGBoost'].merge(meta[['site_id','treatment_conc']],on='site_id',validate='one_to_one'); p=z[[f'p{i}' for i in range(4)]].to_numpy().argmax(1)
cm=pd.DataFrame(confusion_matrix(z.y,p,labels=range(4),normalize='true'),index=classes,columns=classes); cm.to_csv(OUT/'xgboost_confusion_matrix.csv')
z['conc_bin']=pd.qcut(z.treatment_conc.rank(method='first'),3,labels=['low','mid','high']); conc=[]
for b,q in z.groupby('conc_bin',observed=False):
 qp=q[[f'p{i}' for i in range(4)]].to_numpy().argmax(1); conc.append({'concentration_bin':str(b),'sites':len(q),'macro_f1':f1_score(q.y,qp,average='macro'),'balanced_accuracy':balanced_accuracy_score(q.y,qp)})
pd.DataFrame(conc).to_csv(OUT/'concentration_stratified_performance.csv',index=False)

fig,ax=plt.subplots(1,3,figsize=(15,4)); ax[0].bar(ci.model,ci.ba_mean,yerr=[ci.ba_mean-ci.ba_lo,ci.ba_hi-ci.ba_mean],capsize=4); ax[0].set(title='Treatment-bootstrap BA (95% CI)',ylim=(0,1)); im=ax[1].imshow(cm,0,1,cmap='Blues'); ax[1].set(title='XGBoost confusion'); ax[1].set_xticks(range(4),[x.replace('_','\n') for x in classes],rotation=45,ha='right'); ax[1].set_yticks(range(4),classes); fig.colorbar(im,ax=ax[1]); ax[2].plot(nested.fold,nested.all_error,marker='o',label='all'); ax[2].plot(nested.fold,nested.accepted_error,marker='o',label='accepted'); ax[2].set(title='Nested defer test error',xlabel='fold',ylabel='error'); ax[2].legend(); fig.tight_layout(); fig.savefig(OUT/'fig_reliability_additional.png',dpi=300)
print('PURE SHIFT\n',pd.DataFrame(shift).to_string(index=False)); print('NESTED DEFER\n',nested.to_string(index=False)); print('Done')
