"""Stage 1: replicate-well local consistency for RxRx2 morphology embeddings.

Run after rxrx2_embedding_icce_colab.py. It uses its XGBoost treatment-held-out
out-of-fold predictions, so the correctness comparison does not reuse train sites.
"""
from pathlib import Path
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

ROOT=Path(os.environ.get('RXRX2_DATA_ROOT','/content/drive/MyDrive/SUDAL/rxrx2/data'))
OUT=Path(os.environ.get('RXRX2_OUTPUT_ROOT','/content/drive/MyDrive/SUDAL/rxrx2/results_embedding_icce'))
META=Path(os.environ.get('RXRX2_METADATA',str(ROOT/'metadata.csv')))
EMB=Path(os.environ.get('RXRX2_EMBEDDINGS',str(ROOT/'embeddings.csv')))
meta=pd.read_csv(META); emb=pd.read_csv(EMB)
groups={
 'type_I_II_interferons':['IFN-alpha-A','IFNB-1','IFNE','IFN-omega','IFN-gamma'],
 'il1_tnf_superfamily':['IL-1-alpha','IL-1-beta','TNF-alpha','TNF-beta','IL-18','IL-33','IL-36-gamma','TL-1A'],
 'chemokines_cxc_cc':['ENA-78','GCP-2','GRO-alpha','GRO-B','GRO-gamma','IL-8','IP-10','I-TAC','CCL-1','CCL14a','CCL21','CCL28','CCL-4','MCP-1','MCP-2','MCP-3'],
 'angiogenic_vascular_regulatory':['Angiopoietin-1','Angiopoietin-2','Angiopoietin-4','VEGF-165','VEGFC','VEGF-D','EG-VEGF','PLGF-4']}
keep={t:k for k,v in groups.items() for t in v}
site=meta[meta.treatment.isin(keep)].merge(emb,on='site_id',validate='one_to_one')
f=[c for c in site if c.startswith('feature_')]

# Average the four imaging sites inside each well first. This prevents one
# visually noisy field from being counted as an independent biological replicate.
keys=['well_id','treatment','treatment_conc','experiment','plate']
well=site.groupby(keys,as_index=False)[f].mean()
condition=['treatment','treatment_conc','experiment']
records=[]
for ckey,g in well.groupby(condition,sort=False):
 x=g[f].to_numpy(float); x=x/(np.linalg.norm(x,axis=1,keepdims=True)+1e-12)
 if len(g)<2: continue
 sim=cosine_similarity(x); np.fill_diagonal(sim,np.nan)
 # Leave-one-well-out centroid cosine: a well cannot make itself look reliable.
 local=np.nanmean(sim,axis=1)
 for (_,r),score in zip(g.iterrows(),local):
  records.append({'well_id':r.well_id,'treatment':r.treatment,'treatment_conc':r.treatment_conc,
                  'experiment':r.experiment,'plate':r.plate,'local_consistency':score,'n_peer_wells':len(g)-1})
well_score=pd.DataFrame(records)
well_score.to_csv(OUT/'well_local_consistency.csv',index=False)
condition_score=well_score.groupby(condition).agg(mean_local_consistency=('local_consistency','mean'),sd_local_consistency=('local_consistency','std'),n_wells=('well_id','size')).reset_index()
condition_score.to_csv(OUT/'condition_local_consistency.csv',index=False)

# Link unsupervised replicate consistency to treatment-held-out XGBoost errors.
oof=pd.read_csv(OUT/'treatment_heldout_oof_predictions.csv')
xgb=oof[oof.model=='XGBoost'].copy(); probs=xgb[[f'p{i}' for i in range(4)]].to_numpy()
xgb['prediction']=probs.argmax(1); xgb['correct']=(xgb.prediction==xgb.y).astype(int); xgb['confidence']=probs.max(1)
site_to_well=site[['site_id','well_id']]
linked=xgb.merge(site_to_well,on='site_id',validate='one_to_one').merge(well_score[['well_id','local_consistency']],on='well_id',validate='many_to_one')
linked['consistency_quartile']=pd.qcut(linked.local_consistency,4,labels=['Q1 lowest','Q2','Q3','Q4 highest'])
risk=linked.groupby('consistency_quartile',observed=False).agg(sites=('site_id','size'),accuracy=('correct','mean'),mean_confidence=('confidence','mean'),mean_consistency=('local_consistency','mean')).reset_index()
risk['error_rate']=1-risk.accuracy
risk.to_csv(OUT/'consistency_vs_xgboost_risk.csv',index=False)
linked.to_csv(OUT/'xgboost_oof_with_local_consistency.csv',index=False)

fig,ax=plt.subplots(1,2,figsize=(10,4))
ax[0].hist(well_score.local_consistency,bins=30,color='#4C78A8'); ax[0].set(title='Replicate-well local consistency',xlabel='Mean cosine similarity to peer wells',ylabel='Well count')
ax[1].bar(risk.consistency_quartile.astype(str),risk.error_rate,color='#E45756'); ax[1].set(title='Treatment-held-out XGBoost error by consistency',xlabel='Consistency quartile',ylabel='Error rate',ylim=(0,1))
fig.tight_layout(); fig.savefig(OUT/'fig_local_consistency.png',dpi=300)
print('Wells:',len(well_score),'conditions:',len(condition_score)); print(risk.to_string(index=False)); print('Saved:',OUT)
