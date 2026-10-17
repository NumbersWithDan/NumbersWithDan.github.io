"""Rebuild the portfolio's descriptive evidence; never executes legacy notebooks.
Usage: python scripts/uhi/verify.py --source C:/Users/danie/CSML/Project
Dependencies: numpy pandas matplotlib scikit-learn. Run from the repository root.
"""
import argparse, pathlib, json, hashlib, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.dummy import DummyRegressor
from source_io import read_dbf, read_points
from sklearn.model_selection import KFold, GroupKFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

args=argparse.ArgumentParser();args.add_argument('--source',type=pathlib.Path,required=True);root=args.parse_args().source
repo=pathlib.Path(__file__).resolve().parents[2]; out=repo/'assets/data/dallas-heat';img=repo/'images/dallas-heat'
raw=root/'Data/UHI/Shape files/af_trav'; derived=root/'Data/uhi_with_all_bands_and_ejscreen_features'
df=read_dbf(derived.with_suffix('.dbf')); obs=read_dbf(raw.with_suffix('.dbf'));xy=read_points(derived.with_suffix('.shp'))
assert len(xy)==len(df)==len(obs)
assert np.array_equal(df.t_f,obs.t_f)
assert np.isfinite(xy).all() and np.isfinite(df.t_f).all()
y=df.t_f.to_numpy(); ndvi=df.NDVI.to_numpy()
summary={'n':len(df),'temperature_min':float(y.min()),'temperature_max':float(y.max()),'temperature_mean':float(y.mean()),'temperature_sd':float(y.std(ddof=1)),'temperature_p05':float(np.quantile(y,.05)),'temperature_p95':float(np.quantile(y,.95)),'ndvi_temperature_pearson':float(np.corrcoef(y,ndvi)[0,1]),'missing':{k:int(v) for k,v in df.isna().sum().items()},'raw_and_derived_targets_match':True,'versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__,'scikit_learn':sklearn.__version__,'matplotlib':matplotlib.__version__},'sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [raw.with_suffix('.dbf'),raw.with_suffix('.shp'),derived.with_suffix('.dbf'),derived.with_suffix('.shp')]}}
# Descriptive quartiles use stored, unaudited satellite indices. No causal interpretation.
quart=pd.qcut(df.NDVI,4,labels=['Least green','Lower middle','Upper middle','Most green'])
summary['ndvi_quartiles']=[{'quartile':str(k),'n':len(g),'mean_temperature':float(g.t_f.mean())} for k,g in df.groupby(quart,observed=True)]
# A diagnostic model only: original feature construction remains unaudited.
X=df.drop(columns=['t_f','rh','hi_f'])
groups=np.floor((xy-xy.min(axis=0))/2000).astype(int);groups=np.unique(groups,axis=0,return_inverse=True)[1]
summary['spatial_groups_2km']=int(len(np.unique(groups)))
for name,cv,g in [('random',KFold(5,shuffle=True,random_state=42),None),('spatial_2km',GroupKFold(5),groups)]:
 model=make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(n_estimators=60,max_depth=12,min_samples_leaf=8,n_jobs=2,random_state=42))
 scores=cross_validate(model,X,y,cv=cv,groups=g,scoring={'mae':'neg_mean_absolute_error','r2':'r2'},return_train_score=False)
 summary[name]={'mae_f':(-scores['test_mae']).tolist(),'r2':scores['test_r2'].tolist()}
 baseline=cross_validate(DummyRegressor(strategy='mean'),X,y,cv=cv,groups=g,scoring='neg_mean_absolute_error')
 summary[name]['dummy_mean_mae_f']=(-baseline['test_score']).tolist()
print(json.dumps(summary,indent=2))
(out/'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'#f6fcf8','axes.facecolor':'#f6fcf8','text.color':'#163e3d','axes.labelcolor':'#163e3d'})
def save(name):
 plt.savefig(img/(name+'.png'),dpi=160,bbox_inches='tight');plt.close()
fig,ax=plt.subplots(figsize=(9,7));p=ax.scatter((xy[:,0]-xy[:,0].min())/1000,(xy[:,1]-xy[:,1].min())/1000,c=y,s=2,cmap='inferno',rasterized=True,vmin=101,vmax=110)
ax.set_aspect('equal');ax.axis('off');ax.set_title('Afternoon temperatures along the survey routes');fig.colorbar(p,ax=ax,label='Observed air temperature (Â°F)');save('observed-routes')
fig,ax=plt.subplots(figsize=(9,4));ax.hist(y,bins=np.arange(100.85,110.26,.2),color='#16786d',edgecolor='#f6fcf8');ax.set(xlabel='Observed afternoon air temperature (Â°F)',ylabel='Recorded observations',title='Most readings sit much closer together than the extremes');ax.axvline(y.mean(),color='#b34822',ls='--',label=f'Mean: {y.mean():.1f}Â°F');ax.legend();save('temperature-distribution')
fig,ax=plt.subplots(figsize=(9,4));q=summary['ndvi_quartiles'];ax.plot(range(4),[v['mean_temperature'] for v in q],'-o',color='#16786d',lw=3,ms=9);ax.set_xticks(range(4),[v['quartile'] for v in q]);ax.set(ylabel='Mean observed air temperature (Â°F)',xlabel='Quartiles of stored satellite vegetation index',title='A first look at greenery and heat');ax.grid(axis='y',alpha=.2);save('greenery-comparison')
fig,ax=plt.subplots(figsize=(9,4))
for i,k in enumerate(['random','spatial_2km']):
 v=np.array(summary[k]['mae_f']);ax.scatter(np.full(5,i),v,s=55,color='#b34822');ax.plot([i-.2,i+.2],[v.mean()]*2,color='#163e3d',lw=3)
ax.set_xticks([0,1],['Random 5-fold','2 km spatial groups Â· 5-fold']);ax.set(ylabel='Held-out mean absolute error (Â°F)',title='Holding out places makes prediction harder');ax.grid(axis='y',alpha=.2);save('validation-comparison')
# All route points, rounded in km; no demographic attributes or person identifiers.
(out/'observations.json').write_text(json.dumps({'description':'Normalized schematic display positions; not geographic coordinates','points':[[round((a-xy[:,0].min())/np.ptp(xy,axis=0).max(),5),round((b-xy[:,1].min())/np.ptp(xy,axis=0).max(),5),float(t)] for (a,b),t in zip(xy,y)]},separators=(',',':')))
