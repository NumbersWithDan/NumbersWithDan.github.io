"""Spatially evaluate period-specific excess heat and predict citywide scenarios.

No response-derived raster, local response mean, humidity, TES heat score, or
coordinate is an environmental predictor. Centering references and hot thresholds
come only from each training split. Private geographic outputs never enter Git.
"""
import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, average_precision_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import RobustScaler

OUT = Path(__file__).resolve().parents[2] / "assets/data/dallas-heat"
# Fixed before this experiment. No selection on outer-fold performance.
CONFIGS = [("landscape",5,400,10),("combined",5,400,10),
           ("combined",7,400,10),("combined",3,600,20)]
MODEL_SPEC = {"protocol":2,"configs":CONFIGS,"learning_rate":.035,"min_child_weight":12,
    "reg_alpha":.15,"subsample":.9,"colsample_bytree":.9,"tree_method":"hist",
    "seed":42,"outer_folds":5,"inner_folds":3,"blocks_m":1000,"shifts":[0,.5],
    "stress_config":["combined",3,400,20],"stress_buffer_m":500,"neighbors":32,
    "support_distance_m":1000,"support_feature_quantile":.95}

def groups(xy, size=1000, shift=0):
    return np.unique(np.floor(xy/size+shift).astype(int),axis=0,return_inverse=True)[1]

def fit(X,y,config):
    _,depth,trees,strength=config
    m=xgb.XGBRegressor(n_estimators=trees,max_depth=depth,reg_lambda=strength,learning_rate=.035,
        min_child_weight=12,reg_alpha=.15,subsample=.9,colsample_bytree=.9,
        tree_method="hist",n_jobs=2,random_state=42,objective="reg:squarederror")
    return m.fit(X,y)

def metrics(y,p,hot,baseline):
    score=p-baseline
    top=np.zeros(len(y),dtype=bool)
    top[np.argsort(score)[-max(1,int(np.ceil(.25*len(y)))):]]=True
    return {"mae_f":float(mean_absolute_error(y,p)),"r2":float(r2_score(y,p)),
            "spearman":float(spearmanr(y-baseline,score).statistic) if np.ptp(score)>1e-10 else None,
            "average_precision":float(average_precision_score(hot,score)),
            "hot_prevalence":float(hot.mean()),
            "precision_top_quarter":float(precision_score(hot,top,zero_division=0)) if np.ptp(score)>1e-10 else None,
            "recall_top_quarter":float(recall_score(hot,top,zero_division=0)) if np.ptp(score)>1e-10 else None}

def select(X,y,xy,physical):
    folds=list(GroupKFold(3).split(X,y,groups(xy)))
    scores=[]
    for cfg in CONFIGS:
        cols=physical if cfg[0]=="landscape" else list(X)
        err=[]
        for tr,te in folds:
            ref=y[tr].mean()
            m=fit(X.iloc[tr][cols],y[tr]-ref,cfg)
            err.append(mean_absolute_error(y[te],m.predict(X.iloc[te][cols])+ref))
        scores.append(float(np.mean(err)))
    cfg=CONFIGS[int(np.argmin(scores))]
    cols=physical if cfg[0]=="landscape" else list(X)
    ref=float(y.mean())
    return fit(X[cols],y-ref,cfg),cols,ref,cfg,scores

def main(work,only=None):
    if work.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('Keep geographic caches and fitted models outside the public website')
    d=pd.read_pickle(work/'islands.pkl'); X=d['X'].dropna(axis=1,how='all'); xy=d['xy']
    spec=json.loads(json.dumps(MODEL_SPEC))
    spec_path=work/'experiment-spec.json'
    if spec_path.exists():
        assert json.loads(spec_path.read_text())==spec, 'Model settings changed; use a fresh private work folder'
    else:
        assert not list(work.glob('*-models.joblib')), 'Unversioned model cache; use a fresh private work folder'
        spec_path.write_text(json.dumps(spec,indent=2))
    assert len(X)==len(xy) and not np.isinf(X.to_numpy()).any()
    physical=[c for c in X if c.startswith(('TOA_','Brightness_','NDVI','NDBI','Distance_', 'tree_', 'road_', 'vegetated_', 'built_signal_'))]
    added=['NDVI_mean_2130m','NDBI_mean_2130m','NDVI_local_contrast','NDBI_local_contrast',
           'vegetated_fraction_630m','built_signal_fraction_630m','tree_canopy_fraction',
           'road_pixel_fraction_210m','road_pixel_fraction_630m']
    old=[c for c in X if c not in added]
    summary={"protocol":"100m cell means; independent period models; 1km outer blocks and 3 inner spatial folds",
       "grid_m":100,"outer_block_m":1000,"candidates":CONFIGS,"experiment_spec":spec,"periods":{},
       "features":list(X),"raw_counts":d['raw_counts'],
       "feature_cache_sha256":hashlib.sha256((work/'islands.pkl').read_bytes()).hexdigest(),
       "target":"cell air temperature minus equal-cell training-period mean, in Fahrenheit",
       "hot_definition":"above the training-period 75th percentile of 100m cell means",
       "prediction_scope":"environmental citywide scenario for campaign-day conditions, not current weather",
       "classification":"ranking metrics only; heat scores are not calibrated probabilities"}
    results={}
    for period,s in d['surveys'].items():
        if only is not None and period!=only:
            continue
        ids=s['ids']; F=X.iloc[ids].reset_index(drop=True); y=s['temperature']; loc=xy[ids]
        cache=work/(period.lower()+'-models.joblib')
        if cache.exists():
            res=joblib.load(cache)
            assert res['fingerprint']==summary['feature_cache_sha256']
        else:
            pred=np.full((len(y),4),np.nan); ref=np.zeros(len(y)); hot=np.zeros(len(y),dtype=bool)
            fold_id=np.zeros(len(y),int); folds=[]
            for f,(tr,te) in enumerate(GroupKFold(5).split(F,y,groups(loc))):
                assert not np.intersect1d(groups(loc)[tr],groups(loc)[te]).size
                m,cols,reference,cfg,scores=select(F.iloc[tr],y[tr],loc[tr],physical)
                pred[te,0]=m.predict(F.iloc[te][cols])+reference
                pred[te,1]=KNeighborsRegressor(n_neighbors=32,weights='distance').fit(loc[tr],y[tr]).predict(loc[te])
                pred[te,2]=reference
                # Fixed legacy-feature comparator. Hyperparameters also differ;
                # this is not an isolated estimate of the new features' benefit.
                original=fit(F.iloc[tr][old],y[tr]-reference,('combined',7,200,10))
                pred[te,3]=original.predict(F.iloc[te][old])+reference
                ref[te]=reference; hot[te]=y[te]>=np.quantile(y[tr],.75); fold_id[te]=f
                folds.append({'model':m,'cols':cols,'reference':reference,'config':cfg,'inner_mae':scores})
                print(period,'outer',f+1,'config',cfg,'MAE',round(mean_absolute_error(y[te],pred[te,0]),3),flush=True)
            # Repeat inner selection within a second grid placement.
            shifted=np.zeros(len(y))
            for f,(tr,te) in enumerate(GroupKFold(5).split(F,y,groups(loc,shift=.5))):
                m,cols,r,cfg,_=select(F.iloc[tr],y[tr],loc[tr],physical)
                shifted[te]=m.predict(F.iloc[te][cols])+r
                print(period,'shifted fold',f+1,'complete',flush=True)
            # Fixed configuration and 500m spatial gap for the separate regional test.
            transfer=np.zeros(len(y)); region=pd.qcut(loc[:,1],5,labels=False)
            transfer_base=np.zeros(len(y))
            for f in range(5):
                te=np.flatnonzero(region==f); tr=np.flatnonzero(region!=f)
                tr=tr[cKDTree(loc[te]).query(loc[tr])[0]>=500]
                r=y[tr].mean(); m=fit(F.iloc[tr],y[tr]-r,('combined',3,400,20))
                transfer[te]=m.predict(F.iloc[te])+r; transfer_base[te]=r
            print(period,'regional checks complete',flush=True)
            final,cols,reference,cfg,scores=select(F,y,loc,physical)
            print(period,'final fit complete',flush=True)
            city_pred=final.predict(X[cols])
            # Distribution-support diagnostic on model predictors, not a prediction interval.
            med=F[cols].median().fillna(0); scaler=RobustScaler().fit(F[cols].fillna(med))
            z=scaler.transform(F[cols].fillna(med)); zcity=scaler.transform(X[cols].fillna(med))
            nn=cKDTree(z)
            train_distance=nn.query(z,k=2,workers=2)[0][:,1]
            feature_distance=nn.query(zcity,k=1,workers=2)[0]
            threshold=float(np.quantile(train_distance,.95))
            geo_distance=cKDTree(loc).query(xy)[0]
            supported=(feature_distance<=threshold)&(geo_distance<=1000)&X.NDVI.notna().to_numpy()
            print(period,'citywide predictions and support check complete',flush=True)
            res={'pred':pred,'refs':ref,'hot':hot,'fold_id':fold_id,'folds':folds,
                 'shifted':shifted,'transfer':transfer,'transfer_base':transfer_base,
                 'final':final,'cols':cols,'reference':reference,'config':cfg,'inner_scores':scores,
                 'city_pred':city_pred,'supported':supported,'feature_threshold':threshold,
                 'distance':geo_distance,'fingerprint':summary['feature_cache_sha256']}
            joblib.dump(res,cache)
        assert np.isfinite(res['pred']).all()
        prediction=res['pred']; reference=res['refs']; hot=res['hot']
        report={'raw_readings':d['raw_counts'][period],'cells':len(y),'reference_f':res['reference'],
           'standard_deviation_f':float(y.std()),'hot_cutoff_f':float(np.quantile(y,.75)),
           'chosen_config':res['config'],'final_columns':res['cols'],
           'outer_selections':[{'config':f['config'],'inner_mae':f['inner_mae']} for f in res['folds']],
           'models':{name:metrics(y,prediction[:,i],hot,reference) for i,name in enumerate(['environmental','spatial','mean','original_features'])},
           'shifted_mae_f':float(mean_absolute_error(y,res['shifted'])),
           'shifted_r2':float(r2_score(y,res['shifted'])),
           'regional_mae_f':float(mean_absolute_error(y,res['transfer'])),
           'regional_mean_mae_f':float(mean_absolute_error(y,res['transfer_base'])),
           'support_rule':'nearest survey cell <= 1000m; robust-scaled model-feature nearest-neighbor distance <= training leave-one-out 95th percentile; valid point NDVI',
           'support_feature_threshold':res['feature_threshold'],
           'city_land_cells':int((d['city_mask']&~d['water_mask']).sum()),
           'supported_city_land_cells':int((d['city_mask']&~d['water_mask']&res['supported']).sum())}
        summary['periods'][period]=report
        results[period]=res
        result_name='islands-results.json' if only is None else 'islands-results-'+only.lower()+'.json'
        (OUT/result_name).write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
        print(period,json.dumps(report['models']),flush=True)
    if only is None:
        joblib.dump(results,work/'all-period-models.joblib')

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('--work',type=Path,required=True)
    a.add_argument('--only',choices=['Morning','Afternoon','Evening'])
    args=a.parse_args(); main(args.work,args.only)
