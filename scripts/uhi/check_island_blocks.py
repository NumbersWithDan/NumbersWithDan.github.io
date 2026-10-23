"""Fixed-configuration afternoon block sensitivity on equal-weight 100m cells."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import GroupKFold
from model_islands import groups,fit

def main(work):
    d=pd.read_pickle(work/'islands.pkl'); s=d['surveys']['Afternoon']
    X=d['X'].iloc[s['ids']].dropna(axis=1,how='all').reset_index(drop=True)
    xy=d['xy'][s['ids']]; y=s['temperature']; records=[]
    fingerprint=hashlib.sha256((work/'islands.pkl').read_bytes()).hexdigest()
    for size in [100,250,500,1000,2000]:
        for shift in [0,.5]:
            cache=work/f'island-block-{size}-{shift}.npz'
            if cache.exists():
                old=np.load(cache); assert str(old['fingerprint'])==fingerprint
                pred,base,near=old['pred'],old['base'],old['near']
            else:
                pred=np.zeros(len(y)); base=np.zeros(len(y)); near=np.zeros(len(y))
                for tr,te in GroupKFold(5).split(X,y,groups(xy,size,shift)):
                    r=y[tr].mean(); m=fit(X.iloc[tr],y[tr]-r,('combined',4,200,10))
                    pred[te]=m.predict(X.iloc[te])+r; base[te]=r
                    near[te]=cKDTree(xy[tr]).query(xy[te])[0]
                np.savez_compressed(cache,pred=pred,base=base,near=near,fingerprint=fingerprint)
            records.append({'block_m':size,'shift':shift,'mae_f':float(mean_absolute_error(y,pred)),
              'r2':float(r2_score(y,pred)),'mean_mae_f':float(mean_absolute_error(y,base)),
              'median_nearest_training_m':float(np.median(near))})
            print(records[-1],flush=True)
    out=Path(__file__).resolve().parents[2]/'assets/data/dallas-heat/islands-blocks.json'
    out.write_text(json.dumps({'period':'Afternoon','config':['combined',4,200,10],
      'grid_origin':'fixed projected grid; consistent across tested sizes',
      'records':records},indent=2)+'\n')

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--work',type=Path,required=True); main(a.parse_args().work)
