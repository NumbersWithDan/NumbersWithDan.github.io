"""Check fitted fold provenance, published metrics and public coordinate policy."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from model_islands import groups,metrics

def main(work):
    repo=Path(__file__).resolve().parents[2]; out=repo/'assets/data/dallas-heat'
    d=pd.read_pickle(work/'islands.pkl'); r=json.loads((out/'islands-results.json').read_text())
    results=joblib.load(work/'all-period-models.joblib')
    checks=[]
    for period,s in d['surveys'].items():
        m=results[period]; ids=s['ids']; xy=d['xy'][ids]; y=s['temperature']; X=d['X'].iloc[ids].reset_index(drop=True)
        assert np.isfinite(y).all() and len(np.unique(ids))==len(ids)
        assert int(s['readings_per_cell'].sum())==d['raw_counts'][period]
        assert not any(c in m['cols'] for c in ['t_f','rh','hi_f','temp_diff','temp_norm','tes'])
        for f,(tr,te) in enumerate(GroupKFold(5).split(X,y,groups(xy))):
            assert not np.intersect1d(groups(xy)[tr],groups(xy)[te]).size
            assert np.all(m['fold_id'][te]==f)
            fold=m['folds'][f]
            assert np.isclose(fold['reference'],y[tr].mean())
            assert np.allclose(m['refs'][te],y[tr].mean())
            assert np.array_equal(m['hot'][te],y[te]>=np.quantile(y[tr],.75))
            assert np.allclose(fold['model'].predict(X.iloc[te][fold['cols']])+fold['reference'],m['pred'][te,0],atol=1e-5)
            assert np.isclose(min(fold['inner_mae']),fold['inner_mae'][r['candidates'].index(list(fold['config']))])
        recalculated=metrics(y,m['pred'][:,0],m['hot'],m['refs'])
        for key,value in recalculated.items():
            published=r['periods'][period]['models']['environmental'][key]
            assert value is None and published is None or value is not None and np.isclose(value,published)
        assert np.isfinite(m['city_pred']).all()
        checks.append({'period':period,'cells':len(y),'folds':5,'metrics_verified':True})
    display=json.loads((out/'islands-display.json').read_text())
    for s in display['periods'].values():
        pts=np.array(s['points']); assert np.isfinite(pts).all() and (pts[:,:2]>=0).all() and (pts[:,:2]<=1).all()
    forbidden={'latitude','longitude','easting','northing','bounds','xy','coordinates'}
    def scan(x):
        if isinstance(x,dict):
            assert not forbidden.intersection(x)
            for v in x.values(): scan(v)
        elif isinstance(x,list):
            for v in x: scan(v)
    for name in ['islands-results.json','islands-blocks.json','islands-evidence.json','islands-display.json']:
        scan(json.loads((out/name).read_text()))
    record={'period_checks':checks,'normalized_display_only':True,'public_coordinate_keys_absent':True,
      'local_and_global_shap_additivity':'asserted during every render',
      'classification_note':'scores rank heat; they are not calibrated probabilities'}
    (out/'islands-audit.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--work',type=Path,required=True); main(a.parse_args().work)
