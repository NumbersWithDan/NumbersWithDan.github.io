"""Render city-context maps, directional SHAP and supported candidate islands.

All public positions are normalized schematic display units. Fitted models,
geographic coordinates and citywide feature tables remain in private --work.
"""
import argparse
import json
import hashlib
from pathlib import Path
import geopandas as gpd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy.ndimage import label, convolve
from scipy.spatial import cKDTree
import shapely
import xgboost as xgb

REPO=Path(__file__).resolve().parents[2]
OUT=REPO/'assets/data/dallas-heat'
IMG=REPO/'images/dallas-heat'
COLORS=['#157a70','#bb4a29','#705b99']
NAMES={'PM25':'Fine particle concentration','PTRAF':'Traffic proximity indicator',
 'NDVI':'Vegetation at the cell','NDVI_mean_210m':'Vegetation nearby','NDVI_mean_630m':'Neighborhood vegetation',
 'NDVI_mean_2130m':'Wider-area vegetation','NDVI_local_contrast':'Local vegetation contrast',
 'NDBI':'Built-up signal at the cell','NDBI_mean_210m':'Built-up signal nearby',
 'NDBI_mean_630m':'Neighborhood built-up signal','NDBI_mean_2130m':'Wider-area built-up signal',
 'NDBI_local_contrast':'Local built-up contrast','Distance_to_Lake':'Distance to a lake',
 'Distance_to_Stream':'Distance to a stream','Distance_to_Park':'Distance to a park',
 'road_pixel_fraction_210m':'Streets nearby','road_pixel_fraction_630m':'Neighborhood street footprint',
 'vegetated_fraction_630m':'Greener pixel share','built_signal_fraction_630m':'Built-signal pixel share','tree_canopy_fraction':'Tree canopy share',
 'PRE1960PCT':'Older housing share','LOWINCPCT':'Lower-income population share',
 'OVER64PCT':'Older population share','UNEMPPCT':'Unemployment share',
 'MdSalesPr':'Median residential sale price','CoefVar':'Sale price variation',
 'PHHOO':'Owner-occupied housing share','VptResVac':'Residential vacancy share',
 'PPermNCUnt':'New construction permit indicator','PRehabPerm':'Rehabilitation permit indicator',
 'PFclOO1517':'Foreclosure indicator','PCVLnRsPr':'Code violation indicator',
 'PPubSubAll':'Subsidized housing indicator','PAPerm':'Permit activity indicator'}
for b in range(1,8): NAMES[f'TOA_B{b}']=f'Satellite reflectance band {b}'
for b in [10,11]: NAMES[f'Brightness_B{b}_K']=f'Satellite thermal signal {b}'
for c in 'ABCDEFGHI': NAMES['MVA_'+c]='Market area category '+c
NAMES['MVA_Missing']='No market area category'

def friendly(s): return NAMES.get(s,s.replace('_',' '))

def save(fig,name):
    fig.savefig(IMG/(name+'.png'),dpi=170,bbox_inches='tight'); plt.close(fig)

def lines(g):
    output=[]
    for geom in g.geometry:
        for part in shapely.get_parts(geom):
            if part.geom_type=='Polygon': output.append(np.asarray(part.exterior.coords))
            elif part.geom_type=='LineString': output.append(np.asarray(part.coords))
    return output

def main(root,work,overview_only=False):
    d=pd.read_pickle(work/'islands.pkl'); models=joblib.load(work/'all-period-models.joblib')
    report=json.loads((OUT/'islands-results.json').read_text()); X=d['X']; xy=d['xy']; periods=list(models)
    boundary=gpd.read_file(work/'city-boundary.geojson').to_crs(32614)
    roads=gpd.read_file(root/'Data/Dallas City Data/Streets/STREETS.shp').to_crs(32614)
    water=gpd.read_file(root/'Data/Water/Lakes/WaterFeatures.shp').to_crs(32614)
    parks=gpd.read_file(root/'Data/Dallas City Data/Parks/Parks.shp').to_crs(32614)
    parks.geometry=parks.geometry.make_valid()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':'#f6fcf8',
      'axes.facecolor':'#f6fcf8','text.color':'#163e3d','axes.labelcolor':'#163e3d',
      'axes.spines.top':False,'axes.spines.right':False})
    survey_xy=xy[np.unique(np.concatenate([s['ids'] for s in d['surveys'].values()]))]
    bounds=[survey_xy[:,0].min()-1400,survey_xy[:,0].max()+1400,
            survey_xy[:,1].min()-1400,survey_xy[:,1].max()+1400]
    def basemap(ax,citywide=False):
        parks.plot(ax=ax,color='#dce7d8',edgecolor='none',alpha=.6,zorder=0)
        water.plot(ax=ax,color='#c8dde1',edgecolor='none',zorder=1)
        roads.plot(ax=ax,color='#a7b4ae',lw=.22,alpha=.3,zorder=2)
        boundary.boundary.plot(ax=ax,color='#778e84',lw=.65,alpha=.7,zorder=2)
        if citywide:
            b=boundary.total_bounds; ax.set_xlim(b[0]-1000,b[2]+1000); ax.set_ylim(b[1]-1000,b[3]+1000)
        else: ax.set_xlim(*bounds[:2]); ax.set_ylim(*bounds[2:])
        ax.set_aspect('equal'); ax.axis('off')
        ax.text(.98,.98,'N ↑',transform=ax.transAxes,ha='right',va='top')
        ax.text(.02,.015,'Dallas streets · parks · water',transform=ax.transAxes,fontsize=8,color='#637d73')
        # Label source geometries without publishing geographic coordinate fields.
        for text in ['White Rock Lake','Bachman Lake']:
            named=water[water.NAME.fillna('').str.casefold()==text.casefold()]
            if len(named):
                point=named.geometry.union_all().representative_point()
                ax.text(point.x,point.y,text,fontsize=7,color='#36554e',ha='center',zorder=10,clip_on=True,
                        bbox=dict(facecolor='#f6fcf8',alpha=.75,edgecolor='none',pad=1))
    fig,axes=plt.subplots(1,3,figsize=(15,6))
    for ax,p in zip(axes,periods):
        s=d['surveys'][p]; v=s['temperature']-models[p]['reference']
        basemap(ax); im=ax.scatter(*xy[s['ids']].T,c=v,s=6,cmap='RdYlBu_r',vmin=-2.5,vmax=2.5,zorder=4)
        ax.set_title(p+f"\n{len(s['ids'])} survey cells")
    fig.colorbar(im,ax=axes,shrink=.65,pad=.015,label='Observed excess air temperature (°F)')
    save(fig,'period-observations')
    if overview_only:
        return
    fig,ax=plt.subplots(figsize=(10,4.3))
    for p,c in zip(periods,COLORS):
        s=d['surveys'][p]; ax.hist(s['temperature']-models[p]['reference'],bins=np.arange(-5,5,.2),density=True,
                                 histtype='step',lw=2,label=p,color=c)
    ax.axvline(0,color='#36554e',ls='--'); ax.set(xlabel='Degrees above or below the period survey reference (°F)',ylabel='Density',title='Temperature above the survey average'); ax.legend(); save(fig,'period-distribution')
    blocks=json.loads((OUT/'islands-blocks.json').read_text())['records']
    sizes=sorted(set(r['block_m'] for r in blocks))
    errors=np.array([[r['mae_f'] for r in blocks if r['block_m']==s] for s in sizes])
    baseline=np.array([[r['mean_mae_f'] for r in blocks if r['block_m']==s] for s in sizes])
    fig,ax=plt.subplots(figsize=(10,4.5))
    ax.plot(sizes,errors.mean(axis=1),'-o',color=COLORS[0],label='Fixed environmental model')
    ax.fill_between(sizes,errors.min(axis=1),errors.max(axis=1),color=COLORS[0],alpha=.15)
    ax.plot(sizes,baseline.mean(axis=1),'--',color='#738278',label='Training mean')
    ax.axvline(1000,color='#b54b2b',ls=':',label='Primary evaluation scale')
    ax.set(xlabel='Spatial holdout block width (m)',ylabel='Held-out mean absolute error (°F)',title='Prediction error by block size')
    ax.set_xticks(sizes); ax.legend(fontsize=9); fig.tight_layout(); save(fig,'islands-block-sensitivity')
    fig,ax=plt.subplots(figsize=(9,4.5))
    greenery=[]
    for p,c in zip(periods,COLORS):
        s=d['surveys'][p]; valid=X.iloc[s['ids']].NDVI_mean_630m.notna().to_numpy()
        ndvi=X.iloc[s['ids'][valid]].NDVI_mean_630m
        q=pd.qcut(ndvi,4,labels=False,duplicates='drop').to_numpy()
        means=[float((s['temperature'][valid]-models[p]['reference'])[q==i].mean()) for i in range(4)]
        ax.plot(range(4),means,'-o',color=c,label=p,lw=2)
        greenery.append({'period':p,'quartile_mean_excess_f':means})
    ax.set_xticks(range(4),['Least green','Lower middle','Upper middle','Most green'])
    ax.set(ylabel='Mean observed excess temperature (°F)',xlabel='Neighborhood vegetation quartiles',title='Heat and neighborhood vegetation'); ax.axhline(0,c='#7c8e86',lw=.7); ax.legend(); save(fig,'period-greenery')
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    for i,p in enumerate(periods):
        r=report['periods'][p]; m=r['models']; y=i
        axes[0].barh(y-.18,m['mean']['mae_f'],height=.32,color='#87958c',label='Mean baseline' if i==0 else None)
        axes[0].barh(y+.18,m['environmental']['mae_f'],height=.32,color=COLORS[i],label='Environmental model' if i==0 else None)
        axes[0].text(m['environmental']['mae_f']+.015,y+.18,f"{m['environmental']['mae_f']:.2f}°F",va='center')
        axes[1].bar(i,m['environmental']['precision_top_quarter'],color=COLORS[i])
        axes[1].plot([i-.35,i+.35],[m['environmental']['hot_prevalence']]*2,color='#53685c',ls='--')
        axes[1].text(i,m['environmental']['precision_top_quarter']+.025,f"{m['environmental']['precision_top_quarter']:.0%}",ha='center')
    axes[0].set_yticks(range(3),periods); axes[0].invert_yaxis(); axes[0].set(xlabel='Held-out mean absolute error (°F)',title='Temperature prediction error'); axes[0].legend(loc='lower right',fontsize=8)
    axes[1].set_xticks(range(3),periods); axes[1].set(ylim=(0,1),ylabel='Observed hot cells in the highest-scored quarter',title='Hot cell identification'); axes[1].text(.02,.95,'Dashed marks show hot-cell prevalence',transform=axes[1].transAxes,va='top',fontsize=8)
    fig.tight_layout(); save(fig,'islands-validation')
    evidence={'greenery':greenery,'periods':{},'feature_labels':NAMES,'global_features':{},'global_directions':{},
      'local_definition':'above the period training upper-quartile threshold and at least 0.25°F above the mean predicted land cells 500–1500m away; >=3 contiguous 100m cells',
      'support_note':'feature novelty and distance screening; not calibrated uncertainty'}
    land=d['city_mask']&~d['water_mask']&X.NDVI.notna().to_numpy()
    tree=cKDTree(xy[land]); land_ids=np.flatnonzero(land)
    min_cell=np.floor(xy.min(axis=0)/100).astype(int)
    grid=(np.floor(xy/100).astype(int)-min_cell)
    shape=grid.max(axis=0)+1
    # Convolution avoids materializing tens of millions of neighbor indices.
    offsets=np.arange(-15,16)*100
    dx,dy=np.meshgrid(offsets,offsets,indexing='ij')
    distance=np.hypot(dx,dy)
    kernel=((distance>=500)&(distance<=1500)).astype(float)
    maskgrid=np.zeros(shape); maskgrid[grid[land,0],grid[land,1]]=1
    counts=convolve(maskgrid,kernel,mode='constant')
    def annulus_mean(values):
        raster=np.zeros(shape); raster[grid[land,0],grid[land,1]]=values[land]
        sums=convolve(raster,kernel,mode='constant')
        avg=np.divide(sums,counts,out=np.full_like(sums,np.nan),where=counts>=30)
        return avg[grid[:,0],grid[:,1]]
    cases_all={}
    for p in periods:
        res=models[p]; v=res['city_pred']; sup=res['supported']
        model_bytes=res['final'].get_booster().save_raw()
        def explain(indices):
            key=hashlib.sha256(np.asarray(indices,dtype=np.int64).tobytes()+model_bytes).hexdigest()[:20]
            cache=work/f'shap-{p.lower()}-{key}.npz'
            if cache.exists():
                contributions=np.load(cache)['contributions']
            else:
                contributions=res['final'].get_booster().predict(xgb.DMatrix(X.iloc[indices][res['cols']],nthread=2),pred_contribs=True)
                np.savez_compressed(cache,contributions=contributions)
            assert np.allclose(contributions.sum(axis=1),v[indices],atol=.003)
            return contributions
        background=annulus_mean(v); support_fraction=annulus_mean(sup.astype(float))
        contrast=v-background
        threshold=report['periods'][p]['hot_cutoff_f']-res['reference']
        eligible=land&sup&(support_fraction>=.5)&(v>=threshold)&(contrast>=.25)
        raster=np.zeros(shape,dtype=bool); raster[grid[:,0],grid[:,1]]=eligible
        labs,count=label(raster); group=labs[grid[:,0],grid[:,1]]
        patches=[]
        for k in range(1,count+1):
            ids=np.flatnonzero(group==k)
            if len(ids)>=3:
                patches.append((float(v[ids].mean()+contrast[ids].mean()),ids))
        patches.sort(key=lambda z:-z[0]); cases=[]; private_cases=[]
        for score,ids in patches:
            if len(cases)==3: break
            center=xy[ids].mean(axis=0)
            if any(np.linalg.norm(center-c['center'])<2000 for c in private_cases): continue
            representative=int(ids[np.argmax(contrast[ids])])
            near=np.asarray(tree.query_ball_point(xy[representative],1500),dtype=int)
            ann=land_ids[near]
            ann=ann[np.linalg.norm(xy[ann]-xy[representative],axis=1)>=500]
            ci=np.concatenate([[representative],ann])
            contributions=explain(ci)
            delta=contributions[0,:-1]-contributions[1:,:-1].mean(axis=0)
            assert np.isclose(delta.sum(),contrast[representative],atol=.003)
            order=np.argsort(np.abs(delta))[::-1][:6]
            case={'name':chr(65+len(cases)),'cells':len(ids),'area_km2':len(ids)*.01,
              'mean_excess_f':float(v[ids].mean()),'mean_local_contrast_f':float(contrast[ids].mean()),
              'example_excess_f':float(v[representative]),'example_local_contrast_f':float(contrast[representative]),
              'drivers':[{'feature':res['cols'][j],'label':friendly(res['cols'][j]),'contribution_f':float(delta[j]),
                 'feature_value':float(X.iloc[representative][res['cols'][j]]) if pd.notna(X.iloc[representative][res['cols'][j]]) else None,
                 'ring_feature_mean':float(X.iloc[ann][res['cols'][j]].mean()) if X.iloc[ann][res['cols'][j]].notna().any() else None} for j in order]}
            case['ring_sensitivity_f']={}
            for inner,outer in [(300,1000),(500,1500),(750,2000)]:
                nn=land_ids[np.asarray(tree.query_ball_point(xy[representative],outer),dtype=int)]
                nn=nn[np.linalg.norm(xy[nn]-xy[representative],axis=1)>=inner]
                case['ring_sensitivity_f'][f'{inner}_{outer}m']=float(v[representative]-v[nn].mean())
            cases.append(case); private_cases.append({'center':center,'ids':ids,'representative':representative,'annulus':ann,'delta':delta})
        cases_all[p]=private_cases
        survey=d['surveys'][p]
        survey_tree=cKDTree(xy[survey['ids']])
        for case,priv in zip(cases,private_cases):
            dist,k=survey_tree.query(xy[priv['representative']])
            if dist<=250:
                case['nearby_survey']={'distance_m':float(dist),
                  'observed_excess_f':float(survey['temperature'][k]-res['reference']),
                  'heldout_excess_f':float(res['pred'][k,0]-res['reference'])}
        evidence['periods'][p]={'candidate_patches':len(patches),'displayed_cases':cases,
          'supported_land_share':float((land&sup).sum()/land.sum()),
          'predicted_excess_supported_p05_p95':[float(x) for x in np.quantile(v[land&sup],[.05,.95])]}
        fig,axes=plt.subplots(1,2,figsize=(12,6.5))
        for ax in axes: basemap(ax,True)
        axes[0].scatter(*xy[land].T,c=v[land],s=2,cmap='RdYlBu_r',vmin=-2,vmax=2,zorder=3,rasterized=True,alpha=.9)
        im=axes[1].scatter(*xy[land&sup].T,c=v[land&sup],s=2,cmap='RdYlBu_r',vmin=-2,vmax=2,zorder=3,rasterized=True)
        for case,priv in zip(cases,private_cases):
            axes[1].annotate(case['name'],priv['center'],ha='center',va='center',fontsize=11,fontweight='bold',color='white',
                bbox=dict(boxstyle='circle,pad=.4',fc='#183e3d',ec='white'),zorder=15)
        axes[0].set_title('Citywide predictions')
        axes[1].set_title('Predictions with survey support')
        fig.suptitle(p+' predicted heat',fontsize=16,y=.96)
        fig.colorbar(im,ax=axes,shrink=.55,pad=.02,label='Predicted excess air temperature (°F)')
        save(fig,'citywide-'+p.lower())
        # Directional global SHAP, only at observed survey cells.
        ids=d['surveys'][p]['ids']; rng=np.random.default_rng(42); selected=rng.choice(ids,min(1500,len(ids)),replace=False)
        contributions=explain(selected)
        contrib=contributions[:,:-1]; rank=np.argsort(np.abs(contrib).mean(axis=0))[-10:]
        if p=='Afternoon':
            evidence['afternoon_global_features']=[res['cols'][j] for j in rank[::-1]]
        evidence['global_features'][p]=[res['cols'][j] for j in rank[::-1]]
        directions=[]
        for j in rank[::-1]:
            vals=X.iloc[selected][res['cols'][j]].to_numpy()
            if np.isfinite(vals).sum()<20: continue
            q1,q3=np.nanquantile(vals,[.25,.75])
            if q1==q3: continue
            difference=float(contrib[vals>=q3,j].mean()-contrib[vals<=q1,j].mean())
            directions.append({'feature':res['cols'][j],'high_minus_low_mean_shap_f':difference})
        evidence['global_directions'][p]=directions
        fig,ax=plt.subplots(figsize=(10,6))
        for i,j in enumerate(rank):
            val=X.iloc[selected][res['cols'][j]].to_numpy(); valid=np.isfinite(val)
            lo,hi=np.nanquantile(val,[.05,.95]) if valid.any() else (0,1)
            colors=np.clip((np.nan_to_num(val,nan=lo)-lo)/max(hi-lo,1e-8),0,1)
            ax.scatter(contrib[:,j],i+rng.uniform(-.25,.25,len(selected)),c=colors,cmap='coolwarm',s=5,alpha=.45,vmin=0,vmax=1,rasterized=True)
        ax.axvline(0,color='#36554e',lw=1); ax.set_yticks(range(10),[friendly(res['cols'][j]) for j in rank]); ax.set(xlabel='SHAP contribution to predicted excess heat (°F)',title=p+' model feature effects')
        from matplotlib.cm import ScalarMappable
        cb=fig.colorbar(ScalarMappable(cmap='coolwarm'),ax=ax,shrink=.6); cb.set_ticks([0,1],labels=['Lower feature value','Higher feature value'])
        fig.tight_layout(); save(fig,'shap-'+p.lower())
        if private_cases:
            fig,axes=plt.subplots(1,len(private_cases),figsize=(5.5*len(private_cases),4.8),squeeze=False)
            for ax,case,priv in zip(axes[0],cases,private_cases):
                order=np.argsort(np.abs(priv['delta']))[-6:]; vals=priv['delta'][order]
                ax.barh(range(6),vals,color=np.where(vals>0,'#b54b2b','#237e92'))
                ax.set_yticks(range(6),[friendly(res['cols'][j]) for j in order],fontsize=9); ax.axvline(0,color='#36554e',lw=.8)
                ax.set_title(f"Island {case['name']}\n{case['example_local_contrast_f']:+.2f}°F vs surrounding ring")
                ax.set_xlabel('Contribution to local contrast (°F)')
            fig.tight_layout(); save(fig,'local-shap-'+p.lower())
    # Cross-period agreement uses shared cells, with each period's own reference.
    sets=[set(s['ids']) for s in d['surveys'].values()]; common=sorted(set.intersection(*sets))
    measured={}
    for p,s in d['surveys'].items(): measured[p]=dict(zip(s['ids'],s['temperature']-models[p]['reference']))
    evidence['shared_cells']=len(common)
    evidence['observed_period_correlations']={p+' / '+q:float(np.corrcoef([measured[p][i] for i in common],[measured[q][i] for i in common])[0,1]) for j,p in enumerate(periods) for q in periods[j+1:]}
    fig,axes=plt.subplots(1,3,figsize=(13,4.3))
    pairs=[('Morning','Afternoon'),('Afternoon','Evening'),('Morning','Evening')]
    for ax,(p,q),color in zip(axes,pairs,COLORS):
        a=np.array([measured[p][i] for i in common]); b=np.array([measured[q][i] for i in common])
        ax.scatter(a,b,s=4,color=color,alpha=.25,rasterized=True)
        ax.axhline(0,color='#72877b',lw=.7); ax.axvline(0,color='#72877b',lw=.7)
        corr=float(np.corrcoef(a,b)[0,1])
        ax.set(xlabel=p+' excess (°F)',ylabel=q+' excess (°F)',title=f'Observed correlation {corr:.2f}')
    fig.suptitle('Cells measured through the day',fontsize=13)
    fig.tight_layout(); save(fig,'period-persistence')
    # Normalized base geometry for the observed-cell interactive explorer.
    origin=xy.min(axis=0)-100; span=float(np.ptp(xy,axis=0).max()+200)
    display_extent=shapely.box(origin[0],origin[1],origin[0]+span,origin[1]+span)
    def normpaths(frame):
        clipped=gpd.clip(frame,display_extent)
        return [np.round((p-origin)/span,5).tolist() for p in lines(clipped)]
    major=roads[roads.CLASS.astype(str).str.contains('HIGHWAY|MAJOR',case=False)]
    display={'base':{'boundary':normpaths(boundary.simplify(150).to_frame('geometry')),
       'water':normpaths(water.simplify(100).to_frame('geometry')),
       'roads':normpaths(major.simplify(100).to_frame('geometry'))},'periods':{}}
    for p,s in d['surveys'].items():
        display['periods'][p]={'points':np.column_stack([np.round((xy[s['ids']]-origin)/span,5),np.round(s['temperature']-models[p]['reference'],3)]).tolist(),
           'reference_f':models[p]['reference']}
    (OUT/'islands-display.json').write_text(json.dumps(display,separators=(',',':')))
    (OUT/'islands-evidence.json').write_text(json.dumps(evidence,indent=2,allow_nan=False)+'\n')
    joblib.dump(cases_all,work/'island-cases.joblib')
    print(json.dumps(evidence,indent=2),flush=True)

if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__); a.add_argument('--source',type=Path,required=True); a.add_argument('--work',type=Path,required=True)
    a.add_argument('--overview-only',action='store_true')
    args=a.parse_args(); main(args.source,args.work,args.overview_only)
