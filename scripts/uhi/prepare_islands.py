"""Prepare independent period labels and identical citywide environmental features.

All geographic caches stay in a private --work directory outside the website.
Raw observations, not the supplied modeled temperature rasters, supply labels.
"""
import argparse
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from rebuild_features import build, digest


def source_signature(root,work):
    # Version every feature input, including shapefile sidecars and boundary.
    source_paths=[p for p in (root/'Data').rglob('*') if p.is_file() and
        (p.suffix.lower() in ['.shp','.shx','.dbf','.prj','.tif','.txt'] or '.gdb' in str(p.parent))
        and not p.name.endswith('.lock')]
    return {'sources':{str(p.relative_to(root)):digest(p) for p in sorted(source_paths)},
      'boundary':digest(work/'city-boundary.geojson'),
      'extractor':digest(Path(__file__).with_name('rebuild_features.py')),'grid_m':100}

def main(root, work):
    if work.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError('The geographic work directory must be outside the public website')
    work.mkdir(exist_ok=True, parents=True)
    signature=source_signature(root,work)
    signature_path=work/'feature-signature.json'
    if signature_path.exists():
        assert json.loads(signature_path.read_text())==signature, 'Feature sources or code changed; choose a fresh work directory'
    else:
        assert not (work/'rebuilt.pkl').exists(), 'Unversioned feature cache; choose a fresh work directory'
        signature_path.write_text(json.dumps(signature,indent=2))
    boundary = gpd.read_file(work / "city-boundary.geojson").to_crs(32614)
    city = boundary.geometry.make_valid().union_all()
    bounds = city.bounds
    xx, yy = np.meshgrid(np.arange(np.floor(bounds[0]/100)*100+50, bounds[2], 100),
                         np.arange(np.floor(bounds[1]/100)*100+50, bounds[3], 100))
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    city_xy = xy[shapely.contains_xy(city, *xy.T)]
    surveys = {}; raw_counts = {}; hashes = {}
    for period, code in [("Morning", "am"), ("Afternoon", "af"), ("Evening", "pm")]:
        p = root / f"Data/UHI/Shape files/{code}_trav.shp"
        g = gpd.read_file(p).to_crs(32614)
        assert g.t_f.notna().all()
        keys = np.floor(np.column_stack([g.geometry.x, g.geometry.y])/100).astype(int)
        frame = pd.DataFrame({"cx":keys[:,0], "cy":keys[:,1], "temperature":g.t_f.to_numpy()})
        surveys[period] = frame.groupby(["cx", "cy"], as_index=False).temperature.agg(["mean", "size"]).reset_index()
        raw_counts[period] = len(g)
        hashes[code] = {ext: digest(p.with_suffix(ext)) for ext in [".shp", ".dbf", ".prj"]}
    city_keys = np.floor(city_xy/100).astype(int)
    all_keys = np.unique(np.concatenate([city_keys] + [s[["cx","cy"]].to_numpy() for s in surveys.values()]),axis=0)
    positions = {tuple(k):i for i,k in enumerate(all_keys)}
    coords = all_keys*100+50
    points = gpd.GeoDataFrame({"t_f":np.zeros(len(coords))}, geometry=gpd.points_from_xy(*coords.T), crs=32614)
    if not (work / "rebuilt.pkl").exists():
        build(root, work, points_input=points, extended=True)
    data = pd.read_pickle(work / "rebuilt.pkl")
    assert np.array_equal(data["xy"], coords)
    data["city_mask"] = np.array([shapely.contains_xy(city, *p) for p in coords])
    data["water_mask"] = np.zeros(len(coords),dtype=bool)
    lakes = gpd.read_file(root / "Data/Water/Lakes/WaterFeatures.shp").to_crs(32614)
    lake_union = lakes.geometry.make_valid().union_all()
    data["water_mask"] = shapely.contains_xy(lake_union, *coords.T)
    data["surveys"] = {}
    for period, s in surveys.items():
        ids = np.array([positions[tuple(k)] for k in s[["cx","cy"]].to_numpy()])
        data["surveys"][period] = {"ids":ids, "temperature":s["mean"].to_numpy(), "readings_per_cell":s["size"].to_numpy()}
    data["raw_counts"] = raw_counts
    data["label_hashes"] = hashes
    pd.to_pickle(data, work / "islands.pkl")
    print(json.dumps({"feature_locations":len(coords),"city_cells":int(data['city_mask'].sum()),
                      "survey_cells":{p:len(s['ids']) for p,s in data['surveys'].items()},
                      "canopy_nonmissing":int(data['X'].tree_canopy_fraction.notna().sum()),
                      "features":len(data['X'].columns)},indent=2),flush=True)

if __name__ == "__main__":
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument("--source",type=Path,required=True); a.add_argument("--work",type=Path,required=True)
    args=a.parse_args(); main(args.source,args.work)
