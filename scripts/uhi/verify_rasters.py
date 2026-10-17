"""Check saved final-table features against available rasters, without rewriting them."""
import argparse
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform

from source_io import read_dbf, read_points

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--companion-source", type=Path, help="Earlier transfer, only for missing rasters")
args = parser.parse_args()
root = args.source
df = read_dbf(root / "final_uhi_data.dbf")
xy = read_points(root / "final_uhi_data.shp")
_, first = np.unique(xy, axis=0, return_index=True)
df, xy = df.iloc[first], xy[first]
checks = {}
paths = [("NDVI", "Data/ndvi_output.tif"), ("NDBI", "Data/ndbi_output.tif")]
paths += [(c, f"Data/EJScreen/Rasterized/{c}.tif") for c in
          ["LOWINCPCT", "OVER64PCT", "UNEMPPCT", "PRE1960PCT", "PM25", "PTRAF"]]
for col, rel in paths:
    source = root / rel
    source_root = "finished"
    if not source.exists() and args.companion_source:
        source = args.companion_source / rel
        source_root = "earlier_transfer"
    if not source.exists():
        checks[col] = {"available": False}
        continue
    with rasterio.open(source) as raster:
        coords = list(zip(*transform("EPSG:32614", raster.crs, xy[:, 0], xy[:, 1])))
        values = np.array([sample[0] for sample in raster.sample(coords, masked=True)])
        matches = np.isclose(values, df[col].to_numpy(), rtol=1e-5, atol=1e-5, equal_nan=True)
        checks[col] = {
            "source_root": source_root, "shape": raster.shape, "crs": str(raster.crs),
            "resolution": raster.res, "nodata": raster.nodata,
            "matches_saved_fraction": float(matches.mean()),
            "sample_min": float(np.nanmin(values)), "sample_max": float(np.nanmax(values)),
        }
    print(f"{col}: {matches.sum()} / {len(matches)} matches ({source_root})", flush=True)
output = Path(__file__).resolve().parents[2] / "assets/data/dallas-heat/raster-sample-check.json"
output.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf8")
