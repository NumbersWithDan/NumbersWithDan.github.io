"""Recompute final-table park/lake distance features from available polygons."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from pyproj import CRS, Transformer
import shapefile
import shapely
from shapely.ops import transform

from source_io import read_dbf, read_points

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
args = parser.parse_args()
root = args.source
df = read_dbf(root / "final_uhi_data.dbf")
xy = read_points(root / "final_uhi_data.shp")
_, first = np.unique(xy, axis=0, return_index=True)
df, xy = df.iloc[first], xy[first]
target_crs = CRS.from_wkt((root / "final_uhi_data.prj").read_text())
checks = {}
for column, rel in [("Distance_t", "Data/Dallas City Data/Parks/Parks.shp"),
                    ("Distance_1", "Data/Water/Lakes/WaterFeatures.shp")]:
    path = root / rel
    source_crs = CRS.from_wkt(path.with_suffix(".prj").read_text())
    converter = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    # A geometry-only Reader does not need the missing DBF/SHX sidecars.
    reader = shapefile.Reader(shp=str(path))
    polygons = [transform(converter.transform, shapely.geometry.shape(shape.__geo_interface__))
                for shape in reader.iterShapes()]
    invalid_count = sum(not shape.is_valid for shape in polygons)
    # Match GeoPandas' original distance(...).min() behavior without changing
    # invalid polygons or requiring a topology-dependent union operation.
    geometries = np.asarray(polygons, dtype=object)
    points = shapely.points(xy)
    distances = np.concatenate([
        np.nanmin(shapely.distance(points[i:i + 256, None], geometries[None, :]), axis=1)
        for i in range(0, len(points), 256)
    ])
    expected = df[column].to_numpy()
    difference = np.abs(distances - expected)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    checks[column] = {"source": rel, "source_sha256": digest, "polygon_count": len(polygons),
                      "invalid_geometry_count": invalid_count, "geometry_repaired": False,
                      "observations": len(xy), "unit": "meters", "median_absolute_difference_m": float(np.median(difference)),
                      "max_absolute_difference_m": float(difference.max()),
                      "matches_within_1cm": int((difference <= .01).sum())}
    print(json.dumps(checks[column]), flush=True)
output = Path(__file__).resolve().parents[2] / "assets/data/dallas-heat/distance-check.json"
output.write_text(json.dumps(checks, indent=2) + "\n", encoding="utf8")
