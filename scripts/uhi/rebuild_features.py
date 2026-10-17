"""Rebuild one row per survey observation from authoritative local source files.

Source data are read-only. Private caches, models, and projected coordinates go to
--work, outside the public website. See README.md for the complete workflow.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import pyogrio
from pyproj import Transformer
import rasterio
from rasterio.windows import from_bounds, Window
from scipy.ndimage import uniform_filter
import shapely

EJ = ["LOWINCPCT", "OVER64PCT", "UNEMPPCT", "PRE1960PCT", "PM25", "PTRAF"]
MVA = ["MdSalesPr", "CoefVar", "PHHOO", "VptResVac", "PPermNCUnt", "PRehabPerm", "PFclOO1517", "PCVLnRsPr", "PPubSubAll", "PAPerm"]


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def join_once(points, polygons, columns, label, audit):
    """No buffer or nearest-area imputation. Smallest area wins boundary ties."""
    polygons = polygons.to_crs(points.crs).reset_index(drop=True)
    invalid = ~polygons.geometry.is_valid
    audit[label + "_repaired_geometries"] = int(invalid.sum())
    polygons.loc[invalid, "geometry"] = polygons.loc[invalid].geometry.make_valid()
    polygons["_polygon_id"] = np.arange(len(polygons))
    polygons["_area"] = polygons.geometry.area
    pairs = gpd.sjoin(points[["geometry"]], polygons[columns + ["_area", "_polygon_id", "geometry"]],
                      how="left", predicate="intersects")
    audit[label + "_multiple_matches"] = int((pairs.groupby(level=0).size() > 1).sum())
    pairs["_observation"] = pairs.index
    chosen = pairs.sort_values(["_observation", "_area", "_polygon_id"], na_position="last").drop_duplicates("_observation").set_index("_observation").reindex(points.index)
    assert len(chosen) == len(points) and chosen.index.is_unique
    audit[label + "_unmatched"] = int(chosen.index_right.isna().sum())
    return chosen[columns]


def build(root, work):
    work.mkdir(parents=True, exist_ok=True)
    audit = {"sources": {}, "observation_policy": "one original afternoon survey record per row; no buffered neighborhood joins"}
    def track(path):
        p = root / path
        audit["sources"][str(path)] = digest(p)
        return p
    raw_path = track("Data/UHI/Shape files/af_trav.shp")
    track("Data/UHI/Shape files/af_trav.dbf")
    points = gpd.read_file(raw_path).to_crs(32614).reset_index(drop=True)
    assert points.geometry.geom_type.eq("Point").all() and points.t_f.notna().all()
    audit["observations"] = len(points)
    assert not points.geometry.duplicated().any()
    xy = np.column_stack([points.geometry.x, points.geometry.y])
    X = pd.DataFrame(index=points.index)

    # Work only on the survey extent plus 360m for neighborhood summaries.
    band_dir = root / "Data/LandSAT ARD/Landsat 8-9"
    metadata_path = track("Data/LandSAT ARD/Landsat 8-9/MetaData/LC09_L1TP_027037_20231006_20231006_02_T1_MTL.txt")
    metadata = dict(re.findall(r"^\s*([A-Z0-9_]+)\s*=\s*([^\r\n]+)", metadata_path.read_text(), re.M))
    audit["imagery_date"] = metadata["DATE_ACQUIRED"].strip('" ')
    scene = "LC09_L1TP_027037_20231006_20231006_02_T1_"
    with rasterio.open(band_dir / (scene + "B4.TIF")) as ref:
        assert ref.crs.to_epsg() == 32614
        bounds = points.total_bounds + np.array([-360, -360, 360, 360])
        win0 = from_bounds(*bounds, transform=ref.transform)
        window = Window(int(np.floor(win0.col_off)), int(np.floor(win0.row_off)),
                        int(np.ceil(win0.width)) + 2, int(np.ceil(win0.height)) + 2)
        transform = ref.window_transform(window)
        rows, cols = rasterio.transform.rowcol(transform, xy[:, 0], xy[:, 1])
        rows, cols = np.array(rows), np.array(cols)
        reference_transform, reference_shape = ref.transform, ref.shape
    def read(name, folder=band_dir):
        path = folder / (scene + name + ".TIF")
        track(path.relative_to(root))
        with rasterio.open(path) as src:
            assert src.transform == reference_transform and src.shape == reference_shape
            assert src.crs.to_epsg() == 32614
            return src.read(1, window=window)
    qa = read("QA_PIXEL", band_dir / "MetaData")
    saturation = read("QA_RADSAT", band_dir / "MetaData")
    # Landsat 8/9 QA_PIXEL: fill, dilated cloud, cirrus, cloud, shadow, snow.
    clear = ((qa & 63) == 0) & (saturation == 0)
    audit["qa_clear_observations"] = int(clear[rows, cols].sum())
    bands = {}
    for b in [1, 2, 3, 4, 5, 6, 7, 10, 11]:
        dn = read("B" + str(b)).astype(float)
        valid = clear & (dn > 0)
        if b < 10:
            values = (float(metadata[f"REFLECTANCE_MULT_BAND_{b}"]) * dn + float(metadata[f"REFLECTANCE_ADD_BAND_{b}"])) / np.sin(np.deg2rad(float(metadata["SUN_ELEVATION"])))
            values[~valid | (values < 0)] = np.nan
            name = f"TOA_B{b}"
        else:
            radiance = float(metadata[f"RADIANCE_MULT_BAND_{b}"]) * dn + float(metadata[f"RADIANCE_ADD_BAND_{b}"])
            values = float(metadata[f"K2_CONSTANT_BAND_{b}"]) / np.log1p(float(metadata[f"K1_CONSTANT_BAND_{b}"]) / np.maximum(radiance, 1e-12))
            values[~valid | (radiance <= 0)] = np.nan
            name = f"Brightness_B{b}_K"
        bands[b] = values
        X[name] = values[rows, cols]
    for name, a, b in [("NDVI", 5, 4), ("NDBI", 6, 5)]:
        denominator = bands[a] + bands[b]
        values = np.divide(bands[a] - bands[b], denominator, out=np.full_like(denominator, np.nan), where=denominator > 1e-6)
        assert np.nanmin(values) >= -1.000001 and np.nanmax(values) <= 1.000001
        X[name] = values[rows, cols]
        for size in [7, 21]:
            valid = np.isfinite(values)
            count = uniform_filter(valid.astype(float), size=size, mode="constant")
            total = uniform_filter(np.nan_to_num(values), size=size, mode="constant")
            mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count >= .5)
            # Square windows approximately 210m and 630m wide on a 30m grid.
            X[f"{name}_mean_{size * 30}m"] = mean[rows, cols]
    print("Satellite features rebuilt with QA masks and metadata calibration.", flush=True)

    # Distances from original geometry, with explicit topology repair counts.
    for name, rel in [("Park", "Data/Dallas City Data/Parks/Parks.shp"),
                      ("Lake", "Data/Water/Lakes/WaterFeatures.shp"),
                      ("Stream", "Data/Water/Streams/WaterFeatures.shp")]:
        track(rel.replace(".shp", ".prj"))
        shapes = gpd.read_file(track(rel)).to_crs(points.crs)
        bad = ~shapes.geometry.is_valid
        audit[name + "_geometry_repairs"] = int(bad.sum())
        shapes.loc[bad, "geometry"] = shapes.loc[bad].geometry.make_valid()
        tree = shapely.STRtree(shapes.geometry.to_numpy())
        _, distances = tree.query_nearest(points.geometry.to_numpy(), all_matches=False, return_distance=True)
        assert len(distances) == len(points)
        X["Distance_to_" + name] = distances

    # Read source EJScreen polygons inside the survey extent. No rasterization.
    ejpath = root / "Data/EJScreen/EJSCREEN_2023_BG_with_AS_CNMI_GU_VI.gdb"
    bounds = points.to_crs(3857).total_bounds
    ej = pyogrio.read_dataframe(ejpath, layer="EJSCREEN_Full_with_AS_CNMI_GU_VI", bbox=tuple(bounds), columns=EJ + ["ID"])
    assert len(ej) > 0
    audit["EJScreen_source_polygons"] = len(ej)
    # Hash all GDB components so reconstruction inputs are identifiable.
    for p in sorted(ejpath.glob("*")):
        if p.is_file() and p.suffix not in [".lock", ".lnk"]:
            audit["sources"][str(p.relative_to(root))] = digest(p)
    ej_values = join_once(points, ej, EJ, "EJScreen", audit)
    for c in EJ:
        ej_values[c] = pd.to_numeric(ej_values[c], errors="coerce")
        ej_values.loc[ej_values[c] < 0, c] = np.nan
    X = pd.concat([X, ej_values], axis=1)
    mva = gpd.read_file(track("Data/MAV/Model_ResPar/Model_ResPar.shp"))
    track("Data/MAV/Model_ResPar/Model_ResPar.dbf")
    joined = join_once(points, mva, MVA + ["MVACluster"], "MVA", audit)
    X = pd.concat([X, joined[MVA].apply(pd.to_numeric, errors="coerce")], axis=1)
    categories = joined.MVACluster.where(joined.MVACluster.isin(list("ABCDEFGHI")), "Missing")
    for label in list("ABCDEFGHI") + ["Missing"]:
        X["MVA_" + label] = (categories == label).astype(float)
    assert len(X) == len(points) and X.index.is_unique
    assert not np.isinf(X.to_numpy()).any()
    audit["features"] = X.columns.tolist()
    audit["missing"] = X.isna().sum().astype(int).to_dict()
    # This cache is PRIVATE and contains projected coordinates used for validation.
    pd.to_pickle({"X": X, "y": points.t_f.to_numpy(), "xy": xy, "audit": audit}, work / "rebuilt.pkl")
    (work / "source-audit.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf8")
    print(json.dumps({k: v for k, v in audit.items() if k != "sources"}, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    args = parser.parse_args()
    build(args.source, args.work)
