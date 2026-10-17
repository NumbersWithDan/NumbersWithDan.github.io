"""Verify the finished Dallas project without executing the original notebooks.

Requires the author's original data and trusted joblib model. See README.md.
All generated files are written to this website; source files are read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from scipy.spatial import cKDTree
import sklearn
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
import xgboost as xgb

from source_io import read_dbf, read_points


REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets/data/dallas-heat"
IMG = REPO / "images/dallas-heat"
RENAMES = {
    "Distance_t": "Distance_to_Park",
    "Distance_1": "Distance_to_Lake",
    "Distance_2": "Distance_to_River",
    **{f"MVAClust_{i}": f"MVACluster_{letter}" for i, letter in enumerate("ABCDEFGHI", 1)},
    "MVAClust10": "MVACluster_NA",
}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf8")


def hash_file(path):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256")
    return digest.hexdigest()


def evaluate(X, y, weights, splits, params, output_path=None, scale_target=True):
    """No fitted model crosses folds; target scaling is fitted on training rows."""
    folds = []
    prediction = np.full(len(y), np.nan)
    for fold, (train, test) in enumerate(splits, 1):
        mu = float(np.average(y[train], weights=weights[train]))
        sd = float(np.sqrt(np.average((y[train] - mu) ** 2, weights=weights[train]))) if scale_target else 1.0
        model = xgb.XGBRegressor(**params)
        model.fit(X.iloc[train], (y[train] - mu) / sd, sample_weight=weights[train])
        pred = model.predict(X.iloc[test]) * sd + mu
        prediction[test] = pred
        baseline = np.full(len(test), mu)
        result = {
            "fold": fold, "train_rows": len(train), "test_rows": len(test),
            "mae_f": float(mean_absolute_error(y[test], pred, sample_weight=weights[test])),
            "r2": float(r2_score(y[test], pred, sample_weight=weights[test])),
            "mean_baseline_mae_f": float(mean_absolute_error(y[test], baseline, sample_weight=weights[test])),
        }
        folds.append(result)
        print(json.dumps(result), flush=True)
    summary = {
        "folds": folds,
        "mean_fold_mae_f": float(np.mean([f["mae_f"] for f in folds])),
        "mean_fold_r2": float(np.mean([f["r2"] for f in folds])),
        "mean_fold_baseline_mae_f": float(np.mean([f["mean_baseline_mae_f"] for f in folds])),
        "pooled_mae_f": float(mean_absolute_error(y, prediction, sample_weight=weights)),
        "pooled_r2": float(r2_score(y, prediction, sample_weight=weights)),
    }
    if output_path:
        write_json(output_path, summary)
    return summary, prediction


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reuse-validation", action="store_true",
                        help="Reuse completed scores only when input hashes and model settings match")
    args = parser.parse_args()
    root = args.source
    OUT.mkdir(parents=True, exist_ok=True)
    IMG.mkdir(parents=True, exist_ok=True)
    previous_path = OUT / "final-verification.json"
    previous = json.loads(previous_path.read_text(encoding="utf8")) if args.reuse_validation and previous_path.exists() else {}
    final_path = root / "final_uhi_data"
    raw_path = root / "Data/UHI/Shape files/af_trav"
    df = read_dbf(final_path.with_suffix(".dbf")).rename(columns=RENAMES)
    xy = read_points(final_path.with_suffix(".shp"))
    raw = read_dbf(raw_path.with_suffix(".dbf"))
    raw_xy = read_points(raw_path.with_suffix(".shp"))
    assert len(df) == len(xy) and len(raw) == len(raw_xy)
    crs = CRS.from_wkt(final_path.with_suffix(".prj").read_text())
    assert crs.is_projected and crs.axis_info[0].unit_name == "metre"
    transformer = Transformer.from_crs(4326, crs, always_xy=True)
    projected_raw = np.column_stack(transformer.transform(raw_xy[:, 0], raw_xy[:, 1]))
    distance, source_id = cKDTree(projected_raw).query(xy)
    assert distance.max() < 0.01, "Final points do not align with original observations"
    assert np.array_equal(df.t_f.to_numpy(), raw.t_f.to_numpy()[source_id])
    assert len(np.unique(source_id)) == len(raw)
    counts = np.bincount(source_id, minlength=len(raw))
    weights = 1.0 / counts[source_id]
    # Duplicated rows vary in MVA attributes but must have identical physical observations.
    fixed = ["t_f", "rh", "hi_f", "NDVI", "NDBI", "Distance_to_Park", "Distance_to_Lake", "Distance_to_River"]
    assert (df.groupby(source_id)[fixed].nunique(dropna=False) <= 1).all().all()
    # First occurrence is safe only for the proven identical physical features below.
    first = np.array([np.flatnonzero(source_id == i)[0] for i in range(len(raw))])
    observations = df.iloc[first].reset_index(drop=True)
    original_model_path = root / "Models/final_uhi_model.joblib"
    # Only load the author's trusted model file, never a downloaded third-party pickle.
    saved = joblib.load(original_model_path)
    feature_names = saved.get_booster().feature_names
    assert len(feature_names) == len(set(feature_names))
    X = df[feature_names].astype(float)
    assert not {"t_f", "rh", "hi_f", "t_f_normalized"}.intersection(feature_names)
    y = df.t_f.to_numpy()
    params = saved.get_params()
    params.update(n_jobs=2, tree_method="hist")
    params = {k: v for k, v in params.items() if v is not None}
    params = {k: (v.item() if isinstance(v, np.generic) else v) for k, v in params.items()}
    params = {k: v for k, v in params.items() if not (isinstance(v, float) and np.isnan(v))}
    result = {
        "scope": "Finished training-table and saved-model review; not complete raw-feature reconstruction",
        "original_observations": len(raw), "final_training_rows": len(df),
        "extra_join_rows": len(df) - len(raw), "observations_with_multiple_matches": int((counts > 1).sum()),
        "max_matches": int(counts.max()), "max_coordinate_alignment_error_m": float(distance.max()),
        "targets_match_original_by_location": True, "feature_names": feature_names,
        "missing_by_feature": {k: int(v) for k, v in X.isna().sum().items()},
        "saved_model_parameters": params,
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__,
                     "sklearn": sklearn.__version__, "xgboost": xgb.__version__, "matplotlib": matplotlib.__version__},
        "sha256": {str(p.relative_to(root)): hash_file(p) for p in [
            final_path.with_suffix(".dbf"), final_path.with_suffix(".shp"), final_path.with_suffix(".prj"),
            raw_path.with_suffix(".dbf"), raw_path.with_suffix(".shp"), original_model_path,
            root / "UHI Model Final-model comps.ipynb", root / ".ipynb_checkpoints/Full City Prediction-checkpoint.ipynb",
            root / "Final Report_Final.docx"]},
    }
    saved_z = saved.predict(X)
    # Original notebook uses a z score over the joined (not unique-observation) table.
    saved_f = saved_z * y.std(ddof=1) + y.mean()
    result["saved_model_in_sample_only"] = {
        "r2": float(r2_score(y, saved_f)), "mae_f": float(mean_absolute_error(y, saved_f)),
        "warning": "Training fit only, not held-out test performance",
    }
    reuse_validation = (previous.get("sha256") == result["sha256"]
                        and previous.get("saved_model_parameters") == params)
    write_json(OUT / "final-verification.json", result)

    # Reproduce the centimeter-grid evaluation; no scaling or weighting corrections here.
    legacy = ((xy[:, 0] // 0.01).astype(np.int64) * 1000 + (xy[:, 1] // 0.01).astype(np.int64))
    result["legacy_group_count"] = len(np.unique(legacy))
    legacy_splits = list(GroupKFold(10).split(X, y, legacy))
    legacy_folds = []
    z = (y - y.mean()) / y.std(ddof=1)
    if reuse_validation and "legacy_reproduction" in previous:
        legacy_folds = previous["legacy_reproduction"]["fold_r2"]
        print("Reusing verified legacy folds (identical source hashes and settings).", flush=True)
    else:
        for fold, (train, test) in enumerate(legacy_splits, 1):
            model = xgb.XGBRegressor(**params)
            model.fit(X.iloc[train], z[train])
            score = float(r2_score(z[test], model.predict(X.iloc[test])))
            legacy_folds.append(score)
            print(f"legacy fold {fold}: R2={score:.4f}", flush=True)
    result["legacy_reproduction"] = {"fold_r2": legacy_folds, "mean_fold_r2": float(np.mean(legacy_folds)),
                                     "notebook_saved_mean_r2": 0.906482975153126}
    write_json(OUT / "final-verification.json", result)

    # Corrected diagnostics inherit historically tuned parameters, so are sensitivity checks,
    # not a fresh independent estimate from nested hyperparameter selection.
    unique_splits = KFold(5, shuffle=True, random_state=42).split(raw)
    random_splits = [(np.flatnonzero(np.isin(source_id, train)), np.flatnonzero(np.isin(source_id, test)))
                     for train, test in unique_splits]
    cell_xy = np.floor((xy - xy.min(axis=0)) / 2000).astype(int)
    grid_groups = np.unique(cell_xy, axis=0, return_inverse=True)[1]
    result["grid_group_count"] = len(np.unique(grid_groups))
    spatial_splits = list(GroupKFold(5).split(X, y, grid_groups))
    # Contiguous north-south strips, 500m gap between all training and test points.
    bands_raw = pd.qcut(projected_raw[:, 1], 5, labels=False).astype(int)
    band_groups = bands_raw[source_id]
    buffered_splits = []
    for region in range(5):
        test = np.flatnonzero(band_groups == region)
        candidates = np.flatnonzero(band_groups != region)
        nearest = cKDTree(xy[test]).query(xy[candidates])[0]
        train = candidates[nearest >= 500]
        buffered_splits.append((train, test))
    all_predictions = {}
    for name, splits in [("random_observation_groups", random_splits), ("spatial_2km", spatial_splits),
                         ("buffered_north_south_regions", buffered_splits)]:
        print(f"EVALUATING {name}", flush=True)
        for train, test in splits:
            assert not set(source_id[train]).intersection(source_id[test])
        cache_path = OUT / f"{name}-predictions.npz"
        if reuse_validation and name in previous and (name != "spatial_2km" or cache_path.exists()):
            result[name] = previous[name]
            if cache_path.exists():
                all_predictions[name] = np.load(cache_path)["predictions"]
            print(f"Reusing {name} scores (identical source hashes and settings).", flush=True)
        else:
            result[name], all_predictions[name] = evaluate(X, y, weights, splits, params)
            np.savez_compressed(cache_path, predictions=all_predictions[name], source_id=source_id)
        write_json(OUT / "final-verification.json", result)

    # Compute deterministic tree-path SHAP from the saved model on unique observations.
    # This explains the historical fitted model; it is not the interventional-background
    # SHAP variant used in the notebook and must not be presented as identical.
    # A fixed sample makes exact tree-path explanations fast enough to repeat.
    sample = np.sort(np.random.default_rng(42).choice(len(first), min(1024, len(first)), replace=False))
    unique_X = X.iloc[first[sample]]
    saved.get_booster().set_param({"nthread": 2})
    contributions = saved.get_booster().predict(xgb.DMatrix(unique_X), pred_contribs=True)
    assert np.allclose(contributions.sum(axis=1), saved.predict(unique_X), atol=1e-5)
    importance = np.abs(contributions[:, :-1]).mean(axis=0)
    ordering = np.argsort(importance)[::-1]
    result["tree_path_shap_mean_abs"] = [{"feature": feature_names[i], "value_z": float(importance[i])}
                                        for i in ordering]
    result["tree_path_shap_additivity_verified"] = True
    result["tree_path_shap_sample"] = {"n": len(sample), "seed": 42, "market_match": "first stored match per original observation"}
    write_json(OUT / "final-verification.json", result)
    plots(result, xy[first], y[first], all_predictions, source_id, counts)
    print("Finished verification and figures.", flush=True)


def plots(result, unique_xy, y, all_predictions, source_id, counts):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": "#f6fcf8", "axes.facecolor": "#f6fcf8",
                         "text.color": "#163e3d", "axes.labelcolor": "#163e3d"})
    schemes = ["random_observation_groups", "spatial_2km", "buffered_north_south_regions"]
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, name in enumerate(schemes):
        model = [f["mae_f"] for f in result[name]["folds"]]
        base = [f["mean_baseline_mae_f"] for f in result[name]["folds"]]
        ax.scatter(np.full(5, i) - .12, model, color="#b34822", label="XGBoost folds" if i == 0 else None)
        ax.scatter(np.full(5, i) + .12, base, marker="x", color="#16786d", label="Mean baseline folds" if i == 0 else None)
        ax.plot([i - .25, i], [np.mean(model)] * 2, color="#b34822", lw=3)
        ax.plot([i, i + .25], [np.mean(base)] * 2, color="#16786d", lw=3)
    ax.set_xticks(range(3), ["Random locations", "2 km grid groups", "Regions + 500 m buffer"])
    ax.set(ylabel="Held-out mean absolute error (°F)", title="The same model settings face three different tests")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=.2)
    fig.savefig(IMG / "finished-validation.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    shap_data = result["tree_path_shap_mean_abs"][:10][::-1]
    labels = {"PM25": "PM2.5 indicator", "Distance_to_Lake": "Distance to lake", "Distance_to_River": "Distance to stream",
              "Distance_to_Park": "Distance to park", "LOWINCPCT": "Low-income share", "OVER64PCT": "Age 65+ share",
              "PRE1960PCT": "Pre-1960 housing share", "PTRAF": "Traffic proximity indicator", "NDVI": "Vegetation index (NDVI)"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([labels.get(d["feature"], d["feature"].replace("_", " ")) for d in shap_data],
            [d["value_z"] for d in shap_data], color="#16786d")
    ax.set(xlabel="Mean absolute tree-path SHAP contribution (standardized target)",
           title="What the saved model relies on")
    fig.savefig(IMG / "finished-model-explanations.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
    pred = np.bincount(source_id, weights=all_predictions["spatial_2km"]) / counts
    error = y - pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    offsets = (unique_xy - unique_xy.min(axis=0)) / 1000
    points = None
    for ax, values, title in [(axes[0], y, "Observed afternoon temperature"),
                              (axes[1], error, "Held-out residual: observed − predicted")]:
        residual = ax is axes[1]
        points = ax.scatter(offsets[:, 0], offsets[:, 1], c=values, s=2,
                            cmap="RdBu_r" if residual else "inferno", vmin=-3 if residual else 101,
                            vmax=3 if residual else 110, rasterized=True)
        ax.set_aspect("equal")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(points, ax=ax, shrink=.8, label="°F")
    axes[0].set_ylabel("")
    fig.tight_layout()
    fig.savefig(IMG / "finished-held-out-residuals.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
