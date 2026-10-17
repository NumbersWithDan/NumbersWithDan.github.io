"""Block-size sensitivity and nested spatial retraining on rebuilt features.

Only coordinate-free summaries are public. Fitted models and observation-level
predictions remain in --work outside the website. No historical pickle is loaded.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb

SIZES = [100, 250, 500, 1000, 2000]
BASE = dict(n_estimators=200, learning_rate=.045, min_child_weight=20, reg_alpha=.1,
            subsample=.85, colsample_bytree=.9, max_bin=128, n_jobs=2,
            random_state=42, tree_method="hist", objective="reg:squarederror")
CANDIDATES = [("landscape", 3, 15), ("landscape", 5, 10), ("combined", 3, 15), ("combined", 5, 10), ("combined", 7, 10)]
SPEC = json.loads(json.dumps({"protocol": 1, "base": BASE, "sizes": SIZES,
                             "shifts": [0, .5], "inner_candidates": CANDIDATES,
                             "neighbors": 32, "blend_weights": [.5, .5],
                             "outer_folds": 5, "inner_folds": 3, "stress_buffer_m": 500}))
OUT = Path(__file__).resolve().parents[2] / "assets/data/dallas-heat"


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf8")


def groups(xy, size, shift=0):
    cells = np.floor((xy - xy.min(axis=0) + size * shift) / size).astype(int)
    return np.unique(cells, axis=0, return_inverse=True)[1]


def metrics(y, pred):
    return {"mae_f": float(mean_absolute_error(y, pred)), "r2": float(r2_score(y, pred))}


def fit_xgb(X, y, depth=4, strength=10):
    model = xgb.XGBRegressor(**BASE, max_depth=depth, reg_lambda=strength)
    model.fit(X, y)
    return model


def knn(xy, y, test_xy):
    # Spatial interpolation benchmark, never supplied to the environmental model.
    return KNeighborsRegressor(n_neighbors=32, weights="distance", algorithm="kd_tree", n_jobs=2).fit(xy, y).predict(test_xy)


def residual_structure(xy, residual):
    """Binned pair statistics from sampled anchors, not an independence test."""
    rng = np.random.default_rng(2026)
    anchors = rng.choice(len(xy), min(1600, len(xy)), replace=False)
    tree = cKDTree(xy)
    bins = np.arange(0, 2100, 100)
    pairs = [[] for _ in bins[:-1]]
    for i in anchors:
        neighbors = np.asarray(tree.query_ball_point(xy[i], 2000))
        distance = np.linalg.norm(xy[neighbors] - xy[i], axis=1)
        for j in range(len(bins) - 1):
            local = neighbors[(distance > bins[j]) & (distance <= bins[j + 1])]
            if len(local):
                local = rng.choice(local, min(30, len(local)), replace=False)
                pairs[j].extend([(int(i), int(k)) for k in local])
    rows = []
    for j, records in enumerate(pairs):
        a = np.asarray(records)
        r1, r2 = residual[a[:, 0]], residual[a[:, 1]]
        rows.append({"distance_m": int(bins[j + 1]), "pairs": len(a),
                     "correlation": float(np.corrcoef(r1, r2)[0, 1]),
                     "semivariance": float(np.mean((r1 - r2) ** 2) / 2)})
    crossing = next((rows[i]["distance_m"] for i in range(len(rows) - 2)
                     if all(abs(rows[j]["correlation"]) < .1 for j in range(i, i + 3))), None)
    # Predeclared conservative working scale, never chosen by the highest R².
    chosen = next((s for s in SIZES if crossing is not None and s >= 2 * crossing), 2000)
    return {"bins": rows, "decorrelation_threshold": .1, "consecutive_bins": 3,
            "estimated_decorrelation_m": crossing, "selected_block_m": chosen,
            "selection_rule": "smallest tested block >= twice the first of three consecutive 100m bins with |residual correlation| < 0.1; 2km if no crossing",
            "qualification": "Diagnostic working scale from out-of-fold residuals; not a uniquely optimal block size or independent confirmatory test"}


def tune(X, y, xy, selected_size, physical):
    folds = list(GroupKFold(3).split(X, y, groups(xy, selected_size)))
    scores = []
    for feature_set, depth, strength in CANDIDATES:
        columns = physical if feature_set == "landscape" else X.columns.tolist()
        errors = []
        for train, valid in folds:
            model = fit_xgb(X.iloc[train][columns], y[train], depth, strength)
            errors.append(mean_absolute_error(y[valid], model.predict(X.iloc[valid][columns])))
        scores.append({"feature_set": feature_set, "depth": depth, "reg_lambda": strength,
                       "inner_mae_f": float(np.mean(errors))})
    best = min(scores, key=lambda s: s["inner_mae_f"])
    columns = physical if best["feature_set"] == "landscape" else X.columns.tolist()
    model = fit_xgb(X[columns], y, best["depth"], best["reg_lambda"])
    return model, columns, best, scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    args = parser.parse_args()
    work = args.work
    data = pd.read_pickle(work / "rebuilt.pkl")
    X, y, xy = data["X"], data["y"], data["xy"]
    # Explicitly classify every missing/unknown market category.
    X["MVA_Missing"] = (X[["MVA_" + c for c in "ABCDEFGHI"]].sum(axis=1) == 0).astype(float)
    physical = [c for c in X if c.startswith(("TOA_", "Brightness_", "NDVI", "NDBI", "Distance_"))]
    assert len(X) == len(y) == len(xy) == 19491
    assert len(np.unique(xy, axis=0)) == len(y)
    fingerprint = hashlib.sha256((work / "rebuilt.pkl").read_bytes()).hexdigest()
    reuse_nested = False
    if any(work.glob("blocks-*.npz")):
        previous = json.loads((OUT / "retraining.json").read_text())
        old_spec = previous.get("experiment_spec", {})
        # A larger inner search does not change the fixed-configuration block study.
        assert {k: v for k, v in old_spec.items() if k != "inner_candidates"} == {k: v for k, v in SPEC.items() if k != "inner_candidates"}, "Cached fixed experiment settings differ; use a fresh private work directory"
        reuse_nested = old_spec.get("inner_candidates") == SPEC["inner_candidates"]
    summary = {"observations": len(y), "feature_count": len(X.columns), "source_audit": data["audit"],
               "experiment_spec": SPEC,
               "private_feature_cache_sha256": fingerprint,
               "versions": {"python": sys.version.split()[0], "xgboost": xgb.__version__, "pandas": pd.__version__, "numpy": np.__version__},
               "study_scope": "single-day afternoon air-temperature interpolation within the surveyed footprint; citywide and temporal transfer are not established",
               "block_study": []}
    pred100 = None
    for size in SIZES:
        for shift in [0, .5]:
            cache = work / f"blocks-{size}-{shift}.npz"
            g = groups(xy, size, shift)
            if cache.exists():
                cached = np.load(cache)
                assert str(cached["fingerprint"]) == fingerprint
                predictions, nearest = cached["predictions"], cached["nearest"]
            else:
                predictions = np.full((len(y), 4), np.nan)
                nearest = np.zeros(len(y))
                for fold, (train, test) in enumerate(GroupKFold(5).split(X, y, g), 1):
                    assert not np.intersect1d(g[train], g[test]).size
                    model = fit_xgb(X.iloc[train], y[train])
                    predictions[test, 0] = model.predict(X.iloc[test])
                    predictions[test, 1] = knn(xy[train], y[train], xy[test])
                    predictions[test, 2] = y[train].mean()
                    predictions[test, 3] = .5 * predictions[test, 0] + .5 * predictions[test, 1]
                    nearest[test] = cKDTree(xy[train]).query(xy[test])[0]
                    print(f"Block {size}m shift {shift} fold {fold} complete", flush=True)
                np.savez_compressed(cache, predictions=predictions, nearest=nearest, fingerprint=fingerprint)
            assert np.isfinite(predictions).all()
            record = {"block_m": size, "shift_fraction": shift, "groups": int(len(np.unique(g))),
                      "median_nearest_training_m": float(np.median(nearest)),
                      "p90_nearest_training_m": float(np.quantile(nearest, .9)),
                      **{name: metrics(y, predictions[:, i]) for i, name in enumerate(["xgboost", "spatial_knn", "mean", "blend"])}}
            summary["block_study"].append(record)
            write(OUT / "retraining.json", summary)
            print(json.dumps(record), flush=True)
            if size == 100 and shift == 0:
                pred100 = predictions[:, 0]
    summary["residual_structure"] = residual_structure(xy, y - pred100)
    size = summary["residual_structure"]["selected_block_m"]
    write(OUT / "retraining.json", summary)
    print("Selected diagnostic block size:", size, flush=True)

    # Nested spatial model/feature selection at the selected working scale.
    cache = work / "nested.npz"
    settings_path = work / "nested-settings.json"
    if cache.exists() and reuse_nested:
        stored = np.load(cache)
        assert str(stored["fingerprint"]) == fingerprint and int(stored["size"]) == size
        nested_pred = stored["predictions"]
        settings = json.loads(settings_path.read_text())
    else:
        nested_pred = np.full((len(y), 4), np.nan)
        settings = []
        for fold, (train, test) in enumerate(GroupKFold(5).split(X, y, groups(xy, size)), 1):
            model, columns, best, scores = tune(X.iloc[train], y[train], xy[train], size, physical)
            nested_pred[test, 0] = model.predict(X.iloc[test][columns])
            nested_pred[test, 1] = knn(xy[train], y[train], xy[test])
            nested_pred[test, 2] = y[train].mean()
            nested_pred[test, 3] = .5 * nested_pred[test, 0] + .5 * nested_pred[test, 1]
            settings.append({"fold": fold, "selected": best, "search": scores})
            print("Nested fold", fold, best, metrics(y[test], nested_pred[test, 0]), flush=True)
        np.savez_compressed(cache, predictions=nested_pred, fingerprint=fingerprint, size=size)
        write(settings_path, settings)
    summary["nested_spatial"] = {"block_m": size, "outer_folds": 5, "inner_folds": 3,
                                 "configs_per_inner_search": len(CANDIDATES), "fold_selection": settings,
                                 **{name: metrics(y, nested_pred[:, i]) for i, name in enumerate(["xgboost", "spatial_knn", "mean", "blend"])}}
    write(OUT / "retraining.json", summary)

    # Geographic transfer stress test. Regions and buffer fixed before fitting.
    region = pd.qcut(xy[:, 1], 5, labels=False)
    transfer_cache = work / "transfer.npz"
    if transfer_cache.exists():
        stored = np.load(transfer_cache)
        assert str(stored["fingerprint"]) == fingerprint
        transfer = stored["predictions"]
    else:
        transfer = np.full((len(y), 3), np.nan)
        for i in range(5):
            test = np.flatnonzero(region == i)
            train = np.flatnonzero(region != i)
            train = train[cKDTree(xy[test]).query(xy[train])[0] >= 500]
            assert cKDTree(xy[test]).query(xy[train])[0].min() >= 500
            # Fixed shallow configuration: no selection using this stress-test score.
            model = fit_xgb(X.iloc[train], y[train], depth=3, strength=15)
            transfer[test, 0] = model.predict(X.iloc[test])
            transfer[test, 1] = knn(xy[train], y[train], xy[test])
            transfer[test, 2] = y[train].mean()
            print("Buffered region", i + 1, metrics(y[test], transfer[test, 0]), flush=True)
        np.savez_compressed(transfer_cache, predictions=transfer, fingerprint=fingerprint)
    summary["buffered_region_stress_test"] = {"buffer_m": 500, "regions": 5,
                                             **{name: metrics(y, transfer[:, i]) for i, name in enumerate(["xgboost", "spatial_knn", "mean"])}}
    # Full-data selection is for the final exploratory artifact, not evaluation.
    model, columns, best, scores = tune(X, y, xy, size, physical)
    spatial_model = KNeighborsRegressor(n_neighbors=32, weights="distance", algorithm="kd_tree", n_jobs=2).fit(xy, y)
    joblib.dump({"model": model, "columns": columns, "spatial_model": spatial_model,
                 "fixed_blend_weights": [.5, .5], "block_m": size, "source_sha256": fingerprint}, work / "retrained-model.joblib")
    model.get_booster().save_model(work / "retrained-model.ubj")
    summary["final_fit"] = {"selected": best, "columns": columns, "search": scores,
                            "qualification": "fit on all survey observations for exploratory explanations; performance comes from outer held-out predictions"}
    sample = np.sort(np.random.default_rng(42).choice(len(y), 1024, replace=False))
    contributions = model.get_booster().predict(xgb.DMatrix(X.iloc[sample][columns]), pred_contribs=True)
    assert np.allclose(contributions.sum(axis=1), model.predict(X.iloc[sample][columns]), atol=1e-3)
    importance = np.abs(contributions[:, :-1]).mean(axis=0)
    summary["shap"] = {"method": "tree-path", "sample_n": 1024, "seed": 42, "units": "degrees Fahrenheit", "additivity_checked": True,
                       "importance": sorted([{"feature": c, "mean_absolute_contribution_f": float(v)} for c, v in zip(columns, importance)], key=lambda d: -d["mean_absolute_contribution_f"])}
    write(OUT / "retraining.json", summary)
    print("Retraining complete.", flush=True)


if __name__ == "__main__":
    main()
