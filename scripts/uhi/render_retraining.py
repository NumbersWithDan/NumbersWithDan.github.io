"""Regenerate every analysis figure and coordinate-free website display data."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "assets/data/dallas-heat"
IMG = REPO / "images/dallas-heat"


def save(fig, name):
    fig.savefig(IMG / (name + ".png"), dpi=170, bbox_inches="tight")
    plt.close(fig)


def map_axes(ax, xy):
    ax.set_aspect("equal")
    ax.set_xlim(xy[:, 0].min() - 400, xy[:, 0].max() + 400)
    ax.set_ylim(xy[:, 1].min() - 400, xy[:, 1].max() + 400)
    ax.axis("off")
    ax.text(.98, .98, "N ↑", transform=ax.transAxes, ha="right", va="top", fontsize=11)


def main(work):
    data = pd.read_pickle(work / "rebuilt.pkl")
    X, y, xy = data["X"], data["y"], data["xy"]
    report = json.loads((OUT / "retraining.json").read_text())
    assert "shap" in report
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f6fcf8", "axes.facecolor": "#f6fcf8",
                         "text.color": "#163e3d", "axes.labelcolor": "#163e3d"})
    fig, ax = plt.subplots(figsize=(9, 7))
    p = ax.scatter(*xy.T, c=y, cmap="inferno", s=3, vmin=101, vmax=110, rasterized=True)
    map_axes(ax, xy)
    ax.set_title("Afternoon air temperature along the survey routes")
    fig.colorbar(p, ax=ax, shrink=.8, label="Observed air temperature (°F)")
    save(fig, "observed-routes")
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(y, bins=np.arange(100.85, 110.26, .2), color="#16786d", edgecolor="#f6fcf8")
    ax.axvline(y.mean(), ls="--", color="#b34822", label=f"Mean: {y.mean():.1f}°F")
    ax.set(xlabel="Observed afternoon air temperature (°F)", ylabel="Recorded observations",
           title="Most readings sit much closer together than the extremes")
    ax.legend()
    save(fig, "temperature-distribution")
    valid = X.NDVI.notna()
    q = pd.qcut(X.loc[valid, "NDVI"], 4, labels=False)
    quartiles = pd.DataFrame({"temperature": y[valid], "quartile": q}).groupby("quartile").temperature.agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(4), quartiles["mean"], "-o", color="#16786d", lw=3, ms=8)
    ax.set_xticks(range(4), ["Least green", "Lower middle", "Upper middle", "Most green"])
    ax.set(ylabel="Mean observed air temperature (°F)", xlabel="Quartiles of rebuilt, QA-masked NDVI",
           title="A vegetation association worth investigating")
    ax.grid(axis="y", alpha=.2)
    save(fig, "greenery-comparison")

    study = report["block_study"]
    sizes = sorted(set(r["block_m"] for r in study))
    fig, ax = plt.subplots(figsize=(10, 4.7))
    for name, label, color in [("xgboost", "Environmental XGBoost", "#16786d"),
                                ("spatial_knn", "Nearby-temperature interpolation", "#ba542b"),
                                ("blend", "Fixed 50:50 blend", "#6a5898"), ("mean", "Training-mean baseline", "#76847f")]:
        values = np.array([[r[name]["mae_f"] for r in study if r["block_m"] == size] for size in sizes])
        ax.plot(sizes, values.mean(axis=1), "-o", label=label, color=color)
        ax.fill_between(sizes, values.min(axis=1), values.max(axis=1), alpha=.12, color=color)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.set(xlabel="Grid block width (meters)", ylabel="Pooled held-out MAE (°F)",
           title="Performance depends on the distance of the prediction task")
    ax.grid(axis="y", alpha=.2)
    ax.legend(fontsize=9, loc="lower right")
    save(fig, "block-sensitivity")
    rows = report["residual_structure"]["bins"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([r["distance_m"] - 50 for r in rows], [r["correlation"] for r in rows], "-o", color="#16786d")
    ax.axhspan(-.1, .1, color="#16786d", alpha=.1, label="Diagnostic |correlation| < 0.1")
    ax.axhline(0, color="#76847f", lw=1)
    ax.set(xlabel="Separation between sampled observation pairs (meters)", ylabel="Residual pair correlation",
           title="Nearby prediction errors still resemble each other")
    ax.legend(fontsize=9)
    save(fig, "residual-dependence")
    nested = report["nested_spatial"]
    fig, ax = plt.subplots(figsize=(9, 4))
    keys = ["xgboost", "spatial_knn", "blend", "mean"]
    labels = ["Tuned XGBoost", "Nearby readings", "50:50 blend", "Training mean"]
    values = [nested[k]["mae_f"] for k in keys]
    ax.barh(labels[::-1], values[::-1], color=["#76847f", "#6a5898", "#ba542b", "#16786d"])
    for i, v in enumerate(values[::-1]):
        ax.text(v + .01, i, f"{v:.2f}°F", va="center")
    ax.set(xlabel="Pooled held-out MAE (°F); lower is better", title=f"A fresh comparison at {nested['block_m']:,} m blocks")
    ax.set_xlim(0, max(values) * 1.18)
    save(fig, "finished-validation")
    predictions = np.load(work / "nested.npz")["predictions"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, values, title, cmap, vmin, vmax in [
        (axes[0], y, "Observed temperature", "inferno", 101, 110),
        (axes[1], y - predictions[:, 3], "Held-out residual: observed − blended prediction", "RdBu_r", -3, 3)]:
        p = ax.scatter(*xy.T, c=values, s=2, cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
        map_axes(ax, xy)
        ax.set_title(title, fontsize=11)
        fig.colorbar(p, ax=ax, shrink=.75, label="°F")
    fig.tight_layout()
    save(fig, "finished-held-out-residuals")
    importance = report["shap"]["importance"][:10][::-1]
    labels = {"PM25": "PM2.5 indicator", "NDVI": "Vegetation index", "NDBI": "Built-up index",
              "LOWINCPCT": "Low-income share", "OVER64PCT": "Age 65+ share", "PRE1960PCT": "Pre-1960 housing share",
              "Distance_to_Park": "Distance to park", "Distance_to_Lake": "Distance to lake", "Distance_to_Stream": "Distance to stream",
              "NDVI_mean_210m": "Vegetation · 210 m window", "NDVI_mean_630m": "Vegetation · 630 m window"}
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh([labels.get(r["feature"], r["feature"].replace("_", " ")) for r in importance],
            [r["mean_absolute_contribution_f"] for r in importance], color="#16786d")
    ax.set(xlabel="Mean absolute tree-path SHAP contribution (°F)", title="What the retrained environmental model relies on")
    save(fig, "finished-model-explanations")

    # Transparent descriptive screening, not a claim about intervention effects.
    hot_cutoff = float(np.quantile(y, .75))
    eligible = (y >= hot_cutoff) & X[["NDVI_mean_210m", "Distance_to_Park", "PRE1960PCT", "OVER64PCT"]].notna().all(axis=1).to_numpy()
    pool = X.loc[eligible]
    score = ((1 - pool.NDVI_mean_210m.rank(pct=True)) + pool.Distance_to_Park.rank(pct=True)
             + pool.PRE1960PCT.rank(pct=True) + pool.OVER64PCT.rank(pct=True)) / 4
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(*xy.T, c="#d0dbd6", s=2, alpha=.5, rasterized=True)
    p = ax.scatter(*xy[eligible].T, c=score, s=5, cmap="viridis", vmin=0, vmax=1, rasterized=True)
    map_axes(ax, xy)
    ax.set_title("A closer look within the hotter survey observations")
    fig.colorbar(p, ax=ax, shrink=.8, label="Descriptive screening score (0–1)")
    save(fig, "opportunity-screening")
    i = int(score.idxmax())
    public = {"temperature": {"n": len(y), "min_f": float(y.min()), "max_f": float(y.max()), "mean_f": float(y.mean()),
                               "p05_f": float(np.quantile(y, .05)), "p95_f": float(np.quantile(y, .95))},
              "ndvi_quartiles": [{"group": int(i), "n": int(r["size"]), "mean_f": float(r["mean"])} for i, r in quartiles.iterrows()],
              "screening": {"hot_cutoff_f": hot_cutoff, "hot_observations": int((y >= hot_cutoff).sum()),
                            "eligible_observations": int(eligible.sum()), "definition": "equal average of within-hot-pool percentile ranks: lower neighborhood NDVI, greater park distance, older housing share, older population share"},
              "anonymous_example": {"observed_f": float(y[i]), "held_out_environmental_prediction_f": float(predictions[i, 0]),
                                    "held_out_blended_prediction_f": float(predictions[i, 3]),
                                    "neighborhood_ndvi": float(X.loc[i, "NDVI_mean_210m"]),
                                    "park_distance_m": float(X.loc[i, "Distance_to_Park"]),
                                    "pre1960_housing_share": float(X.loc[i, "PRE1960PCT"]),
                                    "older_population_share": float(X.loc[i, "OVER64PCT"]), "screening_score": float(score.loc[i])}}
    (OUT / "page-evidence.json").write_text(json.dumps(public, indent=2) + "\n", encoding="utf8")
    # Display positions have neither a geographic origin nor a physical unit.
    display = (xy - xy.min(axis=0)) / np.ptp(xy, axis=0).max()
    payload = {"description": "Normalized schematic display positions; not geographic coordinates",
               "points": [[round(float(a), 5), round(float(b), 5), float(t)] for (a, b), t in zip(display, y)]}
    (OUT / "observations.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf8")
    print("Regenerated nine figures, anonymous example, and schematic explorer data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    main(parser.parse_args().work)
