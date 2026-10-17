# Rebuild and retrain the Dallas heat project

The current workflow reads original files without changing them. Do not execute the historical notebooks to regenerate this page: they contain writes, hardcoded working directories, and state-dependent cells.

Use Python 3.11 with numpy, pandas, scipy, scikit-learn, matplotlib, joblib, xgboost, geopandas, pyogrio, shapely, pyproj, rasterio, and Pillow. The actual run used XGBoost 2.1.3, GeoPandas 1.1.1, Pyogrio 0.11.1, Shapely 2.1.2, Rasterio 1.4.4, PyProj 3.7.2, numpy 2.2.6, pandas 2.3.3, and scikit-learn 1.5.1. Keep this environment separate from the static website runtime.

From the website repository root, choose a private work folder OUTSIDE this public repository, then run:

```powershell
python scripts/uhi/rebuild_features.py --source 'C:\Users\danie\OneDrive\Documents\Grad School\Classes\CSML Project' --work '<private-work-directory>'
python scripts/uhi/retrain.py --work '<private-work-directory>'
python scripts/uhi/render_retraining.py --work '<private-work-directory>'
```

`rebuild_features.py` reconstructs one row per original afternoon observation, QA-masked and calibrated Landsat features, neighborhood summaries, vector distances, and deterministic polygon joins. Source hashes and repair/coverage counts accompany a private feature cache.

`retrain.py` evaluates five block sizes and two grid placements; chooses a diagnostic scale from out-of-fold residual dependence rather than maximum accuracy; uses nested spatial tuning for the environmental model; and compares mean, spatial interpolation, and a predeclared fixed blend. It also runs a separate buffered-region stress test. Fold prediction caches remain private. Cached results are reused only for matching feature-cache hashes. The working-scale selection is exploratory; a new survey campaign is needed for independent confirmation.

The final environmental model and 32-neighbor spatial interpolator are saved in `retrained-model.joblib` in the private work directory, with feature names and fixed blend weights [0.5, 0.5]. An XGBoost UBJSON export is also saved. The bundle is a trusted local artifact containing training-location information and must not be published. The public website needs no Python or model runtime.

`render_retraining.py` rebuilds every page figure, descriptive evidence, an anonymous measured example, and normalized display-only explorer positions. Maps have no coordinate axes, and public JSON contains no geographic origin or actual coordinates. The screening score uses actual hotter observations and equal ranks of four source features, not citywide extrapolation or estimated intervention effects.

The report is `assets/data/dallas-heat/review.md`; current experiment results are `retraining.json`; page descriptions are supported by `page-evidence.json`. All prediction errors on the page are held-out errors. SHAP explains only the final environmental model, not the full blend or causal effects.

Earlier files (`verify.py`, `verify_finished.py`, `verify_rasters.py`, `verify_distances.py`, and earlier JSON) document previous review stages. They are not the current retraining workflow. Public coordinate fields in those earlier assets have been removed. Do not treat their scores as scores of the rebuilt pipeline.
