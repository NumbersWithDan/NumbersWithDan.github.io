# Dallas heat islands

A portfolio analysis of heat differences within Dallas using the August 5, 2023 morning, afternoon and evening sensor surveys. Source datasets were assembled as of fall 2024. The current workflow fits separate environmental models and maps survey-relative temperature excess across Dallas.

## Reproduce the current analysis

Use Python 3.11 with numpy, pandas, scipy, scikit-learn, matplotlib, joblib, xgboost, geopandas, pyogrio, shapely, pyproj, rasterio and Pillow. The run uses XGBoost 2.1.3, GeoPandas 1.1.1, Pyogrio 0.11.1, Shapely 2.1.2, Rasterio 1.4.4, PyProj 3.7.2, numpy 2.2.6, pandas 2.3.3 and scikit-learn 1.5.1.

Choose a new private working folder outside this public repository. Raw source files remain read-only. Download the Dallas municipal boundary into that folder as `city-boundary.geojson` with projected EPSG 32614 coordinates using the City of Dallas GIS endpoint below. The boundary is contemporary cartographic context, retrieved October 23, 2026, rather than a new heat observation or predictor.

https://gis.dallascityhall.com/arcgis/rest/services/Basemap/CityLimits/MapServer/0/query?where=CITY%20%3D%20%27DALLAS%27&outFields=CITY&outSR=32614&returnGeometry=true&f=geojson

From the repository root run these commands in order, replacing both placeholders with local directories.

```powershell
python scripts/uhi/prepare_islands.py --source '<source-project-directory>' --work '<private-work-directory>'
python scripts/uhi/model_islands.py --work '<private-work-directory>'
python scripts/uhi/check_island_blocks.py --work '<private-work-directory>'
python scripts/uhi/render_islands.py --source '<source-project-directory>' --work '<private-work-directory>'
python scripts/uhi/build_island_page.py
python scripts/uhi/audit_islands.py --work '<private-work-directory>'
```

The preparation step uses the shared feature extractor in `rebuild_features.py`. It takes mean temperature within each fixed 100 m grid cell separately by period. Features are sampled at cell centers through the same extractor for survey and citywide cells. Cells receive equal modeling weight. This changes the estimand from sensor-record temperatures to temperatures averaged along the surveyed road portions of each cell; it does not establish an area-wide thermometer average.

Cloud-masked Landsat provides reflectance, thermal brightness, NDVI, NDBI, neighborhood summaries and landscape contrasts. City vectors provide park/water distances and street-pixel occupancy. EJScreen and residential market polygons provide area context. The supplied canopy field is unusable in this footprint and is excluded. No TES temperature, vulnerability composite, modeled survey-temperature raster, humidity or heat-index field becomes a predictor.

## Targets and validation

The primary target is cell temperature minus the equal-cell training-period mean in Fahrenheit. Constant centering or standardizing changes the units/reference, not predictive information. All training references and upper-quartile hot thresholds are estimated inside the relevant training split. Separate models distinguish morning, afternoon and evening conditions without pretending that three times on one day are three independent weather campaigns.

Five outer folds hold out fixed 1 km spatial cells. Three inner spatial folds compare four landscape/combined-feature and tree-depth configurations. A second grid placement repeats nested tuning. The separate north-south regional test excludes training cells within 500 m of held-out locations and uses a fixed shallow configuration. The block sensitivity experiment uses a single fixed configuration across 100, 250, 500, 1000 and 2000 m with two placements. These experiments remain exploratory on the same campaign, rather than independent confirmation after repeated model development. The supplied raw point files have no route timestamps or sensor identifiers, so within-period collection drift cannot be separated from geographic effects.

The mean baseline and a 32-neighbor inverse-distance interpolator accompany environmental predictions. A fixed legacy-feature comparator also appears in the JSON. It uses a different configuration, so its comparison does not isolate the effect of added features. New results are cell-level metrics and cannot be read as a direct before/after improvement over the historical record-level results.

Hotspot ranking metrics use predictions above the fold's training mean. A hot label means a measured cell at or above that training split's 75th percentile. Average precision, observed prevalence, precision/recall among the highest-scored quarter, MAE, R² and anomaly rank correlation are reported. Predicted heat scores and top-ranked cells are not calibrated odds. No claim of a hot-island probability is made.

## Citywide predictions and local explanations

Environmental models predict a campaign-condition scenario over the municipal 100 m grid. Water and invalid point NDVI cells are excluded from the land analysis. The support diagnostic requires a survey cell within 1 km and a nearest-neighbor distance in robust-scaled model-feature space below the training leave-one-out 95th percentile. It is a heuristic applicability screen, not a confidence bound. Both the complete scenario and screened version are displayed.

Candidate local islands must pass that screen, have at least half their surrounding ring supported, exceed the period's observed upper-quartile threshold, and be at least 0.25°F above the mean predicted valid land cells 500–1500 m away. Patches contain at least three four-connected cells. Rank patches by mean excess plus mean local contrast, then show up to three at least 2 km apart. Candidate boundaries depend on the analysis grid, ring and threshold; they are not verified physical boundaries.

Global SHAP uses exact tree-path contributions at a fixed-seed sample of up to 1500 survey cells. A local example is the highest-contrast cell inside each selected patch. Its local SHAP is the cell contribution minus the mean contribution over its surrounding ring. All feature differences sum to the predicted local contrast; additivity is checked numerically. The chart shows the six largest signed differences. It explains model contrasts, rather than causal intervention effects, with correlated geographic proxies explicitly acknowledged.

## Findings and next experiments

Morning environmental prediction is strongest, with 0.37°F MAE versus 1.27°F for the mean and pooled R² of 0.89. The buffered regional morning error is 0.60°F versus 1.48°F for the mean. Afternoon error is 0.75°F versus 0.85°F, with R² of 0.21; its buffered-region result does not beat the mean. Evening error is 0.44°F versus 0.69°F, with R² of 0.56. The richer feature set did not materially improve afternoon performance against the fixed legacy-feature comparator. Current cell-level results are not directly comparable with earlier raw-record scores.

The remaining high-value experiments are imagery closer to the campaign date, usable tree-canopy/impervious-surface and building-shade data, sensitivity to pollution/location proxies, and route/time metadata or another independent survey day. Centering the target or adding tree depth alone cannot replace those inputs. A calibrated hot/not-hot classifier would be a separate task if actual probabilities are needed; the current project deliberately reports ranked temperature contrasts.

## Outputs and privacy

Current public artifacts are `islands-results.json`, `islands-blocks.json`, `islands-evidence.json`, `islands-display.json`, the scientific PNGs and the portfolio HTML. Geographic coordinates, raw features, fitted models and fold-level predictions stay in the private work directory. Public interactive geometry uses normalized schematic positions with no geographic origin. Source and feature signatures prevent silent reuse when inputs change; model settings are checked before reusing period caches.

The earlier `retrain.py`, `render_retraining.py`, `verify*` scripts and historical JSON document previous review stages. They do not generate the current page or represent the current cell-level model. Never execute the historical source notebooks as a reproduction shortcut; they contain writes and state-dependent operations.

## Primary background

- [EPA heat-island definitions](https://www.epa.gov/heatislands/what-are-heat-islands)
- [Dallas heat campaign summary](https://content.govdelivery.com/accounts/TXDALLAS/bulletins/393ca7d)
- [USGS Landsat Level-1 calibration](https://www.usgs.gov/landsat-missions/using-usgs-landsat-level-1-data-product)
- [USGS quality flags](https://www.usgs.gov/landsat-missions/landsat-collection-2-quality-assessment-bands)
- [Nested cross-validation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html)
