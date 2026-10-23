"""Write the portfolio story from the current spatial experiment artifacts."""
import html
import json
from pathlib import Path
import re
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
D=ROOT/'assets/data/dallas-heat'

def figure(name,alt,caption,eager=False,ident=''):
    w,h=Image.open(ROOT/'images/dallas-heat'/f'{name}.png').size
    return f'<figure{ident}><a class="heat-figure-link" href="../images/dallas-heat/{name}.png" aria-label="Open full-size figure {html.escape(alt)}"><img src="../images/dallas-heat/{name}.png" width="{w}" height="{h}" loading="{"eager" if eager else "lazy"}" alt="{html.escape(alt)}"></a><figcaption>{caption}</figcaption></figure>'

def main():
    r=json.loads((D/'islands-results.json').read_text()); e=json.loads((D/'islands-evidence.json').read_text())
    def friendly(c): return e['feature_labels'].get(c,c.replace('_',' '))
    af=r['periods']['Afternoon']; am=r['periods']['Morning']; pm=r['periods']['Evening']
    model=af['models']['environmental']; base=af['models']['mean']
    reduction=1-model['mae_f']/base['mae_f']
    rows=''.join(f'<tr><th scope="row">{p}</th><td>{v["cells"]:,}</td><td>{v["reference_f"]:.1f}°F</td><td>{v["models"]["mean"]["mae_f"]:.2f}°F</td><td>{v["models"]["environmental"]["mae_f"]:.2f}°F</td><td>{v["models"]["environmental"]["r2"]:.2f}</td></tr>' for p,v in r['periods'].items())
    n=sum(r['raw_counts'].values())
    story=f'''<article class="box page-content heat-story">
<header class="heat-hero">
<p class="heat-eyebrow">Dallas, Texas · Geospatial machine learning</p>
<h2>Same city.<br>Different heat.</h2>
<p class="heat-deck">Finding the hotter pockets of Dallas, then asking what the landscape has to do with them.</p>
<p>An island stands out from what surrounds it. A desert can be hot across a huge area without having that contrast. Heat islands are about the difference. A pocket of city can run warmer than the neighborhoods around it, even when the whole city is having a very hot day.</p>
<p>The broader urban heat island effect compares a city with its outlying surroundings. Here I’m looking within Dallas. I want to find the places that are unusually hot for the same time of day, see whether the pattern lasts into the evening, and understand what the model picks up about those places.</p>
<div class="heat-facts"><div><strong>{n:,}</strong><span>temperature readings</span></div><div><strong>3</strong><span>times of day</span></div><div><strong>{am['models']['environmental']['mae_f']:.2f}°F</strong><span>morning error at 1 km holdouts</span></div></div>
<p class="heat-note">Python · GeoPandas · Rasterio · XGBoost · Spatial validation · SHAP</p>
<a href="#observations" class="button">Start with the data</a> <a href="#citywide" class="heat-text-link">See the citywide predictions →</a>
</header>
<nav class="heat-contents" aria-label="Project sections"><a href="#observations">The observations</a><a href="#target">Defining an island</a><a href="#features">The landscape</a><a href="#blocks">Testing somewhere else</a><a href="#model-check">The results</a><a href="#citywide">Across Dallas</a><a href="#drivers">Inside an island</a></nav>
<section id="observations"><p class="heat-eyebrow">01 / Follow the routes</p><h3>One city through three parts of the day</h3>
<p>On August 5, 2023, Dallas volunteers drove routes with temperature and humidity sensors. The files include {r['raw_counts']['Morning']:,} morning readings, {r['raw_counts']['Afternoon']:,} afternoon readings, and {r['raw_counts']['Evening']:,} evening readings. The <a href="https://content.govdelivery.com/accounts/TXDALLAS/bulletins/393ca7d">city’s campaign summary</a> describes how the survey was collected.</p>
<p>I average the readings within each 100 m cell, separately for each time period. That keeps a slow drive down one block from giving that block more influence just because the sensor collected more readings there. Each occupied cell gets one vote.</p>
{figure('period-observations','Observed temperature anomalies for morning, afternoon and evening over Dallas streets and water','Each dot is a surveyed 100 m cell. Colors compare that cell with the survey reference for its own time period. Muted streets, parks, and lakes provide geographic context.',True)}
<p>The afternoon is the hottest in absolute terms. The interesting question is whether the same places stand out in the morning or stay relatively warm in the evening. I fit separate models for each period so those patterns can differ.</p>
<div class="heat-explorer"><h4>Follow the unusually hot readings</h4><p>Pick a time and move the cutoff. This view shows measured cell averages, rather than model predictions.</p>
<label for="survey-period">Time of day</label><select id="survey-period"><option selected>Morning</option><option>Afternoon</option><option>Evening</option></select>
<label for="heat-cutoff">At least <output id="heat-cutoff-value">+1.0°F</output> above the survey reference</label><input id="heat-cutoff" type="range" min="0" max="3" value="1" step="0.1" disabled>
<p id="heat-count" aria-live="polite">Loading the survey cells.</p><canvas id="heat-map" width="900" height="820" role="img" aria-label="Measured heat anomalies across survey cells" hidden></canvas><p class="heat-note">North is up. Roads and water are schematic context. Counts refer to sampled cells, rather than residents or the whole city.</p></div></section>
<section id="target"><p class="heat-eyebrow">02 / Define the thing to predict</p><h3>Hot compared with what?</h3>
<p>A temperature of 105°F means something different at 7 a.m. than at 3 p.m. My main target is <strong>excess air temperature in °F above the survey reference for that time of day</strong>. The reference is the average of the equally weighted surveyed cells. In the afternoon it is {af['reference_f']:.1f}°F, so a cell at {af['reference_f']+1:.1f}°F has an excess of +1°F.</p>
<div class="heat-callout"><p><strong>Predicted heat excess = cell air temperature − period survey reference</strong></p><p>I keep Fahrenheit because “one degree warmer” is easier to interpret than a standardized score. Subtracting the mean changes the reference point, rather than making the model more accurate. During validation, the reference comes only from the training cells.</p></div>
{figure('period-distribution','Distribution of measured excess temperatures at three times of day','Zero is each period’s survey reference. This is a survey-relative comparison within Dallas, rather than a measured urban–rural temperature difference.')}
{figure('period-persistence','Measured temperature anomalies in cells surveyed at all three times of day','Each point is the same 100 m cell measured at two times. This checks persistence across periods without treating one afternoon as evidence of a permanent neighborhood trait.')}
<p>There are {e['shared_cells']:,} cells measured in all three periods. The morning–afternoon correlation is {e['observed_period_correlations']['Morning / Afternoon']:.2f}, and the afternoon–evening correlation is {e['observed_period_correlations']['Afternoon / Evening']:.2f}. Those comparisons help separate places that stay relatively warm from places whose position changes through the day.</p>
<p>That finds broadly hot areas. To identify a smaller island within them, I add a second check after prediction. A candidate must sit in the hotter quarter of the survey temperature distribution and be at least <strong>0.25°F warmer than the model’s surrounding 500–1,500 m ring</strong>. I group touching cells into patches of at least three cells.</p>
<p class="heat-note">The ring and contrast cutoff are explicit screening choices, rather than a universal scientific definition. The contrast is calculated from predicted environmental surfaces, so nearby measured temperatures never become hidden predictors. It is a way to compare places, rather than a probability that an island exists.</p></section>
<section id="features"><p class="heat-eyebrow">03 / Describe the surroundings</p><h3>What does the model know about a place?</h3>
<p>A thermometer tells me what happened. The other datasets give me a way to describe the landscape where it happened. I bring them onto the same 100 m grid before fitting the models.</p>
<ol class="heat-pipeline"><li><strong>The survey supplies the target.</strong> Morning, afternoon, and evening air-temperature readings become one average per sampled cell and period. Humidity and heat index stay out of the predictors.</li>
<li><strong>Landsat describes the surface.</strong> Satellite bands provide vegetation and built-up signals, along with thermal brightness. NDVI compares near-infrared and red light to indicate vegetation. I summarize the landscape over roughly 210 m, 630 m, and 2.1 km windows, then compare the nearby landscape with the wider area.</li>
<li><strong>City layers describe the setting.</strong> Parks, lakes, and streams provide geometric distance features. The street network adds a proxy for how much of the nearby raster is crossed by roads.</li>
<li><strong>Block groups and market areas add neighborhood context.</strong> EJScreen contributes indicators such as traffic, fine-particle pollution, housing age, and population characteristics. Residential market polygons add housing and development indicators. These describe areas, rather than individual people.</li></ol>
<details><summary>How the sources become model features</summary><p>I calibrate the Landsat bands from their metadata and mask cloud, shadow, snow, fill, and saturation flags. Missing measurements stay missing. Spatial joins attach neighborhood attributes, while distances use a projected map measured in meters. The same extraction runs at surveyed cells and citywide prediction cells.</p><p>The temperature campaign is from August 2023 and the supplied Landsat scene is from October 2023. The satellite features describe landscape conditions from that scene. Their thermal brightness is different from the survey’s air temperature and from a calibrated land-surface temperature product.</p></details>
{figure('period-greenery','Observed temperature anomalies across neighborhood vegetation quartiles for each period','Each period has its own roughly equal-count vegetation groups. This is an observed association across surveyed cells, rather than the estimated cooling effect of adding trees.')}
<p>The vegetation pattern is much clearer in the morning. The least-green quarter averages about {e['greenery'][0]['quartile_mean_excess_f'][0]-e['greenery'][0]['quartile_mean_excess_f'][3]:.2f}°F warmer than the greenest quarter, compared with about {e['greenery'][1]['quartile_mean_excess_f'][0]-e['greenery'][1]['quartile_mean_excess_f'][3]:.2f}°F in the afternoon. That gives the model a clearer landscape signal early in the day and is one clue to the harder afternoon predictions. Vegetation can provide <a href="https://www.epa.gov/heatislands/benefits-trees-and-vegetation">shade and cooling through evaporation</a>, so faster warming in more exposed areas is a plausible part of the story. But these snapshots don’t tell me which places heated fastest. Morning differences could also reflect heat retained overnight. By afternoon, shifting building shade, wind, and surface materials may matter in ways a broad vegetation index misses. Those are explanations I would test with more detailed measurements.</p>
</section>
<section id="blocks"><p class="heat-eyebrow">04 / Make the test match the question</p><h3>A small grid and a fair test do different jobs</h3>
<p>The <strong>100 m analysis grid</strong> decides how I summarize a place. The <strong>1 km holdout blocks</strong> decide which places the model has to predict without seeing their temperatures. Those sizes do different jobs.</p>
<p>If I scatter neighboring cells between training and testing, the model can be tested just one street away from a location it already knows. Holding out whole blocks asks it to bridge a larger gap. I compare 100 m, 250 m, 500 m, 1 km, and 2 km blocks, shifting the grid as another check.</p>
{figure('islands-block-sensitivity','Afternoon held-out errors across block sizes and two grid placements','The environmental configuration stays fixed in this sensitivity test. Shading spans two placements of the holdout grid and is not a confidence interval. The 100 m placements coincide because the sampled cells are already on a 100 m grid.')}
<p>In this cell-based afternoon test, the median distance to a training cell is 100 m with the smallest blocks and about 316 m with 1 km blocks. With 2 km blocks it is roughly 540–580 m. The scores answer increasingly difficult versions of “predict somewhere else.”</p>
<p>I use 1 km blocks as the main test. Earlier held-out residual checks showed nearby errors becoming much less similar over several hundred meters. One kilometre is a conservative working scale above that distance, while 2 km remains a harder check. It is a practical analysis choice, rather than the block size picked for the best score.</p>
<p class="heat-note">A cell can still lie close to the edge of a held-out block, so block width is different from guaranteed separation. I also test larger north–south regions with a 500 m buffer between training and test locations.</p></section>
<section id="model-check"><p class="heat-eyebrow">05 / Compare predictions with observations</p><h3>Can the landscape help find the hotter cells?</h3>
<p>I fit XGBoost models separately for morning, afternoon, and evening. Within each training split, three inner spatial folds compare landscape-only and combined features with different tree depths. Five outer folds test the chosen models on held-out blocks. I repeat that process with a shifted grid.</p>
<p>The simplest baseline predicts the training-period average everywhere. It knows the time of day, but nothing about the location. A nearby-temperature interpolation is a second benchmark inside the survey area. For the citywide map, I use the environmental model, which can work from landscape features alone.</p>
{figure('islands-validation','Spatially held-out temperature errors and precision for hotter survey cells across all three periods','The left panel compares average prediction error with the mean baseline. The right panel asks how often the highest-scored quarter of cells actually falls above its training-period hot cutoff. Dashed marks show the observed prevalence of hot cells.')}
<div class="heat-table-wrap"><table class="heat-table"><caption>Five-fold spatial evaluation at 1 km blocks</caption><thead><tr><th scope="col">Period</th><th scope="col">Survey cells</th><th scope="col">Reference</th><th scope="col">Baseline error</th><th scope="col">Model error</th><th scope="col">R²</th></tr></thead><tbody>{rows}</tbody></table></div>
<p>The morning pattern is the clearest of the three. Its model has an average error of <strong>{am["models"]["environmental"]["mae_f"]:.2f}°F</strong> versus <strong>{am["models"]["mean"]["mae_f"]:.2f}°F</strong> for the baseline, with R² of <strong>{am["models"]["environmental"]["r2"]:.2f}</strong>. Its highest-scored quarter contains <strong>{am["models"]["environmental"]["precision_top_quarter"]:.0%}</strong> observed hot cells. The morning model also beats its baseline in the buffered regional test, at {am["regional_mae_f"]:.2f}°F versus {am["regional_mean_mae_f"]:.2f}°F.</p>
<p>For the afternoon, average absolute error is <strong>{model['mae_f']:.2f}°F</strong>, compared with <strong>{base['mae_f']:.2f}°F</strong> for the baseline. That is about <strong>{reduction:.0%}</strong> less error. Of the model’s highest-scored quarter of afternoon cells, <strong>{model['precision_top_quarter']:.0%}</strong> are observed hot cells, compared with a hot-cell prevalence of <strong>{model['hot_prevalence']:.0%}</strong>.</p>
<p class="heat-note">The shifted-grid afternoon error is {af['shifted_mae_f']:.2f}°F. On the separate buffered-region test, afternoon error is {af['regional_mae_f']:.2f}°F versus {af['regional_mean_mae_f']:.2f}°F for its baseline. Citywide predictions remain exploratory in places beyond the survey. These experiments use one campaign day and are not an independent test on a new day. The supplied readings have no route timestamps, so collection conditions cannot be fully separated from geography.</p></section>
<section id="citywide"><p class="heat-eyebrow">06 / Put the predictions on the city</p><h3>From the sampled streets to Dallas</h3>
<p>I apply the environmental models to land cells across the Dallas city boundary. The left map shows the full prediction surface. The right keeps cells with valid vegetation data, familiar model-feature combinations, and a survey cell within 1 km. Grey areas still have predictions on the left, but do not pass that support screen.</p>
<label for="city-period">Choose a prediction period</label><select id="city-period" data-heat-period><option selected>Morning</option><option>Afternoon</option><option>Evening</option></select>
'''
    for p in r['periods']:
        hidden='' if p=='Morning' else ' hidden'
        ev=e['periods'][p]; cases=ev['displayed_cases']
        story+=f'<div class="heat-period-panel" data-city-period="{p}"{hidden}>'
        story+=figure('citywide-'+p.lower(),f'{p} citywide excess temperature predictions with support screening',f'{p} environmental prediction for campaign-day conditions. About {ev["supported_land_share"]:.0%} of valid city land cells pass the support screen. Labels identify selected candidate patches, rather than confirmed heat-island boundaries.')
        story+=f'<p>The screening rule identifies <strong>{ev["candidate_patches"]}</strong> connected {p.lower()} candidate patches. '
        if cases:
            story+='The lettered examples are the highest-ranked patches that are at least 2 km apart, ranked by mean heat excess plus local contrast.</p>'
            story+='<div class="heat-cases">'
            for c in cases:
                story+=f'<div class="heat-case"><h4>Island {c["name"]}</h4><p><strong>{c["mean_excess_f"]:+.2f}°F</strong> relative to the {p.lower()} survey reference</p><p><strong>{c["mean_local_contrast_f"]:+.2f}°F</strong> average contrast with the surrounding ring</p><p class="heat-note">{c["area_km2"]:.2f} km² across {c["cells"]} connected analysis cells</p>'
                if 'nearby_survey' in c:
                    ns=c['nearby_survey']
                    location='The example cell was surveyed and recorded' if ns['distance_m']<1 else f'A surveyed cell {ns["distance_m"]:.0f} m from the example recorded'
                    story+=f'<p class="heat-note">{location} {ns["observed_excess_f"]:+.2f}°F excess. Its held-out prediction was {ns["heldout_excess_f"]:+.2f}°F.</p>'
                story+='</div>'
            story+='</div>'
        else: story+='No patches pass every support and local-contrast requirement for this period.</p>'
        story+='</div>'
    story+='''<p class="heat-note">The model predicts survey-relative heat under the campaign-day setup, rather than a current forecast. The citywide average is not forced to zero. The support screen is a diagnostic, not a calibrated confidence interval. At least half of a candidate’s surrounding ring must pass the screen, but all valid ring cells contribute to the contrast. Open water and missing vegetation cells are excluded from the heat-island search.</p></section>
<section id="drivers"><p class="heat-eyebrow">07 / Open up the prediction</p><h3>What makes a predicted island stand out?</h3>
<p>SHAP breaks a prediction into feature contributions. In these charts, points to the right push predicted excess heat upward. Red points have higher feature values and blue points have lower values. This shows both the size and direction of the model’s associations.</p>'''
    story+='<label for="explain-period">Choose an explanation period</label><select id="explain-period" data-heat-period><option selected>Morning</option><option>Afternoon</option><option>Evening</option></select>'
    for p in r['periods']:
        hidden='' if p=='Morning' else ' hidden'
        story+=f'<div class="heat-period-panel" data-city-period="{p}"{hidden}>'
        story+=figure('shap-'+p.lower(),f'Directional SHAP values for the {p.lower()} environmental model','Tree-path SHAP at up to 1,500 sampled survey cells. Contributions are in °F relative to the model’s baseline. Correlated indicators can share information, so these explain the fitted model rather than isolated physical effects.')
        phrases=[]
        for direction in [v for v in e['global_directions'][p] if abs(v['high_minus_low_mean_shap_f'])>=.05][:3]:
            adjective='higher' if direction['high_minus_low_mean_shap_f']>0 else 'lower'
            if direction['feature'].startswith('Distance_'): adjective='greater' if adjective=='higher' else 'shorter'
            phrases.append(adjective+' '+friendly(direction['feature']).lower())
        if phrases:
            joined=', '.join(phrases[:-1])+' and '+phrases[-1] if len(phrases)>1 else phrases[0]
            story+=f'<p>The {p.lower()} model associates {joined} with warmer contributions on average. This compares the upper and lower feature-value quarters in the explanation sample, rather than estimating what would happen if I changed one feature in place.</p>'
        story+='</div>'
    # Explain the features readers actually see in the current global chart.
    # Rankings are exported with the evidence so page generation needs no model cache.
    definitions={
      'PM25':'An area-level estimate of fine-particle pollution. It can also distinguish different urban settings.',
      'PTRAF':'An area-level indicator of nearby traffic and road activity.',
      'Distance_to_Lake':'Straight-line distance to the nearest mapped lake polygon.',
      'Distance_to_Park':'Straight-line distance to a park boundary, rather than walking distance or available shade.',
      'Distance_to_Stream':'Straight-line distance to the nearest mapped stream.',
      'PRE1960PCT':'The share of area housing built before 1960.',
      'LOWINCPCT':'The area population share classified as lower income in EJScreen.',
      'OVER64PCT':'The area population share age 65 or older.',
      'UNEMPPCT':'An area-level unemployment indicator.','tree_canopy_fraction':'The source area share covered by tree canopy, when valid values are available.'}
    def definition(c):
        if c in definitions: return definitions[c]
        if c.startswith('Brightness_'): return 'Satellite thermal brightness from the October scene, rather than the August air-temperature target.'
        if c.startswith('TOA_'): return 'Reflected light measured in one satellite wavelength band, used as a surface-description feature.'
        if c=='NDVI_local_contrast': return 'Nearby vegetation signal minus the wider-area signal. Higher means locally greener surroundings.'
        if c=='NDBI_local_contrast': return 'Nearby built-up signal minus the wider-area signal. Higher means a stronger local built-up contrast.'
        if c.startswith('NDVI'): return 'A satellite vegetation signal. Higher values generally indicate more vegetation at the stated scale.'
        if c.startswith('NDBI'): return 'A satellite built-up signal. It can also respond to bare soil, rather than identifying pavement alone.'
        if c.startswith('road_'): return 'The fraction of nearby raster pixels crossed by mapped streets, used as a street-network proxy.'
        if c=='vegetated_fraction_630m': return 'The share of neighborhood pixels whose vegetation index exceeds the stated greener-pixel threshold.'
        if c=='built_signal_fraction_630m': return 'The share of neighborhood pixels with a positive built-up index.'
        if c.startswith('MVA_'): return 'A residential market-area category from the Dallas market value analysis.'
        return 'An area-level housing or development indicator from the Dallas residential market analysis.'
    for p in r['periods']:
        hidden='' if p=='Morning' else ' hidden'
        top=e['global_features'][p][:8]
        story+=f'<div class="heat-table-wrap heat-period-panel" data-city-period="{p}"{hidden}><table class="heat-table"><caption>The features in plain English</caption><thead><tr><th scope="col">Feature</th><th scope="col">What it describes</th></tr></thead><tbody>'+''.join(f'<tr><th scope="row">{friendly(c)}</th><td>{definition(c)}</td></tr>' for c in top)+'</tbody></table></div>'
    story+='<h4>A closer look at the lettered islands</h4><p>Here I explain one representative cell in each lettered patch against the model’s surrounding ring. A positive bar helps explain why that cell is predicted warmer than the ring. A negative bar offsets that contrast. All feature contributions, including those outside the six shown, add up to the model’s local contrast.</p>'
    for p in r['periods']:
        cases=e['periods'][p]['displayed_cases']; hidden='' if p=='Morning' else ' hidden'
        story+=f'<div class="heat-period-panel" data-city-period="{p}"{hidden}>'
        if cases:
            story+=figure('local-shap-'+p.lower(),f'Signed local SHAP explanations for selected {p.lower()} candidate islands','Orange contributions raise the predicted cell-versus-ring contrast and blue contributions reduce it. The example cell has the largest local contrast within its patch, so its value differs from the patch average above.')
            for c in cases:
                positives=[f'{v["label"].lower()} ({v["contribution_f"]:+.2f}°F)' for v in c['drivers'] if v['contribution_f']>0][:2]
                if positives: story+=f'<p><strong>Island {c["name"]}</strong> is a useful example of how the features combine. The largest positive contributions shown include '+ ' and '.join(positives)+'. These describe what separates the modeled cell from its surrounding ring.</p>'
        story+='</div>'
    story+=f'''<p>This is where I would begin a closer investigation. I can identify a warmer pocket, quantify its predicted contrast, and see which landscape and neighborhood features the model uses to distinguish it. The next question is whether those associations hold on another hot day and what a visit to that place adds to the picture.</p>
<p class="heat-note">SHAP describes the prediction, rather than proving that pollution, housing age, or any one feature causes heat. Block-group indicators can act as location proxies. The named patches are candidates for follow-up, rather than estimates of how much a tree-planting or planning intervention would cool them.</p></section>
<section id="project-files" class="heat-resources"><h3>Explore the project</h3><p><a href="https://github.com/NumbersWithDan/NumbersWithDan.github.io/tree/main/scripts/uhi" class="button">Project on GitHub</a></p><p class="heat-note">Original analytical data assembled as of fall 2024. The analysis uses the August 2023 Dallas heat campaign, the supplied October 2023 Landsat scene, and the accompanying neighborhood and city layers. City-boundary map context was retrieved in October 2026.</p><p><a href="machine-learning.html">More machine learning projects →</a></p></section>
</article>'''
    path=ROOT/'data/dallas-heat-islands.html'; text=path.read_text(encoding='utf8')
    text=re.sub(r'<article class="box page-content heat-story">.*?</article>',lambda _:story,text,flags=re.S)
    text=text.replace('<body class="is-preload">','<body class="is-preload heat-project">')
    text=re.sub(r'<meta name="description" content="[^"]*">','<meta name="description" content="Exploring Dallas heat islands with morning, afternoon and evening readings, spatial machine learning, citywide predictions and local SHAP explanations.">',text)
    text=text.replace('This project sits at the intersection of environmental data, mapping, and questions about how cities distribute risk.',
      'Dallas can be hot everywhere and still have places that stand out. This project follows those differences through the day, maps candidate heat islands, and looks at the landscape signals behind their predictions.')
    text=text.replace('Design: <a','Design by <a')
    assert ':' not in re.sub(r'<[^>]+>','',story), 'Visible story text contains a colon'
    path.write_text(text,encoding='utf8')
    print('Wrote portfolio page from current experiment artifacts.')

if __name__=='__main__': main()
