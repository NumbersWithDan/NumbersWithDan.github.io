(() => {
  'use strict';
  const periodControls = document.querySelectorAll('[data-heat-period]');
  function updateCity(value) {
    periodControls.forEach(control => { control.value = value; });
    document.querySelectorAll('[data-city-period]').forEach(panel => { panel.hidden = panel.dataset.cityPeriod !== value; });
  }
  periodControls.forEach(control => control.addEventListener('change', () => updateCity(control.value)));
  if (periodControls.length) updateCity(periodControls[0].value);
  const slider = document.getElementById('heat-cutoff');
  const period = document.getElementById('survey-period');
  const canvas = document.getElementById('heat-map');
  const status = document.getElementById('heat-count');
  if (!slider || !canvas || !period) return;
  canvas.hidden = true;
  fetch('../assets/data/dallas-heat/islands-display.json')
    .then(response => { if (!response.ok) throw new Error('Survey request failed'); return response.json(); })
    .then(data => {
      const ctx = canvas.getContext('2d');
      if (!ctx) throw new Error('Map unavailable');
      const allPoints = Object.values(data.periods).flatMap(survey => survey.points);
      const minX = Math.min(...allPoints.map(p => p[0])) - .015;
      const maxX = Math.max(...allPoints.map(p => p[0])) + .015;
      const minY = Math.min(...allPoints.map(p => p[1])) - .015;
      const maxY = Math.max(...allPoints.map(p => p[1])) + .015;
      function draw() {
        const current = data.periods[period.value];
        const points = current.points;
        const cutoff = Number(slider.value);
        document.getElementById('heat-cutoff-value').textContent = `+${cutoff.toFixed(1)}°F`;
        const style = getComputedStyle(document.documentElement);
        const dark = document.documentElement.classList.contains('dark');
        ctx.fillStyle = style.getPropertyValue('--background').trim();
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const pad = 30;
        const scale = Math.min((canvas.width - 2 * pad) / (maxX - minX), (canvas.height - 2 * pad) / (maxY - minY));
        const offsetX = (canvas.width - (maxX - minX) * scale) / 2;
        const offsetY = (canvas.height - (maxY - minY) * scale) / 2;
        const point = ([x, y]) => [offsetX + (x - minX) * scale, canvas.height - offsetY - (y - minY) * scale];
        function paths(lines, color, width, fill) {
          ctx.strokeStyle = color; ctx.lineWidth = width; ctx.fillStyle = color;
          for (const line of lines) {
            if (!line.length) continue;
            ctx.beginPath();
            line.forEach((p, i) => { const [x, y] = point(p); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
            if (fill) ctx.fill(); else ctx.stroke();
          }
        }
        paths(data.base.water, dark ? '#234a57' : '#d2e6ec', 1, true);
        paths(data.base.roads, dark ? '#44544b' : '#c7d3cb', .5, false);
        paths(data.base.boundary, dark ? '#729283' : '#98ad9f', 1, false);
        let count = 0;
        ctx.fillStyle = style.getPropertyValue('--muted-foreground').trim(); ctx.globalAlpha = .35;
        for (const p of points) { const [x, y] = point(p); ctx.fillRect(x, y, 2.5, 2.5); }
        ctx.globalAlpha = .95; ctx.fillStyle = dark ? '#ffb47e' : '#b3471f';
        for (const p of points) { if (p[2] >= cutoff) { count++; const [x, y] = point(p); ctx.fillRect(x, y, 4, 4); } }
        ctx.globalAlpha = 1;
        status.textContent = `${count.toLocaleString()} of ${points.length.toLocaleString()} ${period.value.toLowerCase()} survey cells (${(100 * count / points.length).toFixed(1)}%) are at least ${cutoff.toFixed(1)}°F above the ${current.reference_f.toFixed(1)}°F reference.`;
        canvas.setAttribute('aria-label', `${status.textContent} Dallas streets and water provide context. North is up.`);
      }
      slider.disabled = false; canvas.hidden = false;
      slider.addEventListener('input', draw); period.addEventListener('change', draw);
      new MutationObserver(draw).observe(document.documentElement, {attributes: true, attributeFilter: ['class']});
      draw();
    })
    .catch(() => { status.textContent = 'The interactive survey could not be loaded. The measured maps above remain available.'; });
})();
