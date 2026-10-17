(() => {
  'use strict';
  const slider = document.getElementById('heat-cutoff');
  const canvas = document.getElementById('heat-map');
  const status = document.getElementById('heat-count');
  if (!slider || !canvas) return;
  canvas.hidden = true;
  fetch('../assets/data/dallas-heat/observations.json')
    .then(response => { if (!response.ok) throw new Error('Observation request failed'); return response.json(); })
    .then(data => {
      const points = data.points;
      const ctx = canvas.getContext('2d');
      if (!ctx || !points.length) throw new Error('Map unavailable');
      const maxX = Math.max(...points.map(p => p[0]));
      const maxY = Math.max(...points.map(p => p[1]));
      function draw() {
        const cutoff = Number(slider.value);
        document.getElementById('heat-cutoff-value').textContent = `${cutoff.toFixed(1)}°F`;
        const style = getComputedStyle(document.documentElement);
        ctx.fillStyle = style.getPropertyValue('--background').trim();
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        const pad = 35;
        const scale = Math.min((canvas.width - 2 * pad) / maxX, (canvas.height - 2 * pad) / maxY);
        const offsetX = (canvas.width - maxX * scale) / 2;
        let count = 0;
        ctx.fillStyle = style.getPropertyValue('--muted-foreground').trim();
        ctx.globalAlpha = 0.22;
        for (const [x, y] of points) ctx.fillRect(offsetX + x * scale, canvas.height - pad - y * scale, 2, 2);
        ctx.globalAlpha = 0.8;
        ctx.fillStyle = document.documentElement.classList.contains('dark') ? '#ffb47e' : '#b3471f';
        for (const [x, y, t] of points) {
          if (t >= cutoff) { count++; ctx.fillRect(offsetX + x * scale, canvas.height - pad - y * scale, 2.5, 2.5); }
        }
        ctx.globalAlpha = 1;
        status.textContent = `${count.toLocaleString()} of ${points.length.toLocaleString()} observations (${(100 * count / points.length).toFixed(1)}%) at or above ${cutoff.toFixed(1)}°F.`;
        canvas.setAttribute('aria-label', `${status.textContent} Spatial plot of survey routes; north is up.`);
      }
      slider.disabled = false;
      canvas.hidden = false;
      slider.addEventListener('input', draw);
      new MutationObserver(draw).observe(document.documentElement, {attributes:true, attributeFilter:['class']});
      draw();
    })
    .catch(() => { status.textContent = 'The interactive observations could not be loaded. The route map and descriptive figures above remain available.'; });
})();
