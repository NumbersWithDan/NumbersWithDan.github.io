(() => {
  'use strict';
  const slider = document.getElementById('weight');
  const rate = document.getElementById('learning-rate');
  const output = document.getElementById('descent-state');
  const point = document.getElementById('loss-point');
  const step = document.getElementById('descent-step');
  let weight = Number(slider.value);
  function render() {
    const loss = (weight - 2) ** 2;
    const inView = weight >= -1 && weight <= 5;
    point.style.display = inView ? '' : 'none';
    point.setAttribute('cx', String(70 + (weight + 1) * 80));
    point.setAttribute('cy', String(200 - loss * 150 / 9));
    slider.value = String(Math.max(-1, Math.min(5, weight)));
    slider.setAttribute('aria-valuetext', inView ? weight.toFixed(2) : 'Outside plotted range; move slider to choose a new start');
    output.textContent = `w = ${weight.toFixed(2)} · Loss = ${loss.toFixed(2)} · Gradient = ${(2 * (weight - 2)).toFixed(2)}` + (inView ? '' : ' · Outside plotted range; reset or move the slider.');
    step.disabled = !inView;
  }
  slider.addEventListener('input', () => { weight = Number(slider.value); render(); });
  step.addEventListener('click', () => { weight -= Number(rate.value) * 2 * (weight - 2); render(); });
  document.getElementById('descent-reset').addEventListener('click', () => { weight = 0; render(); });
  render();
})();
