const fs = require('fs');
const path = require('path');
const { JSDOM } = require('jsdom');

async function main() {
  const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');

  const dom = new JSDOM(html, {
    runScripts: 'dangerously',
    resources: 'usable',
    url: 'http://localhost/',
    pretendToBeVisual: true,
  });

  const { window } = dom;
  // Node 22 has global fetch; expose it to the jsdom window since jsdom doesn't implement fetch itself.
  window.fetch = fetch;

  // Chart.js is loaded via a <script src> to a CDN we can't reach in this sandbox.
  // Stub it out so the app's own logic still runs and we can inspect what it WOULD have charted.
  window.Chart = function(ctx, config) {
    window.__lastChartConfig = config;
    this.destroy = () => {};
  };

  // Canvas 2D context isn't implemented by jsdom; stub getContext so our code doesn't throw.
  window.HTMLCanvasElement.prototype.getContext = () => ({});

  const errors = [];
  window.addEventListener('error', (e) => errors.push(e.error ? e.error.message : e.message));
  window.addEventListener('unhandledrejection', (e) => errors.push('UNHANDLED REJECTION: ' + (e.reason && e.reason.stack ? e.reason.stack : e.reason)));

  await new Promise((resolve) => {
    window.document.addEventListener('DOMContentLoaded', () => setTimeout(resolve, 50));
  });

  // Give the async refreshAll() chain time to complete its fetch calls
  await new Promise((r) => setTimeout(r, 1500));

  const doc = window.document;
  const results = {};

  results.connLabel = doc.getElementById('connLabel').textContent;
  results.netLoadReadout = doc.getElementById('netLoadReadout').textContent;
  results.rawLoadValue = doc.getElementById('rawLoadValue').textContent;
  results.solarValue = doc.getElementById('solarValue').textContent;
  results.statusPill = doc.getElementById('statusPill').textContent;
  results.sheddingContent = doc.getElementById('sheddingContent').innerHTML.slice(0, 200);
  results.batteryContent = doc.getElementById('batteryContent').innerHTML.slice(0, 200);
  results.chartLabelsCount = window.__lastChartConfig ? window.__lastChartConfig.data.labels.length : 'NO CHART CALL';
  results.jsErrors = errors;

  console.log(JSON.stringify(results, null, 2));

  // ---- Now simulate real user interaction: move the hour slider ----
  const hourSlider = doc.getElementById('hourSlider');
  hourSlider.value = '6';
  hourSlider.dispatchEvent(new window.Event('input', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 800));

  console.log('--- after moving hour slider to 6 ---');
  console.log('hourLabel:', doc.getElementById('hourLabel').textContent);
  console.log('netLoadReadout:', doc.getElementById('netLoadReadout').textContent);
  console.log('solarValue (should be near 0 at hour 6):', doc.getElementById('solarValue').textContent);

  // ---- Simulate clicking the heatwave button ----
  const heatBtn = doc.getElementById('btnHeatwave');
  heatBtn.dispatchEvent(new window.Event('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 800));
  console.log('--- after heatwave click ---');
  console.log('tempLabel:', doc.getElementById('tempLabel').textContent);

  // ---- Simulate region switch ----
  const regionButtons = doc.querySelectorAll('#regionSwitch button');
  console.log('region buttons found:', regionButtons.length);
  regionButtons[1].dispatchEvent(new window.Event('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 800));
  console.log('--- after switching region ---');
  console.log('heroRegionLabel:', doc.getElementById('heroRegionLabel').textContent);
  console.log('batteryContent:', doc.getElementById('batteryContent').innerHTML.slice(0, 250));

  // ---- Simulate SMS send ----
  const smsInput = doc.getElementById('smsInput');
  smsInput.value = 'STATUS EMG';
  const smsBtn = doc.getElementById('smsSend');
  smsBtn.dispatchEvent(new window.Event('click', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 800));
  console.log('--- after SMS send ---');
  console.log('smsReply:', doc.getElementById('smsReply').textContent);
  console.log('smsReply class:', doc.getElementById('smsReply').className);

  console.log('--- final JS error count:', errors.length, errors);
}

main().catch(e => { console.error('HARNESS FAILURE:', e); process.exit(1); });
