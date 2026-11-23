/* Real MV3 service worker + real local model, in an isolated Chromium profile.
   This never installs the extension into the user's Chrome profile or contacts Shopee. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const {chromium} = require(process.env.ABSA_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..');
const extension = path.join(root, 'extension');
const cache = path.join(root, '.cache');
const configFile = path.join(extension, 'api_config.local.json');
const connection = fs.existsSync(configFile) ? JSON.parse(fs.readFileSync(configFile, 'utf8')) : {api_url: 'http://127.0.0.1:8000'};
const headers = {'Content-Type': 'application/json', ...(connection.api_key ? {'X-API-Key': connection.api_key} : {})};
fs.mkdirSync(cache, {recursive: true});
const report = [];
const passed = name => {report.push(name); console.log('PASS', name);};

(async () => {
  const health = await (await fetch(connection.api_url + '/health')).json();
  assert.equal(health.model_loaded, true, 'Run the actual backend before the E2E test.');
  const profile = fs.mkdtempSync(path.join(cache, 'chromium-test-'));
  const launchOptions = {
    channel: 'chromium', headless: true, viewport: {width: 1120, height: 900},
    args: [`--disable-extensions-except=${extension}`, `--load-extension=${extension}`]
  };
  let context = await chromium.launchPersistentContext(profile, launchOptions);
  try {
    let worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
    const id = worker.url().split('/')[2];
    const page = await context.newPage();
    page.setDefaultTimeout(20000);
    await page.goto(`chrome-extension://${id}/analyzer.html`);
    await page.locator('.connection.connected').waitFor();
    assert.equal(await page.locator('.connection-text').innerText(), 'Connected');
    assert.equal(await page.locator('.visible-tab').count(), 0);
    passed('MV3 loaded, Connected label is short, manual page has no DOM mode');
    const sample = JSON.parse(fs.readFileSync(path.join(root, 'sample_data/reviews.json'), 'utf8'));
    await page.locator('textarea').fill(sample.reviews.join('\n'));
    assert.match(await page.locator('.preview-count').innerText(), /^7 /);
    await page.locator('.analyze').click();
    await page.locator('.status.success').waitFor({timeout: 60000});
    assert.equal(await page.locator('.total-reviews').innerText(), '7');
    assert.equal(await page.locator('.aspect-row').count(), 11);
    const direct = await (await fetch(connection.api_url + '/analyze', {
      method: 'POST', headers, body: JSON.stringify(sample)
    })).json();
    const mentions = Object.values(direct.summary).reduce((sum, a) => sum + a.positive + a.negative + a.neutral, 0);
    assert.equal(await page.locator('.total-mentions').innerText(), String(mentions));
    assert.equal(await page.locator('.review-card').count(), 7);
    await page.screenshot({path: path.join(cache, 'manual-real-model.png'), fullPage: true});
    passed('Manual Input → service worker → real FastAPI → 11-aspect dashboard agrees with API');
    await page.locator('.aspect-title').first().click();
    assert.equal(await page.locator('.reset-filter').isVisible(), true);
    await page.locator('.reset-filter').click();
    assert.equal(await page.locator('.review-card').count(), 7);
    await page.locator('.clear').click();
    assert.equal(await page.locator('.dashboard').isVisible(), false);
    assert.equal(await page.locator('.review-card').count(), 0);
    passed('Aspect filter and Clear Results work without another model request');

    const malicious = 'Rửa sạch <img src=x onerror="window.__xss=1">';
    await page.locator('textarea').fill(`${malicious}\n${malicious}\n\nGiao hàng chậm.`);
    assert.match(await page.locator('.preview-count').innerText(), /^2 /);
    await page.locator('.analyze').click();
    await page.locator('.status.success').waitFor({timeout: 60000});
    assert.equal(await page.locator('.review-card img').count(), 0);
    assert.equal(await page.evaluate(() => window.__xss), undefined);
    assert.ok((await page.locator('.review-text').first().innerText()).includes('<img'));
    passed('Review text is displayed safely; duplicate and blank lines are removed');

    const popup = await context.newPage();
    await popup.goto(`chrome-extension://${id}/popup.html`);
    await popup.locator('#api-dot.connected').waitFor();
    assert.equal(await popup.locator('#api-status').innerText(), 'Connected');
    passed('Popup shows Connected without model/device details');
    await worker.evaluate(() => { self.__originalFetch = self.fetch; self.fetch = () => Promise.reject(new TypeError('test offline')); });
    await popup.locator('#check').click();
    await popup.getByText('Disconnected', {exact: true}).waitFor();
    await worker.evaluate(() => {self.fetch = self.__originalFetch; delete self.__originalFetch;});
    await popup.locator('#check').click();
    await popup.locator('#api-dot.connected').waitFor();
    passed('Popup shows Disconnected with recovery instructions and reconnects');
    const opened = context.waitForEvent('page');
    await popup.locator('#manual').click();
    const manual = await opened;
    await manual.waitForURL(`chrome-extension://${id}/analyzer.html`);
    await manual.locator('.connection.connected').waitFor();
    passed('Popup Manual Input button opens standalone analyzer');

    // Chromium's command-line side-loaded extension can be disabled by runtime.reload().
    // Reopening the same isolated profile with the source flags verifies a fresh worker instead.
    await context.close();
    context = await chromium.launchPersistentContext(profile, launchOptions);
    const reloaded = await context.newPage();
    await reloaded.goto(`chrome-extension://${id}/analyzer.html`);
    await reloaded.locator('.connection.connected').waitFor();
    passed('Restarting Chromium reloads the unpacked source and analyzer reconnects with a fresh worker');
    fs.writeFileSync(path.join(cache, 'e2e-report.json'), JSON.stringify({passed: report, real_model: health.model_version, device: health.device}, null, 2));
    console.log(JSON.stringify({passed: report.length, device: health.device}));
  } finally {await context.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
