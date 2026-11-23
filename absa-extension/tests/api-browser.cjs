/* Isolated MV3 test: Shopee requests use local fixtures and dummy cookies.
   The test-only host grant substitutes the toolbar activeTab gesture in headless Chrome. */
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const {chromium} = require(process.env.ABSA_PLAYWRIGHT_MODULE || 'playwright');
const root = path.resolve(__dirname, '..'), cache = path.join(root, '.cache');
fs.mkdirSync(cache, {recursive: true});
const fixtureExtension = fs.mkdtempSync(path.join(cache, 'api-test-extension-'));
fs.cpSync(path.join(root, 'extension'), fixtureExtension, {recursive: true});
const manifest = JSON.parse(fs.readFileSync(path.join(fixtureExtension, 'manifest.json')));
manifest.name = 'TEST FIXTURE – Shopee API integration';
manifest.host_permissions.push('https://shopee.vn/*');
fs.writeFileSync(path.join(fixtureExtension, 'manifest.json'), JSON.stringify(manifest));
const report = [];
const pass = text => {report.push(text); console.log('PASS', text);};
const waitUntil = async check => {
  for (let i = 0; i < 200; i++) {if (await check()) return; await new Promise(resolve => setTimeout(resolve, 50));}
  throw new Error('Timed out waiting for browser state');
};

(async () => {
  const context = await chromium.launchPersistentContext(fs.mkdtempSync(path.join(cache, 'api-test-profile-')), {
    channel: 'chromium', headless: true, viewport: {width: 1100, height: 900},
    args: [`--disable-extensions-except=${fixtureExtension}`, `--load-extension=${fixtureExtension}`]
  });
  try {
    const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
    const productUrl = 'https://shopee.vn/product/123/456';
    let blocked = false, cookieSent = false, requests = 0, collectGate = null, collectWaiting = false, bulk = false, stopAt = null;
    const requestedLimits = [];
    const offsets = [], requestTimes = [];
    const fixture = fs.readFileSync(path.join(__dirname, 'shopee_product_fixture.html'), 'utf8');
    await context.route('https://shopee.vn/**', async route => {
      const url = new URL(route.request().url());
      if (url.pathname === '/api/v2/item/get_ratings') {
        requests++; requestedLimits.push(Number(url.searchParams.get('limit')));
        offsets.push(Number(url.searchParams.get('offset'))); requestTimes.push(Date.now());
        const headers = await route.request().allHeaders();
        cookieSent ||= (headers.cookie || '').includes('absa_test_session=fixture-only');
        if (collectGate) {collectWaiting = true; await collectGate;}
        if (blocked || (stopAt !== null && Number(url.searchParams.get('offset')) >= stopAt)) return route.fulfill({status: 403, contentType: 'application/json', body: '{}'});
        assert.equal(url.searchParams.get('shopid'), '123');
        if (bulk) {
          const offset = Number(url.searchParams.get('offset')), limit = Number(url.searchParams.get('limit'));
          return route.fulfill({contentType: 'application/json', body: JSON.stringify({error: 0, data: {
            ratings: Array.from({length: limit}, (_, i) => ({comment: `Rửa sạch, không khô da. Lần ${offset + i + 1}`}))}})});
        }
        return route.fulfill({contentType: 'application/json', body: JSON.stringify({error: 0, data: {ratings: [
          {comment: 'Rửa sạch, không khô da.', author_username: 'MUST NOT RETURN'},
          {comment: 'Giao hàng chậm.', seller_response: 'MUST NOT RETURN'},
          {comment: 'Giao hàng chậm.'}, {comment: ''}
        ]}})});
      }
      return route.fulfill({contentType: 'text/html', body: fixture});
    });
    await context.addCookies([{name: 'absa_test_session', value: 'fixture-only', domain: 'shopee.vn',
      path: '/', httpOnly: true, secure: true, sameSite: 'Lax'}]);
    const page = await context.newPage(); page.setDefaultTimeout(15000);
    await page.goto(productUrl);
    const tabId = await worker.evaluate(async url => (await chrome.tabs.query({})).find(t => t.url === url).id, productUrl);
    const inject = () => worker.evaluate(async tab => {
      await chrome.scripting.executeScript({target: {tabId: tab}, files: ['shared.js', 'product.js', 'sidebar.js', 'content.js']});
      return chrome.tabs.sendMessage(tab, {type: 'ABSA_SHOW'});
    }, tabId);
    const ui = selector => page.locator(`#shopee-absa-analyzer ${selector}`);
    const changePage = (itemid, title) => page.evaluate(({itemid, title}) => {
      history.pushState({}, '', `/product/123/${itemid}`);
      document.querySelector('h1').textContent = title; document.title = `${title} | Shopee`;
    }, {itemid, title});
    const navigate = async (itemid, title) => {await changePage(itemid, title);
      await ui('.current-product-id').filter({hasText: `Mã sản phẩm: ${itemid}`}).waitFor();};
    await inject(); await ui('.connection.connected').waitFor();
    assert.equal(await ui('.connection-text').innerText(), 'Connected');
    assert.equal(await ui('.visible-tab').count(), 0);
    assert.equal(await ui('.tabs .tab').count(), 2);
    assert.equal(await ui('.current-product-name').innerText(), 'Sữa rửa mặt A');
    assert.ok((await ui('.current-product-id').innerText()).includes('456'));
    pass('Sidebar defaults to Shopee, has no DOM mode, and shows current product with Connected only');

    await ui('#absa-api-preset').selectOption('20');
    assert.equal(await ui('#absa-api-limit').inputValue(), '20');
    await ui('#absa-api-limit').fill('37');
    assert.equal(await ui('#absa-api-preset').inputValue(), 'custom');
    for (const value of ['', '0', '1001', '2.5']) {
      await ui('#absa-api-limit').fill(value);
      assert.equal(await ui('.collect-api').isDisabled(), true);
      assert.equal(await ui('#limit-error').isVisible(), true);
    }
    await ui('#absa-api-limit').fill('37');
    assert.equal(await ui('.collect-api').isEnabled(), true);
    pass('Preset and typed counts stay in sync; invalid counts are rejected');

    await ui('.collect-api').click();
    await ui('.preview-count').filter({hasText: '2 đánh giá'}).waitFor();
    assert.equal(requests, 1); assert.equal(requestedLimits[0], 37); assert.equal(cookieSent, true);
    assert.ok(!(await ui('.preview').innerText()).includes('MUST NOT RETURN'));
    assert.equal(await ui('.total-reviews').innerText(), '0');
    pass('Typed count reaches MAIN fetch; session is used, only deduplicated comments return, no automatic inference');

    await ui('.analyze').click();
    await ui('.status.success').waitFor({timeout: 60000});
    assert.equal(await ui('.total-reviews').innerText(), '2');
    assert.equal(await ui('.result-product-name').innerText(), 'Sữa rửa mặt A');
    assert.ok((await ui('.result-product-id').innerText()).includes('456'));
    await page.screenshot({path: path.join(cache, 'api-sidebar-fixture.png')});
    pass('Real FastAPI/model results show the analyzed product name and ID');

    await inject(); assert.equal(await page.locator('#shopee-absa-analyzer').count(), 1);
    const dimensions = await page.locator('#shopee-absa-analyzer').evaluate(host => ({
      width: host.getBoundingClientRect().width, height: host.clientHeight,
      screen: document.documentElement.clientHeight, main: host.shadowRoot.querySelector('.main').clientHeight}));
    assert.equal(dimensions.width, 440); assert.equal(dimensions.height, dimensions.screen);
    assert.ok(dimensions.main < dimensions.height);
    pass('Sidebar remains a single panel with internal scrolling');

    await navigate('999', 'Sữa rửa mặt B');
    assert.equal(await ui('.dashboard').isVisible(), false);
    assert.equal(await ui('.preview').isVisible(), false);
    assert.equal(await ui('.review-card').count(), 0);
    assert.equal(await ui('.analyze').isDisabled(), true);
    assert.equal(await ui('.current-product-name').innerText(), 'Sữa rửa mặt B');
    pass('SPA navigation updates product identity and clears old results and pending reviews');

    await ui('.close').click(); await changePage('456', 'Sữa rửa mặt A'); await inject();
    await ui('.current-product-name').filter({hasText: 'Sữa rửa mặt A'}).waitFor();
    assert.equal(await ui('.dashboard').isVisible(), false);
    pass('Closing and reopening on another product starts without old results');

    let releaseCollection;
    collectGate = new Promise(resolve => {releaseCollection = resolve;}); collectWaiting = false;
    await ui('.collect-api').click(); await waitUntil(() => collectWaiting);
    await navigate('999', 'Sữa rửa mặt B'); releaseCollection(); collectGate = null;
    await waitUntil(() => worker.evaluate(tab => !collectingTabs.has(tab), tabId));
    await new Promise(resolve => setTimeout(resolve, 250));
    assert.equal(await ui('.preview').isVisible(), false);
    assert.equal(await ui('.analyze').isDisabled(), true);
    assert.equal(await ui('.current-product-name').innerText(), 'Sữa rửa mặt B');
    pass('Late API responses are discarded after product changes');

    await ui('.collect-api').click();
    await ui('.preview-count').filter({hasText: '2 đánh giá'}).waitFor();
    // Pause the real worker/model request, then change product before releasing it.
    await worker.evaluate(() => {self.__realFetch = self.fetch; self.__modelWaiting = false; self.__modelFinished = false;
      self.fetch = async (...args) => {if (String(args[0]).endsWith('/analyze')) {
        self.__modelWaiting = true; await new Promise(resolve => {self.__releaseModel = resolve;});
        const response = await self.__realFetch(...args); self.__modelFinished = true; return response;
      } return self.__realFetch(...args);};});
    await ui('.analyze').click();
    await waitUntil(() => worker.evaluate(() => self.__modelWaiting));
    await navigate('456', 'Sữa rửa mặt A');
    await worker.evaluate(() => {self.fetch = self.__realFetch; self.__releaseModel();});
    await waitUntil(() => worker.evaluate(() => self.__modelFinished));
    await new Promise(resolve => setTimeout(resolve, 250));
    assert.equal(await ui('.dashboard').isVisible(), false);
    assert.equal(await ui('.review-card').count(), 0);
    assert.equal(await ui('.current-product-name').innerText(), 'Sữa rửa mặt A');
    pass('In-flight real model results are discarded after product changes');

    blocked = true;
    const beforeBlocked = requests;
    await ui('.collect-api').click();
    await ui('.status.error').filter({hasText: '403'}).waitFor();
    assert.equal(requests, beforeBlocked + 1);
    assert.equal(await ui('.analyze').isDisabled(), true);
    await ui('.manual-tab').click(); await ui('textarea').fill('Rửa sạch, không khô da.');
    assert.equal(await ui('.analyze').isEnabled(), true);
    pass('403 stops without retry; manual input remains available');

    await page.evaluate(() => history.pushState({}, '', '/product/123/777'));
    await ui('.current-product-name').filter({hasText: 'Sản phẩm #777'}).waitFor();
    await page.evaluate(() => {document.querySelector('h1').textContent = 'Sữa rửa mặt C';});
    await ui('.current-product-name').filter({hasText: 'Sữa rửa mặt C'}).waitFor();
    await page.screenshot({path: path.join(cache, 'api-product-changed.png')});
    pass('Lagging SPA headings use the new product ID until its name arrives');

    // Full 1,000-review workflow: mocked Shopee pages, real local model in 50 small requests.
    blocked = false; bulk = true;
    await ui('.api-tab').click(); await ui('#absa-api-preset').selectOption('1000');
    assert.equal(await ui('#absa-api-limit').inputValue(), '1000');
    const base = requests;
    await ui('.collect-api').click();
    await ui('.status.loading').filter({hasText: '/1000'}).waitFor();
    assert.equal(await ui('.job-progress').isVisible(), true);
    await ui('.preview-count').filter({hasText: '1000 đánh giá'}).waitFor({timeout: 60000});
    await ui('.collect-api').waitFor({state: 'visible'});
    await waitUntil(() => ui('.analyze').isEnabled());
    assert.equal(requests - base, 20);
    assert.deepEqual(offsets.slice(base), Array.from({length: 20}, (_, i) => i * 50));
    assert.ok(requestedLimits.slice(base).every(n => n === 50));
    assert.ok(requestTimes.slice(base + 1).every((time, i) => time - requestTimes[base + i] >= 700));
    pass('1000 source reviews are collected in 20 sequential pages with pacing and progress');

    await worker.evaluate(() => {self.__bulkFetch = self.fetch; self.__sizes = []; self.__active = 0; self.__peak = 0;
      self.fetch = async (...args) => {
        if (!String(args[0]).endsWith('/analyze')) return self.__bulkFetch(...args);
        self.__sizes.push(JSON.parse(args[1].body).reviews.length); self.__active++; self.__peak = Math.max(self.__peak, self.__active);
        try {return await self.__bulkFetch(...args);} finally {self.__active--;}
      };});
    await ui('.analyze').click();
    await ui('.status.loading').filter({hasText: '/1000'}).waitFor();
    assert.equal(await ui('.job-progress').isVisible(), true);
    // CPU/Docker can take much longer than CUDA for 50 sequential model requests.
    await ui('.status.success').filter({hasText: '1000'}).waitFor({timeout: 1500000});
    const batching = await worker.evaluate(() => ({sizes: self.__sizes, peak: self.__peak}));
    assert.equal(batching.sizes.length, 50); assert.ok(batching.sizes.every(n => n === 20));
    assert.equal(batching.peak, 1); assert.equal(await ui('.total-reviews').innerText(), '1000');
    assert.equal(await ui('.review-card').count(), 10);
    await ui('.more-reviews').click(); assert.equal(await ui('.review-card').count(), 20);
    assert.ok((await ui('.review-text').first().innerText()).endsWith('Lần 1'));
    assert.ok((await ui('.review-text').last().innerText()).endsWith('Lần 20'));
    assert.equal(await ui('.result-product-name').innerText(), 'Sữa rửa mặt C');
    assert.equal(await ui('.job-progress').isVisible(), false);
    await page.screenshot({path: path.join(cache, 'api-batch-1000.png')});
    pass('1000 reviews run through the real model in 50 sequential batches of 20 with correct totals and display order');

    // Keep a valid partial collection if a later page is rejected; never auto-retry.
    stopAt = 50;
    const beforePartial = requests;
    await ui('.collect-api').click();
    await ui('.status.error').filter({hasText: '403'}).waitFor();
    assert.equal(requests - beforePartial, 2);
    assert.equal(await ui('.preview-count').innerText(), '50 đánh giá');
    assert.ok((await ui('.preview-note').innerText()).includes('Chưa lấy đủ'));
    assert.equal(await ui('.analyze').isEnabled(), true);
    pass('A later blocked source page preserves the earlier reviews and clearly marks the collection incomplete');

    // Closing the panel while page one is in flight must prevent page two.
    stopAt = null; let releaseClose;
    collectGate = new Promise(resolve => {releaseClose = resolve;}); collectWaiting = false;
    const beforeClose = requests;
    await ui('.collect-api').click(); await waitUntil(() => collectWaiting);
    await ui('.close').click(); releaseClose(); collectGate = null;
    await waitUntil(() => worker.evaluate(tab => !collectingTabs.has(tab), tabId));
    await new Promise(resolve => setTimeout(resolve, 1000));
    assert.equal(requests - beforeClose, 1);
    pass('Closing the panel stops scheduling additional collection batches');

    fs.writeFileSync(path.join(cache, 'api-browser-report.json'), JSON.stringify({passed: report,
      network: 'Shopee intercepted by local fixture; no live API access tested', shipping_permissions_unchanged: true}, null, 2));
    console.log(JSON.stringify({passed: report.length, requests, dummy_cookie_used: cookieSent, real_model_batches: batching.sizes.length}));
  } finally {await context.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
