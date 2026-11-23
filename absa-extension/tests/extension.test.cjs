const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const extension = path.resolve(__dirname, '../extension');
const code = file => fs.readFileSync(path.join(extension, file), 'utf8');
function shared() { const box = vm.createContext({URL}); vm.runInContext(code('shared.js'), box); return box.ABSA; }
const plain = value => JSON.parse(JSON.stringify(value));

function worker(tabUrl = 'https://shopee.vn/product/123/456') {
  let listener;
  const calls = {fetch: [], injected: [], tabs: [], messages: []};
  const box = vm.createContext({URL, AbortController, setTimeout, clearTimeout, console});
  box.importScripts = (...names) => {for (const name of names) vm.runInContext(code(name), box);};
  box.chrome = {
    runtime: {id: 'a'.repeat(32), getURL: name => 'chrome-extension://' + 'a'.repeat(32) + '/' + name,
      onMessage: {addListener: fn => {listener = fn;}}},
    scripting: {executeScript: async data => calls.injected.push(data)},
    tabs: {query: async () => [{id: 1, url: tabUrl}], get: async id => ({id, url: tabUrl}), create: async data => calls.tabs.push(data),
      sendMessage: async (...data) => {calls.messages.push(data); return {ok: true};}}
  };
  box.fetch = async (url, options) => {
    if (url.startsWith('chrome-extension://')) return {ok: false, status: 404};
    calls.fetch.push({url, ...options});
    const reviews = options.body ? JSON.parse(options.body).reviews : null;
    const data = reviews ? {model_version: 'TEST', results: reviews.map((text, i) => ({review_id: i, text, aspects: []}))}
      : {model_loaded: true, model_version: 'TEST', device: 'cpu', max_reviews: 100};
    return {ok: true, json: async () => data};
  };
  vm.runInContext(code('background.js'), box);
  const send = (message, id = box.chrome.runtime.id, extra = {}) => new Promise(resolve => {
    if (!listener({productKey: 'https://shopee.vn/123/456', offset: 0, ...message}, {id, ...extra}, resolve)) resolve({ignored: true});
  });
  return {box, calls, send};
}

test('11 aspect order and actual product URL checks', () => {
  const a = shared();
  assert.equal(a.ASPECTS.length, 11);
  assert.equal(a.ASPECTS[0].aspect, 'CleansingEffectiveness');
  assert.equal(a.ASPECTS[10].aspect, 'General');
  for (const url of ['https://shopee.vn/name-i.12.34', 'https://shopee.vn/product/12/34']) assert.ok(a.isProductUrl(url));
  for (const url of ['https://shopee.vn/', 'https://shopee.vn/verify/traffic/error', 'https://shopee.vn.evil.test/product/12/34', 'http://shopee.vn/product/12/34']) assert.ok(!a.isProductUrl(url));
});
test('deduplication keeps content, removes blank and repeated whitespace variants', () => {
  assert.deepEqual(plain(shared().uniqueReviews(['  tốt ', 'tốt', '', 'giao\n hàng'])),
    {reviews: ['tốt', 'giao hàng'], duplicates: 1, empty: 1});
});
test('None omitted; unique label pairs differ from review mentions', () => {
  const rows = shared().summarize([{aspects: [{code: 'A1', sentiment: 'Positive'}, {code: 'A1', sentiment: 'Positive'},
    {code: 'A1', sentiment: 'Negative'}, {code: 'A2', sentiment: 'None'}]}]);
  assert.equal(rows[0].positive, 1); assert.equal(rows[0].negative, 1); assert.equal(rows[0].mentions, 1);
  assert.equal(rows[1].neutral, 0);
});
test('worker accepts a bounded inference batch and preserves review order', async () => {
  const w = worker();
  const reviews = Array.from({length: 20}, (_, i) => `review ${i}`);
  const response = await w.send({type: 'ABSA_ANALYZE', reviews});
  assert.equal(response.ok, true);
  assert.equal(response.data.results.length, 20);
  assert.equal(response.data.results[19].review_id, 19);
  assert.deepEqual(w.calls.fetch.filter(x => x.body).map(x => JSON.parse(x.body).reviews.length), [20]);
  assert.ok(w.calls.fetch.every(x => x.url.startsWith('http://127.0.0.1:8000/') && x.credentials === 'omit'));
});
test('worker rejects invalid payload and ignores external senders', async () => {
  const w = worker();
  assert.equal((await w.send({type: 'ABSA_ANALYZE', reviews: [123]})).ok, false);
  assert.equal(w.calls.fetch.length, 0);
  assert.equal((await w.send({type: 'ABSA_ANALYZE', reviews: Array(21).fill('x')})).ok, false);
  assert.equal(w.calls.fetch.length, 0);
  assert.deepEqual(await w.send({type: 'ABSA_HEALTH'}, 'other-extension'), {ignored: true});
});
test('Open Analyzer on a product injects only local packaged files, does not analyze', async () => {
  const w = worker();
  assert.equal((await w.send({type: 'ABSA_OPEN'})).data.mode, 'sidebar');
  assert.equal(w.calls.injected.length, 1);
  assert.ok(w.calls.injected[0].files.includes('content.js'));
  assert.ok(w.calls.injected[0].files.includes('product.js'));
  assert.ok(!w.calls.injected[0].files.includes('extraction.js'));
  assert.equal(w.calls.fetch.length, 0);
});
test('Open Analyzer outside Shopee opens an extension page without injecting into other sites', async () => {
  const w = worker('https://example.com');
  assert.equal((await w.send({type: 'ABSA_OPEN'})).data.mode, 'manual');
  assert.equal(w.calls.injected.length, 0);
  assert.ok(w.calls.tabs[0].url.endsWith('/analyzer.html'));
});
test('model-unavailable state returns an error rather than fabricated predictions', async () => {
  const w = worker();
  w.box.fetch = async url => url.startsWith('chrome-extension://') ? {ok: false, status: 404}
    : {ok: true, json: async () => ({model_loaded: false, error: 'missing model'})};
  const response = await w.send({type: 'ABSA_ANALYZE', reviews: ['a']});
  assert.equal(response.ok, false); assert.match(response.error, /missing model/);
});
test('manifest permissions are limited to activeTab, scripting and exact configured API origins', () => {
  const manifest = JSON.parse(code('manifest.json'));
  assert.deepEqual(manifest.permissions, ['activeTab', 'scripting']);
  assert.ok(manifest.host_permissions.includes('http://127.0.0.1:8000/*'));
  for (const host of manifest.host_permissions) {
    const url = new URL(host);
    assert.ok(url.protocol === 'https:' || url.origin === 'http://127.0.0.1:8000');
    assert.ok(!url.hostname.includes('*'));
  }
  assert.equal(manifest.manifest_version, 3);
  assert.equal(manifest.content_scripts, undefined);
});

test('private connection config sends its key only as an API header and never in health data', async () => {
  const w = worker(), original = w.box.fetch;
  w.box.fetch = async (url, options) => {
    if (url.endsWith('/api_config.local.json')) return {ok: true, json: async () => ({api_url: 'https://unit-api.trycloudflare.com', api_key: 'TEST-KEY'})};
    return original(url, options);
  };
  const response = await w.send({type: 'ABSA_HEALTH'});
  assert.equal(response.ok, true);
  assert.equal(response.data.api_url, 'https://unit-api.trycloudflare.com');
  assert.equal(response.data.api_key, undefined);
  assert.equal(w.calls.fetch[0].url, 'https://unit-api.trycloudflare.com/health');
  assert.equal(w.calls.fetch[0].headers['X-API-Key'], 'TEST-KEY');
});

test('connection rejects remote HTTP, URL credentials and non-origin paths', () => {
  const box = vm.createContext({URL}); vm.runInContext(code('api_connection.js'), box);
  for (const api_url of ['http://example.com', 'https://user:pass@example.com', 'https://example.com/path', 'https://example.com?key=secret']) {
    assert.throws(() => box.ABSAConnection.validate({api_url}));
  }
});
test('all packaged JavaScript parses without eval or remote imports', () => {
  for (const file of fs.readdirSync(extension).filter(x => x.endsWith('.js'))) {
    new vm.Script(code(file), {filename: file});
    assert.ok(!/\beval\s*\(|new\s+Function\s*\(/.test(code(file)));
  }
});

test('Shopee API fetch is restricted to a real top-level product tab and bounded count', async () => {
  const w = worker();
  assert.equal((await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 50})).ok, false);
  const sender = {tab: {id: 1}, frameId: 0};
  assert.equal((await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 500}, undefined, sender)).ok, false);
  for (const maxReviews of [0, -1, 1.5, 101, '50']) {
    assert.equal((await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews}, undefined, sender)).ok, false);
  }
  for (const offset of [-1, 951, 1.5, '50']) {
    assert.equal((await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 50, offset}, undefined, sender)).ok, false);
  }
  const other = worker('https://example.com');
  assert.equal((await other.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 50}, undefined, sender)).ok, false);
  assert.equal(w.calls.injected.length, 0);
});

test('stale product context cannot collect reviews from a newly opened product', async () => {
  const w = worker();
  const response = await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 37, productKey: 'https://shopee.vn/123/999'}, undefined, {tab: {id: 1}, frameId: 0});
  assert.equal(response.ok, false);
  assert.equal(w.calls.injected.length, 0);
});
test('Shopee collection injects MAIN in the same tab; sanitizes response without cookie permissions', async () => {
  const w = worker();
  w.box.chrome.scripting.executeScript = async data => {
    w.calls.injected.push(data);
    return [{result: {reviews: ['review'], pages: 1, scanned: 2, account: 'MUST NOT FORWARD', cookie: 'MUST NOT FORWARD'}}];
  };
  const response = await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 50}, undefined, {tab: {id: 1}, frameId: 0});
  assert.equal(response.ok, true);
  assert.equal(w.calls.injected[0].world, 'MAIN');
  assert.equal(w.calls.injected[0].args[0].shopid, '123');
  assert.equal(w.calls.injected[0].args[0].itemid, '456');
  assert.equal(response.data.account, undefined);
  assert.equal(response.data.cookie, undefined);
  assert.equal(w.calls.fetch.length, 0);
});
test('navigation to a different product discards collected results', async () => {
  const w = worker(); let reads = 0;
  w.box.chrome.tabs.get = async id => ({id, url: ++reads === 1 ? 'https://shopee.vn/product/123/456' : 'https://shopee.vn/product/123/999'});
  w.box.chrome.scripting.executeScript = async () => [{result: {reviews: ['old product']}}];
  const response = await w.send({type: 'ABSA_FETCH_SHOPEE', maxReviews: 50}, undefined, {tab: {id: 1}, frameId: 0});
  assert.equal(response.ok, false);
  assert.equal(response.data, undefined);
});
