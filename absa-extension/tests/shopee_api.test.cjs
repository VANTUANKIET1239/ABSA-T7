const assert = require('node:assert/strict');
const {test} = require('node:test');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const code = fs.readFileSync(path.join(__dirname, '../extension/shopee_api.js'), 'utf8');
const product = {origin: 'https://shopee.vn', shopid: '123', itemid: '456', maxReviews: 50, offset: 0};
function context(pages, href = 'https://shopee.vn/product/123/456') {
  const calls = [];
  const box = vm.createContext({URL, URLSearchParams, AbortController, location: {href}, clearTimeout,
    setTimeout: (fn, delay) => delay === 800 ? (fn(), 0) : setTimeout(fn, delay)});
  box.document = new Proxy({}, {get() {throw new Error('Never read document.cookie');}});
  box.fetch = async (url, options) => {
    calls.push({url, options});
    const next = pages.shift();
    if (next instanceof Error) throw next;
    return {ok: !next?.status || next.status === 200, status: next?.status || 200,
      headers: {get: () => next?.contentType || 'application/json'},
      json: async () => next?.payload || {data: {ratings: next?.ratings || []}}};
  };
  vm.runInContext(code, box);
  return {box, calls, run: options => box.ABSAShopeeApi.fetchCurrentProductReviews({...product, ...options})};
}
test('both URL formats preserve numeric identifiers as strings', () => {
  const c = context([]);
  const get = c.box.ABSAShopeeApi.productIdentity;
  assert.equal(get('https://shopee.vn/name-i.300458619.27080128342').itemid, '27080128342');
  assert.equal(get('https://shopee.vn/product/123/456').shopid, '123');
  assert.equal(get('https://shopee.vn.evil.test/product/123/456'), null);
  assert.equal(get('https://shopee.vn/verify/traffic/error'), null);
});
test('each call reads one bounded same-origin page and returns only comments', async () => {
  const ratings = Array.from({length: 50}, (_, i) => ({comment: 'text ' + i, author_username: 'PRIVATE', seller_response: 'PRIVATE'}));
  const c = context([{ratings}, {ratings: [{comment: 'last'}, {comment: ''}, {comment: null}]}]);
  const first = await c.run();
  const result = await c.run({offset: 50});
  assert.equal(first.reviews.length, 50); assert.equal(result.reviews.length, 1);
  assert.equal(result.scanned, 3);
  assert.equal(result.pages, 1);
  assert.equal(c.calls.length, 2);
  assert.equal(new URL(c.calls[1].url).searchParams.get('offset'), '50');
  for (const call of c.calls) {
    assert.equal(new URL(call.url).origin, 'https://shopee.vn');
    assert.equal(call.options.credentials, 'same-origin');
    assert.equal(call.options.redirect, 'error');
    assert.equal(call.options.headers.cookie, undefined);
  }
  assert.ok(!JSON.stringify(result).includes('PRIVATE'));
});
test('20-review selection stops after one request even when full', async () => {
  const c = context([{ratings: Array.from({length: 20}, () => ({comment: 'x'}))}]);
  const result = await c.run({maxReviews: 20});
  assert.equal(result.scanned, 20); assert.equal(c.calls.length, 1);
  assert.equal(new URL(c.calls[0].url).searchParams.get('limit'), '20');
});

test('custom page sizes use the exact limit and support the final page at offset 950', async () => {
  for (const count of [1, 37, 50]) {
    const first = Math.min(50, count);
    const c = context([{ratings: Array.from({length: first}, () => ({comment: 'ok'}))},
      {ratings: Array.from({length: count - first}, () => ({comment: 'ok'}))}]);
    const result = await c.run({maxReviews: count, offset: 950});
    assert.equal(result.scanned, count);
    assert.equal(c.calls.length, Math.ceil(count / 50));
    assert.equal(new URL(c.calls.at(-1).url).searchParams.get('limit'), String(count > 50 ? count - 50 : count));
    assert.equal(new URL(c.calls[0].url).searchParams.get('offset'), '950');
  }
});
test('invalid count and non-product location make no requests', async () => {
  const c = context([]);
  for (const maxReviews of [0, -1, 1.5, 51, 10000, '50', null]) assert.ok((await c.run({maxReviews})).error);
  for (const offset of [-1, 951, 1.5, '50']) assert.ok((await c.run({offset})).error);
  const other = context([], 'https://example.com/product/123/456');
  assert.ok((await other.run()).error);
  assert.equal(other.calls.length + c.calls.length, 0);
});
test('HTTP auth/block/rate-limit responses stop immediately without retry', async () => {
  for (const status of [401, 403, 429]) {
    const c = context([{status}]);
    const result = await c.run();
    assert.ok(result.error.includes(String(status)));
    assert.equal(result.reviews.length, 0); assert.equal(c.calls.length, 1);
  }
});
test('Shopee verification error or HTML response is not bypassed', async () => {
  for (const response of [{payload: {error: 90309999}}, {contentType: 'text/html'}]) {
    const c = context([response]);
    assert.ok((await c.run()).error); assert.equal(c.calls.length, 1);
  }
});
test('a rejected later page stops without retrying or claiming earlier pages', async () => {
  const c = context([{status: 429}]);
  const result = await c.run({offset: 50});
  assert.equal(result.reviews.length, 0); assert.equal(result.partial, false);
  assert.ok(result.error); assert.equal(c.calls.length, 1);
});
test('unexpected response shape fails instead of reading arbitrary text', async () => {
  const c = context([{payload: {account: 'PRIVATE'}}]);
  const result = await c.run();
  assert.ok(result.error); assert.equal(result.reviews.length, 0);
});
