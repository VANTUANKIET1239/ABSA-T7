(function (root) {
  'use strict';
  if (root.ABSAProduct) return;
  let previousKey = '', previousTitle = '', staleTitle = null;
  // Product metadata only. Review content is collected exclusively through the API.
  function current() {
    const identity = ABSA.productIdentity(location.href);
    const title = (document.querySelector('[data-sqe="name"] h1, .product-briefing h1, h1')?.textContent ||
      document.querySelector('meta[property="og:title"]')?.content || document.title || '')
      .replace(/\s+/g, ' ').replace(/\s*[|–-]\s*Shopee(?:\s+Việt Nam)?\s*$/i, '').trim().slice(0, 250);
    if ((identity?.key || '') !== previousKey) {
      // SPAs may change the URL before replacing the previous product's heading.
      staleTitle = previousKey && title === previousTitle ? title : null;
    } else if (staleTitle !== null && title !== staleTitle) staleTitle = null;
    previousKey = identity?.key || ''; previousTitle = title;
    return identity ? {...identity, title: title && title !== staleTitle ? title : `Sản phẩm #${identity.itemid}`} : null;
  }
  function watch(callback) {
    let signature = '';
    const check = () => {
      const value = current(), next = JSON.stringify(value);
      if (next !== signature) { signature = next; callback(value); }
    };
    check();
    const timer = setInterval(check, 500);
    addEventListener('popstate', check); addEventListener('pageshow', check);
    return () => {clearInterval(timer); removeEventListener('popstate', check); removeEventListener('pageshow', check);};
  }
  root.ABSAProduct = Object.freeze({current, watch});
})(globalThis);
