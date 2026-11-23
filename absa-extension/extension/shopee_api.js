(function (root) {
  'use strict';
  function productIdentity(value) {
    try {
      const url = new URL(value);
      if (url.protocol !== 'https:' || !(url.hostname === 'shopee.vn' || url.hostname.endsWith('.shopee.vn'))) return null;
      const match = url.pathname.match(/-i\.(\d+)\.(\d+)(?:\/|$)/) || url.pathname.match(/^\/product\/(\d+)\/(\d+)(?:\/|$)/);
      return match ? {origin: url.origin, shopid: match[1], itemid: match[2]} : null;
    } catch { return null; }
  }

  // Self-contained: Chrome serializes this function into the product tab's MAIN world.
  // No document.cookie/chrome.cookies access, exported credentials, or arbitrary endpoints.
  async function fetchCurrentProductReviews(options) {
    const reviews = [];
    let scanned = 0, pages = 0;
    const result = error => ({reviews, source: 'shopee_api', scanned, pages,
      requested: options.maxReviews, partial: Boolean(error && reviews.length), error: error || null});
    // One short page per extension message; the sidebar schedules subsequent pages.
    if (!Number.isInteger(options.maxReviews) || options.maxReviews < 1 || options.maxReviews > 50 ||
        !Number.isInteger(options.offset) || options.offset < 0 || options.offset + options.maxReviews > 1000) return result('Lô lấy đánh giá không hợp lệ.');
    function currentMatches() {
      try {
        const url = new URL(location.href);
        if (url.protocol !== 'https:' || !(url.hostname === 'shopee.vn' || url.hostname.endsWith('.shopee.vn'))) return false;
        const ids = url.pathname.match(/-i\.(\d+)\.(\d+)(?:\/|$)/) || url.pathname.match(/^\/product\/(\d+)\/(\d+)(?:\/|$)/);
        return url.origin === options.origin && ids?.[1] === options.shopid && ids?.[2] === options.itemid;
      } catch { return false; }
    }
    if (!currentMatches()) return result('Sản phẩm đã đổi. Hãy lấy lại đánh giá.');
    const limit = Math.min(50, options.maxReviews - scanned);
    const url = new URL('/api/v2/item/get_ratings', location.href);
    url.search = new URLSearchParams({filter: '0', flag: '1', limit: String(limit), offset: String(options.offset),
      type: '0', shopid: options.shopid, itemid: options.itemid}).toString();
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    let payload;
    try {
      const response = await fetch(url.href, {
        method: 'GET', credentials: 'same-origin', redirect: 'error', cache: 'no-store',
        headers: {accept: 'application/json', 'x-api-source': 'pc', 'x-requested-with': 'XMLHttpRequest'},
        signal: controller.signal
      });
      if (!response.ok) {
        const message = [401, 403, 429].includes(response.status)
          ? `Shopee từ chối truy cập (${response.status}). Hãy kiểm tra tab Shopee hoặc nhập tay.`
          : `Không lấy được đánh giá (${response.status}).`;
        return result(message);
      }
      if (!(response.headers.get('content-type') || '').toLowerCase().includes('application/json')) {
        return result('Shopee yêu cầu đăng nhập hoặc xác minh. Hãy kiểm tra tab Shopee.');
      }
      payload = await response.json();
    } catch (error) {
      return result(error.name === 'AbortError' ? 'Shopee phản hồi quá lâu. Hãy thử lại sau.'
        : 'Không lấy được đánh giá. Hãy kiểm tra tab Shopee hoặc nhập tay.');
    } finally { clearTimeout(timeout); }
    if (!currentMatches()) return result('Sản phẩm đã đổi. Hãy lấy lại đánh giá.');
    if (payload?.error) return result('Shopee từ chối yêu cầu. Hãy kiểm tra tab Shopee hoặc nhập tay.');
    const ratings = payload?.data?.ratings;
    if (ratings === null && payload?.data) return result(null);
    if (!Array.isArray(ratings)) return result('Không đọc được dữ liệu Shopee. Hãy thử nhập tay.');
    if (ratings.length > limit) return result('Shopee trả dữ liệu không hợp lệ.');
    pages++;
    scanned += ratings.length;
    // Only buyer comment is returned to the extension; account/profile fields are ignored.
    for (const rating of ratings) if (typeof rating?.comment === 'string' && rating.comment.trim()) reviews.push(rating.comment);

    return result(null);
  }
  root.ABSAShopeeApi = Object.freeze({productIdentity, fetchCurrentProductReviews});
  if (typeof module !== 'undefined') module.exports = root.ABSAShopeeApi;
})(globalThis);
