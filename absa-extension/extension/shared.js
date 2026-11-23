(function (root) {
  'use strict';
  if (root.ABSA) return;
  const ASPECTS = [
    ['CleansingEffectiveness', 'Hiệu quả làm sạch'], ['HydrationSkinComfort', 'Cấp ẩm và độ dễ chịu'],
    ['SkinCompatibilitySafety', 'Mức độ an toàn'], ['TextureFoaming', 'Kết cấu và tạo bọt'],
    ['Fragrance', 'Mùi hương'], ['IngredientsFormulation', 'Thành phần'], ['Packaging', 'Bao bì'],
    ['PriceValue', 'Giá cả'], ['ShopService', 'Dịch vụ cửa hàng'], ['Delivery', 'Giao hàng'],
    ['General', 'Đánh giá tổng quan']
  ].map(([aspect, name], i) => Object.freeze({code: `A${i + 1}`, aspect, name}));
  const SENTIMENTS = ['Positive', 'Negative', 'Neutral'];
  const LABELS = {Positive: 'Tích cực', Negative: 'Tiêu cực', Neutral: 'Trung tính'};
  function productIdentity(value) {
    try {
      const url = new URL(value);
      if (url.protocol !== 'https:' || !(url.hostname === 'shopee.vn' || url.hostname.endsWith('.shopee.vn'))) return null;
      const ids = url.pathname.match(/-i\.(\d+)\.(\d+)(?:\/|$)/) || url.pathname.match(/^\/product\/(\d+)\/(\d+)(?:\/|$)/);
      return ids ? {key: `${url.origin}/${ids[1]}/${ids[2]}`, url: `${url.origin}/product/${ids[1]}/${ids[2]}`,
        origin: url.origin, shopid: ids[1], itemid: ids[2]} : null;
    } catch { return null; }
  }
  const isProductUrl = value => Boolean(productIdentity(value));
  function uniqueReviews(values) {
    const seen = new Set(), reviews = [];
    let duplicates = 0, empty = 0;
    for (const value of values) {
      const text = String(value).normalize('NFC').replace(/\s+/g, ' ').trim();
      if (!text) { empty++; continue; }
      if (seen.has(text)) { duplicates++; continue; }
      seen.add(text); reviews.push(text);
    }
    return {reviews, duplicates, empty};
  }
  function summarize(results) {
    const rows = ASPECTS.map(a => ({...a, positive: 0, negative: 0, neutral: 0, mentions: 0}));
    const byCode = new Map(rows.map(a => [a.code, a]));
    for (const review of results) {
      const pairs = new Set(), mentioned = new Set();
      for (const a of review.aspects || []) {
        const row = byCode.get(a.code), key = `${a.code}:${a.sentiment}`;
        if (!row || !SENTIMENTS.includes(a.sentiment) || pairs.has(key)) continue;
        pairs.add(key); mentioned.add(a.code); row[a.sentiment.toLowerCase()]++;
      }
      for (const code of mentioned) byCode.get(code).mentions++;
    }
    return rows;
  }
  root.ABSA = Object.freeze({ASPECTS, SENTIMENTS, LABELS, isProductUrl, productIdentity, uniqueReviews, summarize,
    API: 'http://127.0.0.1:8000', MAX_REVIEWS: 1000, SOURCE_BATCH: 50, ANALYSIS_BATCH: 20, SOURCE_DELAY: 800, MAX_CHARS: 4000});
  if (typeof module !== 'undefined') module.exports = root.ABSA;
})(globalThis);
