/* Local inference traffic originates here. Shopee API reads use the current product tab. */
'use strict';
importScripts('shared.js', 'shopee_api.js', 'api_connection.js');
const collectingTabs = new Set();

async function api(path, body) {
  const connection = await ABSAConnection.get();
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 25000);
  try {
    const headers = body ? {'Content-Type': 'application/json'} : {};
    if (connection.api_key) headers['X-API-Key'] = connection.api_key;
    const response = await fetch(connection.api_url + path, {
      method: body ? 'POST' : 'GET', credentials: 'omit', cache: 'no-store',
      headers,
      body: body ? JSON.stringify(body) : undefined, signal: controller.signal
    });
    let data;
    try { data = await response.json(); } catch { throw new Error('API không trả dữ liệu hợp lệ. Kiểm tra Docker và Cloudflare.'); }
    if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : `API trả lỗi ${response.status}.`);
    return data;
  } catch (error) {
    if (error.name === 'AbortError') throw new Error('API phản hồi quá lâu. Thử ít review hơn hoặc kiểm tra log backend.');
    if (error instanceof TypeError) throw new Error('Không kết nối được API. Kiểm tra Docker và Cloudflare.');
    throw error;
  } finally { clearTimeout(timeout); }
}

async function handle(message, sender) {
  switch (message.type) {
    case 'ABSA_STYLES':
      return {css: await (await fetch(chrome.runtime.getURL('sidebar.css'))).text()};
    case 'ABSA_HEALTH': {
      const data = await api('/health');
      if (typeof data.model_loaded !== 'boolean') throw new Error('Địa chỉ hiện tại không phải API ABSA.');
      return {...data, api_url: (await ABSAConnection.get()).api_url};
    }
    case 'ABSA_FETCH_SHOPEE': {
      if (!sender?.tab?.id || sender.frameId !== 0) throw new Error('Chỉ lấy API từ sidebar của trang sản phẩm đang mở.');
      if (!Number.isInteger(message.maxReviews) || message.maxReviews < 1 || message.maxReviews > ABSA.SOURCE_BATCH ||
          !Number.isInteger(message.offset) || message.offset < 0 || message.offset + message.maxReviews > ABSA.MAX_REVIEWS) throw new Error('Lô lấy đánh giá không hợp lệ.');
      const tab = await chrome.tabs.get(sender.tab.id);
      const identity = ABSAShopeeApi.productIdentity(tab.url);
      if (!identity) throw new Error('Hãy mở URL chi tiết sản phẩm Shopee hợp lệ.');
      if (message.productKey !== `${identity.origin}/${identity.shopid}/${identity.itemid}`) throw new Error('Sản phẩm đã đổi. Hãy lấy lại đánh giá.');
      if (collectingTabs.has(tab.id)) throw new Error('Tab này đang lấy review. Hãy đợi hoàn tất.');
      collectingTabs.add(tab.id);
      try {
        const injected = await chrome.scripting.executeScript({target: {tabId: tab.id}, world: 'MAIN',
          func: ABSAShopeeApi.fetchCurrentProductReviews, args: [{...identity, maxReviews: message.maxReviews, offset: message.offset}]});
        const latest = ABSAShopeeApi.productIdentity((await chrome.tabs.get(tab.id)).url);
        if (!latest || ['origin', 'shopid', 'itemid'].some(key => latest[key] !== identity[key])) {
          throw new Error('Trang sản phẩm đã thay đổi trong lúc lấy review. Không sử dụng kết quả cũ.');
        }
        const data = injected[0]?.result;
        if (!data || !Array.isArray(data.reviews) || data.reviews.length > message.maxReviews || data.reviews.some(x => typeof x !== 'string') ||
            !Number.isInteger(data.scanned) || data.scanned < data.reviews.length || data.scanned > message.maxReviews ||
            !Number.isInteger(data.pages) || data.pages < 0 || data.pages > 1) {
          throw new Error('Không nhận được danh sách review hợp lệ từ trang Shopee.');
        }
        // Never forward arbitrary page objects or metadata to the UI/backend.
        return {reviews: data.reviews, source: 'shopee_api', requested: message.maxReviews,
          scanned: Number.isInteger(data.scanned) ? data.scanned : 0, pages: Number.isInteger(data.pages) ? data.pages : 0,
          partial: Boolean(data.partial), error: typeof data.error === 'string' ? data.error : null};
      } finally { collectingTabs.delete(tab.id); }
    }
    case 'ABSA_ANALYZE': {
      if (!Array.isArray(message.reviews) || !message.reviews.length || message.reviews.length > ABSA.ANALYSIS_BATCH ||
          message.reviews.some(x => typeof x !== 'string' || x.length > ABSA.MAX_CHARS)) {
        throw new Error('Mỗi lô tối đa 20 đánh giá, mỗi đánh giá tối đa 4.000 ký tự.');
      }
      const health = await api('/health');
      if (!health.model_loaded) throw new Error(health.error || 'Mô hình chưa sẵn sàng. Kiểm tra /health.');
      const size = Math.max(1, Math.min(20, health.max_reviews || 20));
      const results = [];
      // Short sequential requests respect MV3's fetch-response timeout and API limits.
      for (let start = 0; start < message.reviews.length; start += size) {
        const chunk = message.reviews.slice(start, start + size);
        const data = await api('/analyze', {reviews: chunk});
        if (!Array.isArray(data.results) || data.results.length !== chunk.length) throw new Error('API trả thiếu review; không tổng hợp kết quả không đầy đủ.');
        if (data.model_version !== health.model_version) throw new Error('Model thay đổi giữa các batch. Hãy phân tích lại.');
        for (const [i, result] of data.results.entries()) {
          if (result.review_id !== i || result.text !== chunk[i] || !Array.isArray(result.aspects)) throw new Error('Kết quả API không khớp review gửi đi.');
          results.push({...result, review_id: start + i});
        }
      }
      return {total_reviews: results.length, results, model_version: health.model_version, device: health.device};
    }
    case 'ABSA_OPEN': {
      const [tab] = await chrome.tabs.query({active: true, currentWindow: true});
      if (tab?.id && ABSA.isProductUrl(tab.url)) {
        await chrome.scripting.executeScript({target: {tabId: tab.id}, files: ['shared.js', 'product.js', 'sidebar.js', 'content.js']});
        const shown = await chrome.tabs.sendMessage(tab.id, {type: 'ABSA_SHOW'});
        if (!shown?.ok) throw new Error(shown?.error || 'Không mở được sidebar. Hãy tải lại trang.');
        return {mode: 'sidebar'};
      }
      await chrome.tabs.create({url: chrome.runtime.getURL('analyzer.html')});
      return {mode: 'manual'};
    }
    case 'ABSA_OPEN_MANUAL':
      await chrome.tabs.create({url: chrome.runtime.getURL('analyzer.html')});
      return {mode: 'manual'};
    default: throw new Error('Lệnh extension không hợp lệ.');
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (sender.id !== chrome.runtime.id || !message || typeof message.type !== 'string') return false;
  handle(message, sender).then(data => sendResponse({ok: true, data})).catch(error => sendResponse({ok: false, error: error.message}));
  return true;
});
