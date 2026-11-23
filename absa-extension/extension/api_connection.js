(function (root) {
  'use strict';
  function validate(value) {
    const url = new URL(value.api_url);
    const loopback = ['127.0.0.1', 'localhost'].includes(url.hostname);
    if (!(url.protocol === 'https:' || (url.protocol === 'http:' && loopback)) ||
        url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new Error('Địa chỉ API không hợp lệ.');
    const key = value.api_key || '';
    if (typeof key !== 'string' || /[\r\n]/.test(key)) throw new Error('Khóa API không hợp lệ.');
    return Object.freeze({api_url: url.origin, api_key: key});
  }
  let loading;
  async function read() {
    const response = await fetch(chrome.runtime.getURL('api_config.local.json'), {cache: 'no-store'});
    if (response.status === 404) return validate({api_url: ABSA.API});
    if (!response.ok) throw new Error('Không đọc được cấu hình API.');
    return validate(await response.json());
  }
  // Missing unpacked files can reject fetch instead of producing an HTTP 404.
  const get = () => loading ||= read().catch(error => {
    loading = null;
    if (error instanceof TypeError && /fetch|network|load/i.test(error.message)) return validate({api_url: ABSA.API});
    throw error;
  });
  root.ABSAConnection = Object.freeze({get, validate});
  if (typeof module !== 'undefined') module.exports = root.ABSAConnection;
})(globalThis);
