'use strict';
const $ = value => document.querySelector(value);
async function send(type) {
  const response = await chrome.runtime.sendMessage({type});
  if (!response?.ok) throw new Error(response?.error || 'Extension vừa reload; đóng rồi mở lại popup.');
  return response.data;
}
async function check() {
  $('#api-status').textContent = 'Đang kết nối…'; $('#api-dot').classList.remove('connected');
  try {
    const data = await send('ABSA_HEALTH');
    $('#api-address').textContent = data.api_url || '';
    $('#api-address').style.overflowWrap = 'anywhere';
    $('#api-status').textContent = data.model_loaded ? 'Connected' : 'Chưa sẵn sàng';
    $('#api-dot').classList.toggle('connected', data.model_loaded);
    $('#message').textContent = data.model_loaded ? '' : data.error;
  } catch (error) { $('#api-status').textContent = 'Disconnected'; $('#message').textContent = error.message; }
}
$('#check').addEventListener('click', check);
for (const [selector, type] of [['#open', 'ABSA_OPEN'], ['#manual', 'ABSA_OPEN_MANUAL']]) {
  $(selector).addEventListener('click', async () => {
    try { await send(type); window.close(); } catch (error) { $('#message').textContent = error.message; }
  });
}
chrome.tabs.query({active: true, currentWindow: true}).then(([tab]) => {
  try { $('#domain').textContent = new URL(tab.url).hostname || 'Trang nội bộ Chrome'; }
  catch { $('#domain').textContent = 'Trang nội bộ Chrome'; }
  if (!ABSA.isProductUrl(tab?.url)) $('#message').textContent = 'Mở sản phẩm Shopee hoặc chọn Nhập tay.';
});
check();
