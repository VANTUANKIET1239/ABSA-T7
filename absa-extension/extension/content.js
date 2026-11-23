(function () {
  'use strict';
  if (globalThis.__absaContentReady) return;
  globalThis.__absaContentReady = true;
  let panel = null, opening = null, stopWatching = null;
  chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message.type !== 'ABSA_SHOW' || sender.id !== chrome.runtime.id) return false;
    if (panel?.host.isConnected) { panel.updateProduct(ABSAProduct.current()); panel.shadow.querySelector('.close').focus(); sendResponse({ok: true}); return false; }
    opening ||= ABSASidebar.create({product: ABSAProduct.current(), getProduct: ABSAProduct.current,
      onClose: () => {stopWatching?.(); stopWatching = null; panel = null;}})
      .then(value => {panel = value; stopWatching = ABSAProduct.watch(product => panel?.updateProduct(product));})
      .finally(() => {opening = null;});
    opening.then(() => sendResponse({ok: true})).catch(() => sendResponse({ok: false, error: 'Không mở được sidebar. Hãy tải lại trang.'}));
    return true;
  });
})();
