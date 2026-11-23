(function (root) {
  'use strict';
  if (root.ABSASidebar) return;
  const template = `
    <div class="app">
      <header class="header"><div class="brand-mark" aria-hidden="true">a<span>.</span></div><div class="header-copy"><h1>Phân tích đánh giá</h1></div><button class="icon-button close" aria-label="Đóng">×</button></header>
      <div class="connection"><span class="connection-dot"></span><span class="connection-text">Đang kết nối…</span><button class="link-button reconnect">Kết nối lại</button></div>
      <main class="main">
        <section class="product-card"><span class="context-label">Đang mở</span><a class="current-product-name" target="_blank" rel="noopener noreferrer"></a><p class="current-product-id"></p></section>
        <section class="input-card"><div class="tabs" role="tablist"><button class="tab api-tab" role="tab">Shopee</button><button class="tab manual-tab" role="tab">Nhập tay</button></div>
          <div class="api-input"><label for="absa-api-limit">Số đánh giá tối đa</label><div class="limit-controls"><select id="absa-api-preset" aria-label="Chọn nhanh số đánh giá"><option value="20">20 đánh giá</option><option value="50" selected>50 đánh giá</option><option value="100">100 đánh giá</option><option value="200">200 đánh giá</option><option value="500">500 đánh giá</option><option value="1000">1.000 đánh giá</option><option value="custom">Tự nhập</option></select><input id="absa-api-limit" type="number" min="1" max="1000" step="1" value="50" inputmode="numeric" aria-describedby="limit-help limit-error"></div><p id="limit-help" class="helper">Chọn hoặc nhập từ 1–1.000.</p><p id="limit-error" class="limit-error" role="alert" hidden></p><button class="button secondary collect-api">Lấy đánh giá</button></div>
          <div class="manual-input" hidden><label for="absa-reviews">Nội dung đánh giá</label><textarea id="absa-reviews" rows="5" placeholder="Mỗi dòng một đánh giá"></textarea><p class="helper">Tối đa 1.000 đánh giá · 4.000 ký tự/đánh giá</p></div>
          <div class="preview" hidden><strong class="preview-count"></strong><p class="preview-note"></p><details><summary>Xem trước</summary><ol class="preview-list"></ol></details></div>
          <button class="button primary analyze" disabled>Phân tích</button>
        </section>
        <div class="status" role="status" aria-live="polite"></div>
        <progress class="job-progress" aria-label="Tiến độ xử lý" max="1" value="0" hidden></progress>
        <section class="empty"><div class="empty-icon" aria-hidden="true">▥</div><h2>Chưa có kết quả</h2><p>Lấy đánh giá hoặc nhập nội dung để bắt đầu.</p></section>
        <div class="dashboard" hidden>
          <section class="summary-card"><div class="section-heading"><h2>Kết quả</h2><button class="link-button clear">Xóa kết quả</button></div><div class="result-context"><span class="context-label">Kết quả của</span><strong class="result-product-name"></strong><span class="result-product-id"></span></div><p class="source-note helper"></p><div class="metrics"><div><strong class="total-reviews">0</strong><span>đánh giá đã phân tích</span></div><div><strong class="total-mentions">0</strong><span>lượt khía cạnh–cảm xúc</span></div></div><div class="overall-bar" aria-hidden="true"></div><div class="overall-labels"></div><p class="helper">Tỷ lệ tính trên các lượt khía cạnh–cảm xúc.</p></section>
          <section class="aspect-section"><div class="section-heading"><h2>Cảm xúc theo khía cạnh</h2><span class="small-label">% trong từng khía cạnh</span></div><div class="legend"><span><i class="positive"></i>Tích cực</span><span><i class="negative"></i>Tiêu cực</span><span><i class="neutral"></i>Trung tính</span></div><div class="aspect-list"></div></section>
          <section class="attention-card"><h2>Khía cạnh bị chê nhiều</h2><p class="helper">Theo tỷ lệ tiêu cực, kèm số đánh giá.</p><div class="negative-list"></div></section>
          <section class="review-section"><div class="section-heading"><h2>Chi tiết đánh giá</h2><button class="link-button reset-filter" hidden>Xóa lọc</button></div><p class="review-filter helper"></p><div class="review-list"></div><button class="button secondary more-reviews" hidden>Xem thêm</button></section>
        </div>
        <footer>Đánh giá được xử lý qua API.</footer>
      </main>
    </div>`;
  function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }
  async function create(options = {}) {
    const host = document.createElement('section');
    host.id = 'shopee-absa-analyzer'; host.setAttribute('aria-label', 'Phân tích cảm xúc review');
    if (options.embedded) host.setAttribute('embedded', '');
    const shadow = host.attachShadow({mode: 'open'});
    const stylesheet = new CSSStyleSheet();
    const styles = await chrome.runtime.sendMessage({type: 'ABSA_STYLES'});
    if (!styles?.ok || typeof styles.data?.css !== 'string') throw new Error('Không đọc được CSS của extension.');
    stylesheet.replaceSync(styles.data.css);
    shadow.adoptedStyleSheets = [stylesheet];
    // The template is static packaged markup. All page/API/user text uses textContent below.
    const container = document.createElement('div'); container.className = 'absa-root'; container.innerHTML = template; shadow.append(container);
    (options.mount || document.documentElement).append(host);
    const $ = value => shadow.querySelector(value);
    const request = message => chrome.runtime.sendMessage(message).then(response => {
      if (!response?.ok) throw new Error(response?.error || 'Extension vừa được reload. Hãy tải lại trang và mở lại bảng.');
      return response.data;
    });
    let mode = options.embedded ? 'manual' : 'api', pending = [], results = [], filter = null, shown = 10, generation = 0, busy = false, fetching = false, sourceNote = '';
    let product = options.product || null, pendingProduct = null, resultProduct = null, analysisBatch = ABSA.ANALYSIS_BATCH;
    function showProduct() {
      $('.product-card').hidden = Boolean(options.embedded);
      const name = $('.current-product-name');
      name.textContent = product?.title || 'Chưa mở sản phẩm Shopee';
      if (product) name.href = product.url; else name.removeAttribute('href');
      name.title = product?.title || '';
      $('.current-product-id').textContent = product ? `Mã sản phẩm: ${product.itemid} · Shop: ${product.shopid}` : '';
      $('.result-product-name').textContent = resultProduct?.title || 'Đánh giá nhập tay';
      $('.result-product-id').textContent = resultProduct ? `Mã sản phẩm: ${resultProduct.itemid} · Shop: ${resultProduct.shopid}` : '';
    }
    const previousFocus = document.activeElement;
    function status(text, type = '') {
      $('.status').textContent = text; $('.status').className = `status ${type}`;
      $('.status').setAttribute('role', type === 'error' ? 'alert' : 'status');
    }
    function progress(completed, total) {
      $('.job-progress').hidden = false; $('.job-progress').max = Math.max(1, total); $('.job-progress').value = completed;
    }
    function close() { generation++; host.remove(); if (previousFocus?.isConnected) previousFocus.focus(); options.onClose?.(); }
    $('.close').addEventListener('click', close);
    if (options.embedded) $('.close').hidden = true;
    shadow.addEventListener('keydown', event => { if (event.key === 'Escape' && !options.embedded) close(); });
    function preview(values) {
      const data = ABSA.uniqueReviews(values); pending = data.reviews;
      $('.preview').hidden = !pending.length;
      $('.preview-count').textContent = `${pending.length} đánh giá`;
      $('.preview-note').textContent = [sourceNote, data.duplicates ? `Bỏ ${data.duplicates} đánh giá trùng.` : ''].filter(Boolean).join(' ');
      $('.preview-list').replaceChildren(...pending.slice(0, 5).map(text => el('li', '', text)));
      if (pending.length > 5) $('.preview-list').append(el('li', 'helper', `Còn ${pending.length - 5} đánh giá.`));
      const invalid = pending.length > ABSA.MAX_REVIEWS || pending.some(x => x.length > ABSA.MAX_CHARS);
      $('.analyze').disabled = busy || fetching || !pending.length || invalid;
      $('.analyze').textContent = busy ? 'Đang phân tích…' : fetching ? 'Đang lấy…' : `Phân tích${pending.length ? ` ${pending.length} đánh giá` : ''}`;
      if (invalid) status('Tối đa 1.000 đánh giá, mỗi đánh giá không quá 4.000 ký tự.', 'error');
    }
    const validLimit = () => Number.isInteger($('#absa-api-limit').valueAsNumber) && $('#absa-api-limit').valueAsNumber >= 1 && $('#absa-api-limit').valueAsNumber <= ABSA.MAX_REVIEWS;
    function syncControls() {
      for (const name of ['api', 'manual']) $(`.${name}-tab`).disabled = busy || fetching || Boolean(options.embedded && name !== 'manual');
      for (const selector of ['#absa-api-preset', '#absa-api-limit', '.manual-input textarea']) $(selector).disabled = busy || fetching;
      $('.collect-api').disabled = busy || fetching || !product || !validLimit();
      preview(pending);
    }
    function setMode(next) {
      if (busy || fetching) return;
      generation++; mode = next; pending = []; sourceNote = ''; status('');
      for (const name of ['api', 'manual']) $(`.${name}-input`).hidden = mode !== name;
      for (const name of ['api', 'manual']) {
        $(`.${name}-tab`).classList.toggle('active', mode === name);
        $(`.${name}-tab`).setAttribute('aria-selected', String(mode === name));
      }
      pendingProduct = product ? {...product} : null;
      preview(mode === 'manual' ? $('.manual-input textarea').value.split(/\r?\n/) : []);
    }
    $('.api-tab').addEventListener('click', () => setMode('api'));
    $('.manual-tab').addEventListener('click', () => setMode('manual'));
    $('.manual-input textarea').addEventListener('input', () => { refreshProduct(); pendingProduct = product ? {...product} : null; status(''); preview($('.manual-input textarea').value.split(/\r?\n/)); });
    function validateLimit() {
      const valid = validLimit();
      $('#limit-error').hidden = valid; $('#limit-error').textContent = valid ? '' : 'Nhập số nguyên từ 1 đến 1.000.';
      $('#absa-api-limit').setAttribute('aria-invalid', String(!valid)); syncControls();
    }
    $('#absa-api-preset').addEventListener('change', () => {
      const value = $('#absa-api-preset').value;
      if (value === 'custom') {$('#absa-api-limit').focus(); $('#absa-api-limit').select();}
      else $('#absa-api-limit').value = value;
      validateLimit();
    });
    $('#absa-api-limit').addEventListener('input', () => {
      $('#absa-api-preset').value = ['20', '50', '100', '200', '500', '1000'].includes($('#absa-api-limit').value) ? $('#absa-api-limit').value : 'custom';
      validateLimit();
    });
    $('.collect-api').addEventListener('click', async () => {
      refreshProduct();
      if (busy || fetching) return;
      if (!product || !validLimit()) return;
      const current = ++generation;
      const maxReviews = $('#absa-api-limit').valueAsNumber, selectedProduct = {...product};
      fetching = true; pending = []; sourceNote = ''; syncControls();
      let scanned = 0, error = null;
      const collected = [];
      progress(0, maxReviews); status(`Đang lấy 0/${maxReviews} đánh giá…`, 'loading');
      try {
        while (scanned < maxReviews) {
          if (scanned) await new Promise(resolve => setTimeout(resolve, ABSA.SOURCE_DELAY));
          refreshProduct();
          if (current !== generation || !host.isConnected) return;
          const limit = Math.min(ABSA.SOURCE_BATCH, maxReviews - scanned);
          const data = await request({type: 'ABSA_FETCH_SHOPEE', maxReviews: limit, offset: scanned, productKey: selectedProduct.key});
          refreshProduct();
          if (current !== generation || !host.isConnected) return;
          collected.push(...data.reviews); scanned += data.scanned; error = data.error;
          pendingProduct = {...product}; sourceNote = `Shopee · Đã đọc ${scanned}/${maxReviews} đánh giá.${error ? ' Chưa lấy đủ do lỗi.' : ''}`;
          preview(collected); progress(scanned, maxReviews); status(`Đang lấy ${scanned}/${maxReviews} đánh giá…`, 'loading');
          if (error || data.scanned < limit) break;
        }
        status(error || (pending.length ? 'Đã lấy xong. Bạn có thể phân tích.' : 'Chưa có nội dung đánh giá. Thử nhập tay.'), error ? 'error' : '');
      } catch (error) {
        if (current === generation) {
          if (collected.length) {sourceNote = `Shopee · Đã đọc ${scanned}/${maxReviews} đánh giá. Chưa lấy đủ do lỗi.`; preview(collected);}
          status(error.message, 'error');
        }
      } finally { if (current === generation) {fetching = false; $('.job-progress').hidden = true; syncControls();} }
    });
    async function checkHealth() {
      $('.connection-text').textContent = 'Đang kết nối…'; $('.connection').classList.remove('connected');
      try {
        const data = await request({type: 'ABSA_HEALTH'});
        $('.connection-text').title = data.api_url || '';
        analysisBatch = Math.max(1, Math.min(ABSA.ANALYSIS_BATCH, data.max_reviews || ABSA.ANALYSIS_BATCH));
        $('.connection-text').textContent = data.model_loaded ? 'Connected' : 'Chưa sẵn sàng';
        $('.connection').classList.toggle('connected', data.model_loaded);
        if (!data.model_loaded) status(data.error || 'Chưa tải được mô hình.', 'error');
      } catch (error) { $('.connection-text').textContent = 'Disconnected'; status(error.message, 'error'); }
    }
    $('.reconnect').addEventListener('click', checkHealth);
    function bar(values, className = 'bar') {
      const node = el('div', className), total = values.reduce((a, b) => a + b, 0);
      ABSA.SENTIMENTS.forEach((sentiment, i) => {
        const part = el('span', sentiment.toLowerCase());
        part.style.width = `${total ? values[i] / total * 100 : 0}%`; node.append(part);
      });
      return node;
    }
    const pct = (value, total) => total ? `${(100 * value / total).toFixed(1)}%` : '0%';
    function renderReviews() {
      const visible = results.filter(review => !filter || review.aspects.some(a => a.code === filter.code && (!filter.sentiment || a.sentiment === filter.sentiment)));
      $('.reset-filter').hidden = !filter;
      $('.review-filter').textContent = filter ? `${ABSA.ASPECTS.find(a => a.code === filter.code).name}${filter.sentiment ? ` · ${ABSA.LABELS[filter.sentiment]}` : ''} — ${visible.length} đánh giá` : `${results.length} đánh giá · Chọn khía cạnh để lọc`;
      $('.review-list').replaceChildren(...visible.slice(0, shown).map(review => {
        const card = el('article', 'review-card'); card.append(el('span', 'review-index', `Đánh giá ${review.review_id + 1}`), el('p', 'review-text', review.text));
        const chips = el('div', 'chips');
        for (const a of review.aspects) {
          if (!ABSA.SENTIMENTS.includes(a.sentiment) || !ABSA.ASPECTS.some(x => x.code === a.code)) continue;
          const chip = el('span', `chip ${a.sentiment.toLowerCase()}`, `${a.name} · ${ABSA.LABELS[a.sentiment]}`);
          if (Number.isFinite(a.confidence)) {
            chip.title = `Softmax: ${(a.confidence * 100).toFixed(1)}% (chưa hiệu chuẩn)`;
            chip.append(el('span', 'chip-confidence', ` · ${(a.confidence * 100).toFixed(1)}%`));
          }
          chips.append(chip);
        }
        if (!chips.children.length) chips.append(el('span', 'no-aspect', review.status === 'empty' ? 'Nội dung trống.' : 'Không có khía cạnh phù hợp.'));
        card.append(chips); return card;
      }));
      $('.more-reviews').hidden = visible.length <= shown;
    }
    function selectFilter(code, sentiment) { filter = {code, sentiment}; shown = 10; renderReviews(); $('.review-section').scrollIntoView({block: 'start', behavior: 'smooth'}); }
    function render(data, analyzedProduct) {
      results = data.results; filter = null; shown = 10;
      resultProduct = analyzedProduct; showProduct();
      $('.source-note').textContent = sourceNote || 'Nguồn: nhập tay.';
      $('.dashboard').hidden = false; $('.empty').hidden = true;
      const rows = ABSA.summarize(results), totals = ['positive', 'negative', 'neutral'].map(k => rows.reduce((sum, row) => sum + row[k], 0));
      const total = totals.reduce((a, b) => a + b, 0);
      $('.total-reviews').textContent = String(results.length); $('.total-mentions').textContent = String(total);
      $('.overall-bar').replaceChildren(bar(totals));
      $('.overall-labels').replaceChildren(...ABSA.SENTIMENTS.map((s, i) => el('span', s.toLowerCase(), `${ABSA.LABELS[s]} ${totals[i]} · ${pct(totals[i], total)}`)));
      $('.aspect-list').replaceChildren(...rows.map(row => {
        const card = el('article', 'aspect-row'), heading = el('div', 'aspect-heading');
        const label = el('button', 'aspect-title', `${row.name} · ${row.code}`);
        label.addEventListener('click', () => selectFilter(row.code)); heading.append(label, el('span', 'small-label', `${row.mentions} đánh giá`)); card.append(heading);
        const values = [row.positive, row.negative, row.neutral], sum = values.reduce((a, b) => a + b, 0);
        if (!sum) { card.append(el('span', 'no-aspect', 'Chưa được đề cập')); return card; }
        const track = el('div', 'bar');
        const ratios = el('div', 'ratios');
        ABSA.SENTIMENTS.forEach((sentiment, i) => {
          const button = el('button', sentiment.toLowerCase()); button.style.width = `${100 * values[i] / sum}%`;
          button.setAttribute('aria-label', `${row.name}: ${ABSA.LABELS[sentiment]} ${pct(values[i], sum)}`);
          button.title = `${ABSA.LABELS[sentiment]}: ${values[i]} lượt (${pct(values[i], sum)})`;
          button.addEventListener('click', () => selectFilter(row.code, sentiment)); track.append(button);
          const ratio = el('button', sentiment.toLowerCase(), pct(values[i], sum));
          ratio.addEventListener('click', () => selectFilter(row.code, sentiment)); ratios.append(ratio);
        }); card.append(track, ratios); return card;
      }));
      const negative = rows.filter(a => a.negative).sort((a, b) => (b.negative / (b.positive + b.negative + b.neutral)) - (a.negative / (a.positive + a.negative + a.neutral)) || b.negative - a.negative).slice(0, 3);
      $('.negative-list').replaceChildren(...negative.map(a => {
        const button = el('button', 'negative-row'); button.append(el('span', '', a.name), el('strong', '', `${pct(a.negative, a.positive + a.negative + a.neutral)} · ${a.negative}/${a.mentions} đánh giá`));
        button.addEventListener('click', () => selectFilter(a.code, 'Negative')); return button;
      }));
      if (!negative.length) $('.negative-list').append(el('p', 'helper', 'Không có đánh giá tiêu cực.'));
      renderReviews();
    }
    $('.reset-filter').addEventListener('click', () => { filter = null; shown = 10; renderReviews(); });
    $('.more-reviews').addEventListener('click', () => { shown += 10; renderReviews(); });
    function clearResults() {
      results = []; filter = null; resultProduct = null;
      for (const selector of ['.review-list', '.aspect-list', '.negative-list', '.overall-labels', '.overall-bar']) $(selector).replaceChildren();
      $('.total-reviews').textContent = '0'; $('.total-mentions').textContent = '0';
      $('.source-note').textContent = ''; $('.dashboard').hidden = true; $('.empty').hidden = false; showProduct();
    }
    $('.clear').addEventListener('click', () => {
      if (busy || fetching) return;
      clearResults(); status('Đã xóa kết quả.');
    });
    function updateProduct(next) {
      const changed = (product?.key || null) !== (next?.key || null);
      product = next ? {...next} : null;
      if (changed) {
        generation++; pending = []; pendingProduct = null; sourceNote = ''; busy = false; fetching = false;
        $('.job-progress').hidden = true;
        $('.manual-input textarea').value = ''; clearResults();
        status(product ? 'Đã đổi sản phẩm. Hãy lấy lại đánh giá.' : 'Hãy mở một sản phẩm Shopee.');
      } else {
        if (pendingProduct?.key === product?.key && product) pendingProduct = {...product};
        if (resultProduct?.key === product?.key && product) resultProduct = {...product};
      }
      showProduct(); syncControls();
    }
    function refreshProduct() { if (options.getProduct) updateProduct(options.getProduct()); }
    $('.analyze').addEventListener('click', async () => {
      refreshProduct();
      if (!pending.length || busy) return;
      if ((pendingProduct?.key || null) !== (product?.key || null)) return;
      const current = ++generation; busy = true; syncControls();
      const texts = [...pending], accumulated = [], batchSize = analysisBatch;
      let version = null, device = null;
      $('.dashboard').hidden = true; $('.empty').hidden = true;
      progress(0, texts.length); status(`Đang phân tích 0/${texts.length} đánh giá…`, 'loading');
      try {
        for (let start = 0; start < texts.length; start += batchSize) {
          refreshProduct();
          if (current !== generation || !host.isConnected) return;
          const data = await request({type: 'ABSA_ANALYZE', reviews: texts.slice(start, start + batchSize)});
          refreshProduct();
          if (current !== generation || !host.isConnected) return;
          if (version && data.model_version !== version) throw new Error('Mô hình đã đổi. Hãy phân tích lại.');
          version = data.model_version; device = data.device;
          accumulated.push(...data.results.map(result => ({...result, review_id: start + result.review_id})));
          progress(accumulated.length, texts.length); status(`Đang phân tích ${accumulated.length}/${texts.length} đánh giá…`, 'loading');
        }
        const data = {results: accumulated, total_reviews: accumulated.length, model_version: version, device};
        render(data, product ? {...product} : null); status(`Đã phân tích ${data.total_reviews} đánh giá.`, 'success');
      } catch (error) { if (current === generation) { $('.empty').hidden = false; status(error.message, 'error'); } }
      finally { if (current === generation) {busy = false; $('.job-progress').hidden = true; syncControls();} }
    });
    showProduct(); setMode(mode); syncControls(); checkHealth();
    return {host, shadow, close, updateProduct};
  }
  root.ABSASidebar = Object.freeze({create});
})(globalThis);
