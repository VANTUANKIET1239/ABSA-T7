"""Lazada facial cleanser reviews crawler (Selenium + existing Chrome 9222).

Only reads public pages that the browser can access. Selectors for the CLOSED
star filter are based on the supplied Lazada HTML. The OPEN dropdown was not
supplied, so star selection is checked, with a slower all-stars fallback.
"""

import csv
import os
import re
import time
from collections import Counter
from urllib.parse import quote, urlencode, urlsplit, urlunsplit

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

# ===================== CONFIG =====================
DEBUG_PORT = 9222
TARGET_TOTAL_REVIEWS = 6000  # Có thể sửa thành 12000 nếu cần
MAX_REVIEWS_PER_PRODUCT = 200
MAX_SEARCH_PAGES = 70
MAX_FILTERED_REVIEW_PAGES = 100
MAX_UNFILTERED_REVIEW_PAGES = 80
MIN_COMMENT_LENGTH = 6
DEBUG_SAVE_FILTER_HTML = False  # True: lưu DOM dropdown sau khi mở để debug
SEARCH_WAIT_SECONDS = 20
DETAIL_PRICE_WAIT_SECONDS = 12
SEARCH_SCROLL_STEPS = 5
DEBUG_DIR = "data/lazada_debug"
OUTPUT_FILE = "data/lazada_cleanser_reviews.csv"

# Cùng quota với code Shopee: 1-4 sao trước, sau đó dồn phần thiếu vào 5 sao.
STAR_QUOTAS = {1: 30, 2: 30, 3: 30, 4: 50}

SEARCH_BASE = "https://www.lazada.vn/tag/sửa-rửa-mặt/"
CATALOG_BASE = "https://www.lazada.vn/catalog/"
HEADERS = [
    "product_name", "brand", "rating", "date", "product_url",
    "price", "discount", "comment", "review_text_length", "platform",
]

PRODUCT_CARD_SELECTORS = (
    "div.Ms6aG.MefHh",    # Selector từ code Lazada cũ
    "div.Bm3ON",           # Một số layout danh sách Lazada
    "div[data-qa-locator='product-item']",
)
PRICE_SELECTORS = (
    "div.aBrP0 span.ooOxS", "span.ooOxS", "div.aBrP0",
    "[class*='price-current']", "[class*='price--']",
)
DISCOUNT_SELECTORS = (
    "[class*='discount']", "[class*='Discount']",
    "[class*='saleTag']", "[class*='promotion']",
)


def sleep_brief(seconds=1.2):
    """Giãn cách các tương tác thay vì gọi liên tục."""
    time.sleep(seconds)


def clean_text(value):
    return re.sub(r"\s+", " ", value or "").strip()


def normalize_url(url):
    if not url:
        return ""
    if url.startswith("//"):
        url = "https:" + url
    if url.startswith("/"):
        url = "https://www.lazada.vn" + url
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, "", ""))


def review_key(row):
    # HTML mẫu không có ID review ổn định, nên dùng fingerprint tương đối.
    # Hai review thật có cùng sản phẩm/ngày/sao/nội dung có thể bị coi là trùng.
    return (
        normalize_url(row.get("product_url", "")),
        str(row.get("rating", "")).strip(),
        clean_text(row.get("date", "")),
        clean_text(row.get("comment", "")).casefold(),
    )


def connect_chrome():
    print("[*] Kết nối Chrome tại cổng", DEBUG_PORT)
    options = Options()
    options.add_experimental_option("debuggerAddress", f"127.0.0.1:{DEBUG_PORT}")
    return webdriver.Chrome(options=options)


def build_search_url(page, source="catalog"):
    """Thử /catalog/ (ổn định hơn) và link /tag/ do người dùng cung cấp.

    Dùng %20 thay cho dấu + trong query để khớp URL Lazada người dùng đã mở.
    """
    if source == "tag":
        params = urlencode({
            "q": "sửa rửa mặt",
            "catalog_redirect_tag": "true",
            "page": page,
        }, quote_via=quote)
        return f"{SEARCH_BASE}?{params}"

    params = urlencode({"q": "sửa rửa mặt", "page": page}, quote_via=quote)
    return f"{CATALOG_BASE}?{params}"


def is_product_url(href):
    """Chỉ chấp nhận URL sản phẩm thuộc Lazada, không lấy URL điều hướng/tìm kiếm."""
    url = normalize_url(href)
    if not url:
        return False
    parts = urlsplit(url)
    domain = (parts.hostname or "").lower()
    if domain != "lazada.vn" and not domain.endswith(".lazada.vn"):
        return False
    path = parts.path.lower()
    if not path.endswith(".html"):
        return False
    return ("/products/" in path or "pdp-i" in path) and bool(
        re.search(r"(?:-i|pdp-i)\d+", path)
    )


def find_product_card(anchor):
    """Lấy container card gần nhất; tránh giá của card hàng xóm."""
    current = anchor
    for _ in range(7):
        if current is None or getattr(current, "name", None) in ("body", "html"):
            break
        if current.name in ("div", "li", "article"):
            cls = current.get("class", [])
            if (current.get("data-qa-locator") == "product-item"
                    or "Bm3ON" in cls or "Ms6aG" in cls):
                return current
        current = current.parent

    # Giao diện mới không còn class card: tìm tổ tiên gần nhất chứa cả giá.
    current = anchor
    for _ in range(5):
        if current is None or getattr(current, "name", None) in ("body", "html"):
            break
        if current.select_one("span.ooOxS, div.aBrP0, [class*='price-current']"):
            return current
        current = current.parent

    # Cuối cùng chỉ trả chính anchor để không đọc nhầm dữ liệu toàn trang.
    return anchor


def extract_title(anchor, card):
    """Lazada có thể để tiêu đề ở title, alt ảnh hoặc .RfADt."""
    candidates = [anchor.get("title"), anchor.get("aria-label")]
    for node in (anchor.select_one("img[alt]"), card.select_one(".RfADt a"),
                 card.select_one(".RfADt")):
        if node is not None:
            candidates.extend([node.get("title"), node.get("alt"),
                               node.get_text(" ", strip=True)])
    candidates.append(anchor.get_text(" ", strip=True))
    for val in candidates:
        val = clean_text(val)
        if len(val) >= 8 and not re.fullmatch(r"[\d.,]+\s*₫?", val):
            return val
    return ""

def first_text(element, selectors):
    for selector in selectors:
        found = element.select_one(selector)
        if found:
            value = clean_text(found.get_text(" ", strip=True))
            if value:
                return value
    return ""


def extract_price(card):
    price = first_text(card, PRICE_SELECTORS)
    if price:
        return price
    text = clean_text(card.get_text(" ", strip=True))
    match = re.search(r"(?:₫\s*[\d.,]+|[\d.,]+\s*₫)", text)
    return clean_text(match.group(0)) if match else ""  # Unknown != 0đ


def extract_discount(card):
    # Ưu tiên nhãn discount thật, tránh match nhầm phần trăm khác trên card.
    for selector in DISCOUNT_SELECTORS:
        for element in card.select(selector):
            value = clean_text(element.get_text(" ", strip=True))
            match = re.search(r"-?\s*(100|\d{1,2})\s*%", value)
            if match:
                return f"{match.group(1)}%"

    text = clean_text(card.get_text(" ", strip=True))
    match = re.search(r"(?:giảm\s*)?-\s*(100|\d{1,2})\s*%", text, re.I)
    if not match:
        match = re.search(r"giảm\s*(100|\d{1,2})\s*%", text, re.I)
    return f"{match.group(1)}%" if match else "0%"


def parse_product_cards(html):
    """Dựa trên tất cả link sản phẩm, không phụ thuộc duy nhất vào class card.

    Vấn đề phiên bản cũ: tìm thấy class card nhưng selector link bên trong
    không khớp thì trả về 0 ngay, không bao giờ thử fallback theo link.
    """
    soup = BeautifulSoup(html, "html.parser")
    results, seen_urls = [], set()

    # Hỗ trợ cả link /products/<slug>-i123-s456.html và /products/pdp-i123.html
    anchors = (a for a in soup.select("a[href]") if is_product_url(a.get("href", "")))
    for anchor in anchors:
        link = normalize_url(anchor.get("href"))
        if link in seen_urls:
            continue
        card = find_product_card(anchor)
        title = extract_title(anchor, card)
        if not title:
            continue

        seen_urls.add(link)
        results.append({
            "product_name": title,
            "product_url": link,
            "price": extract_price(card),
            "discount": extract_discount(card),
            "brand": "",
        })
    return results


def save_search_debug(driver, page, source):
    """Lưu bằng chứng chẩn đoán nếu không lấy được sản phẩm."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    prefix = os.path.join(DEBUG_DIR, f"search_page_{page}_{source}")
    html_path = prefix + ".html"
    screenshot_path = prefix + ".png"
    report_path = prefix + ".txt"

    html = driver.page_source
    soup = BeautifulSoup(html, "html.parser")
    links = [a.get("href", "") for a in soup.select("a[href]")]
    product_links = [h for h in links if is_product_url(h)]
    body = soup.body.get_text(" ", strip=True) if soup.body else soup.get_text(" ", strip=True)
    try:
        driver.save_screenshot(screenshot_path)
    except WebDriverException:
        screenshot_path = "(không chụp được)"
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"Requested source: {source}\nFinal URL: {driver.current_url}\n")
        f.write(f"Page title: {driver.title}\nHTML length: {len(html)}\n")
        f.write(f"Total links: {len(links)}\n")
        f.write(f"Product links (recognized): {len(product_links)}\n")
        for css in PRODUCT_CARD_SELECTORS:
            f.write(f"Selector {css}: {len(soup.select(css))}\n")
        f.write("\nFirst 35 href values:\n" + "\n".join(links[:35]))
        f.write("\n\nVisible page text (first 3000 chars):\n" + body[:3000])

    looks_blocked = bool(re.search(
        r"captcha|robot|unusual traffic|access denied|verify you are human|xác minh|truy cập bị từ chối",
        body[:5000] + " " + driver.title, re.I
    ))
    print(f"    [DEBUG] URL thực tế: {driver.current_url}")
    print(f"    [DEBUG] Tiêu đề: {driver.title}")
    print(f"    [DEBUG] Link sản phẩm trong DOM: {len(product_links)}")
    if looks_blocked:
        print("    [!] Trang có dấu hiệu CAPTCHA/xác minh. Hãy kiểm tra Chrome bằng tay.")
    print(f"    [DEBUG] HTML: {html_path}")
    print(f"    [DEBUG] Screenshot: {screenshot_path}")
    print(f"    [DEBUG] Thống kê: {report_path}")


def get_product_list(driver, page):
    """Mở danh sách, cuộn, chờ SPA render; khi lỗi thử URL thứ hai."""
    for source in ("catalog", "tag"):
        url = build_search_url(page, source)
        print(f"\n[*] Tìm 'sửa rửa mặt' - trang {page} ({source}): {url}")
        try:
            driver.get(url)
            WebDriverWait(driver, 15).until(
                lambda d: d.execute_script("return document.readyState") in ("interactive", "complete")
            )

            start = time.monotonic()
            found_by_url = {}
            # Tìm cả trong DOM ban đầu lẫn sau khi cuộn (lazy loading/virtual list).
            attempt = 0
            while time.monotonic() - start < SEARCH_WAIT_SECONDS:
                for p in parse_product_cards(driver.page_source):
                    found_by_url[p["product_url"]] = p
                if found_by_url:
                    break
                if attempt % 2 == 1:
                    driver.execute_script("window.scrollBy(0, 650);")
                attempt += 1
                sleep_brief(1.2)

            if found_by_url:
                for _ in range(SEARCH_SCROLL_STEPS):
                    driver.execute_script("window.scrollBy(0, 900);")
                    sleep_brief(0.8)
                    for p in parse_product_cards(driver.page_source):
                        found_by_url[p["product_url"]] = p
                driver.execute_script("window.scrollTo(0, 0);")
                sleep_brief(0.4)
                for p in parse_product_cards(driver.page_source):
                    found_by_url[p["product_url"]] = p

                products = list(found_by_url.values())
                print(f"    [+] Tìm thấy {len(products)} sản phẩm hợp lệ.")
                print("    Ví dụ:", products[0]["product_name"][:100])
                return products
        except (TimeoutException, WebDriverException) as exc:
            print(f"    [!] Lỗi mở trang: {type(exc).__name__}: {str(exc)[:160]}")

        print("    [!] Không trích xuất được sản phẩm từ URL này.")
        try:
            save_search_debug(driver, page, source)
        except Exception as exc:
            print(f"    [!] Không lưu được debug: {exc}")

    print("    [!] Cả hai URL đều trả về 0. Xem file trong thư mục data/lazada_debug/.")
    return []

def scroll_to_reviews(driver):
    # Scroll trực tiếp vào review section, không cuộn số lần cố định.
    try:
        section = WebDriverWait(driver, 12).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "#module_product_review, .pdp-mod-review-v2"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", section)
    except TimeoutException:
        for _ in range(7):
            driver.execute_script("window.scrollBy(0, 900);")
            sleep_brief(0.5)
            if driver.find_elements(By.CSS_SELECTOR, ".pdp-mod-review-v2"):
                break

    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, ".pdp-mod-review-v2"))
    )
    sleep_brief(0.8)


def parse_product_detail(html):
    """Lấy giá / giảm giá / brand từ TRANG CHI TIẾT (PDP) Lazada.

    HTML được cung cấp có các class *-v2 dưới #module_product_price_v2.
    Không tìm giá qua text của toàn trang vì dễ lấy nhầm phí vận chuyển,
    giá voucher hoặc sản phẩm gợi ý. Giá là giá hiện tại, không phải giá gốc.
    """
    soup = BeautifulSoup(html, "html.parser")
    module = soup.select_one("#module_product_price_v2")
    if module is None:
        module = soup.select_one("#module_product_price, .pdp-mod-product-price-v2, .pdp-mod-product-price")
    price_root = module if module is not None else soup

    price = first_text(price_root, (
        ".pdp-v2-product-price-content-salePrice-amount",   # HTML mẫu mới
        ".pdp-product-price .pdp-price",                    # Một số giao diện cũ
        ".pdp-price",                                      # Fallback trong module
    ))
    original_price = first_text(price_root, (
        ".pdp-v2-product-price-content-originalPrice-amount",
        ".pdp-product-price .pdp-price_del",
        ".pdp-price_del",
    ))
    raw_discount = first_text(price_root, (
        ".pdp-v2-product-price-content-originalPrice-discount",
        ".pdp-product-price .pdp-product-price__discount",
    ))

    # Chỉ dùng meta khi không có module giá hiển thị; meta có thể không phải
    # giá theo biến thể đang chọn nên giá trên PDP luôn được ưu tiên.
    if not price:
        meta = soup.select_one("meta[property='product:price:amount'], meta[itemprop='price']")
        if meta:
            price = clean_text(meta.get("content", ""))

    discount = ""
    percentage = re.search(r"(-?)\s*(100|\d{1,2})\s*%", raw_discount)
    if percentage:
        discount = f"{percentage.group(2)}%"  # cùng format với Shopee (23%)
    elif price and original_price:
        def vnd_number(text):
            digits = re.sub(r"[^0-9]", "", text or "")
            return int(digits) if digits else 0

        sale_value = vnd_number(price)
        original_value = vnd_number(original_price)
        if original_value > 0 and 0 < sale_value < original_value:
            discount = f"{round((1 - sale_value / original_value) * 100)}%"
        elif original_value > 0 and sale_value >= original_value:
            discount = "0%"
    elif price and module is not None:
        # Không có giá gốc / nhãn sale trong module: giá không hiển thị giảm.
        discount = "0%"

    brand = first_text(soup, (
        ".pdp-product-brand-v2__brand-link",  # HTML mẫu: Eucerin
        ".pdp-product-brand__brand-link",
    ))
    if not brand:
        for selector in ("meta[property='product:brand']", "meta[name='brand']", "meta[itemprop='brand']"):
            node = soup.select_one(selector)
            if node:
                brand = clean_text(node.get("content", ""))
                if brand:
                    break

    return {
        "price": price,
        "original_price": original_price,
        "discount": discount,
        "brand": brand,
    }


def fetch_product_detail(driver, product):
    """Đọc sau driver.get(product_url), TRƯỚC khi cuộn đến reviews."""
    try:
        WebDriverWait(driver, DETAIL_PRICE_WAIT_SECONDS).until(
            lambda d: bool(d.find_elements(By.CSS_SELECTOR,
                "#module_product_price_v2 .pdp-v2-product-price-content-salePrice-amount, "
                "#module_product_price .pdp-price, "
                "meta[property='product:price:amount']"))
        )
    except TimeoutException:
        print("      [!] Chưa thấy module giá trong thời gian chờ, thử đọc HTML hiện tại.")

    detail = parse_product_detail(driver.page_source)
    if not detail["price"]:
        print("      [!] Không đọc được giá ở trang chi tiết; giữ giá danh sách nếu có.")
    if detail["price"]:
        product["price"] = detail["price"]
    if detail["discount"]:
        product["discount"] = detail["discount"]
    if detail["brand"]:
        product["brand"] = detail["brand"]
    return detail


def detect_brand(driver):
    """Hỗ trợ đọc thương hiệu từ PDP và meta khi cần."""
    return parse_product_detail(driver.page_source)["brand"]


def get_star_count(item):
    # Chỉ đếm sao trong từng review, KHÔNG lấy sao trung bình .mod-rating.
    star_block = item.select_one(".item-middle .review-star")
    if not star_block:
        return 0
    selected = star_block.select('.i-rate-star path[style*="rgb(255, 200, 60)"]')
    return len(selected) if 1 <= len(selected) <= 5 else 0


def parse_review_items(html, product):
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for item in soup.select(".pdp-mod-review-main .mod-reviews > .item"):
        rating = get_star_count(item)
        if rating not in (1, 2, 3, 4, 5):
            continue

        parts = [
            clean_text(node.get_text(" ", strip=True))
            for node in item.select(".item-content-main-content-reviews-item")
        ]
        comment = clean_text(" ".join(part for part in parts if part))
        if len(comment) < MIN_COMMENT_LENGTH:
            continue

        date_el = item.select_one("span.time")
        date = clean_text(date_el.get_text(" ", strip=True)) if date_el else ""
        rows.append({
            "product_name": product["product_name"],
            "brand": product.get("brand", ""),
            "rating": rating,
            "date": date,
            "product_url": product["product_url"],
            "price": product.get("price", ""),
            "discount": product.get("discount", "0%"),
            "comment": comment,
            "review_text_length": len(comment),
            "platform": "Lazada",
        })
    return rows


def selected_filter_text(driver):
    elements = driver.find_elements(
        By.CSS_SELECTOR, ".pdp-mod-filterSort-v2 .oper:first-child .condition"
    )
    return clean_text(elements[0].text) if elements else ""


def is_selected_star(text, star):
    return bool(re.search(rf"(?<!\d){star}\s*(?:sao|star)(?!\w)", text, re.I))


def select_star_filter(driver, star):
    """Open rating dropdown, click the target and VERIFY that the UI accepted it.

    Open-dropdown markup was not included in the user-supplied HTML; matching
    text is deliberately best-effort and must be verified before scraping.
    """
    if is_selected_star(selected_filter_text(driver), star):
        return True

    try:
        trigger = WebDriverWait(driver, 7).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".pdp-mod-filterSort-v2 .oper:first-child"))
        )
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", trigger)
        driver.execute_script("arguments[0].click();", trigger)
        sleep_brief(0.4)
        if DEBUG_SAVE_FILTER_HTML:
            os.makedirs("data", exist_ok=True)
            with open("data/lazada_open_filter_debug.html", "w", encoding="utf-8") as f:
                f.write(driver.page_source)

        # Tìm đúng văn bản '1 Sao', '1 Sao (N)', '1 Star', ... trong menu.
        # Leaf matching tránh click một container chứa toàn bộ text của trang.
        js_find_option = r"""
            const star = arguments[0];
            const rex = new RegExp('^\\s*' + star + '\\s*(?:sao|star|★|⭐)(?:\\s*\\(\\d+\\))?\\s*$', 'i');
            const nodes = [...document.querySelectorAll('body *')];
            for (const node of nodes) {
                if (node.children.length !== 0) continue;
                const txt = (node.textContent || '').replace(/\u00a0/g, ' ').trim();
                if (!rex.test(txt) || !node.getClientRects().length) continue;
                let target = node.closest('li,button,[role="option"],[class*="dropdown-item"],[class*="option-item"]');
                if (!target) target = node;
                if (!target.getClientRects().length) continue;
                return target;
            }
            return null;
        """
        choice = WebDriverWait(driver, 5).until(
            lambda d: d.execute_script(js_find_option, star)
        )
        driver.execute_script("arguments[0].click();", choice)
        WebDriverWait(driver, 8).until(
            lambda d: is_selected_star(selected_filter_text(d), star)
        )
        # Đợi AJAX render reviews đúng số sao; nếu không có review, chấp nhận.
        def ready(d):
            soup = BeautifulSoup(d.page_source, "html.parser")
            items = soup.select(".pdp-mod-review-main .mod-reviews > .item")
            if not items:
                return True
            return all(get_star_count(item) == star for item in items)

        WebDriverWait(driver, 10).until(ready)
        return True
    except (TimeoutException, WebDriverException) as exc:
        print(f"      [!] Không xác minh được filter {star} sao: {type(exc).__name__}")
        return False


def current_review_page(driver):
    elements = driver.find_elements(
        By.CSS_SELECTOR, ".review-pagination .iweb-pagination-item-active"
    )
    if elements:
        value = elements[0].get_attribute("title") or clean_text(elements[0].text)
        if value.isdigit():
            return int(value)
    return None


def click_next_review_page(driver):
    next_buttons = driver.find_elements(
        By.CSS_SELECTOR, ".review-pagination li.iweb-pagination-next"
    )
    if not next_buttons:
        return False
    next_li = next_buttons[0]
    if (next_li.get_attribute("aria-disabled") == "true"
            or "iweb-pagination-disabled" in (next_li.get_attribute("class") or "")):
        return False

    old_page = current_review_page(driver)
    old_first = ""
    first_items = driver.find_elements(By.CSS_SELECTOR, ".mod-reviews > .item")
    if first_items:
        old_first = clean_text(first_items[0].text)

    try:
        button = next_li.find_element(By.CSS_SELECTOR, "button")
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", button)
        driver.execute_script("arguments[0].click();", button)

        def page_changed(d):
            new_page = current_review_page(d)
            now = d.find_elements(By.CSS_SELECTOR, ".mod-reviews > .item")
            first_changed = bool(now and clean_text(now[0].text) != old_first)
            if old_page is not None and new_page is not None:
                # Tránh đọc DOM cũ khi số trang đã tăng nhưng reviews AJAX chưa đổi.
                return new_page > old_page and (first_changed or not old_first)
            return first_changed

        WebDriverWait(driver, 10).until(page_changed)
        sleep_brief(0.3)
        return True
    except (TimeoutException, WebDriverException):
        print("      [!] Không chuyển được trang review hoặc AJAX không tải.")
        return False


def collect_pages_for_star(driver, product, star, quota, used_keys):
    rows = []
    local_keys = set()
    for page_no in range(1, MAX_FILTERED_REVIEW_PAGES + 1):
        if len(rows) >= quota:
            break
        parsed = parse_review_items(driver.page_source, product)
        if not parsed and page_no == 1:
            print(f"      - {star} sao: không có review hợp lệ ở trang đầu.")

        for row in parsed:
            if row["rating"] != star:
                continue  # Không cho review sai rating lọt vào quota.
            key = review_key(row)
            if key in used_keys or key in local_keys:
                continue
            local_keys.add(key)
            rows.append(row)
            if len(rows) >= quota:
                break

        if len(rows) >= quota or not click_next_review_page(driver):
            break
    print(f"      [+] {star} sao: lấy {len(rows)}/{quota} review mới.")
    return rows


def collect_unfiltered_fallback(driver, product, quotas, total_remaining, used_keys):
    """Fallback khi dropdown sao thay đổi: đọc nhiều trang rồi phân nhóm.

    Có giới hạn số trang, vì vậy chỉ đảm bảo ưu tiên trong phần đã quét,
    không đảm bảo thu thập toàn bộ các review 1-4 sao của sản phẩm lớn.
    """
    print("      [!] Fallback: quét các trang 'Tất cả Sao' rồi ưu tiên review <5 sao.")
    driver.get(product["product_url"])
    scroll_to_reviews(driver)
    buckets = {star: [] for star in range(1, 6)}
    local_keys = set()
    for _ in range(MAX_UNFILTERED_REVIEW_PAGES):
        for row in parse_review_items(driver.page_source, product):
            key = review_key(row)
            if key in used_keys or key in local_keys:
                continue
            local_keys.add(key)
            buckets[row["rating"]].append(row)
        if not click_next_review_page(driver):
            break

    chosen = []
    for star in range(1, 5):
        chosen.extend(buckets[star][:min(quotas[star], total_remaining - len(chosen))])
    chosen.extend(buckets[5][:max(0, total_remaining - len(chosen))])
    print(f"      [+] Fallback thu {len(chosen)} review trong các trang đã quét.")
    return chosen


def fetch_reviews(driver, product, existing_counts, used_keys, global_remaining):
    product_url = product["product_url"]
    existing_for_product = sum(existing_counts.values())
    available = min(MAX_REVIEWS_PER_PRODUCT - existing_for_product, global_remaining)
    if available <= 0:
        return []

    driver.get(product_url)
    detail = fetch_product_detail(driver, product)
    print(f"      [PDP] Giá: {product['price'] or 'N/A'}"
          f" | Giá gốc: {detail['original_price'] or 'N/A'}"
          f" | Giảm: {product['discount'] or 'N/A'}"
          f" | Brand: {product.get('brand') or 'N/A'}"
          f" | Đã có: {existing_for_product} | Còn quota: {available}")
    scroll_to_reviews(driver)

    collected = []
    in_product_keys = set(used_keys)
    remaining_quotas = {
        star: max(0, cap - existing_counts.get(star, 0))
        for star, cap in STAR_QUOTAS.items()
    }
    for star in range(1, 5):
        quota = min(remaining_quotas[star], available - len(collected))
        if quota <= 0:
            continue
        if not select_star_filter(driver, star):
            return collect_unfiltered_fallback(
                driver, product, remaining_quotas, available, used_keys
            )
        new_rows = collect_pages_for_star(driver, product, star, quota, in_product_keys)
        collected.extend(new_rows)
        in_product_keys.update(review_key(row) for row in new_rows)
        if len(collected) >= available:
            return collected
        sleep_brief(0.6)

    five_quota = available - len(collected)
    if five_quota > 0:
        print(f"      -> Dồn {five_quota} chỉ tiêu còn thiếu sang 5 sao.")
        if not select_star_filter(driver, 5):
            return collect_unfiltered_fallback(
                driver, product, remaining_quotas, available, used_keys
            )
        five_rows = collect_pages_for_star(driver, product, 5, five_quota, in_product_keys)
        collected.extend(five_rows)
    return collected


def load_existing_data():
    seen = set()
    per_product = {}
    total = 0
    if not os.path.isfile(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0:
        return seen, per_product, total

    with open(OUTPUT_FILE, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != HEADERS:
            raise ValueError(
                f"CSV '{OUTPUT_FILE}' có header khác phiên bản mới. "
                "Hãy chọn OUTPUT_FILE mới hoặc chuyển đổi schema trước; "
                "script sẽ KHÔNG ghi đè file cũ."
            )
        for row in reader:
            total += 1
            seen.add(review_key(row))
            url = normalize_url(row.get("product_url", ""))
            if url:
                per_product.setdefault(url, Counter())
                try:
                    star = int(row.get("rating", 0))
                except (ValueError, TypeError):
                    star = 0
                per_product[url][star] += 1
    return seen, per_product, total


def append_reviews(rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(OUTPUT_FILE) or ".", exist_ok=True)
    create_header = not os.path.isfile(OUTPUT_FILE) or os.path.getsize(OUTPUT_FILE) == 0
    with open(OUTPUT_FILE, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS)
        if create_header:
            writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def main():
    seen_keys, per_product, saved_total = load_existing_data()
    print(f"[*] File: {OUTPUT_FILE} | Đã có: {saved_total}/{TARGET_TOTAL_REVIEWS}")
    if saved_total >= TARGET_TOTAL_REVIEWS:
        print("[+] Đã đạt chỉ tiêu. Không ghi thêm.")
        return

    driver = connect_chrome()
    visited_in_run = set()
    try:
        for page in range(1, MAX_SEARCH_PAGES + 1):
            if saved_total >= TARGET_TOTAL_REVIEWS:
                break
            products = get_product_list(driver, page)
            if not products:
                print("[!] Hết sản phẩm hoặc trang không tải được. Dừng để kiểm tra.")
                break
            for i, product in enumerate(products, 1):
                if saved_total >= TARGET_TOTAL_REVIEWS:
                    break
                url = product["product_url"]
                if url in visited_in_run:
                    continue
                visited_in_run.add(url)
                counts = per_product.get(url, Counter())
                if sum(counts.values()) >= MAX_REVIEWS_PER_PRODUCT:
                    continue
                print(f"\n[{i}/{len(products)}] {product['product_name'][:90]}")
                try:
                    rows = fetch_reviews(
                        driver, product, counts, seen_keys,
                        global_remaining=TARGET_TOTAL_REVIEWS - saved_total,
                    )
                    # Phòng trường hợp review xuất hiện lặp lại giữa các trang.
                    unique_rows = []
                    for row in rows:
                        key = review_key(row)
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)
                        unique_rows.append(row)
                    append_reviews(unique_rows)
                    if unique_rows:
                        per_product.setdefault(url, Counter()).update(
                            row["rating"] for row in unique_rows
                        )
                    saved_total += len(unique_rows)
                    print(f"      [SAVE] +{len(unique_rows)} | Tổng: {saved_total}/{TARGET_TOTAL_REVIEWS}")
                except (TimeoutException, WebDriverException) as exc:
                    print(f"      [!] Lỗi tải sản phẩm: {type(exc).__name__}: {str(exc)[:100]}")
                sleep_brief(1.3)
    except KeyboardInterrupt:
        print("\n[!] Đã dừng bằng Ctrl+C. Các sản phẩm đã lưu vẫn giữ nguyên.")
    finally:
        print(f"\n[+] Kết thúc. File '{OUTPUT_FILE}': {saved_total} review.")
        # Không driver.quit() để Chrome debug do người dùng mở vẫn hoạt động.


if __name__ == "__main__":
    main()
