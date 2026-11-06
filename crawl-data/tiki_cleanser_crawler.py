"""Crawl Tiki cleanser reviews through JSON APIs with incremental CSV output.

Verify star filters and use bounded all-star scans when unsupported."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urljoin

import requests

# Selenium is only used to read cookies from your already-open Chrome instance.
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
except ImportError:
    webdriver = None
    Options = None


SEARCH_KEYWORD = "sữa rửa mặt"
TARGET_TOTAL_REVIEWS = 6000
MAX_REVIEWS_PER_PRODUCT = 200
STAR_QUOTAS = {1: 30, 2: 30, 3: 30, 4: 50}

MAX_SEARCH_PAGES = 40
SEARCH_PAGE_SIZE = 48
MAX_REVIEW_PAGES_PER_STAR = 60
MAX_UNFILTERED_REVIEW_PAGES = 150
REVIEW_PAGE_SIZE = 20
MIN_COMMENT_LENGTH = 6

OUTPUT_FILE = "data/tiki_cleanser_reviews.csv"
DEBUG_DIR = "data/tiki_debug"
DEBUG_SAVE_FAILED_RESPONSE = True

USE_CHROME_COOKIES = True
CHROME_DEBUG_PORT = 9222
FETCH_MISSING_PRODUCT_DETAILS = True
REQUEST_DELAY = (0.8, 1.8)
PRODUCT_DELAY = (1.0, 2.0)
REQUEST_TIMEOUT = 15
MAX_REQUEST_RETRIES = 3

BASE = "https://tiki.vn"
SEARCH_APIS = (
    f"{BASE}/api/v2/products",
    f"{BASE}/api/personalish/v1/blocks/listings",
)
DETAIL_API = f"{BASE}/api/v2/products/{{product_id}}"
REVIEWS_API = f"{BASE}/api/v2/reviews"

CSV_HEADERS = [
    "product_name", "brand", "rating", "date", "product_url",
    "price", "discount", "comment", "review_text_length",
    "attributes", "platform", "product_id", "review_id", "original_price",
]

VN_TZ = timezone(timedelta(hours=7))


def pause(delay=REQUEST_DELAY):
    time.sleep(random.uniform(*delay))


def tidy(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()


def get_first(record, *names):
    for name in names:
        value = record.get(name)
        if value is not None and value != "":
            return value
    return None


def numeric(value):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    # Support '125.000 ₫', '125,000', '23%' when needed.
    text = str(value).strip().replace("₫", "").replace("đ", "").replace("%", "")
    text = re.sub(r"\s+", "", text)
    if not text:
        return None
    if re.fullmatch(r"-?\d{1,3}(?:\.\d{3})+", text):
        text = text.replace(".", "")
    elif re.fullmatch(r"-?\d{1,3}(?:,\d{3})+", text):
        text = text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def integer(value):
    number = numeric(value)
    return int(number) if number is not None else None


def get_brand(record):
    brand = record.get("brand")
    if isinstance(brand, dict):
        brand = get_first(brand, "name", "brand_name")
    if isinstance(brand, str) and brand.strip():
        return tidy(brand)
    return tidy(get_first(record, "brand_name", "brandName"))


def product_url(record, product_id):
    path = get_first(record, "url_path", "product_url", "url", "url_key")
    if path:
        return urljoin(BASE + "/", str(path).lstrip("/"))
    return f"{BASE}/p{product_id}.html"


def timestamp_to_vn(value):
    if value is None or value == "":
        return ""
    try:
        if isinstance(value, (int, float)) or str(value).isdigit():
            ts = float(value)
            if ts > 100_000_000_000:  # Unix milliseconds
                ts /= 1000
            return datetime.fromtimestamp(ts, VN_TZ).strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError, OSError):
        pass
    value = tidy(value)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo:
            dt = dt.astimezone(VN_TZ)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value


def discount_value(price, original_price, rate):
    raw = numeric(rate)
    if raw is not None:
        return f"{abs(int(round(raw)))}%"
    if price is not None and original_price is not None and original_price > 0:
        return f"{max(0, round((original_price - price) / original_price * 100))}%"
    return ""  # Unknown, not necessarily 0%.


def product_from_json(record):
    if not isinstance(record, dict):
        return None
    pid = get_first(record, "id", "product_id")
    if pid is None:
        return None

    review_count = get_first(record, "review_count", "reviews_count", "reviewCount")
    if review_count is not None and integer(review_count) == 0:
        return None

    price = integer(get_first(record, "price", "sale_price", "final_price"))
    original = integer(get_first(record, "original_price", "list_price", "market_price"))
    rate = get_first(record, "discount_rate", "discount_percent", "discountRate")
    return {
        "product_id": str(pid),
        "product_name": tidy(get_first(record, "name", "product_name", "title")),
        "product_url": product_url(record, pid),
        "brand": get_brand(record),
        "price": price,
        "original_price": original,
        "discount": discount_value(price, original, rate),
        "review_count": integer(review_count),
    }


def merge_product_detail(product, record):
    if not isinstance(record, dict):
        return product
    detail = product_from_json({**record, "review_count": product.get("review_count", 1)})
    if not detail:
        return product
    # Prefer product-detail metadata, but never overwrite a valid value with empty.
    for key in ("product_name", "brand", "price", "original_price", "discount"):
        if detail[key] not in (None, ""):
            product[key] = detail[key]
    return product


def comment_text(record):
    title = tidy(record.get("title"))
    content = tidy(record.get("content"))
    if title and title.casefold() not in {"bình thường", "không có tiêu đề"}:
        if not content:
            return title
        return f"{title}. {content}"
    return content


def review_row(review, product):
    rating = integer(review.get("rating"))
    content = comment_text(review)
    if rating not in (1, 2, 3, 4, 5) or len(content) < MIN_COMMENT_LENGTH:
        return None

    votes = review.get("votes") or []
    attributes = ", ".join(
        tidy(v.get("attribute_name")) for v in votes
        if isinstance(v, dict) and tidy(v.get("attribute_name"))
    )
    rid = get_first(review, "id", "review_id")
    return {
        "product_name": product["product_name"],
        "brand": product.get("brand", ""),
        "rating": rating,
        "date": timestamp_to_vn(review.get("created_at")),
        "product_url": product["product_url"],
        "price": product.get("price") if product.get("price") is not None else "",
        "discount": product.get("discount", ""),
        "comment": content,
        "review_text_length": len(content),
        "attributes": attributes,
        "platform": "Tiki",
        "product_id": product["product_id"],
        "review_id": str(rid) if rid is not None else "",
        "original_price": product.get("original_price") if product.get("original_price") is not None else "",
    }


def row_key(row):
    # Review ID is preferred; fallback avoids duplicating reviews without IDs.
    pid = str(row.get("product_id") or row.get("product_url") or "").strip()
    rid = str(row.get("review_id") or "").strip()
    if pid and rid:
        return (pid, "id", rid)
    content = tidy(row.get("comment")).casefold()
    date = tidy(row.get("date"))
    rating = str(row.get("rating") or "").strip()
    return (pid, "fallback", hashlib.sha256(f"{rating}|{date}|{content}".encode()).hexdigest())


class APIError(RuntimeError):
    pass


class TikiClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            "Accept": "application/json, text/plain, */*",
            "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.6,en;q=0.5",
            "Referer": BASE + "/",
        })
        guest_token = os.environ.get("TIKI_GUEST_TOKEN", "").strip()
        if guest_token:
            self.session.headers["x-guest-token"] = guest_token
        self.search_api = None
        if USE_CHROME_COOKIES:
            self.load_chrome_cookies()

    def load_chrome_cookies(self):
        if webdriver is None or Options is None:
            print("[!] Selenium chưa cài đặt; thử API không kèm Chrome cookies.")
            return
        driver = None
        try:
            options = Options()
            options.add_experimental_option("debuggerAddress", f"127.0.0.1:{CHROME_DEBUG_PORT}")
            driver = webdriver.Chrome(options=options)
            if "tiki.vn" not in driver.current_url:
                driver.get(BASE)
                time.sleep(2)
            for ck in driver.get_cookies():
                self.session.cookies.set(
                    ck["name"], ck["value"],
                    domain=ck.get("domain", ".tiki.vn"),
                    path=ck.get("path", "/"),
                )
            print(f"[+] Đã nạp {len(self.session.cookies)} cookies từ Chrome {CHROME_DEBUG_PORT}.")
        except Exception as exc:
            print(f"[!] Không đọc được cookie Chrome: {exc}")
            print("    Thử tiếp với API công khai; nếu bị 401/403 hãy kiểm tra phiên Chrome.")
        # Deliberately not quitting the user's attached Chrome browser.

    def get(self, url, params=None, save_debug_as=None):
        for attempt in range(1, MAX_REQUEST_RETRIES + 1):
            try:
                resp = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT)
                if resp.status_code in (401, 403):
                    raise APIError(f"HTTP {resp.status_code}: không được phép truy cập {resp.url}")
                if resp.status_code == 429:
                    raise APIError(f"HTTP 429: Tiki giới hạn tần suất truy cập {resp.url}")
                if resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                if save_debug_as and DEBUG_SAVE_FAILED_RESPONSE:
                    self.save_debug(save_debug_as, payload)
                return payload
            except APIError:
                raise
            except (requests.RequestException, ValueError) as exc:
                if attempt == MAX_REQUEST_RETRIES:
                    raise APIError(f"API lỗi ({url}): {exc}") from exc
                delay = min(2 ** attempt + random.uniform(0.2, 0.8), 8)
                print(f"    [!] Request lỗi ({attempt}/{MAX_REQUEST_RETRIES}): {exc}; thử lại sau {delay:.1f}s")
                time.sleep(delay)
        raise APIError("Không thể hoàn tất request")

    @staticmethod
    def save_debug(name, payload):
        Path(DEBUG_DIR).mkdir(parents=True, exist_ok=True)
        path = Path(DEBUG_DIR) / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"    [DEBUG] API JSON: {path}")

    @staticmethod
    def extract_records(payload):
        """Return a product list from supported envelopes without guessing ratings."""
        if not isinstance(payload, dict):
            return []
        data = payload.get("data", [])
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for name in ("products", "items", "data", "records"):
                if isinstance(data.get(name), list):
                    return data[name]
        return []

    def search(self, page):
        params = {
            "q": SEARCH_KEYWORD,
            "page": page,
            "limit": SEARCH_PAGE_SIZE,
            "sort": "top_seller",
        }
        endpoints = (self.search_api,) if self.search_api else SEARCH_APIS
        last_error = ""
        for endpoint in endpoints:
            try:
                payload = self.get(endpoint, params=params)
                records = self.extract_records(payload)
                if page == 1 and not records:
                    self.save_debug("search_page_1_empty.json", payload)
                    print(f"    [!] API {endpoint} trả về 0 sản phẩm; thử endpoint khác nếu có.")
                    continue
                self.search_api = endpoint
                return [p for p in (product_from_json(item) for item in records) if p]
            except APIError as exc:
                last_error = str(exc)
                if self.search_api:
                    raise
                print(f"    [!] {last_error}")
        if page == 1:
            print(f"[!] Không tìm được sản phẩm qua API. {last_error}")
        return []

    def enrich_product(self, product):
        if not FETCH_MISSING_PRODUCT_DETAILS:
            return product
        missing = any(product.get(k) in (None, "") for k in ("price", "discount", "brand"))
        if not missing:
            return product
        try:
            payload = self.get(DETAIL_API.format(product_id=product["product_id"]))
            detail = payload.get("data", payload) if isinstance(payload, dict) else {}
            if isinstance(detail, dict):
                merge_product_detail(product, detail)
        except APIError as exc:
            if any(code in str(exc) for code in ("HTTP 429", "HTTP 403", "HTTP 401")):
                raise
            print(f"    [!] Chi tiết sản phẩm không đọc được: {exc}")
        return product

    @staticmethod
    def extract_reviews(payload):
        if not isinstance(payload, dict):
            return [], {}
        data = payload.get("data", [])
        if isinstance(data, list):
            return data, payload.get("paging", {}) or {}
        if isinstance(data, dict):
            reviews = data.get("data") or data.get("reviews") or []
            paging = data.get("paging") or payload.get("paging") or {}
            return (reviews if isinstance(reviews, list) else []), paging
        return [], {}

    def reviews(self, product_id, page, star=None, mode="sort", all_star_sort="score|desc,id|desc,stars|all"):
        params = {
            "product_id": product_id,
            "page": page,
            "limit": REVIEW_PAGE_SIZE,
            "include": "comments",
        }
        if star is None:
            params["sort"] = all_star_sort
        elif mode == "sort":
            params["sort"] = f"score|desc,id|desc,stars|{star}"
        elif mode == "stars":
            params["sort"] = "score|desc,id|desc,stars|all"
            params["stars"] = star
        else:
            raise ValueError(f"Unknown mode: {mode}")

        payload = self.get(REVIEWS_API, params=params)
        return self.extract_reviews(payload)


class CSVStore:
    def __init__(self, filename):
        self.path = Path(filename)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.seen_keys = set()
        self.product_counts = Counter()
        self.rating_counts = defaultdict(Counter)
        self.total = 0
        if self.path.exists() and self.path.stat().st_size > 0:
            with self.path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames != CSV_HEADERS:
                    raise ValueError(
                        f"CSV cũ có cột {reader.fieldnames}, nhưng chương trình cần {CSV_HEADERS}. "
                        "Dùng OUTPUT_FILE mới hoặc chuyển đổi dữ liệu trước; chương trình không ghi đè."
                    )
                for row in reader:
                    key = row_key(row)
                    self.seen_keys.add(key)
                    pid = str(row.get("product_id") or "")
                    self.product_counts[pid] += 1
                    self.rating_counts[pid][integer(row.get("rating"))] += 1
                    self.total += 1
        print(f"[*] CSV: {self.path} | Đã có {self.total}/{TARGET_TOTAL_REVIEWS} review.")

    def remaining(self, product_id):
        return max(0, min(TARGET_TOTAL_REVIEWS - self.total,
                          MAX_REVIEWS_PER_PRODUCT - self.product_counts[str(product_id)]))

    def save(self, rows):
        new_rows = []
        for row in rows:
            pid = str(row.get("product_id") or "")
            key = row_key(row)
            if key in self.seen_keys:
                continue
            if self.total + len(new_rows) >= TARGET_TOTAL_REVIEWS:
                break
            if self.product_counts[pid] + sum(1 for r in new_rows if r["product_id"] == pid) >= MAX_REVIEWS_PER_PRODUCT:
                continue
            new_rows.append(row)
            self.seen_keys.add(key)

        if not new_rows:
            return 0
        write_header = not self.path.exists() or self.path.stat().st_size == 0
        # Append with UTF-8; another BOM would corrupt the middle of the CSV.
        file_encoding = "utf-8-sig" if write_header else "utf-8"
        with self.path.open("a", encoding=file_encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_HEADERS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerows(new_rows)
            f.flush()

        for row in new_rows:
            pid = str(row["product_id"])
            self.product_counts[pid] += 1
            self.rating_counts[pid][row["rating"]] += 1
            self.total += 1
        return len(new_rows)


def take_filtered_star(client, product, star, quota, occupied_keys):
    """Fetch one rating bucket and verify every returned rating; otherwise use all-star fallback."""
    if quota <= 0:
        return [], True

    for mode in ("sort", "stars"):
        found = []
        local_seen = set()
        supported = True
        try:
            for page in range(1, MAX_REVIEW_PAGES_PER_STAR + 1):
                reviews, paging = client.reviews(product["product_id"], page, star, mode=mode)
                if not reviews:
                    break
                for record in reviews:
                    rating = integer(record.get("rating"))
                    if rating is not None and rating != star:
                        supported = False
                        break
                if not supported:
                    break

                for record in reviews:
                    row = review_row(record, product)
                    if not row:
                        continue
                    key = row_key(row)
                    if key in occupied_keys or key in local_seen:
                        continue
                    local_seen.add(key)
                    found.append(row)
                    if len(found) >= quota:
                        break
                if len(found) >= quota:
                    break

                last_page = integer(paging.get("last_page")) if isinstance(paging, dict) else None
                if (last_page is not None and page >= last_page) or len(reviews) < REVIEW_PAGE_SIZE:
                    break
                pause()
        except APIError as exc:
            if any(code in str(exc) for code in ("HTTP 429", "HTTP 403", "HTTP 401")):
                raise
            print(f"    [!] Filter {star} sao ({mode}) lỗi: {exc}")
            supported = False
        if supported:
            print(f"    [+] {star} sao: {len(found)} review hợp lệ (filter={mode}).")
            return found, True
        print(f"    [!] Filter {star} sao ({mode}) bị bỏ qua/không hoạt động.")
    return [], False


def take_unfiltered_fallback(client, product, low_star_requirements, occupied_keys, capacity):
    """Scan bounded review pages, collecting low-star reviews before five-star reviews."""
    by_star = {star: [] for star in (1, 2, 3, 4, 5)}
    local_seen = set()
    for page in range(1, MAX_UNFILTERED_REVIEW_PAGES + 1):
        try:
            reviews, paging = client.reviews(
                product["product_id"], page,
                all_star_sort="score|asc,id|desc,stars|all",
            )
        except APIError as exc:
            if any(code in str(exc) for code in ("HTTP 429", "HTTP 403", "HTTP 401")):
                raise
            print(f"    [!] Quét fallback thất bại tại trang {page}: {exc}")
            break
        if not reviews:
            break
        for record in reviews:
            row = review_row(record, product)
            if not row:
                continue
            key = row_key(row)
            if key in occupied_keys or key in local_seen:
                continue
            star = row["rating"]
            if star < 5 and len(by_star[star]) >= low_star_requirements.get(star, 0):
                continue
            if star == 5 and len(by_star[5]) >= capacity:
                continue
            by_star[star].append(row)
            local_seen.add(key)

        low_ready = all(len(by_star[s]) >= low_star_requirements.get(s, 0) for s in range(1, 5))
        low_total = sum(len(by_star[s]) for s in range(1, 5))
        if low_ready and low_total + len(by_star[5]) >= capacity:
            break
        last_page = integer(paging.get("last_page")) if isinstance(paging, dict) else None
        if (last_page is not None and page >= last_page) or len(reviews) < REVIEW_PAGE_SIZE:
            break
        pause()
    return by_star


def crawl_product(client, product, store):
    product = client.enrich_product(product)
    pid = product["product_id"]
    capacity = store.remaining(pid)
    if capacity <= 0:
        return []
    print(f"  Giá={product['price']} | giảm={product['discount'] or 'N/A'} | brand={product['brand'] or 'N/A'}")
    collected = []
    occupied = set(store.seen_keys)
    needed = {}
    invalid_filters = set()

    for star in (1, 2, 3, 4):
        available = capacity - len(collected)
        if available <= 0:
            break
        quota = max(0, STAR_QUOTAS[star] - store.rating_counts[pid][star])
        quota = min(quota, available)
        if not quota:
            continue
        rows, valid_filter = take_filtered_star(client, product, star, quota, occupied)
        if not valid_filter:
            invalid_filters.add(star)
            needed[star] = quota
        else:
            for row in rows:
                occupied.add(row_key(row))
            collected.extend(rows)

    five_from_fallback = []
    if invalid_filters and len(collected) < capacity:
        print(f"    [*] Fallback trang review thường cho sao: {sorted(invalid_filters)}")
        scans = take_unfiltered_fallback(
            client, product, needed, occupied, capacity - len(collected)
        )
        for star in (1, 2, 3, 4):
            for row in scans[star][:needed.get(star, 0)]:
                if len(collected) >= capacity:
                    break
                occupied.add(row_key(row))
                collected.append(row)
        five_from_fallback = scans[5]

    needed_five = capacity - len(collected)
    if needed_five:
        for row in five_from_fallback:
            if needed_five <= 0:
                break
            if row_key(row) not in occupied:
                occupied.add(row_key(row))
                collected.append(row)
                needed_five -= 1

    if needed_five:
        five_rows, valid_filter = take_filtered_star(client, product, 5, needed_five, occupied)
        if valid_filter:
            for row in five_rows:
                occupied.add(row_key(row))
            collected.extend(five_rows)
        else:
            print("    [!] Filter 5 sao không hoạt động; thử quét trang reviews thường.")
            scans = take_unfiltered_fallback(client, product, {}, occupied, needed_five)
            collected.extend(scans[5][:needed_five])

    # Unsupported filters may return empty lists; retry without a filter.
    if not collected and (product.get("review_count") or 0) > 0 and not invalid_filters:
        print("    [!] Không có review qua filter; thử API review không lọc sao.")
        requirements = {
            star: min(capacity, max(0, STAR_QUOTAS[star] - store.rating_counts[pid][star]))
            for star in (1, 2, 3, 4)
        }
        fallback = take_unfiltered_fallback(client, product, requirements, occupied, capacity)
        for star in (1, 2, 3, 4):
            for row in fallback[star][:requirements[star]]:
                if len(collected) >= capacity:
                    break
                collected.append(row)
        for row in fallback[5]:
            if len(collected) >= capacity:
                break
            collected.append(row)

    return collected[:capacity]


def main():
    store = CSVStore(OUTPUT_FILE)
    if store.total >= TARGET_TOTAL_REVIEWS:
        print("[+] Đã đủ target, không cần crawl thêm.")
        return
    client = TikiClient()
    processed_ids = set()

    print(f"[*] Keyword='{SEARCH_KEYWORD}' | Max tổng={TARGET_TOTAL_REVIEWS} | Max/SP={MAX_REVIEWS_PER_PRODUCT}")
    try:
        for page in range(1, MAX_SEARCH_PAGES + 1):
            if store.total >= TARGET_TOTAL_REVIEWS:
                break
            print(f"\n[*] Tìm sản phẩm trang {page}/{MAX_SEARCH_PAGES} ...")
            products = client.search(page)
            print(f"    [+] {len(products)} sản phẩm có thể crawl.")
            if not products:
                print("[!] Hết sản phẩm hoặc API chưa trả về kết quả; kiểm tra thư mục debug.")
                break

            for index, product in enumerate(products, start=1):
                if store.total >= TARGET_TOTAL_REVIEWS:
                    break
                pid = product["product_id"]
                if pid in processed_ids or store.remaining(pid) <= 0:
                    continue
                processed_ids.add(pid)
                print(f"\n[{index}/{len(products)}] {product['product_name'][:95]} | id={pid}")
                try:
                    rows = crawl_product(client, product, store)
                    saved = store.save(rows)
                    print(f"    => Lưu {saved} review | Tổng: {store.total}/{TARGET_TOTAL_REVIEWS}")
                except APIError as exc:
                    print(f"    [!] API lỗi: {exc}")
                    if "HTTP 429" in str(exc) or "HTTP 403" in str(exc) or "HTTP 401" in str(exc):
                        print("    [!] Dừng crawl vì giới hạn hoặc từ chối truy cập.")
                        return
                pause(PRODUCT_DELAY)
            pause(REQUEST_DELAY)
    except KeyboardInterrupt:
        print("\n[!] Dừng thủ công (Ctrl+C). Dữ liệu đã lưu vẫn được giữ nguyên.")
    except APIError as exc:
        print(f"\n[!] Dừng do lỗi API: {exc}")
    finally:
        client.session.close()
        print(f"\n[+] Hoàn tất. {store.total} review trong {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
