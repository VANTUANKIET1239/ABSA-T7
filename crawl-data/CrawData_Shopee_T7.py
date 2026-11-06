import os
import time
import re
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

TARGET_TOTAL_REVIEWS = 6000
MAX_REVIEWS_PER_PRODUCT = 200
OUTPUT_FILE = "data/shopee_reviews.csv"


def get_cookies_from_chrome():
    """Lấy cookie từ Chrome đang mở qua cổng 9222."""
    print("=" * 60)
    print("[*] BƯỚC 1: KẾT NỐI VÀ LẤY COOKIE")
    print("=" * 60)
    options = Options()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

    temp_driver = webdriver.Chrome(options=options)

    if "shopee.vn" not in temp_driver.current_url:
        temp_driver.get("https://shopee.vn")
        time.sleep(3)

    raw_cookies = temp_driver.get_cookies()
    cookie_dict = {cookie['name']: cookie['value'] for cookie in raw_cookies}
    print(f"[+] Thành công! Lấy được {len(cookie_dict)} cookies.\n")
    return cookie_dict, temp_driver


def get_product_list(driver, page_index):
    """Lấy danh sách sản phẩm, giá và mức giảm giá."""
    url = f"https://shopee.vn/search?keyword=sửa%20rửa%20mặt&page={page_index}&sortBy=sales"
    print(f"[*] BƯỚC 2: QUÉT SẢN PHẨM TRANG {page_index + 1}")

    driver.get(url)

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, ".shopee-search-item-result__item"))
        )
    except:
        print("    [!] Không tìm thấy sản phẩm, có thể do mạng hoặc hết trang.")
        return []

    for i in range(4):
        print(f"    - Đang tải mượt dữ liệu... ({i + 1}/4)", end="\r", flush=True)
        driver.execute_script("window.scrollBy(0, 1500);")
        time.sleep(0.8)
    print("    - Đã cuộn xong! Đang trích xuất danh sách...        ")

    soup = BeautifulSoup(driver.page_source, "html.parser")
    products = []
    item_cards = soup.select(".shopee-search-item-result__item a.contents")

    for card in item_cards:
        try:
            link = card.get("href")
            if not link: continue
            if link.startswith('/'): link = "https://shopee.vn" + link

            match = re.search(r'-i\.(\d+)\.(\d+)', link)
            if not match: continue

            shop_id, item_id = match.groups()

            text_content = card.get_text(separator=" ")
            card_text_list = card.get_text(separator="\n").split("\n")

            price_match = re.search(r'(?:₫\s*[\d\.]+|[\d\.]+\s*₫)(?:\s*-\s*(?:₫\s*[\d\.]+|[\d\.]+\s*₫))?', text_content)

            if price_match:
                print("Tìm thấy giá:", price_match.group(0))
            else:
                print("Không tìm thấy")
            price = price_match.group(0) if price_match else "0₫"

            discount_match = re.search(r'-?(\d+)%', text_content)
            discount = f"{discount_match.group(1)}%" if discount_match else "0%"

            title_candidates = [t.strip() for t in card_text_list if
                                len(t.strip()) > 10 and '₫' not in t and 'Đã bán' not in t]
            title = title_candidates[0] if title_candidates else "Unknown Smartphone"

            products.append({
                "product_name": title,
                "product_url": link.split('?')[0],
                "price": price,
                "discount": discount,
                "shop_id": shop_id,
                "item_id": item_id
            })
        except Exception:
            continue

    print(f"[+] Hoàn tất! Phát hiện {len(products)} sản phẩm.\n")
    return products


def fetch_reviews_via_stealth_api(product, driver):
    """Lấy review theo ưu tiên số sao và thông tin cửa hàng."""
    product_reviews = []

    print(f"      -> Đang vào trang SP để nạp Token & Dữ liệu Shop...", end="\r", flush=True)
    driver.get(product['product_url'])
    time.sleep(3)
    driver.set_script_timeout(15)

    js_get_shop_info = """
    let shopData = {
        shop_reviews: "N/A", response_rate: "N/A", join_date: "N/A",
        total_products: "N/A", response_time: "N/A", followers: "N/A"
    };
    document.querySelectorAll('label').forEach(labelEl => {
        let label = labelEl.innerText.trim().toLowerCase();
        let parent = labelEl.parentElement;
        let valEl = parent ? parent.querySelector('span') : null;
        if (!valEl) return;
        let value = valEl.innerText.trim();

        if(label === 'đánh giá') shopData.shop_reviews = value;
        else if(label === 'tỉ lệ phản hồi') shopData.response_rate = value;
        else if(label === 'tham gia') shopData.join_date = value;
        else if(label === 'sản phẩm') shopData.total_products = value;
        else if(label === 'thời gian phản hồi') shopData.response_time = value;
        else if(label === 'người theo dõi') shopData.followers = value;
    });
    return shopData;
    """
    try:
        shop_info = driver.execute_script(js_get_shop_info)
        print(f"      -> Uy tín Shop: {shop_info['followers']} Follow | {shop_info['shop_reviews']} Đánh giá")
    except:
        shop_info = {"shop_reviews": "N/A", "response_rate": "N/A", "join_date": "N/A", "total_products": "N/A",
                     "response_time": "N/A", "followers": "N/A"}

    total_fetched_for_product = 0

    def fetch_for_star(star_level, max_quota):
        nonlocal total_fetched_for_product
        print(f"\n        >>> Đang quét đánh giá {star_level} sao (Tối đa {max_quota} cái):")

        current_star_fetched = 0
        offset = 0

        while current_star_fetched < max_quota:
            api_url = f"https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&limit=50&offset={offset}&type={star_level}&shopid={product['shop_id']}&itemid={product['item_id']}"

            js_script = """
            var callback = arguments[1];
            var url = arguments[0];
            fetch(url, {
                method: 'GET',
                headers: { 'accept': 'application/json', 'x-api-source': 'pc', 'x-requested-with': 'XMLHttpRequest' }
            })
            .then(response => response.json())
            .then(data => callback(data))
            .catch(error => callback({"error": 9999, "error_msg": error.toString()}));
            """

            try:
                data = driver.execute_async_script(js_script, api_url)

                if data.get("error"):
                    error_code = data.get("error")
                    if error_code == 90309999:
                        print("        [!] Bị Shopee chặn (90309999)! Hãy mở trình duyệt kéo Captcha.")
                    else:
                        print(f"        [!] Lỗi API: {error_code}")
                    return False

                ratings = data.get("data", {}).get("ratings")
                if not ratings:
                    print(f"        + Đã hết đánh giá {star_level} sao.")
                    break

                valid_count = 0
                for r in ratings:
                    if current_star_fetched >= max_quota: break

                    comment = r.get("comment", "")
                    if comment and len(comment.strip()) > 5:
                        timestamp = r.get("ctime")
                        date_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S') if timestamp else ""

                        product_reviews.append({
                            "product_name": product["product_name"],
                            "rating": r.get("rating_star"),
                            "date": date_str,
                            "product_url": product["product_url"],
                            "price": product["price"],
                            "discount": product["discount"],

                            "shop_reviews": shop_info["shop_reviews"],
                            "shop_response_rate": shop_info["response_rate"],
                            "shop_join_date": shop_info["join_date"],
                            "shop_total_products": shop_info["total_products"],
                            "shop_response_time": shop_info["response_time"],
                            "shop_followers": shop_info["followers"],

                            "comment": comment.replace("\n", " ").strip(),
                            "platform": "shopee"
                        })
                        valid_count += 1
                        current_star_fetched += 1
                        total_fetched_for_product += 1

                print(f"        + Offset {offset}: Lấy được {valid_count} comment hợp lệ.")
                if len(ratings) < 50: break

                offset += 50
                time.sleep(1.5)

            except Exception as e:
                print(f"        [!] Lỗi thực thi JS: {str(e)[:30]}")
                return False

        return True


    stars_config = [{"star": 1, "max": 30}, {"star": 2, "max": 30}, {"star": 3, "max": 30}, {"star": 4, "max": 50}]

    for config in stars_config:
        success = fetch_for_star(config["star"], config["max"])
        if not success: return product_reviews

    max_5_star = MAX_REVIEWS_PER_PRODUCT - total_fetched_for_product
    if max_5_star > 0:
        print(
            f"\n        >>> (Dồn chỉ tiêu) Còn thiếu {max_5_star}/{MAX_REVIEWS_PER_PRODUCT} đánh giá, chuyển qua lấy 5 sao:")
        fetch_for_star(5, max_5_star)

    return product_reviews


def save_data(new_data):
    if not new_data: return
    df = pd.DataFrame(new_data)
    file_exists = os.path.exists(OUTPUT_FILE)
    df.to_csv(OUTPUT_FILE, mode='a', index=False, header=not file_exists, encoding="utf-8-sig")


def main():
    cookies_dict, driver = get_cookies_from_chrome()
    total_collected = 0
    page = 0

    print("=" * 60)
    print(f"[*] BƯỚC 3: BẮT ĐẦU CÀO {TARGET_TOTAL_REVIEWS} ĐÁNH GIÁ")
    print("=" * 60)

    try:
        while total_collected < TARGET_TOTAL_REVIEWS:
            products = get_product_list(driver, page)
            if not products:
                print("[-] Không tìm thấy thêm sản phẩm nào.")
                break

            for idx, product in enumerate(products):
                if total_collected >= TARGET_TOTAL_REVIEWS: break

                short_name = product['product_name'][:45] + "..." if len(product['product_name']) > 45 else product[
                    'product_name']
                print(
                    f"\n[{idx + 1}/{len(products)}] Đang cào: {short_name} | Giá: {product['price']} | Giảm: {product['discount']}")

                reviews = fetch_reviews_via_stealth_api(product, driver)
                save_data(reviews)

                total_collected += len(reviews)
                print(
                    f"      => Đã lưu {len(reviews)} comment vào file. (Tiến độ tổng: {total_collected}/{TARGET_TOTAL_REVIEWS})")

            page += 1

    except KeyboardInterrupt:
        print("\n[!] Dừng thủ công (Ctrl+C).")
    finally:
        print("\n" + "=" * 60)
        print(f"[+] KẾT THÚC! Đã thu thập {total_collected} đánh giá vào '{OUTPUT_FILE}'.")
        print("=" * 60)


if __name__ == "__main__":
    main()