import Options
import requests
import pandas as pd
import time
import random
from datetime import datetime
from tqdm import tqdm
from selenium import webdriver

# ==========================================
# 1. CẤU HÌNH HEADERS & THÔNG SỐ CƠ BẢN
# ==========================================
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 6.3; Win64; x64; rv:83.0) Gecko/20100101 Firefox/83.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'vi-VN,vi;q=0.8,en-US;q=0.5,en;q=0.3',
    'Referer': 'https://tiki.vn/dien-thoai-samsung-galaxy-m31-128gb-6gb-hang-chinh-hang-p58259141.html',
    'x-guest-token': '8jWSuIDBb2NGVzr6hsUZXpkP1FRin7lY',
    'Connection': 'keep-alive',
    'TE': 'Trailers',
}



params_list = {
    'limit': '48',
    'category': '1795',
    'page': '1',
    'sort': 'top_seller',
    'src': 'c1795',
    'urlKey': 'dien-thoai-smartphone'
}

params_review = {
    'sort': 'score|desc,id|desc,stars|all',
    'limit': '20',
    'include': 'comments'
}
def get_cookies_from_chrome():
    """BƯỚC 1: Lấy Cookie từ Chrome đang mở qua cổng 9222"""
    print("=" * 60)
    print("[*] BƯỚC 1: KẾT NỐI VÀ LẤY COOKIE")
    print("=" * 60)
    options = Options()
    options.add_experimental_option("debuggerAddress", "127.0.0.1:9222")

    temp_driver = webdriver.Chrome(options=options)

    if "tiki.vn" not in temp_driver.current_url:
        temp_driver.get("https://tiki.vn")
        time.sleep(3)

    raw_cookies = temp_driver.get_cookies()
    # Chuyển đổi list cookie của Selenium thành dictionary cho requests
    cookie_dict = {cookie['name']: cookie['value'] for cookie in raw_cookies}
    print(f"[+] Thành công! Lấy được {len(cookie_dict)} cookies.\n")
    return cookie_dict, temp_driver


# Hàm xử lý dữ liệu từng comment
# Hàm xử lý dữ liệu từng comment
def comment_parser(json_data, p_name, p_url, p_price):
    d = dict()
    d['product_name'] = p_name
    d['rating'] = json_data.get('rating')

    created_at = json_data.get('created_at')
    d['date'] = datetime.fromtimestamp(created_at).strftime('%Y-%m-%d %H:%M:%S') if created_at else None

    d['product_url'] = p_url
    d['price'] = p_price

    title = json_data.get('title', '')
    content = json_data.get('content', '')
    if title and title != "Bình thường":
        d['comment'] = f"{title}. {content}".strip()
    else:
        d['comment'] = content.strip()

    # --- ĐOẠN MỚI THÊM VÀO ĐỂ XỬ LÝ VOTES ---
    votes = json_data.get('votes', [])
    if votes:
        # Lấy ra danh sách các attribute_name và nối lại bằng dấu phẩy
        attributes_list = [vote.get('attribute_name', '') for vote in votes]
        d['attributes'] = ', '.join(attributes_list)
    else:
        d['attributes'] = None # Nếu người dùng không chọn thuộc tính nào
    # ----------------------------------------

    d['platform'] = 'Tiki'
    return d

# ==========================================
# 2. LẤY DANH SÁCH SẢN PHẨM
# ==========================================
product_info = []

cookies, driver = get_cookies_from_chrome()

print("--- BƯỚC 1: ĐANG LẤY DANH SÁCH SẢN PHẨM ---")

for i in range(1, 40):  # Tùy chỉnh số trang bạn muốn lấy ở đây
    params_list['page'] = i
    print(f'Đang quét trang {i}...')
    try:
        response = requests.get('https://tiki.vn/api/personalish/v1/blocks/listings', headers=headers,
                                params=params_list, timeout=10)
        if response.status_code == 200:
            data = response.json().get('data', [])
            for record in data:
                # Lấy số lượng đánh giá của sản phẩm, nếu không có thì mặc định là 0
                review_count = record.get('review_count', 0)

                # CHỈ LẤY NHỮNG SẢN PHẨM CÓ TỪ 1 ĐÁNH GIÁ TRỞ LÊN
                if review_count > 0:
                    product_info.append({
                        'id': record.get('id'),
                        'product_name': record.get('name'),
                        'price': record.get('price'),
                        'product_url': f"https://tiki.vn/{record.get('url_path')}" if record.get('url_path') else None
                    })
    except Exception as e:
        print(f"Lỗi khi lấy danh sách SP trang {i}: {e}")

    time.sleep(random.uniform(2, 5))

print(f"-> Thu thập được tổng cộng {len(product_info)} sản phẩm CÓ ĐÁNH GIÁ.\n")

# ==========================================
# 3. LẤY ĐÁNH GIÁ TỪNG SẢN PHẨM
# ==========================================
result = []
print("--- BƯỚC 2: BẮT ĐẦU LẤY ĐÁNH GIÁ SẢN PHẨM ---")

for p in tqdm(product_info, desc="Tiến độ tổng"):
    params_review['product_id'] = p['id']

    # --- ĐOẠN MỚI: GHI LOG TÊN VÀ LINK SẢN PHẨM ---
    tqdm.write(f"\n{'=' * 60}")
    tqdm.write(f"▶ Đang cào SP : {p['product_name']}")
    tqdm.write(f"▶ Link        : {p['product_url']}")
    tqdm.write(f"{'=' * 60}")
    # ----------------------------------------------

    page = 1
    while True:
        params_review['page'] = page
        try:
            response = requests.get('https://tiki.vn/api/v2/reviews', headers=headers, params=params_review,
                                    cookies=cookies, timeout=10)

            if response.status_code == 200:
                json_data = response.json()
                data = json_data.get('data', [])

                if not data:
                    tqdm.write("  -> Đã hết bình luận. Chuyển SP tiếp theo.")
                    break

                for comment in data:
                    result.append(comment_parser(comment, p['product_name'], p['product_url'], p['price']))

                paging = json_data.get('paging', {})
                last_page = paging.get('last_page', 1)

                # In tiến độ trang đang cào
                tqdm.write(f"  -> Đã quét xong trang {page}/{last_page}")

                if page >= last_page:
                    break



            else:
                tqdm.write(f"  -> Lỗi API {response.status_code} tại trang {page}")
                break

        except Exception as e:
            tqdm.write(f"  -> Lỗi kết nối tại trang {page}: {e}")
            break

        page += 1
        time.sleep(random.uniform(1.5, 3.5))

# ==========================================
# 4. LƯU DỮ LIỆU RA CSV
# ==========================================
df_comment = pd.DataFrame(result)

if not df_comment.empty:
    # Thêm 'attributes' vào danh sách cột
    columns_order = ['product_name', 'rating', 'date', 'product_url', 'price', 'comment', 'attributes', 'platform']
    df_comment = df_comment[columns_order]

    df_comment.to_csv('tiki_reviews_final.csv', index=False, encoding='utf-8-sig')
    print(f"\n HOÀN THÀNH! Đã lưu {len(df_comment)} đánh giá vào file 'tiki_reviews_final.csv'.")
else:
    print("\n Không có dữ liệu đánh giá nào được lấy thành công.")