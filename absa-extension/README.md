# Shopee Review Analyzer — PhoBERT ABSA local

Chrome Extension Manifest V3 và FastAPI dùng **checkpoint PhoBERT–BiRNN Focal Loss tốt nhất của nghiên cứu**. Không huấn luyện lại, không gọi dịch vụ AI bên ngoài và không dùng database cho extension. Website Django hiện tại không bị thay đổi.

**Cách chạy hiện tại: [Docker CUDA + Cloudflare](./DOCKER.md).** API được tách thành container riêng, dùng GPU NVIDIA với PyTorch 2.14.1/CUDA 13.0; extension dùng HTTPS và khóa truy cập từ `extension/api_config.local.json`. Phần chạy Python local dưới đây là phương án dự phòng. Khóa local không nằm trong Git/ZIP source. Inference vẫn chạy bằng checkpoint trên máy đang chạy Docker.

## Chạy backend trên máy hiện tại

Mở PowerShell tại thư mục gốc repository:

```powershell
cd C:\Users\PC\PycharmProjects\ABSA-T7
& .\.venv-bertopic\Scripts\python.exe -m pip install -r .\absa-extension\backend\requirements.txt
& .\.venv-bertopic\Scripts\python.exe -X utf8 .\absa-extension\backend\run_local.py
```

Chờ `Application startup complete`. Kiểm tra **http://127.0.0.1:8000/health**: `model_loaded` phải là `true`. API docs: **http://127.0.0.1:8000/docs**. Có thể mở [run_local.py](./backend/run_local.py) và nhấn Run trong PyCharm, chọn interpreter `.venv-bertopic/Scripts/python.exe`.

Backend chỉ bind `127.0.0.1:8000`, một process, không reload để tránh nạp model hai lần. Nếu cổng 8000 bị Django cũ chiếm, dừng server Python cũ trước; web Docker ở cổng 8080 vẫn chạy độc lập. Extension cần FastAPI, không gọi API Django.

## Load unpacked trên Chrome

1. Mở `chrome://extensions`, bật **Developer mode**.
2. Nhấn **Load unpacked**, chọn **[extension/](./extension/)** bên trong dự án này.
3. Ghim **Shopee Review Analyzer** từ biểu tượng Extensions.
4. Nhấn biểu tượng extension để xem domain hiện tại và trạng thái API.
5. Ở URL chi tiết sản phẩm Shopee, **Mở công cụ** mở sidebar. Ở trang khác, nút này mở trang nhập tay thuộc extension.
6. **Nhập tay** luôn hoạt động, kể cả khi không truy cập được Shopee.

Bản mẫu ba file ở thư mục gốc đã được thay bằng cấu trúc đầy đủ. Nếu trước đây đã Load unpacked thư mục `absa-extension/`, hãy bỏ bản mẫu cũ và chọn lại `absa-extension/extension/`. Không cần npm, build frontend hoặc Chrome Web Store để sử dụng.

## Cách sử dụng

**Trên Shopee:** mở đúng trang sản phẩm → **Mở công cụ** → tab **Shopee** → chọn nhanh 20/50/100/200/500/1.000 hoặc nhập số nguyên **1–1.000** → **Lấy đánh giá** → kiểm tra bản xem trước → **Phân tích**. Danh sách chọn và ô nhập số đồng bộ với nhau. Chức năng lấy review từ DOM đã được bỏ; chỉ còn API Shopee và nhập tay.

Extension lấy `shopid` và `itemid` từ URL, gọi `/api/v2/item/get_ratings` ngay trong tab đó bằng `fetch` với `credentials: "same-origin"`. Trình duyệt tự gửi cookie hợp lệ của phiên Shopee hiện tại; extension **không đọc hoặc xuất cookie/token, không chuyển cookie sang Python, không cần quyền `cookies`**. Chỉ nội dung `comment` của người mua được chuyển về extension để xem trước và phân tích; tên tài khoản và phản hồi shop không được lấy.

**Nhận diện sản phẩm:** khung **Đang mở** hiển thị tên, mã sản phẩm và mã shop. Kết quả có mục **Kết quả của** ghi rõ sản phẩm được phân tích. Khi URL đổi sang sản phẩm khác (kể cả điều hướng trong SPA), toàn bộ bản xem trước, nội dung nhập tay và kết quả cũ được xóa; phản hồi đang chờ của sản phẩm trước bị bỏ qua. Đóng rồi mở công cụ sẽ đọc lại sản phẩm hiện tại. Nếu URL đổi trước khi trang cập nhật tên, tạm hiển thị mã sản phẩm mới để không gắn tên cũ vào sản phẩm khác. Kiểm tra mỗi 500 ms khi bảng mở và ngay trước/sau thao tác lấy hoặc phân tích. [product.js](./extension/product.js) chỉ đọc thông tin tên/URL, không thu thập nội dung review.

API đọc tuần tự từ offset 0, tối đa 20 trang cho 1.000 review nguồn, mỗi trang tối đa 50 review và nghỉ 800 ms trước trang tiếp theo; không lọc theo số sao hay lấy ngẫu nhiên. Mỗi trang là một message ngắn tới service worker, không giữ một message chờ toàn bộ 1.000 review. Dừng sớm nếu trang trả ít hơn số yêu cầu. Số review phân tích có thể ít hơn số nguồn vì bỏ comment rỗng và trùng. Khi lỗi HTTP 401/403/429, lỗi xác minh hoặc cấu trúc phản hồi thay đổi, quá trình dừng, không tự thử lại. Nếu một trang sau bị lỗi, phần đã lấy được giữ lại và đánh dấu **chưa đầy đủ**. Đây là endpoint nội bộ, có thể thay đổi và không bảo đảm lấy đủ hoặc đại diện cho toàn bộ đánh giá sản phẩm. Nếu chuyển sản phẩm hoặc đóng bảng, các lô tiếp theo không được lên lịch, phản hồi cũ không được dùng.

Gặp đăng nhập/CAPTCHA/kiểm tra truy cập thì tự thao tác trong trình duyệt hoặc dùng Nhập tay; extension không vượt bảo vệ. Sau khi cập nhật source, **Reload extension ở `chrome://extensions` và tải lại tab Shopee**, rồi mở lại công cụ.

**Nhập tay:** mỗi dòng là một review. Bỏ dòng rỗng và chuỗi trùng sau chuẩn hóa khoảng trắng trước khi gửi. Tối đa 1.000 review, 4.000 ký tự/review; nội dung vượt giới hạn bị báo lỗi, không cắt âm thầm. Ví dụ có trong [reviews.json](./sample_data/reviews.json). Upload CSV chưa nằm trong MVP.

**Xử lý theo lô:** tiến độ hiển thị số đánh giá đã đọc/đã phân tích. Phân tích tuần tự tối đa **20 review/request**; 1.000 câu cần 50 request, mỗi request vẫn dùng mini-batch model mặc định 16. Không gửi 1.000 câu trong một request hay chạy 50 request song song. Dashboard chỉ tổng hợp sau khi tất cả lô thành công, giữ đúng thứ tự review và kiểm tra cùng phiên bản model. Lỗi một lô sẽ dừng, không trình bày kết quả một phần như đã hoàn thành. Backend giữ giới hạn mặc định 100 review/request; không cần nâng giới hạn hoặc khởi động lại backend để dùng chức năng 1.000 câu.

**Đọc dashboard:** một review có thể đóng góp nhiều khía cạnh. Tỷ lệ tổng thể dùng tổng lượt **khía cạnh–cảm xúc**, không phải tỷ lệ review tích cực. Mẫu số của mỗi thanh là Positive + Negative + Neutral của riêng khía cạnh đó. Không đề cập không được coi là Neutral. Hiển thị đủ 11 khía cạnh, dòng không xuất hiện ghi “Chưa được đề cập”.

Nhấn tên khía cạnh, màu hoặc tỷ lệ để lọc các review đã phân tích. “Điểm cần chú ý” xếp theo tỷ lệ tiêu cực và kèm số lượng; không phải SHAP/IPA hoặc kết luận đại diện toàn bộ sản phẩm. Confidence là softmax của lớp thắng trong bốn lớp, **chưa hiệu chuẩn**. Xóa kết quả chỉ xóa khỏi phiên xem; không có database hoặc Chrome storage lưu review.

## Tích hợp model thật

- Mặc định: [checkpoint BiRNN Focal Loss](../train-models/Models/Cleanser/best_model_birnn_cleanser_focal_loss_VI.pt). Không sao chép model vào source extension hoặc gửi trọng số sang Chrome.
- [model_service.py](./backend/model_service.py) đối chiếu trực tiếp class `PhoBERT_BiRNN` trong [training source](../train-models/TrainPhoBert_BiLSTM_Beauty.py): encoder 768 → RNN tanh hai chiều hidden 256 → masked mean pooling → dropout 0,3 → linear 512 → 44 → batch × 11 × 4. Nạp toàn bộ encoder/head với `strict=True`, không dùng `AutoModelForSequenceClassification`.
- Tokenizer/config `vinai/phobert-base`, chỉ đọc cache local. `max_length=128`, padding/truncation như nghiên cứu. Không tải lại base weights vì checkpoint đã chứa encoder fine-tune.
- `model.eval()`, `torch.inference_mode()`, load một lần khi startup. Tự chọn CUDA nếu có, CPU nếu không; batch mặc định 16.
- [preprocessing.py](./backend/preprocessing.py) dùng trực tiếp hàm chuẩn hóa/teencode/phủ định/ghép cụm từ trong [source tiền xử lý](../reprocessing/preprocess_cleanser_reviews.py). Phân đoạn bằng cùng Java/VnCoreNLP local; một process Java cho một request, kiểm tra số nhóm đầu ra để không lệch review. Không chạy lại pipeline CSV.
- Mapping lớp **0=Positive, 1=Negative, 2=Neutral, 3=None**. Một lớp cho mỗi khía cạnh như model gốc; None bị bỏ khỏi kết quả đề cập. Version chứa hash checkpoint và source tiền xử lý.

| ID | Aspect | Hiển thị |
|---|---|---|
| A1 | CleansingEffectiveness | Hiệu quả làm sạch |
| A2 | HydrationSkinComfort | Cấp ẩm và độ dễ chịu |
| A3 | SkinCompatibilitySafety | Mức độ an toàn |
| A4 | TextureFoaming | Kết cấu và tạo bọt |
| A5 | Fragrance | Mùi hương |
| A6 | IngredientsFormulation | Thành phần |
| A7 | Packaging | Bao bì |
| A8 | PriceValue | Giá cả |
| A9 | ShopService | Dịch vụ cửa hàng |
| A10 | Delivery | Giao hàng |
| A11 | General | Đánh giá tổng quan |

## API và cấu hình

`GET /health`: trạng thái model, thiết bị, version, giới hạn. Nếu thiếu model, server vẫn mở để báo lỗi; `/analyze` trả 503, **không trả dự đoán giả**. Đã có checkpoint thật nên không thêm mock mode vào bản chạy. Dữ liệu giả chỉ nằm trong unit tests và được đánh dấu là fixtures.

`POST /analyze` nhận:

```json
{"reviews": ["Rửa sạch, không khô da nhưng mùi hơi hắc.", "Giao hàng chậm."]}
```

Response gồm `total_reviews`, `results`, `summary` đủ A1–A11, `model_version`, `device`. Mỗi kết quả có `review_id`, `text`, `aspects` (`code`, `aspect`, `name`, `sentiment`, `confidence`) và `status`. Danh sách rỗng trả kết quả rỗng; review rỗng sau tiền xử lý có `status="empty"`, không sinh Neutral. Sai kiểu trả 422, quá giới hạn trả 413, model chưa sẵn sàng trả 503. API giữ nguyên thứ tự và phần tử trùng trong request; extension bỏ trùng trước khi gửi.

Sidebar chia dữ liệu thành các message tối đa 20 review, đợi từng lô qua service worker/FastAPI xong rồi mới gửi lô tiếp theo, sau đó ghép đúng ID. Service worker từ chối message phân tích lớn hơn 20 câu để không tạo một tác vụ MV3 quá dài. Batch lỗi hoặc model version đổi sẽ báo lỗi thay vì trình bày kết quả một phần như kết quả đầy đủ.

Biến môi trường trong [.env.example](./backend/.env.example) không được tự load. Ví dụ PowerShell:

```powershell
$env:ABSA_DEVICE="cpu"
$env:ABSA_BATCH_SIZE="8"
& .\.venv-bertopic\Scripts\python.exe .\absa-extension\backend\run_local.py
```

| Biến | Mặc định |
|---|---|
| `ABSA_RESEARCH_ROOT` | Thư mục ABSA-T7 |
| `ABSA_CHECKPOINT` | BiRNN Focal Loss trong repository |
| `ABSA_MODEL_NAME` | `vinai/phobert-base`, hoặc folder config/tokenizer local |
| `ABSA_DEVICE` | `auto`; có thể dùng `cpu`, `cuda`, `cuda:0` |
| `ABSA_BATCH_SIZE` | 16 |
| `ABSA_MAX_REVIEWS` | 100/request ở backend; UI tối đa 1.000, chia request không quá 20 |
| `ABSA_MAX_REVIEW_CHARS` | 4000 |
| `ABSA_MAX_TOTAL_CHARS` | 150000/request |
| `ABSA_TORCH_THREADS` | 4 |
| `VNCORENLP_DIR` | `vncorenlp_model` của repository |
| `JAVA_EXECUTABLE` | `java` từ PATH |

Muốn tạo môi trường riêng:

```powershell
cd absa-extension
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
python backend/run_local.py
```

Chọn PyTorch CPU/CUDA theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/). Cần Java và config/tokenizer đã có trong cache của tài khoản chạy API. Khi chuyển source ra khỏi repository, đặt `ABSA_RESEARCH_ROOT` tới ABSA-T7 để tìm tiền xử lý và model. Checkpoint không nằm trong ZIP/Git.

## Kiến trúc, quyền và giới hạn

`Popup/Sidebar → runtime messaging → background service worker → FastAPI → tiền xử lý local → PhoBERT → dashboard`.

Riêng bước thu thập API: `Sidebar → service worker → chrome.scripting (MAIN world của tab sản phẩm) → fetch cùng origin Shopee → comment → bản xem trước`. Chỉ khi bấm Phân tích, nội dung mới đi qua luồng FastAPI ở trên. Không thực hiện request Shopee từ backend Python.

Chỉ yêu cầu `activeTab`, `scripting` và host permission của API loopback. Không có cookies, quyền tabs toàn cục, webRequest hay tất cả website. Chỉ inject khi bấm Mở công cụ trên URL sản phẩm Shopee hợp lệ; không có content script tự chạy. Sidebar Shadow DOM không thay layout Shopee, CSS được lấy từ file đóng gói qua background. Không có eval, CDN hoặc code thực thi tải từ xa. Nội dung review/sản phẩm/API dùng `textContent`, không nội suy vào HTML động.

Backend chỉ cho CORS của `chrome-extension://...`, không cho trang Shopee hoặc website bất kỳ gọi trực tiếp qua trình duyệt. API request từ service worker dùng `credentials: omit`. Khi chạy Docker/Cloudflare, `/analyze` còn kiểm tra `X-API-Key`; CORS không thay thế xác thực. Không log nội dung review; lỗi validation không phản chiếu input. Review là dữ liệu, không phải chỉ dẫn thực thi.

Shopee đã chuyển lần kiểm tra trực tiếp sang `verify/traffic/error` với thông báo Page Unavailable. Luồng API và nhận diện sản phẩm đã được kiểm tra bằng fixtures trong Chromium, **chưa xác nhận được với trang review/API Shopee thật hoạt động trong phiên này**. Không vượt trang kiểm tra. Nhập tay chạy model thật không phụ thuộc Shopee.

## Debug

- **Disconnected:** mở `/health`, kiểm tra terminal, interpreter, Java và cổng 8000. HTML/404 có thể là server Django cũ.
- **Model unavailable:** xem `error` trong `/health`; kiểm tra checkpoint đúng kiến trúc, cache tokenizer, đường dẫn tài nguyên. Không thay bằng model sentiment khác.
- **Sidebar không hiện:** Reload extension trong `chrome://extensions`, sau đó tải lại tab Shopee rồi Mở công cụ. Tab cũ có thể giữ content-script context đã mất hiệu lực.
- **API Shopee bị từ chối:** thử mở trang đánh giá bình thường và hoàn tất đăng nhập/xác minh nếu Shopee yêu cầu, rồi chủ động thử lại hoặc dùng Nhập tay. Không sao chép cookie sang backend. Luồng API nằm trong [shopee_api.js](./extension/shopee_api.js); endpoint nội bộ có thể thay đổi.
- **Console:** popup → Inspect; `chrome://extensions` → service worker Inspect; Shopee → DevTools Console, chọn content-script context khi cần.
- **CPU chậm:** giảm số review/batch size. Không tăng workers vì mỗi worker nạp thêm model.

## Kiểm thử

Từ thư mục gốc ABSA-T7:

```powershell
& .\.venv-bertopic\Scripts\python.exe -m pip install -r absa-extension/backend/requirements-dev.txt
& .\.venv-bertopic\Scripts\python.exe -m unittest discover -s absa-extension/tests -p "test_*.py" -v
node --test absa-extension/tests/extension.test.cjs absa-extension/tests/shopee_api.test.cjs
```

Unit tests dùng fixtures riêng, không gọi model và không sửa CSV/database. Kiểm tra mapping/None, denominator, giới hạn input, model lỗi, CORS, phân đoạn, message validation, chunking và quyền.

E2E tùy chọn, cần API/model thật đang chạy:

```powershell
cd absa-extension
npm install
npx playwright install chromium
npm run test:e2e
npm run test:api-browser
```

Chromium dùng profile thử riêng, không cài vào Chrome cá nhân và không truy cập Shopee thật. Kiểm tra service worker MV3, popup, nhập tay, sidebar, tổng hợp/chi tiết/lọc, XSS, mất kết nối và khởi động lại browser để nạp source/worker mới. Nút Reload trong giao diện Chrome cá nhân chưa được tự động thao tác; cách dùng thủ công là Reload extension rồi tải lại tab như hướng dẫn Debug. Ảnh/báo cáo thử ở `.cache/`, không vào ZIP.

Bài thử `test:api-browser` chặn toàn bộ request URL Shopee và trả dữ liệu mô phỏng, không gửi request tới Shopee thật. Nó dùng cookie HttpOnly giả trong profile thử để xác nhận trình duyệt gửi cookie cùng origin và phần trả về chỉ chứa comment. Bản sao extension trong bài thử có quyền host Shopee **chỉ để thay thao tác bấm toolbar `activeTab` trong headless**; manifest phát hành vẫn chỉ có `activeTab`, `scripting` và host API loopback. Bước phân tích comment sau đó gọi FastAPI/model thật; kiểm tra nhập số tùy chọn, HTTP 403 dừng không thử lại, đổi sản phẩm, đóng/mở lại, tên trang cập nhật chậm, phản hồi API/model cũ bị loại khi chuyển sản phẩm.

Bản cập nhật Docker/Cloudflare đã đạt **15 unit tests Python, 25 unit tests JavaScript**; **8 kiểm tra E2E qua HTTPS được chạy lại với model CUDA trong Docker**, gồm kết quả hiển thị, lọc review và kết nối lại. Khóa API và host/CORS được kiểm tra khi triển khai. Mẫu 20 câu qua HTTPS khớp nhãn CPU, mất khoảng 0,6–0,8 giây ở lượt CUDA sau khi khởi động (xem [đo tốc độ và tối ưu tài nguyên](./DOCKER.md)). Luồng chia lô trước đó đã đạt 15 kiểm tra trong browser, trong đó 1.000 câu dùng 20 trang Shopee mô phỏng và 50 request tới model CUDA thật, kiểm tra thứ tự và chỉ một request phân tích chạy tại mỗi thời điểm. Bài thử đầy đủ 1.000 câu có thể mất nhiều phút trên CPU. Các kiểm thử API Shopee mô phỏng không được coi là bằng chứng truy cập endpoint Shopee thật thành công.

## Đóng gói

```powershell
& .\.venv-bertopic\Scripts\python.exe .\absa-extension\build_zip.py
```

ZIP ở `dist/shopee-review-analyzer.zip`, chỉ gồm source, tài liệu, dữ liệu ví dụ; không checkpoint, `.env`, venv/cache. Giải nén trước khi Load unpacked và chọn `extension/`.

Tài liệu chính thức: [Chrome network requests](https://developer.chrome.com/docs/extensions/develop/concepts/network-requests), [chrome.scripting](https://developer.chrome.com/docs/extensions/reference/api/scripting), [FastAPI lifespan](https://fastapi.tiangolo.com/advanced/events/), [Playwright extension tests](https://playwright.dev/docs/chrome-extensions).
