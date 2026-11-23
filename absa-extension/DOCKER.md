# API ABSA qua Docker và Cloudflare

Luồng hiện tại: **extension → HTTPS Cloudflare Tunnel → FastAPI trong Docker → VnCoreNLP/PhoBERT–BiRNN**. API không phụ thuộc server Python chạy bằng PyCharm. Docker chạy trên máy của bạn; Cloudflare cung cấp đường truy cập HTTPS, không chuyển model sang máy chủ cloud.

Container dùng **CUDA trên GPU NVIDIA**, PyTorch 2.14.1 + CUDA 13.0, cùng checkpoint BiRNN Focal Loss và tiền xử lý của nghiên cứu. Một worker nạp model một lần. Extension vẫn hỗ trợ 1.000 review, lấy từng trang 50 review và phân tích tuần tự từng lô tối đa 20, mini-batch model 16. Java/VnCoreNLP vẫn xử lý trên CPU; đúng JAR và hai file wordsegmenter từ repository được đưa vào image Linux để tránh chi phí đọc qua Windows bind mount ở mỗi lô. Không đổi kiến trúc, trọng số hoặc cách tính dự đoán.

Kiểm thử trên RTX 3060 Ti qua cùng URL Cloudflare với cùng 20 câu: bản CPU/Windows bind mất **19,89 giây**; bản CUDA với tài nguyên VnCoreNLP trong image mất **1,188 giây ở lượt đầu**, rồi **0,593 và 0,782 giây**. Nhãn của 20 câu khớp bản CPU, checkpoint và mã tiền xử lý có cùng hash. Đo riêng tiền xử lý giảm từ 16,714 xuống 0,346 giây. Cải thiện tổng thể đến từ cả GPU và cách lưu tài nguyên trên Linux; các số này là phép thử cụ thể, không phải cam kết thời gian cho mọi review hay điều kiện mạng.

## Chuẩn bị và chạy

Mở Docker Desktop ở chế độ Linux containers, dùng WSL2 và driver NVIDIA hỗ trợ GPU trong WSL2. Cấu hình Compose cấp một GPU cho API và đặt `ABSA_DEVICE=cuda`; nếu GPU không sẵn sàng, API báo lỗi thay vì tự chuyển về CPU. Xem [Docker Desktop GPU](https://docs.docker.com/desktop/features/gpu/) và [Compose GPU reservations](https://docs.docker.com/compose/how-tos/gpu-support/).

Trên máy Windows này, Docker Desktop nhìn thấy GPU qua `nvidia-smi` nhưng thiếu thư viện driver khi gọi CUDA (`cuInit` lỗi 500). Compose vì vậy mount chỉ đọc `/usr/lib/wsl/lib` và `/usr/lib/wsl/drivers` từ host, đồng thời đặt `LD_LIBRARY_PATH=/usr/lib/wsl/lib`. Driver không được sao chép vào image hay ZIP. Cấu hình này dành cho WSL2; trên Linux thuần cần bỏ hai bind WSL và biến `LD_LIBRARY_PATH`, giữ GPU reservation. NVIDIA mô tả trường hợp thư viện WSL thiếu trong container tại [nvidia-container-toolkit #520](https://github.com/NVIDIA/nvidia-container-toolkit/issues/520).

Từ thư mục gốc ABSA-T7:

```powershell
& .\.venv-bertopic\Scripts\python.exe -X utf8 .\absa-extension\prepare_docker.py
docker compose -f absa-extension/compose.yaml up -d --build api
docker compose -f absa-extension/compose.yaml ps
```

Script chuẩn bị xuất tokenizer từ cache có sẵn và tạo khóa API trong `.docker/api_key.txt` nếu chưa có. Checkpoint/tokenizer được mount chỉ đọc; checkpoint không đóng vào image. Tài nguyên VnCoreNLP được COPY từ chính file local khi build, không tải lại. Nếu thay chúng, cần build lại image. Không sửa CSV hoặc database web.

Kiểm tra **http://127.0.0.1:8082/health**: `model_loaded` phải là `true` và `device` là `cuda`. Cổng 8082 độc lập với web 8080 và Python local 8000. Healthcheck container cũng yêu cầu CUDA để tránh báo healthy khi chạy nhầm CPU.

Kiểm tra GPU/PyTorch trong container:

```powershell
docker compose -f absa-extension/compose.yaml exec api nvidia-smi
docker compose -f absa-extension/compose.yaml exec api python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Lần build đầu cần tải thêm các thư viện CUDA. GPU dùng chung với những ứng dụng khác trên máy; nếu thiếu VRAM, hãy đóng tác vụ GPU không cần thiết hoặc giảm `ABSA_BATCH_SIZE` trong Compose. Không cần thay checkpoint. Sau khi sửa Compose, chạy lại `up -d --build api`.

Mở HTTPS:

```powershell
docker compose -f absa-extension/compose.yaml --profile tunnel up -d cloudflared
docker compose -f absa-extension/compose.yaml logs --tail 100 cloudflared
& .\.venv-bertopic\Scripts\python.exe -X utf8 .\absa-extension\configure_extension.py --from-tunnel
```

Lệnh cuối tự đọc URL tunnel, ghi cấu hình riêng cho extension và cập nhật đúng host permission/CSP của URL đó. Sau đó vào `chrome://extensions` → **Reload** extension và tải lại tab Shopee. Popup hiển thị địa chỉ API đang dùng bên dưới Connected.

Nếu đã có URL, có thể dùng:

```powershell
& .\.venv-bertopic\Scripts\python.exe -X utf8 .\absa-extension\configure_extension.py --url https://TEN-TUNNEL.trycloudflare.com
```

API cần header `X-API-Key` cho `/analyze`; extension tự gửi. `/health` không yêu cầu khóa. Khóa không được trả trong health, in ra log hoặc chuyển cho trang Shopee. `.docker/` và `extension/api_config.local.json` được Git ignore và loại khỏi ZIP source. Bản extension đang load trực tiếp từ source sử dụng file cấu hình này; ZIP source không chứa khóa, nên cần cấu hình riêng sau khi giải nén. Muốn cấu hình trên máy khác, truyền `--key-file` tới bản sao khóa do bạn cung cấp riêng.

## Chạy lại, đổi URL và dừng

```powershell
# Khởi động lại các container đã có
docker compose -f absa-extension/compose.yaml --profile tunnel up -d

# Sau khi tunnel đổi URL, cập nhật extension rồi Reload
& .\.venv-bertopic\Scripts\python.exe -X utf8 .\absa-extension\configure_extension.py --from-tunnel

# Xem log
docker compose -f absa-extension/compose.yaml logs --tail 60 api cloudflared

# Chỉ tắt truy cập công khai
docker compose -f absa-extension/compose.yaml stop cloudflared

# Dừng API và tunnel của extension
docker compose -f absa-extension/compose.yaml --profile tunnel down
```

Các lệnh này chỉ tác động project `absa-api`, không dừng web `cleanser-insight`. Máy, Docker Desktop và Internet phải tiếp tục hoạt động. Quick Tunnel là URL tạm, có thể đổi khi tạo lại tunnel, không có cam kết uptime; muốn địa chỉ cố định cần named tunnel và hostname riêng. Xem [tài liệu Cloudflare Quick Tunnels](https://developers.cloudflare.com/tunnel/get-started/quick-tunnels/).

Để kiểm tra trực tiếp container thay vì tunnel, chạy `configure_extension.py --url http://127.0.0.1:8082`, rồi Reload extension. Để quay về Python local không có khóa, bỏ file `extension/api_config.local.json` và Reload; mặc định sẽ gọi `http://127.0.0.1:8000`.
