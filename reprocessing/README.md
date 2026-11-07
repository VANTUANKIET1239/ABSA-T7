# Tiền xử lý và rà soát ngôn ngữ

Chạy các lệnh từ thư mục gốc dự án. Công cụ chính là
[preprocess_cleanser_reviews.py](preprocess_cleanser_reviews.py).

## Chuẩn hóa văn bản

Chế độ thêm `cleaned_comment` giữ nguyên mọi dòng và metadata:

```powershell
.venv-preprocessing/Scripts/python.exe -X utf8 reprocessing/preprocess_cleanser_reviews.py --input input.csv --append-text-only
```

Chế độ tiền xử lý đầy đủ dùng `--input input.csv --output processed.csv`.
Tùy chọn `--filter-vietnamese` chỉ là bộ lọc đơn giản, mặc định tắt;
không thay thế quy trình duyệt ngôn ngữ bên dưới. Chế độ append không lọc dòng.

## Rà soát ngôn ngữ

```powershell
.venv-preprocessing/Scripts/python.exe -X utf8 reprocessing/preprocess_cleanser_reviews.py --input input.csv --audit-languages --output-dir language_audit
```

Lệnh chỉ đọc CSV nguồn, xuất `candidates.json` và `summary.json`.
Các tín hiệu từ vựng chỉ chọn ứng viên; không tự kết luận ngôn ngữ hay tự xóa.
Nhãn trường tiếng Việt và tiêu đề đánh giá chung được bỏ qua khi tìm tín hiệu.
Đây không phải mô hình nhận diện mọi ngôn ngữ.

Đọc từng ứng viên rồi tạo `decisions.json` theo mẫu:

```json
{
  "source_sha256": "chép source_sha256 từ summary.json",
  "decisions": {
    "2": {"status": "english", "reason": "Nội dung hoàn toàn tiếng Anh"},
    "3": {"status": "retained", "reason": "Tiếng Việt không dấu"},
    "4": {"status": "mixed", "reason": "Có cả tiếng Việt và tiếng Anh"}
  }
}
```

Khóa quyết định là `source_record` trong ứng viên: bản ghi CSV tính cả header,
review đầu tiên là 2; không phải số dòng vật lý nếu comment có xuống dòng.
Phải có quyết định cho tất cả ứng viên và chỉ các ứng viên của đúng nguồn đó.
Trạng thái hợp lệ: `retained`, `english`, `english_short`, `mixed`,
`english_other_language`, `uncertain`.

```powershell
.venv-preprocessing/Scripts/python.exe -X utf8 reprocessing/preprocess_cleanser_reviews.py --input input.csv --audit-languages --output-dir language_audit --language-decisions decisions.json
```

Lệnh xác minh SHA-256 và xuất các danh sách CSV riêng; vẫn không sửa nguồn.
`english_reviews.csv` gồm `english` và `english_short`. Các nhóm còn lại nằm
trong các báo cáo riêng để người dùng quyết định có loại hay không.

## Loại các dòng đã duyệt

Ví dụ chỉ loại những dòng trong báo cáo tiếng Anh:

```powershell
.venv-preprocessing/Scripts/python.exe -X utf8 reprocessing/preprocess_cleanser_reviews.py --input input.csv --remove-reviewed language_audit/english_reviews.csv --output filtered.csv --expected-sha256 SHA256_TU_SUMMARY
```

Giữ nguyên các cột, giá trị và thứ tự của các dòng còn lại. Công cụ đối chiếu
vị trí cùng toàn bộ giá trị của mỗi dòng cần loại. `--output` không được tồn tại.
Nếu bỏ `--output`, công cụ cập nhật nguồn sau khi tạo bản sao lưu đã xác minh.
Không đưa file crawl gốc vào thao tác cập nhật trực tiếp nếu muốn bảo toàn nguồn.

Chức năng loại dòng cũng dùng được cho danh sách ngoài chủ đề đã duyệt:
truyền `--record-column filter_source_record --removal-label relevance`.
Các công cụ audit thiếu dấu và sàng lọc ngoài chủ đề vẫn được giữ riêng.

Nếu lọc một bộ dữ liệu đã có unique/mapping/nhãn, lệnh này chỉ cập nhật CSV
được chỉ định; không tự đồng bộ các bản dẫn xuất đó.

## Kiểm thử

```powershell
.venv-preprocessing/Scripts/python.exe -X utf8 -m unittest discover -s reprocessing -p "test_*.py"
```
