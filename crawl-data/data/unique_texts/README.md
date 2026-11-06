# Bộ text duy nhất

Nguồn: `combined_reviews_brand_filled_language_filtered.csv`, cột `cleaned_comment`.

- Review nguồn: 16,355.
- Text khác rỗng: 16,354.
- Nội dung duy nhất: 12,750.
- Bản lặp dư: 3,604.
- Text rỗng: 1, giữ trong bảng ánh xạ với trạng thái `empty_text`.

## Cách tạo nhóm

Chuẩn hóa Unicode NFC, chữ thường và khoảng trắng trên cột nguồn. Giữ nguyên dấu tiếng Việt, phủ định, chữ số và dấu `_`. Không gộp theo độ giống ngữ nghĩa. Không lọc câu ngắn/khen chung, không gán nhãn aspect/sentiment và không chia train/validation/test.

## Các file

- [unique_texts.csv](unique_texts.csv): mỗi nội dung có một dòng. `text_group_id` là SHA-256 của text chuẩn hóa, có tiền tố `txt_`. `duplicate_count` tính cả bản đại diện. `cleaned_comment` chứa text chuẩn hóa dùng làm khóa nhóm.
- `representative_source_row` là vị trí bản ghi dữ liệu đầu tiên của nhóm, tính từ 1, không tính header; không phải số dòng vật lý vì một review có thể chứa xuống dòng. `representative_comment` là ví dụ comment gốc của bản ghi đó.
- [review_text_mapping.csv](review_text_mapping.csv): giữ mọi bản ghi, thứ tự, giá trị nguồn và metadata; thêm `source_row`, `text_group_id`, `text_status`. Dùng `text_group_id` để nối kết quả trên bộ unique về toàn bộ review.
- [deduplication_summary.json](deduplication_summary.json): thống kê, quy tắc và SHA-256 để xác định chính xác CSV nguồn/các CSV đầu ra.

Một nhóm có thể chứa review từ nhiều thương hiệu, sàn hoặc rating. Metadata của một bản đại diện không được coi là metadata chung của nhóm; xem bảng ánh xạ khi phân tích theo các thuộc tính đó.

Tạo lại từ thư mục gốc dự án:

```powershell
python reprocessing/create_unique_text_dataset.py
```
