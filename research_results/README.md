# Kết quả nghiên cứu sữa rửa mặt - bản nháp

Bộ kết quả phục vụ kiểm tra sáu phần: tiền xử lý, tổng quan dữ liệu, aspect/sentiment,
mô hình ABSA, rating/SHAP và IPA. Giữ nguyên dữ liệu, nhãn, checkpoint, evaluator và
các kết quả Business đã có.

## Xem kết quả

- [PDF bản nháp](output/pdf/cleanser_results_draft.pdf).
- [Các biểu đồ](figures/): PNG 300 dpi và PDF cho từng hình, gồm hình riêng cho từng sàn/thương hiệu.
- [Các bảng CSV](tables/): số liệu đầy đủ, không chỉ các cột rút gọn trong PDF.
- [Nguồn và kiểm tra](provenance.json): SHA-256 đầu vào, quy tắc đếm và phạm vi thí nghiệm.

## Tạo lại

Từ thư mục gốc dự án, chạy:

```powershell
.venv-bertopic/Scripts/python.exe -B -X utf8 research_results/build_results.py
```

Script dùng tokenizer và trọng số PhoBERT đã có trong cache cục bộ, chạy suy luận
hai checkpoint BiRNN để thu lỗi thực tế và bootstrap ghép cặp, không huấn luyện lại
ABSA. Kết quả suy luận được cache trong `.cache/`, có kiểm tra SHA của test,
checkpoint và evaluator trước khi tái sử dụng. Không gọi API bên ngoài.

Tạo PDF bằng môi trường có `reportlab`, `pandas`, `Pillow`:

```powershell
python -B -X utf8 research_results/render_report.py
```

Trong phiên tạo ban đầu, môi trường Python đi kèm Codex có đủ các thư viện PDF và
được dùng cho bước này; môi trường `.venv-bertopic` có các thư viện mô hình và biểu đồ.
Script PDF hiện dùng Arial trong `C:/Windows/Fonts/` để hiển thị tiếng Việt.

## Quy tắc đọc kết quả

- Tổng quan và sentiment: mỗi review nguồn hợp lệ tính một lần, không nhân duplicate_count.
- ABSA: dùng tập text duy nhất và split hiện có. Giữ 11 aspect, gồm General.
- Thống kê cặp nhãn giữ cả nhiều sentiment trong cùng aspect; mô hình hiện giữ sentiment cuối.
- Negative rate mô tả: review có negative / review đề cập aspect, gồm cả review có nhiều sentiment.
- SHAP/IPA: 10 aspect, bỏ General khỏi đầu vào; giữ các review chỉ có General.
- SHAP interaction trong bộ này được tính lại bằng cùng tiền xử lý và XGBoost của IPA,
  tránh khác biệt cách đọc price/discount của bộ KANO cũ. Các đầu ra cũ không bị sửa.
- Importance tính mean trị tuyệt đối SHAP trên toàn bộ review. Performance dùng sentiment
  tham chiếu, loại các vị trí không đề cập hoặc có nhiều sentiment trong cùng aspect.
- Bootstrap 1.000 lần chỉ phản ánh lấy mẫu tập test với checkpoint cố định, không thay thế nhiều seed huấn luyện.
- Các mô hình hồi quy theo nhóm được fit riêng; importance và quadrant là tương đối trong nhóm.
- Ví dụ lỗi được chọn có chủ đích để minh họa; chưa kết luận nhãn tham chiếu sai.

PDF là bản nháp kiểm tra, không phải bản thảo nghiên cứu cuối cùng. Các kiểm chứng
chưa thực hiện được ghi rõ trong PDF, không thay bằng kết quả giả định.
