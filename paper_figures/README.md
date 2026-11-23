# Hình và bảng dùng trong nghiên cứu

Bộ này giữ 10 hình đã chọn và 13 bảng theo ảnh tham chiếu. Mỗi mã vẽ tạo một
hình hoàn chỉnh, gồm các panel của hình đó. Chạy mã không huấn luyện lại mô hình.

## Chạy

Từ thư mục gốc dự án:

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 paper_figures/build_all.py
```

Có thể chạy trực tiếp từng tệp trong [scripts/](scripts/) bằng PyCharm, từ bất kỳ
working directory nào. Hai cờ `--tables-only` và `--figures-only` giới hạn đầu ra.
Kiểm tra số liệu và nguồn sau khi chạy:

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 paper_figures/verify_results.py
```

- [PNG 300 DPI](output/png/) và [PDF vector](output/pdf/): đúng 10 hình.
- [CSV trình bày](tables/): đúng 13 bảng; số dùng dấu phẩy thập phân.
- [Nguồn và SHA-256](data/manifest.json): đường dẫn đầu vào chuẩn trong dự án.
- [Các dòng tham khảo](data/literature_reference_rows.json): chép từ ảnh người dùng,
  chưa được đối chiếu độc lập với hai bài báo gốc.

## Nguồn chuẩn

Các hàm ở [common/data_utils.py](common/data_utils.py) đọc trực tiếp dữ liệu trong
[crawl-data](../crawl-data/data/), [kết quả đánh giá ABSA](../evaluation/results/cleanser_models_comparison/outputs/),
[Business](../Business/) và [topic-modeling](../topic-modeling/data/). Không duy trì
thêm hai lớp CSV source/processed trùng nhau. Các bảng định dạng là sản phẩm xuất,
không được dùng làm đầu vào tính toán.

Giữ nguyên 12 [ảnh loss](../train-models/loss_curve/cleanser/) và 12
[lịch sử huấn luyện](../train-models/Histories/Cleanser/). Hình 4 chỉ đọc lịch sử BiRNN–Focal.

Giữ nguyên hai bộ LDA: [K=12](../topic-modeling/data/lda_cleanser_bertopic_corpus/)
(C_v=0,6664) và [K=16](../topic-modeling/data/lda_cleanser_bertopic_corpus_k16/)
(C_v=0,6448), kể cả mô hình, CSV và PDF/PNG. Hai bộ này dùng cùng bản lịch sử
12.741 nội dung với [BERTopic 16 topic](../topic-modeling/data/bertopic_16_topics_reproduced/).
Tập ABSA cuối có 12.690 nội dung; hai phiên bản không được coi là cùng một tập.

Mã tính lại số liệu Business nằm tại [ipa_shap_cleanser.py](../Business/ipa_shap_cleanser.py),
[IPA theo sàn](../Business/ipa_by_platform_cleanser.py),
[IPA theo thương hiệu](../Business/ipa_top5_brands_cleanser.py) và
[so sánh ngưỡng](../Business/IPA-M_mean_median.py). Chúng chỉ xuất số liệu chuẩn;
mọi hình dùng trong bài được vẽ trong thư mục này.

## Quy tắc

- Thống kê: 16.292 review có nội dung và nhãn; một review tính một lần.
- ABSA: 11 khía cạnh, gồm A11; Weighted F1 và Accuracy lấy từ TXT đánh giá gốc.
- SHAP/IPA: 10 khía cạnh cụ thể, không dùng A11 làm biến mô hình.
- Importance = mean(|SHAP_Pos| + |SHAP_Neg|), chuẩn hóa trên 10 khía cạnh.
- Performance loại trạng thái không đề cập và nhiều sentiment cùng khía cạnh.
- PRED(25) trong bảng hồi quy là tỷ lệ 0–1, không phải số phần trăm 0–100.
- Mean/median chỉ đổi ngưỡng IPA; giữ nguyên tọa độ Importance/Performance.

[Danh sách dọn file](cleanup_manifest.json) ghi đường dẫn, lý do và SHA-256 của
các file được loại. [Dấu kiểm tra nguồn được giữ](data/protected_sources.json)
giúp xác nhận dữ liệu, lịch sử, ảnh loss và hai bộ LDA không đổi sau khi dọn.
