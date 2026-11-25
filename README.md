# ABSA-T7

Mã nguồn nghiên cứu phân tích cảm xúc theo khía cạnh (ABSA) cho đánh giá sữa rửa mặt tiếng Việt từ Shopee, Lazada và Tiki; kết hợp khám phá chủ đề, huấn luyện mô hình và phân tích mức độ hài lòng của khách hàng.

## Cấu trúc thư mục

| Thư mục | Mục đích |
| --- | --- |
| [crawl-data/](./crawl-data/) | Thu thập đánh giá từ các sàn; lưu dữ liệu gốc, dữ liệu tổng hợp và thống kê mô tả. |
| [reprocessing/](./reprocessing/) | Làm sạch, chuẩn hóa văn bản, rà soát ngôn ngữ, loại phản hồi của người bán, tạo tập nội dung duy nhất và đồng bộ nhãn. |
| [topic-modeling/](./topic-modeling/) | Khám phá chủ đề bằng BERTopic và LDA để hỗ trợ xác định các khía cạnh; lưu kết quả và từ khóa chủ đề. |
| [train-models/](./train-models/) | Chia dữ liệu train/validation/test, huấn luyện các mô hình ABSA dựa trên PhoBERT; lưu checkpoint, lịch sử và đường cong loss. |
| [evaluation/](./evaluation/) | Đánh giá, so sánh các mô hình ABSA trên tập test và lưu báo cáo kết quả. |
| [Business/](./Business/) | Phân tích mức độ hài lòng bằng hồi quy, SHAP và IPA; so sánh theo sàn, thương hiệu và ngưỡng mean/median. |
| [absa-extension/](./absa-extension/) | Tiện ích Chrome phân tích đánh giá Shopee hoặc nội dung nhập tay |
| [paper_figures/](./paper_figures/) | Tạo và kiểm tra hình, bảng dùng trong bài nghiên cứu từ dữ liệu và kết quả đã có. |
| [research_results/](./research_results/) | Giữ lệnh xuất kết quả cũ, chuyển tiếp sang bộ tạo hình và bảng trong `paper_figures/`. |
| [tests/](./tests/) | Kiểm thử các bước xử lý và đầu ra của mô hình chủ đề. |
