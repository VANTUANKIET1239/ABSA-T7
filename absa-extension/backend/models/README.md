# Tài nguyên model

Không cần sao chép trọng số vào đây. Mặc định API dùng checkpoint tốt nhất tại `train-models/Models/Cleanser/best_model_birnn_cleanser_focal_loss_VI.pt`, config/tokenizer `vinai/phobert-base` trong cache local và VnCoreNLP của repository.

Kiến trúc và forward pass đối chiếu với `TrainPhoBert_BiLSTM_Beauty.py`, class `PhoBERT_BiRNN`: encoder 768, RNN tanh hai chiều hidden 256, masked mean pooling, dropout 0,3, linear 512 → 44, reshape batch × 11 × 4. Toàn bộ encoder/head được nạp `strict=True`; không tải thêm base weights. Không dùng `AutoModelForSequenceClassification`.

Mapping theo thứ tự 0/1/2/3: Positive, Negative, Neutral, None. Confidence là softmax của lớp được chọn trong bốn lớp của mỗi khía cạnh, chưa được hiệu chuẩn. None không được tính là Neutral. Không hỗ trợ tự thay checkpoint của kiến trúc khác.
