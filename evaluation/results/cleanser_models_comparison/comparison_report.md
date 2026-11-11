# Đánh giá các model ABSA sữa rửa mặt

Đã đánh giá **12 checkpoint** BiLSTM, MultiTask, BiGRU và BiRNN với 3 kịch bản baseline, class_weights, focal_loss trên cùng **1.904 câu test**. Đã bổ sung 3 kết quả RNN mới vào các bảng và tính lại thứ hạng trên đủ 12 bản.

Kết quả lấy từ hàm `evaluate_full_absa` và các lớp model tương ứng trong [evaluation_2_cleanser.py](../../evaluation_2_cleanser.py), giữ nguyên logic đánh giá. Checkpoint được nạp với `strict=True` để bảo đảm mọi tên và kích thước trọng số khớp lớp model. Không huấn luyện lại.

RNN trong báo cáo là **BiRNN**, dùng lớp `PhoBERT_BiRNN` có RNN hai chiều với nonlinearity `tanh`. Chín kết quả trước được giữ nguyên sau khi kiểm tra SHA-256 checkpoint; chỉ chạy thêm ba checkpoint BiRNN mới.

Tokenizer: `vinai/phobert-base`; MAX_LEN=128; batch size=32; GPU: NVIDIA GeForce RTX 3060 Ti; PyTorch: 2.15.0+cu132. Dùng float32 theo code gốc, không thêm mixed precision.

## 1. Macro-F1 của từng model và kịch bản

Các số trong bảng là **%**. ACD là phát hiện aspect; SPC là phân loại sentiment; ACD+SPC là dự đoán đúng cặp aspect–sentiment. Xếp hạng tốt nhất/tệ nhất theo **Macro-F1 ACD+SPC**.

| Model | Kịch bản | Macro-F1 ACD | Macro-F1 SPC | Macro-F1 ACD+SPC |
|---|---|---:|---:|---:|
| PhoBERT + BiLSTM | baseline | 95,77 | 82,43 | **63,62** (thấp nhất) |
| PhoBERT + BiLSTM | class_weights | 95,64 | 82,06 | 63,73 |
| PhoBERT + BiLSTM | focal_loss | 95,42 | 83,78 | 68,45 |
| PhoBERT MultiTask | baseline | 95,83 | 84,12 | 67,75 |
| PhoBERT MultiTask | class_weights | 95,89 | 84,39 | 69,39 |
| PhoBERT MultiTask | focal_loss | 95,43 | 83,22 | 69,82 |
| PhoBERT + BiGRU | baseline | 95,89 | 82,86 | 65,14 |
| PhoBERT + BiGRU | class_weights | 95,90 | 83,31 | 65,19 |
| PhoBERT + BiGRU | focal_loss | 95,83 | 84,07 | 68,53 |
| PhoBERT + BiRNN | baseline | 95,91 | 83,15 | 64,40 |
| PhoBERT + BiRNN | class_weights | 95,89 | 83,38 | 65,83 |
| PhoBERT + BiRNN | focal_loss | 95,74 | 84,62 | **70,01** (cao nhất) |

## 2. Bảng xếp hạng theo Macro-F1 ACD+SPC

| Hạng | Model | Kịch bản | Macro-F1 ACD+SPC (%) |
|---:|---|---|---:|
| 1 | PhoBERT + BiRNN | focal_loss | 70,01 |
| 2 | PhoBERT MultiTask | focal_loss | 69,82 |
| 3 | PhoBERT MultiTask | class_weights | 69,39 |
| 4 | PhoBERT + BiGRU | focal_loss | 68,53 |
| 5 | PhoBERT + BiLSTM | focal_loss | 68,45 |
| 6 | PhoBERT MultiTask | baseline | 67,75 |
| 7 | PhoBERT + BiRNN | class_weights | 65,83 |
| 8 | PhoBERT + BiGRU | class_weights | 65,19 |
| 9 | PhoBERT + BiGRU | baseline | 65,14 |
| 10 | PhoBERT + BiRNN | baseline | 64,40 |
| 11 | PhoBERT + BiLSTM | class_weights | 63,73 |
| 12 | PhoBERT + BiLSTM | baseline | 63,62 |

## 3. So sánh bản tốt nhất và tệ nhất theo Table 5

- **Bản tốt nhất (B): PhoBERT + BiRNN + focal_loss**, Macro-F1 ACD+SPC **70,01%**.
- **Bản tệ nhất (W): PhoBERT + BiLSTM + baseline**, Macro-F1 ACD+SPC **63,62%**.
- Chênh lệch tổng thể: **+6,39 điểm phần trăm**.

### F1 ACD+SPC theo aspect

Đây là cột F1-score thuộc nhóm **Sentiment Detect (ACD+SPC)** trong Table 5. Chênh lệch = B − W, tính từ số chưa làm tròn; giá trị âm nghĩa là B thấp hơn W ở riêng aspect đó.

| Aspect | B: BiRNN + focal_loss (%) | W: BiLSTM + baseline (%) | Chênh lệch (điểm %) | Support |
|---|---:|---:|---:|---:|
| CleansingEffectiveness | 68,48 | 62,29 | +6,20 | 603 |
| HydrationSkinComfort | 68,43 | 57,99 | +10,44 | 534 |
| SkinCompatibilitySafety | 80,51 | 81,24 | -0,73 | 711 |
| TextureFoaming | 74,91 | 67,15 | +7,76 | 299 |
| Fragrance | 88,51 | 86,00 | +2,51 | 439 |
| IngredientsFormulation | 56,67 | 37,54 | +19,14 | 135 |
| Packaging | 75,44 | 60,48 | +14,96 | 644 |
| PriceValue | 58,68 | 49,81 | +8,87 | 317 |
| ShopService | 56,96 | 57,68 | -0,72 | 427 |
| Delivery | 58,84 | 58,17 | +0,67 | 490 |
| General | 82,71 | 81,49 | +1,23 | 1.143 |
| **Macro Avg** | **70,01** | **63,62** | **+6,39** | **5.742** |

### F1 phát hiện aspect (ACD) theo Table 5

| Aspect | B: BiRNN + focal_loss (%) | W: BiLSTM + baseline (%) | Chênh lệch (điểm %) |
|---|---:|---:|---:|
| CleansingEffectiveness | 95,46 | 95,75 | -0,30 |
| HydrationSkinComfort | 97,08 | 97,34 | -0,26 |
| SkinCompatibilitySafety | 95,85 | 95,81 | +0,04 |
| TextureFoaming | 96,80 | 96,34 | +0,46 |
| Fragrance | 99,48 | 99,26 | +0,22 |
| IngredientsFormulation | 92,43 | 93,12 | -0,69 |
| Packaging | 97,41 | 97,41 | 0,00 |
| PriceValue | 96,10 | 95,88 | +0,21 |
| ShopService | 92,82 | 93,42 | -0,60 |
| Delivery | 98,49 | 98,29 | +0,20 |
| General | 91,20 | 90,86 | +0,35 |
| **Macro Avg** | **95,74** | **95,77** | **-0,03** |

## 4. Các kết quả BiRNN mới bổ sung

| Kịch bản BiRNN | Macro-F1 ACD (%) | Macro-F1 SPC (%) | Macro-F1 ACD+SPC (%) |
|---|---:|---:|---:|
| baseline | 95,91 | 83,15 | 64,40 |
| class_weights | 95,89 | 83,38 | 65,83 |
| focal_loss | 95,74 | 84,62 | 70,01 |

Bản dẫn đầu trước khi bổ sung RNN là PhoBERT MultiTask + focal_loss (69,82%). Bản dẫn đầu hiện tại là PhoBERT + BiRNN + focal_loss (70,01%), chênh lệch +0,20 điểm phần trăm (tính từ giá trị chưa làm tròn).

## 5. Nhận xét

- PhoBERT + BiRNN + focal_loss dẫn đầu theo Macro-F1 ACD+SPC; hơn PhoBERT + BiLSTM + baseline 6,39 điểm phần trăm.
- Bản tốt nhất tổng thể cao hơn ở 9/11 aspect, thấp hơn ở 2/11 aspect so với bản thấp nhất. Điểm tổng thể không đồng nghĩa tốt hơn ở mọi aspect.
- Chênh lệch dương lớn nhất về joint F1: IngredientsFormulation +19,14; Packaging +14,96; HydrationSkinComfort +10,44; PriceValue +8,87 điểm phần trăm.
- Các aspect mà bản tốt nhất tổng thể thấp hơn: SkinCompatibilitySafety (-0,73); ShopService (-0,72) điểm phần trăm.
- Macro-F1 ACD cao nhất: PhoBERT + BiRNN + baseline (95,91%). Macro-F1 SPC cao nhất: PhoBERT + BiRNN + focal_loss (84,62%).
- Trong lần đánh giá này, focal_loss có Macro-F1 ACD+SPC cao nhất trong cả bốn kiến trúc. Đây là điểm test của các checkpoint được cung cấp.

## 6. Output đánh giá chi tiết của từng checkpoint

Mỗi tệp bên dưới chứa output gốc gồm bảng metric tổng hợp (micro/macro/weighted/accuracy), Table 5 theo 11 aspect và bảng chi tiết 33 cặp aspect–sentiment; kèm checkpoint, lớp model và SHA-256 nguồn. [Tải báo cáo và toàn bộ 12 output](evaluation_outputs.zip).

| Model | Kịch bản | Output chi tiết | Lớp model |
|---|---|---|---|
| PhoBERT + BiLSTM | baseline | [bilstm_baseline.txt](outputs/bilstm_baseline.txt) | `PhoBERT_BiLSTM` |
| PhoBERT + BiLSTM | class_weights | [bilstm_class_weights.txt](outputs/bilstm_class_weights.txt) | `PhoBERT_BiLSTM` |
| PhoBERT + BiLSTM | focal_loss | [bilstm_focal_loss.txt](outputs/bilstm_focal_loss.txt) | `PhoBERT_BiLSTM` |
| PhoBERT MultiTask | baseline | [multitask_baseline.txt](outputs/multitask_baseline.txt) | `PhoBERT_MultiTask` |
| PhoBERT MultiTask | class_weights | [multitask_class_weights.txt](outputs/multitask_class_weights.txt) | `PhoBERT_MultiTask` |
| PhoBERT MultiTask | focal_loss | [multitask_focal_loss.txt](outputs/multitask_focal_loss.txt) | `PhoBERT_MultiTask` |
| PhoBERT + BiGRU | baseline | [bigru_baseline.txt](outputs/bigru_baseline.txt) | `PhoBERT_BiGRU` |
| PhoBERT + BiGRU | class_weights | [bigru_class_weights.txt](outputs/bigru_class_weights.txt) | `PhoBERT_BiGRU` |
| PhoBERT + BiGRU | focal_loss | [bigru_focal_loss.txt](outputs/bigru_focal_loss.txt) | `PhoBERT_BiGRU` |
| PhoBERT + BiRNN | baseline | [birnn_baseline.txt](outputs/birnn_baseline.txt) | `PhoBERT_BiRNN` |
| PhoBERT + BiRNN | class_weights | [birnn_class_weights.txt](outputs/birnn_class_weights.txt) | `PhoBERT_BiRNN` |
| PhoBERT + BiRNN | focal_loss | [birnn_focal_loss.txt](outputs/birnn_focal_loss.txt) | `PhoBERT_BiRNN` |

## 7. Dữ liệu và cách tính giữ nguyên

- Test: [test_cleanser.csv](../../../train-models/data/test_cleanser.csv). Mỗi dòng được tính một lần, không nhân theo duplicate_count.
- Checkpoint: [thư mục Models/Cleanser](../../../train-models/Models/Cleanser). Chỉ đọc trọng số, không ghi đè checkpoint.
- Nhãn gồm 11 aspect và positive/negative/neutral; `None` là trạng thái nội bộ cho aspect không được đề cập.
- Nếu một aspect có nhiều sentiment trong một câu, dataset vẫn lấy sentiment cuối như code gốc. Kết quả ở đây phản ánh đúng quy tắc đó.
- Macro-F1 ACD là trung bình macro ACD của từng aspect theo evaluator. Macro-F1 SPC được tính trên các vị trí có aspect trong gold. Macro-F1 ACD+SPC là trung bình F1 của 33 cặp.
- Kiểm tra: cả 12 bản đều đánh giá 1.904 dòng; support theo aspect khớp nhau; các bảng trong báo cáo được đối chiếu với output in ra; mã đánh giá và CSV test giữ nguyên SHA-256.
- SHA-256 evaluator: `c1b19fb9ab7ffbf10e54df395f5366cc32a12ed060db7955defcdbe6da5e80c1`.
- SHA-256 test: `c73c37713d51beacacf90a2b06c1fa8ec2bd818b01cdda51b2fd0c965b26abca`.
- Phiên bilstm, multitask, bigru: bắt đầu 2026-11-09T15:27:10.575805+00:00; hoàn tất 2026-11-09T15:28:27.946925+00:00 (UTC).
- Phiên birnn: bắt đầu 2026-11-10T12:22:13.302952+00:00; hoàn tất 2026-11-10T12:22:39.252024+00:00 (UTC).

