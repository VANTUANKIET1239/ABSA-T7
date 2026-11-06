# Chạy BERTopic

Đầu vào mặc định là [bộ text duy nhất](../crawl-data/data/unique_texts/unique_texts.csv).
Mặc định `--text-unit clause`: mỗi mệnh đề là một văn bản huấn luyện.

- Tách từ `representative_comment`, sau đó mới chuẩn hóa dấu và bỏ dấu câu.
  Có thể chỉ định cột khác bằng `--split-column`. Cùng một đoạn được dùng tạo
  embeddings và keyword, không ghép đoạn embedding với keyword của cả review.
  Các đoạn được chuẩn hóa teencode bằng cùng quy tắc tiền xử lý hiện có
  (ví dụ `ko` → `không`, `shop` → `cửa hàng`), sau đó ghép các cụm từ theo từ điển.
  Chế độ câu/mệnh đề không chạy lại bộ tách từ VnCoreNLP/Java.
- Tách ở dấu kết thúc câu, xuống dòng, nhãn trường review, dấu chấm phẩy,
  dấu phẩy và từ chuyển ý như “nhưng”. Giữ số thập phân; giữ các phần sau dấu
  phẩy khi đoạn trước chứa phủ định để tránh làm mất phạm vi phủ định.
  Đây là quy tắc heuristic, không phải bộ phân tích cú pháp tiếng Việt.
- Chỉ giữ đoạn có ít nhất 2 từ và 6 ký tự, còn token ngoài stopwords/phủ định đơn lẻ.
  Điều chỉnh số từ bằng `--min-unit-words`. Mapping ghi cả đoạn bị loại cùng lý do.
- Các đoạn có cùng text đã chuẩn hóa chỉ học một lần, có `unit_id` ổn định.
  Mapping giữ tất cả lần xuất hiện trong review, kể cả đoạn lặp trong cùng review.
  `duplicate_count` luôn chỉ là metadata, không dùng để nhân bản hay làm trọng số.
- Kết quả mới mặc định tại `topic-modeling/data/bertopic_clause_texts`, cạnh script.
  `--text-unit sentence` chỉ tách câu/nhãn trường và xuất tại `bertopic_sentence_texts`.
  `--text-unit review` dùng cách chạy toàn review cũ, xuất tại `bertopic_unique_texts`;
  khi đó `--embedding-column` và `--topic-column` mới có tác dụng.
- Nếu nguồn đã mất dấu câu, không thể khôi phục chắc chắn ranh giới câu.
  Nên dùng cột gốc còn dấu câu thay vì `cleaned_comment`.

## Chỉ số và cách chọn cấu hình

Mỗi trial ghi C_v, số topic và tỷ lệ outlier, cùng các chỉ số sau. Keyword là tối đa
10 từ/cụm có trọng số dương, bỏ chuỗi rỗng; các chỉ số keyword không tính Topic -1.

| Chỉ số | Cách tính | Hướng đọc |
| --- | --- | --- |
| `largest_topic_share` | Số đoạn trong topic lớn nhất / mọi đoạn huấn luyện | Phát hiện một cụm chiếm phần lớn corpus |
| `largest_topic_share_assigned` | Số đoạn trong topic lớn nhất / các đoạn khác -1 | Mẫu số dùng trong điểm tổng hợp |
| `topic_diversity` | Số keyword khác nhau / tổng số keyword của các topic | Cao hơn nghĩa là ít keyword lặp giữa topic |
| `keyword_overlap_mean`, `keyword_overlap_max` | Trung bình và lớn nhất của Jaccard keyword giữa từng cặp topic | Thấp hơn nghĩa là ít trùng keyword chính xác |
| `valid_topic_ratio` | Số topic có ít nhất 2 keyword / mọi topic khác -1 | Phát hiện topic không có đủ biểu diễn |

Mặc định `--selection-metric balanced` chọn điểm lớn nhất theo công thức:

```text
max(C_v, 0) * topic_diversity * (1 - keyword_overlap_mean)
    * (1 - largest_topic_share_assigned) * (1 - outlier_rate) * valid_topic_ratio
```

Đây là điểm heuristic để xếp hạng, không phải chỉ số chuẩn và không đảm bảo topic
đã đúng aspect. Nó ưu tiên ít trùng keyword, hạn chế cụm quá lớn, nhiều outlier và
topic thiếu keyword. Nếu bằng điểm, chọn C_v cao hơn; nếu vẫn bằng, giữ trial trước.
Dùng `--selection-metric c_v` để chọn theo C_v như trước; các chỉ số khác vẫn được ghi.
Baseline hyperparameters được giữ nguyên trong `baseline_params.json`.

## Kết quả xuất

- `text_units.csv`: các đoạn duy nhất dùng huấn luyện, kèm text tạo embedding/keyword
  và metadata của review xuất hiện đầu tiên.
- `review_unit_mapping.csv`: mọi đoạn → `input_row`/`text_group_id`, lý do giữ/loại,
  và topic/probability sau khi huấn luyện. `input_row` tính từ 1, không tính header CSV.
- `text_units_with_topics.csv`: topic cho từng đoạn duy nhất. `Count` trong
  `topic_info.csv` đếm đoạn duy nhất; `Source_Review_Count` đếm review chứa đoạn của topic.
- `reviews_with_topics.csv`: mỗi review gốc một dòng, `Topics` là danh sách JSON
  topic từ các đoạn của nó, có thể chứa -1; `Topic_Count` không tính -1.
  Review không có đoạn hợp lệ vẫn giữ lại với `Topics=[]`.
- `aspect_candidates.csv`: keyword, gợi ý aspect, đoạn đại diện và review gốc.
- `grid_search_results.csv`, `best_params.json`, `evaluation_metrics.json`:
  chỉ số từng trial và cấu hình thắng. Điểm tổng hợp thắng có thể không có C_v cao nhất.
- `topic_keyword_overlap.csv`: Jaccard và keyword chung của từng cặp topic,
  hỗ trợ kiểm tra topic trùng. `training_summary.txt` tóm tắt kết quả.

Dùng Python 3.12 (64-bit) và môi trường riêng trong thư mục gốc dự án:

```powershell
python -m venv .venv-bertopic
.\.venv-bertopic\Scripts\python.exe -m pip install -r topic-modeling/requirements-bertopic.txt
```

## CUDA cho bước tạo embeddings

Cài thêm bản PyTorch CUDA vào đúng môi trường BERTopic:

```powershell
.\.venv-bertopic\Scripts\python.exe -m pip install --upgrade -r topic-modeling/requirements-bertopic-cuda.txt
```

File CUDA dùng PyTorch `2.15.0+cu132` từ kho chính thức, dành cho driver hỗ trợ
CUDA 13.2. Máy hiện tại có RTX 3060 Ti 8 GB, driver 610.88 hỗ trợ CUDA 13.3.
Các thư viện còn lại cài bằng file requirements cơ sở ở trên.

Script mặc định `--device auto`: chọn CUDA nếu PyTorch nhận được GPU, nếu không
chọn CPU. Log và `run_config.json` ghi thiết bị tạo embeddings. Dùng `--device cuda`
để yêu cầu GPU và báo lỗi nếu CUDA chưa khả dụng; `--device cpu` để chạy CPU.

```powershell
.\.venv-bertopic\Scripts\python.exe -X utf8 topic-modeling/bertopic_cleanser_aspects.py --device cuda
```

Giữ `batch_size=32` khi bắt đầu. CUDA áp dụng cho SBERT tạo embeddings;
UMAP, HDBSCAN và C_v trong pipeline hiện tại vẫn chạy trên CPU.
Trong PyCharm/VS Code cần chọn interpreter `.venv-bertopic/Scripts/python.exe`.

Đã kiểm tra trên máy này: PyTorch `2.15.0+cu132`, `cuda.is_available() == True`,
SBERT nằm trên `cuda:0`. Chạy thử 128 review từ `cleaned_comment` tạo ma trận
`(128, 768)` hợp lệ, vector đã chuẩn hóa; `pip check` không có dependency hỏng.

Chạy kiểm thử logic (không tải mô hình):

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 -m unittest discover -s topic-modeling -p test_bertopic_cleanser_aspects.py -v
```

Kiểm tra CSV:

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 topic-modeling/bertopic_cleanser_aspects.py --check-input
```

Chỉ chuẩn bị các đoạn và mapping trên toàn bộ CSV, chưa tải mô hình:

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 topic-modeling/bertopic_cleanser_aspects.py --prepare-only
```

Chạy một cấu hình gốc để kiểm tra toàn bộ pipeline, gồm C_v:

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 topic-modeling/bertopic_cleanser_aspects.py --single-run --output-dir topic-modeling/data/bertopic_clause_baseline
```

Chạy Grid Search mặc định 27 tổ hợp trên mệnh đề, chọn bằng nhiều tiêu chí:

```powershell
.\.venv-bertopic\Scripts\python.exe -B -X utf8 topic-modeling/bertopic_cleanser_aspects.py
```

Trong PyCharm, chọn interpreter `.venv-bertopic/Scripts/python.exe`. Trong VS Code,
các task `BERTopic: tests`, `BERTopic: baseline` và `BERTopic: grid search` dùng
interpreter này trực tiếp. Không dùng `.venv/Scripts/python.exe` cũ nếu nó báo
`No Python at ...Python310...`: môi trường đó đang trỏ tới Python đã bị gỡ.

Nếu báo thiếu thư viện, kiểm tra đúng interpreter và các dependency:

```powershell
.\.venv-bertopic\Scripts\python.exe -m pip check
```

SBERT cần tải mô hình `keepitreal/vietnamese-sbert` lần đầu nếu máy chưa có cache.
`--coherence-sample N` chỉ tính C_v trên mẫu cố định N review; điểm đó là ước tính,
không phải điểm trên toàn bộ corpus. `--skip-coherence` chỉ dùng với `--single-run`.

Tham khảo: [cài đặt BERTopic](https://github.com/MaartenGr/BERTopic),
[Gensim và các dependency](https://pypi.org/project/gensim/).

## Kiểm tra pipeline mệnh đề ngày 06/11/2026

- 34 kiểm thử logic thành công, gồm tách câu, bảo vệ phủ định, chuẩn hóa teencode,
  loại trùng đoạn, ánh xạ nhiều topic về review và chọn cấu hình theo nhiều tiêu chí.
- Toàn bộ 12.750 review tạo 46.053 đoạn; 5.074 đoạn ngắn/không có keyword bị loại.
  Còn 29.458 đoạn duy nhất để học, đại diện cho 40.979 lần xuất hiện trong 12.505 review.
  Xem [các đoạn huấn luyện](data/bertopic_clause_texts/text_units.csv) và
  [mapping đầy đủ](data/bertopic_clause_texts/review_unit_mapping.csv).
- Đã chạy SBERT → UMAP → HDBSCAN → c-TF-IDF → C_v và xuất kết quả trên mẫu
  1.000 review lấy từ CSV unique bằng `sample(n=1000, random_state=42)`:
  2.784 đoạn duy nhất, thử baseline và cấu hình `15/3/10`.
- Baseline thắng theo balanced: C_v **0,4169**, score **0,2797**, **28 topic**,
  outlier **21,59%**, topic lớn nhất **6,50%** mọi đoạn huấn luyện,
  diversity **0,9357**, overlap trung bình **0,0029**.
- Cấu hình `15/3/10` có C_v **0,4198**, **69 topic**, nhưng score **0,2566** nên
  không thắng. Ví dụ này xác nhận lựa chọn mới sử dụng nhiều tiêu chí.
- Đây là kết quả chạy thử trên mẫu, chưa chạy toàn bộ Grid Search trên 29.458 đoạn.
  Chưa thể so sánh trực tiếp với điểm mô hình toàn review do khác corpus và đơn vị học.

Xem [điểm hai cấu hình](data/bertopic_clause_smoke_test/grid_search_results.csv),
[đoạn/review đại diện](data/bertopic_clause_smoke_test/aspect_candidates.csv) và
[chỉ số mô hình thắng](data/bertopic_clause_smoke_test/evaluation_metrics.json).

## Kết quả kiểm tra môi trường ngày 05/11/2026

Các số liệu dưới đây thuộc pipeline toàn review trước khi bổ sung tách mệnh đề.

- `pip check`: không có dependency hỏng/xung đột.
- 11 kiểm thử logic: thành công.
- Đã chạy SBERT → UMAP → HDBSCAN → c-TF-IDF → C_v và xuất CSV trên mẫu
  1.000 review hợp lệ, lấy bằng `sample(n=1000, random_state=42)` từ CSV tổng thể
  `combined_reviews_brand_filled_language_filtered.csv` trước khi chuyển đầu vào sang bộ unique.
- Thử hai cấu hình: bộ gốc và bộ `min_cluster_size=15`, `min_samples=3`,
  `n_neighbors=10`; giữ `min_df=3`, `max_df=0.85`.
- Bộ gốc không có đủ keyword để tính C_v trên mẫu này, được ghi là
  `invalid_coherence`. Bộ còn lại đạt C_v **0.6685**, **5 topic**, outlier **1.2%**.
- Đây là kết quả kiểm tra trên mẫu; chưa chạy hết 27 cấu hình trên 16.337 review.

Xem [điểm từng cấu hình](data/library_smoke_test/grid_search_results.csv),
[tham số thắng](data/library_smoke_test/best_params.json),
[tóm tắt kết quả](data/library_smoke_test/training_summary.txt),
[phiên bản thư viện](data/library_smoke_test/environment.json).

Task `BERTopic: library smoke test` chạy lại trên mẫu đã lưu, sử dụng mô hình
SBERT trong cache của máy (`HF_HUB_OFFLINE=1`). Các task baseline/grid search
không ép chế độ offline, cho phép tải mô hình nếu chưa có cache.
