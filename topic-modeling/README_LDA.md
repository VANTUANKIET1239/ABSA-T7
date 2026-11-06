# Tuning LDA theo hai bước

Mặc định dùng [unique_texts.csv](../crawl-data/data/unique_texts/unique_texts.csv),
cột `cleaned_comment`, giữ dấu `_` trong từ ghép. `duplicate_count` là metadata,
không dùng nhân bản hoặc làm trọng số. Các trial dùng chung corpus, stopwords và
từ vựng; không thay đổi bộ lọc trong cùng một Grid Search.

## Bước 1: chọn K

Chạy mọi K từ **1 đến 20**, bước 1, với các hyperparameters cố định:

| Tham số | Giá trị mặc định |
| --- | --- |
| `alpha` | `symmetric` |
| `eta` | `auto` |
| `passes` | 20 |
| `iterations` | 200 |

Chọn K có **C_v hữu hạn cao nhất** trong bước này. Dùng `--min-k`, `--max-k`,
`--k-step` để đổi dải K; `--k-alpha`, `--k-eta`, `--passes`, `--iterations` đổi
hyperparameters cố định. Không dùng `--k-alpha auto` khi thử K=1 vì Gensim có
thể sinh alpha không hữu hạn. K=1 vẫn được giữ, dù không phân tách aspect.

## Bước 2: tuning hyperparameters tại K đã chọn

Khóa K thắng bước 1, chỉ quét các hyperparameters sau tại K đó:

| Tham số | Grid mặc định |
| --- | --- |
| `alpha` | `symmetric`, `asymmetric`, `auto` |
| `eta` | `auto`, `symmetric` |
| `passes` | Giá trị `--passes` (20) |
| `iterations` | Giá trị `--iterations` (200) |

Bước 2 mặc định có 6 trial; nếu K=1 thì có 4 trial vì bỏ alpha=auto. Cấu hình
thắng bước 1 luôn được đưa vào bước 2 làm mốc so sánh; nếu không nằm trong grid
được truyền thì thêm một trial. Với mặc định có **20 + 6 = 26 trial**, hoặc
**20 + 4 = 24 trial** nếu K=1 thắng.

Chọn C_v hữu hạn cao nhất trong bước 2, lưu và xuất model đó. Khi bằng nhau,
giữ trial được thử trước. Perplexity và topic diversity chỉ để tham khảo.
Đây là tối ưu tuần tự: K phụ thuộc cấu hình cố định của bước 1, không đảm bảo
là tổ hợp tốt nhất nếu thử đồng thời mọi K với mọi hyperparameters.

Cấu hình gốc được giữ riêng: K=6, alpha=symmetric, eta=auto, passes=20,
iterations=200; dải K gốc trước các thay đổi là 6,8,...,20.

## Chạy

Chạy từ thư mục gốc dự án:

```powershell
.\.venv-bertopic\Scripts\python.exe -m pip install -r topic-modeling/requirements-lda.txt
.\.venv-bertopic\Scripts\python.exe topic-modeling/lda_cleanser_aspects.py --check-input
.\.venv-bertopic\Scripts\python.exe topic-modeling/lda_cleanser_aspects.py
```

Gensim LDA dùng **CPU**, không nhận `--device cuda`. Trong PyCharm chọn cùng
interpreter `.venv-bertopic/Scripts/python.exe` đã cài các dependency ở trên.

Để tuning thêm passes và iterations:

```powershell
.\.venv-bertopic\Scripts\python.exe topic-modeling/lda_cleanser_aspects.py --grid-passes 10 20 --grid-iterations 100 200
```

Lệnh này vẫn chỉ dùng passes=20, iterations=200 ở bước chọn K, rồi thử thêm
passes/iterations trong bước 2. Bước 2 có 24 trial (16 nếu K=1); cộng 20 trial
chọn K thành **44 trial**, hoặc **36 nếu K=1**. Có thể
giới hạn grid bằng `--grid-alpha symmetric auto --grid-eta auto`, hoặc truyền
prior dạng số dương như `--grid-alpha 0.1 0.5 --grid-eta 0.01 auto`.
`--passes`/`--iterations` cũng là giá trị dùng ở bước 2 nếu không truyền grid tương ứng.
`--skip-perplexity` bỏ chỉ số
phụ; `--skip-figures` không vẽ biểu đồ và không cần matplotlib.

Kết quả mặc định trong [data/lda_cleanser/](data/lda_cleanser/):

- `lda_k_coherence_results.csv`: mọi trial chọn K với các hyperparameters cố định.
- `k_selection.json`: K thắng, C_v và cấu hình của bước 1.
- `lda_hyperparameter_results.csv`: mọi trial tuning tại K đã chọn.
- `lda_grid_search_results.csv`: gộp hai bước; cột `Stage` phân biệt `k_search`
  và `hyperparameter_search`, `Trial` đánh số riêng trong mỗi bước. Cả bảng theo
  bước và bảng gộp được cập nhật sau từng trial, kể cả trial lỗi/C_v không hợp lệ.
- `best_params.json`: params, C_v và số trial thắng bước 2, kèm kết quả chọn K.
- `baseline_params.json`: cấu hình gốc, tách riêng với cấu hình thắng.
- `run_config.json`: input, cột text, interpreter, grid, tham số cố định và bộ lọc.
- `best_lda_model`, `lda_dictionary`: model thắng thực tế và dictionary tương ứng.
- Các CSV review-topic, keyword, aspect candidate và phân bố topic giữ nguyên.
- `lda_coherence_vs_k.png`: đường C_v chọn K ở bước 1 với hyperparameters cố định;
  `lda_topic_words.png`: keyword của model thắng bước 2.
  `lda_training_summary.txt` ghi tóm tắt cả hai bước.

C_v dùng toàn bộ corpus sau lọc token. Không so sánh trực tiếp điểm LDA với
BERTopic nếu hai lần chạy khác corpus, tokenization hoặc cách tính coherence.
