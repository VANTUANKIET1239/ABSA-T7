"""Đọc dữ liệu nguồn và tính thống kê dùng chung cho các hình, bảng nghiên cứu.

Tên hàm mô tả nội dung dữ liệu, độc lập với số thứ tự hình hoặc bảng.
Các hàm load_* đọc kết quả đã lưu; các hàm thống kê tính từ dữ liệu review.
"""
from functools import lru_cache
import json
import re

import numpy as np
import pandas as pd

from .config import ASPECTS, DATA, PROJECT

# Dữ liệu review và kết quả huấn luyện gốc.
REVIEW_PATH = PROJECT / 'crawl-data/data/combined_reviews_brand_filled_language_filtered.csv'
RAW_REVIEW_PATH = PROJECT / 'crawl-data/data/combined_reviews.csv'
UNIQUE_TEXT_PATH = PROJECT / 'crawl-data/data/unique_texts/unique_texts.csv'
SPLIT_PATHS = {
    split: PROJECT / f'train-models/data/{split}_cleanser.csv'
    for split in ['train', 'val', 'test']
}
EVALUATIONS = PROJECT / 'evaluation/results/cleanser_models_comparison/outputs'
HISTORY_PATH = PROJECT / 'train-models/Histories/Cleanser/training_history_birnn_cleanser_focal_loss_VI.csv'

# Kết quả hồi quy, SHAP và IPA đã được tính trong thư mục Business.
IPA_ROOT = PROJECT / 'Business/customer_satisfaction_xai_ipa_results_cleanser'
CUTOFF_ROOT = PROJECT / 'Business/customer_satisfaction_xai_ipa_mean_median_results_cleanser'
PLATFORM_ROOT = PROJECT / 'Business/ipa_by_platform_results_cleanser'
BRAND_ROOT = PROJECT / 'Business/ipa_top5_brands_results_cleanser'
SHAP_SUMMARY_PATH = IPA_ROOT / 'xai_aspect_summary.csv'
IPA_OVERALL_PATH = IPA_ROOT / 'ipa_attributes.csv'
RATING_REGRESSION_PATH = IPA_ROOT / 'model_comparison_oof.csv'
IPA_PLATFORM_PATH = PLATFORM_ROOT / 'ipa_all_groups.csv'
IPA_BRAND_PATH = BRAND_ROOT / 'ipa_all_groups.csv'
PLATFORM_SUMMARY_PATH = PLATFORM_ROOT / 'group_run_summary.csv'
BRAND_SUMMARY_PATH = BRAND_ROOT / 'group_run_summary.csv'
IPA_MEAN_MEDIAN_PATH = CUTOFF_ROOT / 'ipa_mean_median_comparison.csv'
IPA_CUTOFF_PATH = CUTOFF_ROOT / 'ipa_mean_median_thresholds.csv'
RESULT_CSV_PATHS = (
    SHAP_SUMMARY_PATH, IPA_OVERALL_PATH, RATING_REGRESSION_PATH,
    IPA_PLATFORM_PATH, IPA_BRAND_PATH, PLATFORM_SUMMARY_PATH,
    BRAND_SUMMARY_PATH, IPA_MEAN_MEDIAN_PATH, IPA_CUTOFF_PATH,
)

# Các lần chạy mô hình chủ đề được chọn để so sánh.
TOPIC_ROOT = PROJECT / 'topic-modeling/data'
SENTIMENT_ORDER = ['positive', 'negative', 'neutral']
LABEL_PATTERN = re.compile(r'\{\s*([^#{}]+?)\s*#\s*([^{}]+?)\s*\}')
TOPIC_RUNS = [
    ('BERTopic', 'bertopic_16_topics_reproduced'),
    ('LDA (K tốt nhất)', 'lda_cleanser_bertopic_corpus'),
    ('LDA (K=16)', 'lda_cleanser_bertopic_corpus_k16'),
]


def read_csv(path):
    """Đọc CSV UTF-8, giữ ô trống thành chuỗi rỗng để xử lý nhãn và văn bản."""
    return pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)


def load_training_history():
    """Lịch sử loss theo epoch của BiRNN dùng Focal Loss."""
    return read_csv(HISTORY_PATH)


def load_shap_aspect_summary():
    """Mức độ quan trọng của từng khía cạnh từ kết quả SHAP."""
    return read_csv(SHAP_SUMMARY_PATH)


def load_ipa_overall():
    """Tọa độ Importance/Performance và nhóm IPA trên toàn bộ review."""
    return read_csv(IPA_OVERALL_PATH)


def load_rating_regression_metrics():
    """Chỉ số đánh giá hồi quy rating từ dự đoán ngoài fold (OOF)."""
    result = read_csv(RATING_REGRESSION_PATH)
    if result.PRED25.max() > 1:
        # Đưa kết quả cũ ở thang phần trăm về tỷ lệ 0–1.
        result['PRED25'] = result.PRED25 / 100
    return result


def load_ipa_by_platform():
    """Kết quả IPA riêng cho từng sàn thương mại điện tử."""
    return read_csv(IPA_PLATFORM_PATH)


def load_ipa_by_brand():
    """Kết quả IPA riêng cho từng thương hiệu được chọn."""
    return read_csv(IPA_BRAND_PATH)


def load_platform_run_summary():
    """Danh sách sàn và thông tin tổng hợp của từng lần chạy IPA."""
    return read_csv(PLATFORM_SUMMARY_PATH)


def load_brand_run_summary():
    """Danh sách thương hiệu và thông tin tổng hợp của từng lần chạy IPA."""
    return read_csv(BRAND_SUMMARY_PATH)


def load_ipa_mean_median_comparison():
    """Đối chiếu nhóm IPA của từng khía cạnh khi đổi ngưỡng mean/median."""
    return read_csv(IPA_MEAN_MEDIAN_PATH)


def load_ipa_cutoffs():
    """Ngưỡng Importance và Performance theo trung bình hoặc trung vị."""
    return read_csv(IPA_CUTOFF_PATH)


def canonical_groups(series):
    """Chuẩn hóa khoảng trắng và cách viết tên sàn/thương hiệu."""
    names = series.astype(str).str.normalize('NFKC').str.split().str.join(' ')
    counts = pd.DataFrame({'key': names.str.casefold(), 'name': names}).value_counts().reset_index(name='n')
    preferred = (counts.sort_values(['key', 'n', 'name'], ascending=[True, False, True])
                 .drop_duplicates('key').set_index('key')['name'].to_dict())
    preferred.update({'lazada': 'Lazada', 'shopee': 'Shopee', 'tiki': 'Tiki'})
    return names.str.casefold().map(preferred)


@lru_cache(maxsize=1)
def _read_reviews():
    """Đọc và chuẩn bị review một lần cho mỗi tiến trình chạy."""
    data = read_csv(REVIEW_PATH)
    has_text = data['cleaned_comment'].str.strip().ne('')
    has_label = data['label'].str.strip().ne('')
    data = data.loc[has_text & has_label].copy().reset_index(drop=True)
    data['Platform'] = canonical_groups(data.platform)
    data['Brand'] = canonical_groups(data.brand)
    data['Whitespace_Tokens'] = data.cleaned_comment.str.split().str.len()
    data['rating'] = pd.to_numeric(data.rating, errors='raise')
    return data


def reviews():
    """Review có nội dung và nhãn hợp lệ; trả bản sao để không sửa cache chung."""
    return _read_reviews().copy()


@lru_cache(maxsize=1)
def label_matrix():
    """Ma trận có/không của từng cặp khía cạnh–cảm xúc trong mỗi review."""
    data = reviews()
    matrix = np.zeros((len(data), len(ASPECTS), len(SENTIMENT_ORDER)), dtype=bool)
    for index, value in enumerate(data.label):
        for aspect, sentiment in set(LABEL_PATTERN.findall(value)):
            if aspect in ASPECTS and sentiment in SENTIMENT_ORDER:
                matrix[index, ASPECTS.index(aspect), SENTIMENT_ORDER.index(sentiment)] = True
    return matrix


def sentiment_counts():
    """Đếm nhãn cảm xúc và số review đề cập từng khía cạnh."""
    matrix = label_matrix()
    result = pd.DataFrame(matrix.sum(axis=0), columns=SENTIMENT_ORDER)
    result.insert(0, 'Aspect', ASPECTS)
    result['Mentioned_Reviews'] = matrix.any(axis=2).sum(axis=0)
    result['Label_Pairs'] = matrix.sum(axis=(0, 2))
    return result


def platform_overview():
    """Thống kê số review, sản phẩm, thương hiệu và tỷ trọng theo sàn."""
    data = reviews()
    result = data.groupby('Platform').agg(
        Reviews=('label', 'size'), Products=('product_url', 'nunique'), Brands=('Brand', 'nunique')
    ).reset_index()
    result['Share_Pct'] = 100 * result.Reviews / len(data)
    return result


def rating_counts():
    """Đếm review theo sàn và số sao, luôn có đủ cột từ 1 đến 5 sao."""
    data = reviews()
    counts = pd.crosstab(data['Platform'], data['rating'])
    counts = counts.reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    return counts.rename(columns=str).rename_axis('Platform').reset_index()


def review_length_summary():
    """Thống kê độ dài theo đơn vị cách nhau bằng khoảng trắng, toàn bộ và theo sàn."""
    data = reviews()
    rows = []
    for name, values in [('Toàn bộ', data)] + [(p, data.loc[data.Platform.eq(p)]) for p in ['Lazada', 'Shopee', 'Tiki']]:
        lengths = values.Whitespace_Tokens
        rows.append({'Group': name, 'Mean': lengths.mean(), 'P50': lengths.median(),
                     'P95': lengths.quantile(.95), 'Max': lengths.max(), 'N': len(lengths)})
    return pd.DataFrame(rows)


def split_counts():
    """Số nội dung trong các tập train, validation và test đã lưu."""
    return pd.DataFrame([
        {'Split': split, 'N': len(read_csv(path))}
        for split, path in SPLIT_PATHS.items()
    ])


def dataset_summary():
    """Quy mô dữ liệu gốc, review hợp lệ, nội dung duy nhất và các tập chia."""
    return {'raw_reviews': len(read_csv(RAW_REVIEW_PATH)),
            'valid_reviews': len(reviews()),
            'unique_texts': len(read_csv(UNIQUE_TEXT_PATH)),
            'split_counts': split_counts().set_index('Split').N.to_dict()}


def absa_model_metrics():
    """Đọc F1 và Accuracy từ dòng ACD+SPC trong báo cáo TXT của 12 mô hình."""
    rows = []
    for model in ['birnn', 'bilstm', 'bigru', 'multitask']:
        for scenario in ['baseline', 'class_weights', 'focal_loss']:
            path = EVALUATIONS / f'{model}_{scenario}.txt'
            lines = path.read_text(encoding='utf-8-sig').splitlines()
            line = next(line for line in lines if re.match(r'\|\s*ACD\+SPC\s*\|', line))
            fields = [field.strip() for field in line.split('|') if field.strip()]
            rows.append({'Model': model, 'Scenario': scenario, 'F1_Macro': float(fields[8]) * 100,
                         'Weighted_F1': float(fields[9]) * 100, 'Accuracy': float(fields[10]) * 100})
    return pd.DataFrame(rows)


def birnn_aspect_comparison():
    """So sánh F1 từng khía cạnh của BiRNN với CE, class weights và Focal Loss."""
    values = {aspect: {'Aspect': aspect} for aspect in ASPECTS}
    for scenario in ['baseline', 'class_weights', 'focal_loss']:
        text = (EVALUATIONS / f'birnn_{scenario}.txt').read_text(encoding='utf-8-sig')
        for line in text.splitlines():
            fields = [field.strip() for field in line.split('|') if field.strip()]
            if len(fields) == 8 and fields[0] in values:
                values[fields[0]][scenario] = float(fields[6])
                values[fields[0]]['Support'] = int(fields[7])
    result = pd.DataFrame(values.values())
    result['Delta_Focal_CE'] = result.focal_loss - result.baseline
    return result


def topic_model_metrics():
    """Đọc số topic, số nội dung và C_v của các lần chạy BERTopic/LDA đã chọn."""
    rows = []
    for label, folder in TOPIC_RUNS:
        path = TOPIC_ROOT / folder
        name = 'single_run_params.json' if label == 'BERTopic' else 'best_params.json'
        params = json.loads((path / name).read_text(encoding='utf-8-sig'))
        metrics = json.loads((path / 'evaluation_metrics.json').read_text(encoding='utf-8-sig'))
        rows.append({'Model': label, 'Topics': metrics['n_topics'], 'Reviews': metrics['n_documents'],
                     'Native_Cv': params['c_v'] if label == 'BERTopic' else params['coherence_cv']})
    return pd.DataFrame(rows)


def bertopic_topics():
    """Ghép số nội dung và sáu từ khóa đứng đầu của từng topic BERTopic."""
    path = TOPIC_ROOT / 'bertopic_16_topics_reproduced'
    counts = read_csv(path / 'topic_frequency.csv').set_index('Topic')
    words = read_csv(path / 'topic_keyword_weights.csv')
    return pd.DataFrame([{'Topic': topic, 'Count': int(counts.loc[topic, 'Count']),
                          'Keywords': ', '.join(part.sort_values('Rank').head(6).Word.str.replace('_', ' ', regex=False))}
                         for topic, part in words.loc[words.Topic.ge(0)].groupby('Topic', sort=True)])


def ordered(frame, aspects=ASPECTS):
    """Sắp xếp các dòng theo thứ tự khía cạnh dùng chung trong nghiên cứu."""
    return frame.set_index('Aspect').reindex(aspects).reset_index()


def number(value, digits=2):
    """Định dạng số với dấu phẩy thập phân và dấu chấm phân tách hàng nghìn."""
    return f'{float(value):,.{digits}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def input_files():
    """Liệt kê nguồn đầu vào để ghi đường dẫn và SHA-256 vào manifest."""
    files = {REVIEW_PATH, RAW_REVIEW_PATH, UNIQUE_TEXT_PATH, HISTORY_PATH,
             DATA / 'literature_reference_rows.json', *RESULT_CSV_PATHS}
    files.update(EVALUATIONS.glob('*.txt'))
    files.update(SPLIT_PATHS.values())
    for label, folder in TOPIC_RUNS:
        path = TOPIC_ROOT / folder
        files.add(path / ('single_run_params.json' if label == 'BERTopic' else 'best_params.json'))
        files.add(path / 'evaluation_metrics.json')
    files.update(TOPIC_ROOT / 'bertopic_16_topics_reproduced' / name
                 for name in ['topic_frequency.csv', 'topic_keyword_weights.csv'])
    return sorted(files)
