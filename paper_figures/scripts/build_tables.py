"""Export only the thirteen tables selected in the user's manuscript screenshots."""
import _bootstrap
import csv
import json
from common.config import ASPECTS, CODES, DATA, LABELS, LOSSES, MODEL_NAMES, Q_NAMES, TABLES
from common.data_utils import (
    absa_model_metrics, bertopic_topics, birnn_aspect_comparison,
    dataset_summary, load_ipa_cutoffs, load_ipa_mean_median_comparison,
    load_ipa_overall, load_rating_regression_metrics, number as f,
    platform_overview, review_length_summary, topic_model_metrics,
)

TABLE_STEMS = [
    'table01_dataset_overview', 'table02_platform_overview', 'table03_train_validation_test',
    'table04_review_length', 'table05_topic_model_comparison', 'table06_bertopic_topics',
    'table07_aspect_definitions', 'table08_absa_configurations', 'table09_birnn_aspect_f1',
    'table10_rating_regression', 'table11_ipa_overall', 'table12_ipa_cutoffs', 'table13_ipa_quadrant_changes',
]
SCOPES = [
    'Làm sạch bụi bẩn, dầu thừa và kiểm soát dầu',
    'Độ ẩm, mềm mịn, khô căng và cảm giác sau rửa',
    'Khả năng dung nạp của da, phù hợp da và phản ứng bất lợi',
    'Kết cấu sản phẩm, hạt và khả năng tạo bọt',
    'Đặc điểm mùi và cảm nhận về mùi',
    'Thành phần, hoạt chất, pH và công thức',
    'Thiết kế, tình trạng bao bì và cách đóng gói',
    'Giá và mức độ đáng tiền',
    'Tư vấn, hỗ trợ và thực hiện đơn hàng',
    'Tốc độ và trải nghiệm vận chuyển',
    'Đánh giá chung chưa gắn khía cạnh cụ thể',
]


def write(index, headers, rows):
    path = TABLES / (TABLE_STEMS[index - 1] + '.csv')
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    print(f'[+] {path.name}')


def main():
    TABLES.mkdir(parents=True, exist_ok=True)
    meta = dataset_summary()
    write(1, ['Giai đoạn', 'Số lượng'], [
        ['Thu thập ban đầu', f(meta['raw_reviews'], 0)],
        ['Hợp lệ sau tiền xử lý', f(meta['valid_reviews'], 0)],
        ['Nội dung đánh giá duy nhất', f(meta['unique_texts'], 0)],
    ])
    data = platform_overview()
    write(2, ['Nền tảng', 'Review', 'Tỷ trọng (%)', 'Thương hiệu', 'Sản phẩm'],
          [[r.Platform, f(r.Reviews, 0), f(r.Share_Pct), f(r.Brands, 0), f(r.Products, 0)] for r in data.itertuples()])
    write(3, ['Tập dữ liệu', 'Số lượng'], [[label, f(meta['split_counts'][key], 0)] for key, label in
          [('train', 'Train (70%)'), ('val', 'Validation (15%)'), ('test', 'Test (15%)')]])
    data = review_length_summary()
    write(4, ['Nhóm', 'Trung bình', 'Trung vị', 'P95', 'Dài nhất'],
          [[r.Group, f(r.Mean), f(r.P50, 0), f(r.P95, 0), f(r.Max, 0)] for r in data.itertuples()])
    write(5, ['Mô hình', 'K', 'C_v'],
          [[r.Model, r.Topics, f(r.Native_Cv, 4)] for r in topic_model_metrics().itertuples()])
    write(6, ['Chủ đề', 'Số nội dung', 'Sáu từ khóa đầu'],
          [[r.Topic, f(r.Count, 0), r.Keywords] for r in bertopic_topics().itertuples()])
    write(7, ['Ký hiệu', 'Khía cạnh', 'Phạm vi nhận xét'],
          [[CODES[a], LABELS[a] if a != 'General' else 'Đánh giá chung', scope] for a, scope in zip(ASPECTS, SCOPES)])

    references = json.loads((DATA / 'literature_reference_rows.json').read_text(encoding='utf-8'))
    ref = references['absa']
    rows = [[ref['Architecture'], '-', '-', f(ref['Weighted_F1']), f(ref['Accuracy'])]]
    rows.extend([[MODEL_NAMES[r.Model], LOSSES[r.Scenario], f(r.F1_Macro), f(r.Weighted_F1), f(r.Accuracy)]
                 for r in absa_model_metrics().itertuples()])
    write(8, ['Kiến trúc', 'Hàm mất mát', 'F1 Macro (%)', 'Weighted F1 (%)', 'Accuracy (%)'], rows)
    write(9, ['Khía cạnh', 'CE', 'WCE', 'Focal', 'Δ Focal–CE', 'Mẫu'],
          [[f'{CODES[r.Aspect]} – {LABELS[r.Aspect]}', f(r.baseline), f(r.class_weights), f(r.focal_loss),
            ('+' if r.Delta_Focal_CE >= 0 else '−') + f(abs(r.Delta_Focal_CE)), f(r.Support, 0)]
           for r in birnn_aspect_comparison().itertuples()])

    ref = references['regression']
    rows = [[ref['Model'], f(ref['RMSE'], 3), '-', '-', '-', '-', '-']]
    rows.extend([[r.Model, f(r.RMSE, 4), f(r.MAE, 4), f(r.R2, 4), f(r.MBRE, 4), f(r.MIBRE, 4), f(r.PRED25, 4)]
                 for r in load_rating_regression_metrics().itertuples()])
    write(10, ['Mô hình', 'RMSE', 'MAE', 'R²', 'MBRE', 'MIBRE', 'PRED(25)'], rows)
    write(11, ['Khía cạnh', 'Importance (%)', 'Performance', 'Đề xuất IPA'],
          [[f'{CODES[r.Aspect]} – {LABELS[r.Aspect]}', f(r.Importance_Pct), f(r.Performance_0_100),
            Q_NAMES[r.IPA_Quadrant]] for r in load_ipa_overall().sort_values('Importance_Pct', ascending=False).itertuples()])
    write(12, ['Phương pháp', 'Ngưỡng Importance (%)', 'Ngưỡng Performance'],
          [['Trung bình' if r.Cut_Method == 'mean' else 'Trung vị', f(r.Importance_Cut, 3), f(r.Performance_Cut, 3)]
           for r in load_ipa_cutoffs().itertuples()])
    comparison = load_ipa_mean_median_comparison().set_index('Aspect')
    changed = comparison.loc[comparison.IPA_Quadrant_Mean.ne(comparison.IPA_Quadrant_Median)]
    changed = changed.sort_values('Importance_Pct', ascending=False).reset_index()
    write(13, ['Khía cạnh', 'Ngưỡng trung bình', 'Ngưỡng trung vị'],
          [[f'{CODES[r.Aspect]} – {LABELS[r.Aspect]}', Q_NAMES[r.IPA_Quadrant_Mean],
            Q_NAMES[r.IPA_Quadrant_Median]] for r in changed.itertuples()])


if __name__ == '__main__':
    main()
