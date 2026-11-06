"""Khám phá topic review sữa rửa mặt bằng BERTopic và Grid Search theo C_v.

Mỗi text duy nhất học một lần; duplicate_count chỉ là metadata.
Dùng --check-input để kiểm tra dữ liệu, --single-run để chạy một cấu hình.
Các aspect gợi ý cần được kiểm tra thủ công trước khi dùng làm nhãn ABSA.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from itertools import product
from pathlib import Path
from time import perf_counter

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
DEFAULT_INPUT = 'crawl-data/data/unique_texts/unique_texts.csv'
DEFAULT_OUTPUT = str(BASE_DIR / 'data' / 'bertopic_unique_texts')
EMBEDDING_MODEL_NAME = 'keepitreal/vietnamese-sbert'

BASELINE_PARAMS = {
    'min_cluster_size': 30,
    'min_samples': 5,
    'n_neighbors': 20,
    'min_df': 3,
    'max_df': 0.85,
}

# Chỉ lọc ở tầng keyword; giữ phủ định và mô tả trải nghiệm.
TOPIC_STOPWORDS = {
    'mình', 'tôi', 'tui', 'em', 'anh', 'chị', 'bạn', 'mọi_người',
    'cho', 'với', 'để', 'thì', 'là', 'và', 'của', 'mà', 'có',
    'những', 'các', 'một', 'này', 'kia', 'đó', 'đây', 'ấy',
    'nha', 'nhé', 'ạ', 'ha', 'hihi', 'haha', 'hehe', 'kkk',
    'nói_chung', 'nói_chung_là', 'sản_phẩm', 'sp',
    'sữa_rửa_mặt', 'sữa', 'rửa_mặt', 'srm',
}

ASPECT_KEYWORDS = {
    'Cleansing_Effectiveness': {
        'làm_sạch', 'làm_sạch_sâu', 'sạch_sâu', 'sạch_da', 'không_sạch',
        'rửa_sạch', 'bụi_bẩn', 'bã_nhờn', 'kiềm_dầu', 'kiểm_soát_dầu',
        'dầu_thừa', 'lỗ_chân_lông', 'sạch', 'nhờn',
    },
    'Hydration_Skin_Comfort': {
        'dưỡng_ẩm', 'cấp_ẩm', 'giữ_ẩm', 'khô_căng', 'không_khô_căng',
        'khô_da', 'căng_rít', 'căng_da', 'mềm_mịn', 'mịn_màng',
        'ẩm_mượt', 'dịu_nhẹ', 'dễ_chịu',
    },
    'Safety_Skin_Compatibility': {
        'kích_ứng', 'không_kích_ứng', 'châm_chích', 'rát_da', 'nóng_rát',
        'cay_mắt', 'ngứa_da', 'đỏ_da', 'mẩn_đỏ', 'nổi_mụn', 'lên_mụn',
        'đẩy_mụn', 'da_nhạy_cảm', 'không_gây_mụn',
    },
    'Foaming_Texture': {
        'tạo_bọt', 'bọt_mịn', 'ít_bọt', 'nhiều_bọt', 'bọt',
        'kết_cấu', 'dạng_gel', 'dạng_kem', 'gel', 'lỏng', 'đặc',
        'dễ_rửa', 'khó_rửa',
    },
    'Fragrance': {
        'mùi_hương', 'mùi_thơm', 'không_mùi', 'mùi_hắc', 'mùi_hóa_chất',
        'hương_liệu', 'mùi', 'thơm', 'hắc', 'nồng',
    },
    'Ingredients_Formulation': {
        'thành_phần', 'độ_ph', 'cân_bằng_độ_ph', 'salicylic_acid',
        'niacinamide', 'hyaluronic_acid', 'tràm_trà', 'rau_má',
        'cồn', 'sulfate', 'hương_liệu',
    },
    'Packaging': {
        'bao_bì', 'đóng_gói', 'chai_lọ', 'nắp_chai', 'vòi_bơm',
        'tem_nhãn', 'niêm_phong', 'rò_rỉ', 'đổ_vỡ', 'chai', 'tuýp',
    },
    'Price_Value': {
        'giá_cả', 'giá_thành', 'giá_rẻ', 'giá_cao', 'giảm_giá',
        'mã_giảm_giá', 'đáng_tiền', 'đắt', 'rẻ', 'giá', 'sale',
    },
    'Service_Delivery': {
        'giao_hàng', 'giao_nhanh', 'giao_chậm', 'vận_chuyển',
        'giao', 'shipper', 'chăm_sóc_khách_hàng', 'cửa_hàng',
        'tư_vấn', 'đổi_trả', 'dịch_vụ',
    },
}


def resolve_input_path(name: str) -> Path:
    path = Path(name).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, BASE_DIR / path, PROJECT_DIR / path]
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    checked = '\n'.join(f'    - {candidate}' for candidate in candidates)
    raise FileNotFoundError(
        f'Không tìm thấy CSV: {name}\n'
        f'Các đường dẫn đã kiểm tra:\n{checked}\n'
        'Dùng --input "đường/dẫn/đến/file.csv" để chỉ định CSV hiện có.'
    )


def resolve_output_path(name: str) -> Path:
    path = Path(name).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def choose_column(df: pd.DataFrame, preferred: str, fallbacks: tuple[str, ...]) -> str:
    for col in (preferred, *fallbacks):
        if col in df.columns:
            if col != preferred:
                print(f'[!] Không có {preferred!r}; sử dụng {col!r}.')
            return col
    raise ValueError(f'Không tìm thấy cột {preferred!r} hoặc một trong {fallbacks}. '
                     'Kiểm tra các cột CSV và tùy chọn --embedding-column/--topic-column.')


def read_dataset(input_path: Path, embedding_column: str, topic_column: str):
    if not input_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy CSV: {input_path}')
    df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    e_col = choose_column(df, embedding_column,
                          ('normalized_comment', 'representative_comment', 'comment', 'cleaned_comment'))
    t_col = choose_column(df, topic_column, ('topic_comment', 'cleaned_comment'))

    df = df.copy()
    df['_embedding_text'] = df[e_col].fillna('').astype(str).str.strip()
    if e_col in ('cleaned_comment', 'topic_comment'):
        df['_embedding_text'] = df['_embedding_text'].str.replace('_', ' ', regex=False)
        df['_embedding_text'] = df['_embedding_text'].str.replace(r'\s+', ' ', regex=True).str.strip()
    df['_topic_text'] = df[t_col].fillna('').astype(str).str.strip()
    mask = df['_embedding_text'].str.len().ge(6) & df['_topic_text'].str.len().ge(3)
    valid = df.loc[mask].copy().reset_index(drop=True)
    print(f'[*] CSV: {len(df):,} dòng; đủ dữ liệu text: {len(valid):,} dòng; loại {len(df) - len(valid):,} dòng rỗng/ngắn.')
    if not len(valid):
        raise ValueError('Không có review hợp lệ sau khi lọc text.')
    return valid, e_col, t_col


def aspect_suggestions(topic_words: list[str]) -> str:
    """Gợi ý tối đa hai aspect từ keyword để kiểm tra thủ công."""
    tokens = set(topic_words)
    scores = sorted(
        ((len(tokens & keywords), name) for name, keywords in ASPECT_KEYWORDS.items()),
        reverse=True,
    )
    matched = [name for score, name in scores if score > 0]
    return '; '.join(matched[:2]) if matched else 'Needs_manual_review'


def coherence_cv(topic_model, docs: list[str], max_docs: int):
    from gensim import corpora
    from gensim.models import CoherenceModel

    topic_ids = sorted(topic_id for topic_id in topic_model.get_topics() if topic_id != -1)
    if not topic_ids:
        print('[!] Tất cả review đều thuộc Topic -1. Bỏ qua C_v.')
        return None

    topic_words = [
        [word for word, score in (topic_model.get_topic(topic_id) or []) if word and score > 0]
        for topic_id in topic_ids
    ]
    topic_words = [words for words in topic_words if len(words) >= 2]
    if not topic_words:
        print('[!] Không đủ từ khóa để tính Coherence C_v.')
        return None

    if max_docs > 0 and len(docs) > max_docs:
        sampled = pd.Series(docs).sample(n=max_docs, random_state=42).tolist()
        print(f'[*] C_v dùng mẫu {max_docs:,}/{len(docs):,} reviews (không phải full-corpus).')
    else:
        sampled = docs
        print(f'[*] C_v dùng toàn bộ {len(sampled):,} reviews.')

    analyzer = topic_model.vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in sampled]
    tokenized_docs = [tokens for tokens in tokenized_docs if tokens]
    dictionary = corpora.Dictionary(tokenized_docs)
    # Corpus lấy mẫu có thể thiếu keyword của một số topic.
    filtered = [[w for w in words if w in dictionary.token2id] for words in topic_words]
    filtered = [words for words in filtered if len(words) >= 2]
    if not filtered:
        print('[!] Tập kiểm tra không chứa đủ keywords để tính C_v.')
        return None

    score = CoherenceModel(
        topics=filtered, texts=tokenized_docs, dictionary=dictionary,
        coherence='c_v', processes=1,  # Tránh multiprocessing trên Windows.
    ).get_coherence()
    return float(score)


def model_configuration(params: dict, n_docs: int | None = None) -> dict:
    """Cấu hình dùng chung khi tạo mô hình và lưu JSON."""
    return {
        'embedding_model': EMBEDDING_MODEL_NAME,
        'encode': {'normalize_embeddings': True, 'convert_to_numpy': True},
        'umap': {
            'n_neighbors': min(params['n_neighbors'], n_docs - 1) if n_docs else params['n_neighbors'],
            'n_components': 5, 'min_dist': 0.0, 'metric': 'cosine', 'random_state': 42,
        },
        'hdbscan': {
            'min_cluster_size': params['min_cluster_size'], 'min_samples': params['min_samples'],
            'metric': 'euclidean', 'cluster_selection_method': 'eom', 'prediction_data': True,
        },
        'vectorizer': {
            'stop_words': sorted(TOPIC_STOPWORDS), 'ngram_range': (1, 2),
            'token_pattern': r'(?u)\b\w\w+\b', 'min_df': params['min_df'], 'max_df': params['max_df'],
        },
        'ctfidf': {'bm25_weighting': True, 'reduce_frequent_words': True},
        'bertopic': {'calculate_probabilities': False, 'top_n_words': 10, 'verbose': True},
    }


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def create_model(params: dict, embedding_model, n_docs: int):
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    config = model_configuration(params, n_docs)
    return BERTopic(
        embedding_model=embedding_model,
        umap_model=UMAP(**config['umap']),
        hdbscan_model=HDBSCAN(**config['hdbscan']),
        vectorizer_model=CountVectorizer(**config['vectorizer']),
        ctfidf_model=ClassTfidfTransformer(**config['ctfidf']),
        **config['bertopic'],
    )


def parameter_candidates(args, reference: dict, n_docs: int) -> list[dict]:
    """Thử baseline trước, sau đó duyệt lưới và bỏ tổ hợp trùng."""
    if args.single_run:
        return [reference.copy()]
    grid = {
        'min_cluster_size': args.grid_min_cluster_sizes,
        'min_samples': args.grid_min_samples,
        'n_neighbors': args.grid_n_neighbors,
        'min_df': args.grid_min_df or [reference['min_df']],
        'max_df': args.grid_max_df or [reference['max_df']],
    }
    combinations = [BASELINE_PARAMS.copy(), reference.copy()]
    combinations.extend(dict(zip(grid, values)) for values in product(*grid.values()))
    unique = {}
    for params in combinations:
        params['n_neighbors'] = min(params['n_neighbors'], n_docs - 1)
        unique.setdefault(tuple(params[key] for key in BASELINE_PARAMS), params)
    return list(unique.values())


def validate_params(params: dict, n_docs: int):
    if not 2 <= params['min_cluster_size'] <= n_docs:
        raise ValueError(f'min_cluster_size phải từ 2 đến số review ({n_docs}).')
    if not 1 <= params['min_samples'] < n_docs:
        raise ValueError(f'min_samples phải từ 1 đến {n_docs - 1}.')
    if params['n_neighbors'] < 2:
        raise ValueError('n_neighbors phải >= 2.')
    if params['min_df'] < 1 or not (0 < params['max_df'] <= 1):
        raise ValueError('min_df >= 1 và 0 < max_df <= 1.')


def grid_search(candidates, embedding_model, embeddings, topic_docs, args, out: Path):
    """Chọn lần fit có C_v cao nhất, bỏ qua tổ hợp lỗi."""
    rows = []
    best = None
    best_score = -math.inf
    for trial, params in enumerate(candidates, start=1):
        print(f'\n[*] Grid Search {trial}/{len(candidates)}: {params}')
        started = perf_counter()
        row = {'trial': trial, **params, 'c_v': None, 'n_topics': None,
               'outlier_rate': None, 'status': 'failed', 'error': ''}
        model = topics = probabilities = None
        try:
            validate_params(params, len(topic_docs))
            model = create_model(params, embedding_model, len(topic_docs))
            topics, probabilities = model.fit_transform(topic_docs, embeddings=embeddings)
            row['n_topics'] = len(set(topics) - {-1})
            row['outlier_rate'] = sum(topic == -1 for topic in topics) / len(topics)
            score = coherence_cv(model, topic_docs, args.coherence_sample)
            if score is None or not math.isfinite(score):
                row['status'] = 'invalid_coherence'
                row['error'] = 'Không có C_v hữu hạn để so sánh.'
                print('[!] Bỏ qua cấu hình không có C_v hữu hạn.')
            else:
                row.update(c_v=score, status='ok')
                print(f'[+] C_v={score:.4f}; topics={row["n_topics"]}; outliers={row["outlier_rate"]:.2%}')
                if score > best_score:
                    write_json(out / 'best_params.json', {
                        'trial': trial, 'c_v': score, 'params': params,
                        'model_configuration': model_configuration(params, len(topic_docs)),
                        'n_topics': row['n_topics'], 'outlier_rate': row['outlier_rate'],
                        'coherence_sample': args.coherence_sample,
                    })
                    best_score = score
                    best = (model, topics, probabilities, score)
        except Exception as exc:
            row['status'] = 'failed'
            row['error'] = f'{type(exc).__name__}: {exc}'
            print(f'[!] Tổ hợp {trial} thất bại: {row["error"]}')
        finally:
            model = topics = probabilities = None
        row['elapsed_seconds'] = round(perf_counter() - started, 3)
        rows.append(row)
        # Lưu từng trial để giữ tiến độ khi chương trình bị ngắt.
        pd.DataFrame(rows).to_csv(out / 'grid_search_results.csv', index=False, encoding='utf-8-sig')

    if best is None:
        raise RuntimeError(f'Không có cấu hình nào tính được C_v hợp lệ. Xem {out / "grid_search_results.csv"}')
    print(f'\n[+] C_v tốt nhất trong {len(candidates)} cấu hình: {best_score:.4f}')
    return best


def export_outputs(df, docs_for_topic, topic_model, topics, probabilities, out: Path, coherence_score):
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    result = df.drop(columns=['_embedding_text', '_topic_text']).copy()
    result['Topic'] = topics
    probs = np.asarray(probabilities) if probabilities is not None else None
    if probs is not None and probs.ndim == 1 and len(probs) == len(result):
        result['Topic_Probability'] = probs
    else:
        result['Topic_Probability'] = pd.NA
    result.to_csv(out / 'reviews_with_topics.csv', index=False, encoding='utf-8-sig')

    info = topic_model.get_topic_info()
    info.to_csv(out / 'topic_info.csv', index=False, encoding='utf-8-sig')

    text_to_raw = defaultdict(list)
    raw_col = next(col for col in ('comment', 'representative_comment', 'normalized_comment', 'cleaned_comment')
                   if col in result.columns)
    for topic_text, raw in zip(docs_for_topic, result[raw_col].astype(str)):
        if len(text_to_raw[topic_text]) < 4 and raw not in text_to_raw[topic_text]:
            text_to_raw[topic_text].append(raw)

    rows = []
    for entry in info.itertuples(index=False):
        tid = int(entry.Topic)
        if tid == -1:
            continue
        scored_words = topic_model.get_topic(tid) or []
        words = [word for word, score in scored_words if word and score > 0]
        representation = topic_model.get_representative_docs(tid) or []
        representative_comments = []
        for topic_doc in representation:
            for raw in text_to_raw.get(topic_doc, [topic_doc]):
                if raw not in representative_comments:
                    representative_comments.append(raw)
                if len(representative_comments) >= 3:
                    break
            if len(representative_comments) >= 3:
                break
        if not representative_comments:
            representative_comments = result.loc[result['Topic'].eq(tid), raw_col].dropna().astype(str).head(3).tolist()
        rows.append({
            'Topic': tid,
            'Count': int(entry.Count),
            'Keywords': ', '.join(words),
            'Suggested_Aspect': aspect_suggestions(words),
            'Manual_Aspect': '',
            'Representative_Review_1': representative_comments[0] if len(representative_comments) > 0 else '',
            'Representative_Review_2': representative_comments[1] if len(representative_comments) > 1 else '',
            'Representative_Review_3': representative_comments[2] if len(representative_comments) > 2 else '',
        })
    pd.DataFrame(rows, columns=[
        'Topic', 'Count', 'Keywords', 'Suggested_Aspect', 'Manual_Aspect',
        'Representative_Review_1', 'Representative_Review_2', 'Representative_Review_3',
    ]).to_csv(out / 'aspect_candidates.csv', index=False, encoding='utf-8-sig')

    counts_note = ('Counts refer to unique texts; duplicate_count is metadata, not a training weight.\n'
                   if 'text_group_id' in result.columns else '')
    (out / 'training_summary.txt').write_text(
        f'Input valid reviews: {len(result)}\n'
        f'Clusters (excluding -1): {len(rows)}\n'
        f'Outliers -1: {int(result["Topic"].eq(-1).sum())}\n'
        f'Outlier rate: {result["Topic"].eq(-1).mean():.4f}\n'
        f'C_v coherence: {coherence_score if coherence_score is not None else "N/A"}\n'
        f'{counts_note}'
        'Suggested_Aspect is keyword-based only. Manual validation required.\n',
        encoding='utf-8',
    )
    print(f'\n[+] Kết quả đã lưu ở {out.resolve()}')
    for name in ['reviews_with_topics.csv', 'topic_info.csv', 'aspect_candidates.csv', 'training_summary.txt']:
        print('    -', name)


def main():
    parser = argparse.ArgumentParser(description='BERTopic cho đánh giá sữa rửa mặt')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    parser.add_argument('--embedding-column', default='normalized_comment',
                        help='Ưu tiên cột chỉ định; tự dùng representative_comment cho CSV unique')
    parser.add_argument('--topic-column', default='topic_comment',
                        help='Ưu tiên cột chỉ định; tự dùng cleaned_comment cho CSV unique')
    parser.add_argument('--check-input', action='store_true', help='Kiểm tra CSV/cột mà không huấn luyện')
    parser.add_argument('--min-cluster-size', type=int, default=BASELINE_PARAMS['min_cluster_size'])
    parser.add_argument('--min-samples', type=int, default=BASELINE_PARAMS['min_samples'])
    parser.add_argument('--n-neighbors', type=int, default=BASELINE_PARAMS['n_neighbors'])
    parser.add_argument('--min-df', type=int, default=BASELINE_PARAMS['min_df'])
    parser.add_argument('--max-df', type=float, default=BASELINE_PARAMS['max_df'])
    parser.add_argument('--single-run', action='store_true', help='Chỉ fit một lần với các tham số CLI, không Grid Search')
    parser.add_argument('--grid-min-cluster-sizes', type=int, nargs='+', default=[15, 30, 50])
    parser.add_argument('--grid-min-samples', type=int, nargs='+', default=[3, 5, 10])
    parser.add_argument('--grid-n-neighbors', type=int, nargs='+', default=[10, 20, 30])
    parser.add_argument('--grid-min-df', type=int, nargs='+', help='Mặc định giữ --min-df')
    parser.add_argument('--grid-max-df', type=float, nargs='+', help='Mặc định giữ --max-df')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--coherence-sample', type=int, default=0,
                        help='0 = full corpus; N > 0 = lấy mẫu N reviews để tính C_v')
    parser.add_argument('--skip-coherence', action='store_true', help='Chỉ dùng với --single-run')
    args = parser.parse_args()
    if args.skip_coherence and not args.single_run:
        parser.error('Grid Search cần C_v để chọn tham số; --skip-coherence chỉ dùng với --single-run.')
    if args.coherence_sample < 0 or args.batch_size < 1:
        parser.error('--coherence-sample phải >= 0 và --batch-size phải >= 1.')

    df, embedding_col, topic_col = read_dataset(
        resolve_input_path(args.input), args.embedding_column, args.topic_column
    )
    print(f'[*] Embedding source: {embedding_col}; topic keyword source: {topic_col}')
    if 'text_group_id' in df.columns:
        print('[*] Mỗi dòng là một text duy nhất; giữ text_group_id/duplicate_count, không nhân bản theo tần suất.')
    print(f'[*] Top 3 mẫu embedding: {df["_embedding_text"].head(3).tolist()}')
    print(f'[*] Top 3 mẫu topic text: {df["_topic_text"].head(3).tolist()}')
    if args.check_input:
        print('[+] Đã kiểm tra xong cột và văn bản CSV.')
        if len(df) < max(10, args.min_cluster_size):
            print('[!] Tập đang kiểm tra quá nhỏ để chạy mô hình với min_cluster_size hiện tại.')
        else:
            print('[+] Có thể thử huấn luyện BERTopic với cấu hình hiện tại.')
        return

    if len(df) < 10:
        raise ValueError('Cần tối thiểu 10 review hợp lệ để chạy mô hình; dữ liệu thực tế nên lớn hơn nhiều.')
    reference = {key: getattr(args, key) for key in BASELINE_PARAMS}
    candidates = parameter_candidates(args, reference, len(df))
    if args.single_run:
        validate_params(reference, len(df))
    else:
        # Kiểm tra Gensim trước bước tạo embeddings.
        from gensim.models import CoherenceModel  # noqa: F401
        print(f'[*] Sẽ thử {len(candidates)} cấu hình, chọn theo C_v cao nhất.')

    out = resolve_output_path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'baseline_params.json', {
        'description': 'Cấu hình gốc trước khi thêm Grid Search, không bị thay bằng params thắng.',
        'params': BASELINE_PARAMS,
        'model_configuration': model_configuration(BASELINE_PARAMS),
        'batch_size': 32, 'coherence_sample': 0,
    })
    write_json(out / 'run_config.json', {
        'mode': 'single_run' if args.single_run else 'grid_search',
        'input': str(resolve_input_path(args.input).resolve()), 'valid_reviews': len(df),
        'embedding_column': embedding_col, 'topic_column': topic_col,
        'batch_size': args.batch_size, 'coherence_sample': args.coherence_sample,
        'skip_coherence': args.skip_coherence, 'reference_params': reference,
        'candidates': candidates,
        'selection': 'Maximum finite C_v; ties keep the first trial (baseline is tried first).',
    })

    from sentence_transformers import SentenceTransformer

    embed_docs = df['_embedding_text'].tolist()
    topic_docs = df['_topic_text'].tolist()
    print('[*] Tải SentenceTransformer:', EMBEDDING_MODEL_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print(f'[*] Tạo embeddings từ {embedding_col}...')
    embeddings = embedding_model.encode(
        embed_docs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    if args.single_run:
        model = create_model(reference, embedding_model, len(df))
        print(f'[*] Huấn luyện BERTopic trên {topic_col}, dùng embeddings từ {embedding_col}...')
        topics, probabilities = model.fit_transform(topic_docs, embeddings=embeddings)
        cv = None
        if not args.skip_coherence:
            try:
                cv = coherence_cv(model, topic_docs, args.coherence_sample)
                if cv is not None and not math.isfinite(cv):
                    cv = None
                if cv is not None:
                    print(f'[+] C_v: {cv:.4f}')
            except Exception as exc:
                print(f'[!] Không tính được C_v: {type(exc).__name__}: {exc}')
        write_json(out / 'single_run_params.json', {
            'params': reference, 'model_configuration': model_configuration(reference, len(df)), 'c_v': cv,
        })
    else:
        model, topics, probabilities, cv = grid_search(
            candidates, embedding_model, embeddings, topic_docs, args, out,
        )

    info = model.get_topic_info()
    print('\n' + info[['Topic', 'Count', 'Name']].to_string(index=False))
    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        words = [word for word, score in (model.get_topic(topic_id) or []) if word and score > 0]
        print(f'Topic {topic_id}: {", ".join(words)}')

    export_outputs(df, topic_docs, model, topics, probabilities, out, cv)


if __name__ == '__main__':
    main()
