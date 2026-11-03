"""BERTopic khám phá chủ đề trong reviews SỮA RỬA MẶT (Shopee, Lazada, Tiki).

Đầu vào từ preprocess_cleanser_reviews.py:
    data/cleanser_reviews_preprocessed.csv
    - normalized_comment: câu tự nhiên, thích hợp để tạo sentence embeddings
    - topic_comment: đã tách từ/ghép cụm _, dùng cho c-TF-IDF/keywords
    - cleaned_comment: fallback nếu không có topic_comment

Chạy:
    python bertopic_cleanser_aspects.py --check-input
    python bertopic_cleanser_aspects.py
    python bertopic_cleanser_aspects.py --input data/cleanser_reviews_preprocessed.csv --output-dir data/bertopic_cleanser

Coherence c_v tính trên analyzer của CountVectorizer (giữ cùng unigram/bigram).
Lưu ý: topics là các cụm ngữ nghĩa, KHÔNG tự động là aspect nhãn chuẩn.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = 'data/cleanser_reviews_preprocessed.csv'
DEFAULT_OUTPUT = 'data/bertopic_cleanser'
EMBEDDING_MODEL_NAME = 'keepitreal/vietnamese-sbert'

# Chỉ dọn từ hư / lời thoại và các từ chung quá phổ biến ở tầng *biểu diễn topic*.
# KHÔNG xóa phủ định/đánh giá hoặc chi tiết đặc trưng sữa rửa mặt.
# 'sữa_rửa_mặt' quá chung trong một corpus toàn sữa rửa mặt; bỏ riêng khỏi keywords.
TOPIC_STOPWORDS = {
    'mình', 'tôi', 'tui', 'em', 'anh', 'chị', 'bạn', 'mọi_người',
    'cho', 'với', 'để', 'thì', 'là', 'và', 'của', 'mà', 'có',
    'những', 'các', 'một', 'này', 'kia', 'đó', 'đây', 'ấy',
    'nha', 'nhé', 'ạ', 'ha', 'hihi', 'haha', 'hehe', 'kkk',
    'nói_chung', 'nói_chung_là', 'sản_phẩm', 'sp',
    'sữa_rửa_mặt', 'sữa', 'rửa_mặt', 'srm',
}

# Danh mục aspect dùng để GỢI Ý cho người nghiên cứu, không phải nhãn dự đoán ABSA.
# Từ khóa được lấy từ ngữ cảnh sữa rửa mặt, không giới hạn mỹ phẩm thuần chay.
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
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, BASE_DIR / path, BASE_DIR.parent / path]
    return next((p for p in candidates if p.exists()), candidates[0])


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
                     'Hãy chạy preprocess_cleanser_reviews.py trước.')


def read_dataset(input_path: Path, embedding_column: str, topic_column: str):
    if not input_path.is_file():
        raise FileNotFoundError(f'Không tìm thấy CSV: {input_path}')
    df = pd.read_csv(input_path, encoding='utf-8-sig', low_memory=False)
    e_col = choose_column(df, embedding_column, ('normalized_comment', 'comment'))
    t_col = choose_column(df, topic_column, ('topic_comment', 'cleaned_comment'))

    # Dùng cùng mask cho embedding + topic để thứ tự index gắn ngược chính xác.
    df = df.copy()
    df['_embedding_text'] = df[e_col].fillna('').astype(str).str.strip()
    df['_topic_text'] = df[t_col].fillna('').astype(str).str.strip()
    mask = df['_embedding_text'].str.len().ge(6) & df['_topic_text'].str.len().ge(3)
    valid = df.loc[mask].copy().reset_index(drop=True)
    print(f'[*] CSV: {len(df):,} dòng; đủ dữ liệu text: {len(valid):,} dòng; loại {len(df) - len(valid):,} dòng rỗng/ngắn.')
    if not len(valid):
        raise ValueError('Không có review hợp lệ sau khi lọc text.')
    return valid, e_col, t_col


def aspect_suggestions(topic_words: list[str]) -> str:
    """Tối đa hai gợi ý dựa trên trùng khớp từ khóa, KHÔNG là aspect nhãn chính thức."""
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
        # Coherence trên mẫu chỉ là ƯỚC TÍNH. Mặc định 0 = toàn bộ dữ liệu.
        sampled = pd.Series(docs).sample(n=max_docs, random_state=42).tolist()
        print(f'[*] C_v dùng mẫu {max_docs:,}/{len(docs):,} reviews (không phải full-corpus).')
    else:
        sampled = docs
        print(f'[*] C_v dùng toàn bộ {len(sampled):,} reviews.')

    analyzer = topic_model.vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(doc) for doc in sampled]
    tokenized_docs = [tokens for tokens in tokenized_docs if tokens]
    dictionary = corpora.Dictionary(tokenized_docs)
    # Có thể có từ top topic chưa có mặt nếu corpus đã bị sampling.
    filtered = [[w for w in words if w in dictionary.token2id] for words in topic_words]
    filtered = [words for words in filtered if len(words) >= 2]
    if not filtered:
        print('[!] Tập kiểm tra không chứa đủ keywords để tính C_v.')
        return None

    score = CoherenceModel(
        topics=filtered, texts=tokenized_docs, dictionary=dictionary,
        coherence='c_v', processes=1,  # thuận tiện trên Windows / PyCharm
    ).get_coherence()
    return float(score)


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
    raw_col = 'comment' if 'comment' in result.columns else (
        'normalized_comment' if 'normalized_comment' in result.columns else 'cleaned_comment'
    )
    for topic_text, raw in zip(docs_for_topic, result[raw_col].astype(str)):
        if len(text_to_raw[topic_text]) < 4 and raw not in text_to_raw[topic_text]:
            text_to_raw[topic_text].append(raw)

    rows = []
    for entry in info.itertuples(index=False):
        tid = int(entry.Topic)
        if tid == -1:
            continue
        scored_words = topic_model.get_topic(tid) or []
        words = [word for word, _ in scored_words]
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
        # Nếu không có representative docs, dùng bình luận thuộc cluster.
        if not representative_comments:
            representative_comments = result.loc[result['Topic'].eq(tid), raw_col].dropna().astype(str).head(3).tolist()
        rows.append({
            'Topic': tid,
            'Count': int(entry.Count),
            'Keywords': ', '.join(words),
            'Suggested_Aspect': aspect_suggestions(words),
            'Manual_Aspect': '',  # do nghiên cứu viên xác định
            'Representative_Review_1': representative_comments[0] if len(representative_comments) > 0 else '',
            'Representative_Review_2': representative_comments[1] if len(representative_comments) > 1 else '',
            'Representative_Review_3': representative_comments[2] if len(representative_comments) > 2 else '',
        })
    pd.DataFrame(rows, columns=[
        'Topic', 'Count', 'Keywords', 'Suggested_Aspect', 'Manual_Aspect',
        'Representative_Review_1', 'Representative_Review_2', 'Representative_Review_3',
    ]).to_csv(out / 'aspect_candidates.csv', index=False, encoding='utf-8-sig')

    (out / 'training_summary.txt').write_text(
        f'Input valid reviews: {len(result)}\n'
        f'Clusters (excluding -1): {len(rows)}\n'
        f'Outliers -1: {int(result["Topic"].eq(-1).sum())}\n'
        f'Outlier rate: {result["Topic"].eq(-1).mean():.4f}\n'
        f'C_v coherence: {coherence_score if coherence_score is not None else "N/A"}\n'
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
    parser.add_argument('--embedding-column', default='normalized_comment')
    parser.add_argument('--topic-column', default='topic_comment')
    parser.add_argument('--check-input', action='store_true', help='Kiểm tra CSV/cột mà không huấn luyện')
    parser.add_argument('--min-cluster-size', type=int, default=30)
    parser.add_argument('--min-samples', type=int, default=5)
    parser.add_argument('--n-neighbors', type=int, default=20)
    parser.add_argument('--min-df', type=int, default=3)
    parser.add_argument('--max-df', type=float, default=0.85)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--coherence-sample', type=int, default=0,
                        help='0 = full corpus; N > 0 = lấy mẫu N reviews để tính C_v')
    parser.add_argument('--skip-coherence', action='store_true')
    args = parser.parse_args()

    df, embedding_col, topic_col = read_dataset(
        resolve_input_path(args.input), args.embedding_column, args.topic_column
    )
    print(f'[*] Embedding source: {embedding_col}; topic keyword source: {topic_col}')
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
    if len(df) < args.min_cluster_size:
        raise ValueError(f'Số review hợp lệ ({len(df)}) phải >= min_cluster_size ({args.min_cluster_size}).')
    if args.min_df < 1 or not (0 < args.max_df <= 1):
        raise ValueError('min_df >= 1 và 0 < max_df <= 1')

    from sentence_transformers import SentenceTransformer
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    from umap import UMAP
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer

    embed_docs = df['_embedding_text'].tolist()
    topic_docs = df['_topic_text'].tolist()
    print('[*] Tải SentenceTransformer:', EMBEDDING_MODEL_NAME)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    print('[*] Tạo embeddings từ câu chuẩn hóa, KHÔNG underscore...')
    embeddings = embedding_model.encode(
        embed_docs,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    umap_model = UMAP(
        n_neighbors=min(args.n_neighbors, len(df) - 1),
        n_components=5, min_dist=0.0, metric='cosine', random_state=42,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        metric='euclidean', cluster_selection_method='eom',
        prediction_data=True,
    )
    vectorizer = CountVectorizer(
        stop_words=sorted(TOPIC_STOPWORDS),
        ngram_range=(1, 2),  # giữ cả bigrams cho từ chưa được ghép _
        token_pattern=r'(?u)\b\w\w+\b',
        min_df=args.min_df, max_df=args.max_df,
    )
    ctfidf = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
    model = BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        calculate_probabilities=False,  # tránh matrix lớn N x N_topics
        top_n_words=10,
        verbose=True,
    )
    print('[*] Huấn luyện BERTopic trên topic_comment, dùng embeddings từ normalized_comment...')
    topics, probabilities = model.fit_transform(topic_docs, embeddings=embeddings)

    info = model.get_topic_info()
    print('\n' + info[['Topic', 'Count', 'Name']].to_string(index=False))
    for topic_id in model.get_topics():
        if topic_id == -1:
            continue
        words = [word for word, _ in (model.get_topic(topic_id) or [])]
        print(f'Topic {topic_id}: {", ".join(words)}')

    cv = None
    if not args.skip_coherence:
        print('\n[*] Tính C_v coherence...')
        try:
            cv = coherence_cv(model, topic_docs, args.coherence_sample)
            if cv is not None:
                print(f'[+] C_v: {cv:.4f}')
        except Exception as exc:
            print(f'[!] Không tính được C_v: {type(exc).__name__}: {exc}')

    export_outputs(df, topic_docs, model, topics, probabilities,
                   resolve_output_path(args.output_dir), cv)


if __name__ == '__main__':
    main()
