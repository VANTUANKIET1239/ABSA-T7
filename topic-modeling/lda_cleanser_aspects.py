"""LDA TOPIC MODELING cho review SỮA RỬA MẶT (Shopee / Lazada / Tiki).

Đầu vào mặc định: bộ text duy nhất, cột cleaned_comment (giữ dấu _).

Các bước:
  1) Lọc văn bản hợp lệ, vectorize từ + bigram theo chuẩn tương tự BERTopic.
  2) Chọn K theo C_v với hyperparameters cố định, rồi tuning tại K đã chọn.
  3) Lưu topic words, document-topic mixtures, đại diện review, biểu đồ, model.

Chạy:
  python lda_cleanser_aspects.py --check-input
  python lda_cleanser_aspects.py
  python lda_cleanser_aspects.py --grid-passes 10 20 --grid-iterations 100 200
  python lda_cleanser_aspects.py --grid-alpha symmetric --grid-eta auto

Cài: pip install pandas numpy scikit-learn gensim matplotlib scipy
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd


# ===================== CẤU HÌNH MẶC ĐỊNH =====================
BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = 'crawl-data/data/unique_texts/unique_texts.csv'
DEFAULT_OUTPUT = str(BASE_DIR / 'data' / 'lda_cleanser')
RANDOM_STATE = 42
BASELINE_PARAMS = {
    'num_topics': 6, 'alpha': 'symmetric', 'eta': 'auto',
    'passes': 20, 'iterations': 200,
}
FIXED_PARAMS = {
    'random_state': RANDOM_STATE, 'chunksize': 2000,
    'update_every': 0, 'minimum_probability': 0.0,
}
DEFAULT_GRID_ALPHA = ['symmetric', 'asymmetric', 'auto']
DEFAULT_GRID_ETA = ['auto', 'symmetric']

# Tương ứng TOPIC_STOPWORDS của bertopic_cleanser_aspects.py.
# Chỉ bỏ từ rỗng nghĩa + tên loại sản phẩm vốn xuất hiện khắp corpus.
# KHÔNG bỏ 'không', 'chưa', 'khô_căng', 'kích_ứng', 'giá_cả',
# 'hiệu_quả', 'phù_hợp'... vì đây là tín hiệu để khám phá aspect.
TOPIC_STOPWORDS = {
    'mình', 'tôi', 'tui', 'em', 'anh', 'chị', 'bạn', 'mọi_người',
    'cho', 'với', 'để', 'thì', 'là', 'và', 'của', 'mà', 'có',
    'những', 'các', 'một', 'này', 'kia', 'đó', 'đây', 'ấy',
    'nha', 'nhé', 'ạ', 'ha', 'hihi', 'haha', 'hehe', 'kkk',
    'nói_chung', 'nói_chung_là', 'sản_phẩm', 'sp',
    'sữa_rửa_mặt', 'sữa', 'rửa_mặt', 'srm',
}

# KHÔNG coi các nhãn này là ground truth: chỉ giúp người nghiên cứu
# đọc topic + bình luận mẫu và đặt aspect bằng tay.
ASPECT_KEYWORDS = {
    'Cleansing_Effectiveness': {
        'làm_sạch', 'làm_sạch_sâu', 'sạch', 'sạch_sâu', 'không_sạch',
        'rửa_sạch', 'bụi_bẩn', 'bã_nhờn', 'kiềm_dầu', 'kiểm_soát_dầu',
        'dầu_thừa', 'lỗ_chân_lông', 'nhờn', 'sạch_da',
    },
    'Hydration_Skin_Comfort': {
        'dưỡng_ẩm', 'cấp_ẩm', 'giữ_ẩm', 'khô_căng', 'không_khô_căng',
        'khô_da', 'căng_rít', 'căng_da', 'mềm_mịn', 'mịn_màng',
        'ẩm_mượt', 'dịu_nhẹ', 'dễ_chịu',
    },
    'Safety_Skin_Compatibility': {
        'kích_ứng', 'không_kích_ứng', 'châm_chích', 'rát_da',
        'nóng_rát', 'cay_mắt', 'ngứa_da', 'đỏ_da', 'mẩn_đỏ',
        'nổi_mụn', 'lên_mụn', 'đẩy_mụn', 'da_nhạy_cảm',
        'da_dầu', 'da_khô', 'da_mụn', 'không_gây_mụn',
    },
    'Foaming_Texture': {
        'tạo_bọt', 'bọt_mịn', 'ít_bọt', 'nhiều_bọt', 'bọt',
        'kết_cấu', 'dạng_gel', 'dạng_kem', 'gel', 'lỏng',
        'đặc', 'dễ_rửa', 'khó_rửa',
    },
    'Fragrance': {
        'mùi_hương', 'mùi_thơm', 'không_mùi', 'mùi_hắc',
        'mùi_hóa_chất', 'hương_liệu', 'mùi', 'thơm', 'hắc', 'nồng',
    },
    'Ingredients_Formulation': {
        'thành_phần', 'độ_ph', 'cân_bằng_độ_ph', 'salicylic_acid',
        'niacinamide', 'hyaluronic_acid', 'tràm_trà', 'rau_má',
        'cồn', 'sulfate', 'ph',
    },
    'Packaging': {
        'bao_bì', 'đóng_gói', 'chai_lọ', 'nắp_chai',
        'vòi_bơm', 'tem_nhãn', 'niêm_phong', 'rò_rỉ',
        'đổ_vỡ', 'chai', 'tuýp',
    },
    'Price_Value': {
        'giá_cả', 'giá_thành', 'giá_rẻ', 'giá_cao',
        'giảm_giá', 'mã_giảm_giá', 'đáng_tiền',
        'đắt', 'rẻ', 'giá', 'sale',
    },
    'Service_Delivery': {
        'giao_hàng', 'giao_nhanh', 'giao_chậm', 'vận_chuyển',
        'giao', 'shipper', 'chăm_sóc_khách_hàng',
        'cửa_hàng', 'tư_vấn', 'đổi_trả', 'dịch_vụ',
    },
}

# Tùy chọn loại token tên hãng khỏi bag-of-words (không loại review/brand).
# Mặc định TẮT: nên so sánh cả hai cấu hình nếu topic bị chi phối bởi brand.
BRAND_TOKENS = {
    'hada_labo', 'hadalabo', 'la_roche_posay', 'larocheposay',
    'cetaphil', 'acnes', 'cerave', 'simple', 'nivea', 'nivea_men',
    'bioderma', 'hatomugi', 'senka', 'thorakao', 'eucerin',
    'cosrx', 'garnier', 'vichy', 'cocoon', 'biore', 'oxy',
    'naruko', 'uriage', 'himalaya', 'kiehls', 'kiehl',
}


def resolve_input(name: str) -> Path:
    path = Path(name).expanduser()
    if path.is_absolute():
        return path
    candidates = [Path.cwd() / path, BASE_DIR / path, BASE_DIR.parent / path]
    return next((p for p in candidates if p.exists()), candidates[0])


def resolve_output(name: str) -> Path:
    path = Path(name).expanduser()
    return path if path.is_absolute() else Path.cwd() / path


def load_dataset(path: Path, requested_column: str) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise FileNotFoundError(f'Không thấy file CSV: {path}. Chạy preprocess_cleanser_reviews.py trước.')
    df = pd.read_csv(path, encoding='utf-8-sig', low_memory=False)
    candidates = list(dict.fromkeys([requested_column, 'topic_comment', 'cleaned_comment']))
    column = next((col for col in candidates if col in df.columns), None)
    if column is None:
        raise ValueError(f'Không có cột {candidates}. Các cột hiện có: {list(df.columns)}')
    if column != requested_column:
        print(f'[!] Không có cột {requested_column!r}; dùng {column!r}.')
    # Lưu vị trí dòng gốc để người nghiên cứu tra cứu.
    df = df.copy()
    df['LDA_Source_Row'] = np.arange(len(df), dtype=int)
    df['_lda_text'] = df[column].fillna('').astype(str).str.strip().str.lower()
    df = df.loc[df['_lda_text'].str.len().ge(3)].copy()
    if df.empty:
        raise ValueError('Không có review có nội dung hợp lệ sau khi lọc text rỗng.')
    return df, column


def build_tokens(df: pd.DataFrame, *, min_df: int, max_df: float,
                 ngram_max: int, remove_brand_tokens: bool):
    from sklearn.feature_extraction.text import CountVectorizer

    stops = set(TOPIC_STOPWORDS)
    if remove_brand_tokens:
        stops.update(BRAND_TOKENS)
    # Giữ dấu _ trong token; bigrams của scikit-learn chứa khoảng trắng,
    # nhưng cả Gensim Dictionary và Coherence đều dùng đúng các chuỗi đó.
    vectorizer = CountVectorizer(
        stop_words=sorted(stops),
        ngram_range=(1, ngram_max),
        token_pattern=r'(?u)\b\w\w+\b',
        min_df=min_df,
        max_df=max_df,
    )
    docs = df['_lda_text'].tolist()
    try:
        vectorizer.fit(docs)
    except ValueError as exc:
        raise ValueError('Vocabulary rỗng: giảm --min-df hoặc tăng --max-df, '
                         'đồng thời kiểm tra cột topic_comment.') from exc
    vocabulary = set(vectorizer.get_feature_names_out())
    analyzer = vectorizer.build_analyzer()
    tokens = [[w for w in analyzer(doc) if w in vocabulary] for doc in docs]
    valid_mask = np.array([bool(words) for words in tokens], dtype=bool)
    kept = df.iloc[np.flatnonzero(valid_mask)].copy().reset_index(drop=True)
    kept_tokens = [words for words, valid in zip(tokens, valid_mask) if valid]
    print(f'[*] Review có text: {len(df):,}; đưa vào LDA: {len(kept):,}; '
          f'bỏ do không còn token: {int((~valid_mask).sum()):,}.')
    print(f'[*] Vocab size: {len(vocabulary):,}; n-grams = (1, {ngram_max}).')
    if not kept_tokens:
        raise ValueError('Không còn văn bản để huấn luyện LDA sau lọc từ vựng.')
    return kept, kept_tokens, vectorizer


def aspect_suggestions(words: list[str]) -> str:
    # Với n-gram, xét cả cụm và thành phần token; chỉ mang tính gợi ý.
    present = set(words)
    for phrase in words:
        present.update(phrase.split(' '))
    ranked = sorted(
        ((len(present & cues), aspect) for aspect, cues in ASPECT_KEYWORDS.items()),
        key=lambda item: (-item[0], item[1]),
    )
    matches = [aspect for count, aspect in ranked if count > 0]
    return '; '.join(matches[:2]) if matches else 'Needs_manual_review'


def topic_diversity(model, topn: int) -> float:
    words = [word for topic in range(model.num_topics)
             for word, _ in model.show_topic(topic, topn=topn)]
    return len(set(words)) / len(words) if words else float('nan')


def parse_prior(value: str):
    if value in {'symmetric', 'asymmetric', 'auto'}:
        return value
    try:
        prior = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Prior cần là symmetric/asymmetric/auto hoặc số dương.') from exc
    if not math.isfinite(prior) or prior <= 0:
        raise argparse.ArgumentTypeError('Prior dạng số phải hữu hạn và > 0.')
    return prior


def k_candidates(args) -> list[dict]:
    return [{
        'num_topics': k, 'alpha': args.k_alpha, 'eta': args.k_eta,
        'passes': args.passes, 'iterations': args.iterations,
    } for k in range(args.min_k, args.max_k + 1, args.k_step)]


def parameter_candidates(args, selected_k: int) -> list[dict]:
    dimensions = {
        'num_topics': [selected_k],
        'alpha': args.grid_alpha,
        'eta': args.grid_eta,
        'passes': args.grid_passes if args.grid_passes is not None else [args.passes],
        'iterations': args.grid_iterations if args.grid_iterations is not None else [args.iterations],
    }
    candidates = []
    for values in product(*(list(dict.fromkeys(values)) for values in dimensions.values())):
        params = dict(zip(dimensions, values))
        # Gensim không ước lượng alpha='auto' ổn định khi chỉ có một topic.
        if params['num_topics'] == 1 and params['alpha'] == 'auto':
            continue
        candidates.append(params)
    return candidates


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def run_grid_search(candidates, corpus, dictionary, tokenized_docs,
                    output_dir: Path, *, topn: int, skip_perplexity: bool,
                    results_filename: str = 'lda_grid_search_results.csv',
                    stage: str | None = None, previous_results=None):
    from gensim.models import CoherenceModel, LdaModel

    scores = []
    best_score = -float('inf')
    best_model = best_params = None
    for trial, params in enumerate(candidates, 1):
        print(f'\n[*] Trial {trial}/{len(candidates)}: {params}', flush=True)
        started = perf_counter()
        row = {'Trial': trial, 'K': params['num_topics'],
               **{key: value for key, value in params.items() if key != 'num_topics'},
               'Coherence_Cv': float('nan'), 'Log_Perplexity': float('nan'),
               'Topic_Diversity': float('nan'), 'Status': 'ok', 'Error': ''}
        if stage is not None:
            row['Stage'] = stage
        try:
            model = LdaModel(corpus=corpus, id2word=dictionary, **params, **FIXED_PARAMS)
            if (not np.isfinite(model.alpha).all() or not np.isfinite(model.eta).all()
                    or not np.isfinite(model.get_topics()).all()):
                raise FloatingPointError('Model có alpha/eta/topic probabilities không hữu hạn.')
            topic_words = [[w for w, _ in model.show_topic(tid, topn=topn)]
                           for tid in range(model.num_topics)]
            coherence = float(CoherenceModel(
                topics=topic_words, texts=tokenized_docs, dictionary=dictionary,
                coherence='c_v', processes=1,
            ).get_coherence())
            row['Coherence_Cv'] = coherence
            row['Topic_Diversity'] = topic_diversity(model, topn)
            if not skip_perplexity:
                row['Log_Perplexity'] = float(model.log_perplexity(corpus))
            if math.isfinite(coherence):
                if coherence > best_score:
                    best_score, best_model, best_params = coherence, model, params.copy()
            else:
                row.update(Status='invalid_coherence', Error='C_v không hữu hạn.')
        except (ValueError, RuntimeError, ArithmeticError) as exc:
            row.update(Status='failed', Error=f'{type(exc).__name__}: {exc}')
        row['Elapsed_Seconds'] = perf_counter() - started
        scores.append(row)
        current_results = pd.DataFrame(scores)
        current_results.to_csv(output_dir / results_filename, index=False, encoding='utf-8-sig')
        if results_filename != 'lda_grid_search_results.csv':
            combined = (pd.concat([previous_results, current_results], ignore_index=True)
                        if previous_results is not None else current_results)
            combined.to_csv(output_dir / 'lda_grid_search_results.csv',
                            index=False, encoding='utf-8-sig')
        print(f'    C_v={row["Coherence_Cv"]:.4f}; diversity={row["Topic_Diversity"]:.4f}; '
              f'status={row["Status"]}; {row["Elapsed_Seconds"]:.1f}s', flush=True)
        if row['Error']:
            print(f'    [!] {row["Error"]}', flush=True)
    if best_model is None:
        raise RuntimeError(f'Không có cấu hình có C_v hợp lệ. Xem {results_filename}.')
    return best_model, best_params, best_score, pd.DataFrame(scores)


def run_two_stage_search(args, corpus, dictionary, tokenized_docs, output_dir: Path,
                         run_config: dict):
    print('\n[*] BƯỚC 1: Chọn K với alpha/eta/passes/iterations cố định.', flush=True)
    k_model, selected_params, k_score, k_results = run_grid_search(
        k_candidates(args), corpus, dictionary, tokenized_docs, output_dir,
        topn=args.top_n_words, skip_perplexity=args.skip_perplexity,
        results_filename='lda_k_coherence_results.csv', stage='k_search')
    selected_k = selected_params['num_topics']
    valid_k_results = k_results.loc[k_results['Status'].eq('ok') &
                                    np.isfinite(k_results['Coherence_Cv'])]
    selection = {
        'selected_k': selected_k, 'coherence_cv': k_score,
        'params': selected_params, 'fixed_params': FIXED_PARAMS,
        'trial': int(valid_k_results.loc[valid_k_results['Coherence_Cv'].idxmax(), 'Trial']),
    }
    write_json(output_dir / 'k_selection.json', selection)
    del k_model
    print(f'\n[+] Bước 1 chọn K={selected_k}; C_v={k_score:.4f}.', flush=True)

    candidates = parameter_candidates(args, selected_k)
    if selected_params not in candidates:
        candidates.insert(0, selected_params.copy())
    run_config['hyperparameter_search'].update(selected_k=selected_k, candidates=candidates)
    write_json(output_dir / 'run_config.json', run_config)
    print(f'[*] BƯỚC 2: Giữ K={selected_k}, thử {len(candidates)} cấu hình hyperparameters.', flush=True)
    if selected_k == 1 and 'auto' in args.grid_alpha:
        print('[*] Bỏ alpha=auto tại K=1 để tránh lỗi số học trong Gensim.', flush=True)
    model, params, score, tuning_results = run_grid_search(
        candidates, corpus, dictionary, tokenized_docs, output_dir,
        topn=args.top_n_words, skip_perplexity=args.skip_perplexity,
        results_filename='lda_hyperparameter_results.csv', stage='hyperparameter_search',
        previous_results=k_results)
    results = pd.concat([k_results, tuning_results], ignore_index=True)
    return model, params, score, results, selection


def write_figures(results_df: pd.DataFrame, best_k: int, best_model,
                  output_dir: Path, topn: int) -> None:
    # Agg: chạy được trên terminal, PyCharm, VPS không GUI.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 5))
    k_results = results_df.loc[results_df['Stage'].eq('k_search')]
    valid = k_results.loc[k_results['Status'].eq('ok') &
                          np.isfinite(k_results['Coherence_Cv'])]
    per_k = valid.groupby('K', as_index=False)['Coherence_Cv'].max().sort_values('K')
    ax.plot(per_k['K'], per_k['Coherence_Cv'], marker='o')
    score = float(per_k.loc[per_k['K'].eq(best_k), 'Coherence_Cv'].iloc[0])
    ax.scatter([best_k], [score], s=90, zorder=5)
    ax.annotate(f'Selected K={best_k}\nStage 1 C_v={score:.4f}',
                (best_k, score), xytext=(9, 12), textcoords='offset points')
    ax.set(xlabel='Number of Topics (K)', ylabel='Coherence (C_v)',
           title='LDA – Stage 1: Select K with Fixed Hyperparameters')
    ax.set_xticks(sorted(k_results['K'].unique().astype(int).tolist()))
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / 'lda_coherence_vs_k.png', dpi=200)
    plt.close(fig)

    cols = 3
    rows = math.ceil(best_k / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(16, max(4, 3.8 * rows)))
    axes = np.atleast_1d(axes).ravel()
    for topic_id in range(best_k):
        words_scores = best_model.show_topic(topic_id, topn=topn)
        words = [w for w, _ in words_scores][::-1]
        scores = [float(s) for _, s in words_scores][::-1]
        axes[topic_id].barh(words, scores)
        axes[topic_id].set_title(f'Topic {topic_id}')
        axes[topic_id].set_xlabel('Word probability')
    for ax in axes[best_k:]:
        ax.axis('off')
    fig.suptitle(f'LDA – Facial Cleanser Topic Keywords (K={best_k})')
    fig.tight_layout()
    fig.savefig(output_dir / 'lda_topic_words.png', dpi=200)
    plt.close(fig)


def export_best(best_model, dictionary, corpus, df: pd.DataFrame,
                output_dir: Path, topn: int, representative_n: int) -> None:
    topic_rows = []
    aspect_rows = []
    doc_distributions: list[list[tuple[int, float]]] = []

    for tid in range(best_model.num_topics):
        terms = best_model.show_topic(tid, topn=topn)
        words = [w for w, _ in terms]
        print(f'[Topic {tid:02d}] ' + ', '.join(words))
        for rank, (word, weight) in enumerate(terms, 1):
            topic_rows.append({'Topic': tid, 'Rank': rank, 'Word': word,
                               'Weight': float(weight)})
        aspect_rows.append({'Topic': tid, 'Keywords': ', '.join(words),
                            'Suggested_Aspect': aspect_suggestions(words),
                            'Manual_Aspect': ''})

    # Gensim trả hỗn hợp topic của mỗi document: không ép rằng mỗi
    # review thực sự chỉ có một khía cạnh.
    dominant = []
    dominant_prob = []
    runner_up = []
    runner_up_prob = []
    third = []
    third_prob = []
    for bow in corpus:
        distribution = sorted(best_model.get_document_topics(
            bow, minimum_probability=0.0), key=lambda pair: pair[1], reverse=True)
        doc_distributions.append(distribution)
        dominant.append(int(distribution[0][0]))
        dominant_prob.append(float(distribution[0][1]))
        runner_up.append(int(distribution[1][0]) if len(distribution) > 1 else pd.NA)
        runner_up_prob.append(float(distribution[1][1]) if len(distribution) > 1 else pd.NA)
        third.append(int(distribution[2][0]) if len(distribution) > 2 else pd.NA)
        third_prob.append(float(distribution[2][1]) if len(distribution) > 2 else pd.NA)

    df = df.drop(columns=['_lda_text']).copy()
    df['LDA_Topic'] = dominant
    df['LDA_Topic_Probability'] = dominant_prob
    df['LDA_Second_Topic'] = runner_up
    df['LDA_Second_Probability'] = runner_up_prob
    df['LDA_Third_Topic'] = third
    df['LDA_Third_Probability'] = third_prob
    df.to_csv(output_dir / 'lda_document_topics.csv', index=False, encoding='utf-8-sig')

    distribution = (df['LDA_Topic'].value_counts().sort_index()
                    .rename_axis('Topic').reset_index(name='Count'))
    distribution['Percentage'] = (100 * distribution['Count'] / len(df)).round(2)
    distribution.to_csv(output_dir / 'lda_topic_distribution.csv',
                        index=False, encoding='utf-8-sig')

    raw_column = next((c for c in ('comment', 'normalized_comment', 'cleaned_comment',
                                    'topic_comment') if c in df.columns), None)
    for row in aspect_rows:
        topic_id = row['Topic']
        subset = (df.loc[df['LDA_Topic'].eq(topic_id)]
                    .sort_values('LDA_Topic_Probability', ascending=False))
        row['Count'] = len(subset)
        comments: list[str] = []
        if raw_column is not None:
            for val in subset[raw_column].dropna().astype(str):
                if val and val not in comments:
                    comments.append(val)
                if len(comments) >= representative_n:
                    break
        for i in range(representative_n):
            row[f'Representative_Review_{i + 1}'] = (
                comments[i] if i < len(comments) else '')
    aspect_df = pd.DataFrame(aspect_rows)
    aspect_df.to_csv(output_dir / 'lda_aspect_candidates.csv',
                     index=False, encoding='utf-8-sig')
    pd.DataFrame(topic_rows).to_csv(output_dir / 'lda_best_topic_words.csv',
                                   index=False, encoding='utf-8-sig')

    best_model.save(str(output_dir / 'best_lda_model'))
    dictionary.save(str(output_dir / 'lda_dictionary'))
    print('[+] Xuất file review + topic, keywords, aspect candidates, phân bố và model.')


def main() -> None:
    parser = argparse.ArgumentParser(description='LDA hai bước: chọn K theo C_v, rồi tuning hyperparameters tại K đó')
    parser.add_argument('--input', default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT)
    parser.add_argument('--text-column', default='cleaned_comment')
    parser.add_argument('--check-input', action='store_true')
    parser.add_argument('--min-k', type=int, default=1)
    parser.add_argument('--max-k', type=int, default=20)
    parser.add_argument('--k-step', type=int, default=1,
                        help='Mặc định thử mọi K từ 1 đến 20.')
    parser.add_argument('--min-df', type=int, default=3)
    parser.add_argument('--max-df', type=float, default=0.85)
    parser.add_argument('--ngram-max', type=int, default=2)
    parser.add_argument('--passes', type=int, default=BASELINE_PARAMS['passes'])
    parser.add_argument('--iterations', type=int, default=BASELINE_PARAMS['iterations'])
    parser.add_argument('--k-alpha', type=parse_prior, default=BASELINE_PARAMS['alpha'],
                        help='Alpha cố định khi chọn K; mặc định symmetric.')
    parser.add_argument('--k-eta', type=parse_prior, default=BASELINE_PARAMS['eta'],
                        help='Eta cố định khi chọn K; mặc định auto.')
    parser.add_argument('--grid-alpha', type=parse_prior, nargs='+', default=DEFAULT_GRID_ALPHA,
                        help='Mặc định: symmetric asymmetric auto; cũng nhận số dương.')
    parser.add_argument('--grid-eta', type=parse_prior, nargs='+', default=DEFAULT_GRID_ETA,
                        help='Mặc định: auto symmetric; cũng nhận số dương.')
    parser.add_argument('--grid-passes', type=int, nargs='+',
                        help='Ví dụ: 10 20; nếu không truyền, dùng --passes.')
    parser.add_argument('--grid-iterations', type=int, nargs='+',
                        help='Ví dụ: 100 200; nếu không truyền, dùng --iterations.')
    parser.add_argument('--top-n-words', type=int, default=10)
    parser.add_argument('--representative-n', type=int, default=3)
    parser.add_argument('--remove-brand-tokens', action='store_true',
                        help='Bỏ token tên hãng ra khỏi từ vựng topic (không bỏ review).')
    parser.add_argument('--skip-perplexity', action='store_true',
                        help='Không tính log perplexity để huấn luyện nhanh hơn.')
    parser.add_argument('--skip-figures', action='store_true',
                        help='Không vẽ biểu đồ; không cần matplotlib.')
    args = parser.parse_args()

    if not (1 <= args.min_k <= args.max_k and args.k_step >= 1):
        parser.error('Cần 1 <= min-k <= max-k và k-step >= 1.')
    if args.min_df < 1 or not (0 < args.max_df <= 1):
        parser.error('min-df >= 1 và 0 < max-df <= 1.')
    if args.ngram_max not in (1, 2):
        parser.error('ngram-max cần là 1 hoặc 2.')
    if args.passes < 1 or args.iterations < 1 or args.top_n_words < 2 or args.representative_n < 1:
        parser.error('passes/iterations/representative-n >= 1; top-n-words >= 2.')
    if any(value < 1 for value in (args.grid_passes or []) + (args.grid_iterations or [])):
        parser.error('Mọi giá trị grid-passes/grid-iterations phải >= 1.')
    if 'asymmetric' in args.grid_eta or args.k_eta == 'asymmetric':
        parser.error('eta chỉ nhận auto, symmetric hoặc số dương; không nhận asymmetric.')
    if args.min_k == 1 and args.k_alpha == 'auto':
        parser.error('K=1 không hỗ trợ alpha=auto ổn định. Đổi --k-alpha hoặc tăng --min-k.')
    first_candidates = k_candidates(args)
    ks = [params['num_topics'] for params in first_candidates]

    data_path = resolve_input(args.input)
    df, column = load_dataset(data_path, args.text_column)
    print(f'[*] File: {data_path.resolve()}')
    print(f'[*] Tổng số review có text: {len(df):,}; dùng cột: {column!r}')
    print('[*] Sample:', df['_lda_text'].iloc[:2].tolist())
    print(f'[*] Bước 1: {len(first_candidates)} cấu hình, K={ks}; '
          f'alpha={args.k_alpha}, eta={args.k_eta}, passes={args.passes}, iterations={args.iterations}.')
    print('[*] Bước 2: chỉ tuning hyperparameters tại K thắng bước 1.')
    if args.check_input:
        print('[+] Đầu vào hợp lệ. Không cần tải Gensim/huấn luyện khi dùng --check-input.')
        return

    from gensim import corpora
    if not args.skip_figures:
        try:
            import matplotlib  # noqa: F401
        except ImportError as exc:
            raise ImportError('Thiếu matplotlib trong Python đang chạy. Cài bằng '
                              f'"{sys.executable}" -m pip install matplotlib '
                              'hoặc dùng --skip-figures.') from exc

    df, tokenized_docs, vectorizer = build_tokens(
        df, min_df=args.min_df, max_df=args.max_df,
        ngram_max=args.ngram_max, remove_brand_tokens=args.remove_brand_tokens)
    if len(df) < max(ks):
        raise ValueError('Số review hợp lệ phải >= max K. Giảm --max-k.')
    dictionary = corpora.Dictionary(tokenized_docs)
    corpus = [dictionary.doc2bow(words) for words in tokenized_docs]
    if len(dictionary) < 2:
        raise ValueError('C_v cần ít nhất 2 từ trong vocabulary. Giảm min-df hoặc tăng max-df.')
    if len(dictionary) < max(ks):
        print(f'[!] Vocabulary size ({len(dictionary)}) ít hơn max K ({max(ks)}). '
              'Nên giảm max K hoặc giảm min_df.')

    output = resolve_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'baseline_params.json', {
        'description': 'Cấu hình LDA gốc trước Grid Search; K gốc là 6,8,...,20.',
        'params': BASELINE_PARAMS, 'fixed_params': FIXED_PARAMS,
        'original_k_range': list(range(6, 21, 2)),
        'vectorizer': {'min_df': 3, 'max_df': 0.85, 'ngram_max': 2,
                       'remove_brand_tokens': False},
    })
    run_config = {
        'search_strategy': 'k_then_hyperparameters',
        'input': str(data_path.resolve()), 'text_column': column,
        'python_executable': sys.executable, 'valid_documents': len(df),
        'vocabulary_size': len(dictionary),
        'k_search': {'candidates': first_candidates},
        'hyperparameter_search': {
            'selected_k': None, 'candidates': None,
            'grid_alpha': args.grid_alpha, 'grid_eta': args.grid_eta,
            'grid_passes': args.grid_passes if args.grid_passes is not None else [args.passes],
            'grid_iterations': args.grid_iterations if args.grid_iterations is not None else [args.iterations],
            'include_stage1_winner': True,
        },
        'fixed_params': FIXED_PARAMS, 'top_n_words': args.top_n_words,
        'skip_perplexity': args.skip_perplexity,
        'coherence': {'metric': 'c_v', 'processes': 1, 'documents': len(tokenized_docs)},
        'vectorizer': {'min_df': args.min_df, 'max_df': args.max_df,
                       'ngram_max': args.ngram_max, 'remove_brand_tokens': args.remove_brand_tokens},
        'selection': ('Stage 1: maximum finite C_v with fixed hyperparameters selects K. '
                      'Stage 2: maximum finite C_v at that K selects hyperparameters; '
                      'include the stage 1 winner. Ties keep the first trial.'),
    }
    write_json(output / 'run_config.json', run_config)
    print('[*] Python:', sys.executable)
    print('[*] Gensim LDA dùng CPU; mọi trial dùng cùng corpus và từ vựng.')
    if args.remove_brand_tokens:
        print('[*] Đã loại token tên brand khỏi bag-of-words (metadata brand vẫn giữ).')

    best_model, best_params, best_score, results_df, k_selection = run_two_stage_search(
        args, corpus, dictionary, tokenized_docs, output, run_config)
    best_k = best_params['num_topics']
    valid_results = results_df.loc[results_df['Status'].eq('ok') &
                                   np.isfinite(results_df['Coherence_Cv'])]
    k_results = results_df.loc[results_df['Stage'].eq('k_search')]
    tuning_results = valid_results.loc[valid_results['Stage'].eq('hyperparameter_search')]
    write_json(output / 'best_params.json', {
        'params': best_params, 'fixed_params': FIXED_PARAMS, 'coherence_cv': best_score,
        'stage': 'hyperparameter_search',
        'trial': int(tuning_results.loc[tuning_results['Coherence_Cv'].idxmax(), 'Trial']),
        'k_selection': k_selection,
    })
    print('\n[*] Bảng C_v chọn K ở bước 1:\n', k_results.round(4).to_string(index=False))
    print(f'\n[+] Best params tại K={best_k}: {best_params}; C_v = {best_score:.4f}')

    export_best(best_model, dictionary, corpus, df,
                output, args.top_n_words, args.representative_n)
    if not args.skip_figures:
        write_figures(results_df, best_k, best_model, output, args.top_n_words)

    summary = (f'LDA – SỮA RỬA MẶT\n'
               f'Input: {data_path.resolve()}\n'
               f'Text column: {column}\n'
               f'Valid LDA documents: {len(df):,}\n'
               f'Vocabulary: {len(dictionary):,}\n'
               'Search strategy: select K first, then tune hyperparameters at fixed K.\n'
               f'K tested: {ks}\n'
               f'Stage 1 selected K: {k_selection["selected_k"]}; C_v: {k_selection["coherence_cv"]:.6f}\n'
               f'Stage 1 fixed params: {k_selection["params"]}\n'
               f'Stage 1 trials: {len(k_results)}; stage 2 trials: {len(results_df) - len(k_results)}\n'
               f'Trials: {len(results_df)}; successful: {len(valid_results)}\n'
               f'Best K: {best_k}\n'
               f'Best params: {best_params}\n'
               f'Best C_v: {best_score:.6f}\n'
               f'Topic diversity (best): {topic_diversity(best_model, args.top_n_words):.6f}\n'
               f'n-gram: (1,{args.ngram_max})\n'
               f'min_df: {args.min_df}, max_df: {args.max_df}\n'
               f'passes: {best_params["passes"]}, iterations: {best_params["iterations"]}\n'
               f'remove_brand_tokens: {args.remove_brand_tokens}\n'
               'Suggested_Aspect không phải nhãn ABSA; cần kiểm định thủ công.\n'
               'LDA là mô hình topic hỗn hợp; một review có thể liên quan nhiều chủ đề.\n'
               'C_v chỉ so sánh trực tiếp với BERTopic khi dùng cùng corpus, '
               'tokenization, n-grams, stopwords, và cách tính coherence.\n')
    (output / 'lda_training_summary.txt').write_text(summary, encoding='utf-8')
    print(f'\n[+] Hoàn tất. Kết quả: {output.resolve()}')
    for name in ('lda_grid_search_results.csv', 'lda_k_coherence_results.csv',
                 'lda_hyperparameter_results.csv', 'k_selection.json', 'best_params.json',
                 'baseline_params.json', 'run_config.json', 'lda_coherence_vs_k.png',
                 'lda_best_topic_words.csv', 'lda_document_topics.csv',
                 'lda_aspect_candidates.csv', 'lda_topic_distribution.csv',
                 'lda_topic_words.png', 'lda_training_summary.txt'):
        if (output / name).exists():
            print('    -', name)


if __name__ == '__main__':
    main()
