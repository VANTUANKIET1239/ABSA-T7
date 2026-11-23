"""Read canonical experiment outputs; derive descriptive statistics without copies."""
from functools import lru_cache
import json
import re

import numpy as np
import pandas as pd

from .config import ASPECTS, DATA, PROJECT

REVIEW_PATH = PROJECT / 'crawl-data/data/combined_reviews_brand_filled_language_filtered.csv'
IPA_ROOT = PROJECT / 'Business/customer_satisfaction_xai_ipa_results_cleanser'
CUTOFF_ROOT = PROJECT / 'Business/customer_satisfaction_xai_ipa_mean_median_results_cleanser'
PLATFORM_ROOT = PROJECT / 'Business/ipa_by_platform_results_cleanser'
BRAND_ROOT = PROJECT / 'Business/ipa_top5_brands_results_cleanser'
TOPIC_ROOT = PROJECT / 'topic-modeling/data'
EVALUATIONS = PROJECT / 'evaluation/results/cleanser_models_comparison/outputs'
HISTORY_PATH = PROJECT / 'train-models/Histories/Cleanser/training_history_birnn_cleanser_focal_loss_VI.csv'
SENTIMENT_ORDER = ['positive', 'negative', 'neutral']
PATTERN = re.compile(r'\{\s*([^#{}]+?)\s*#\s*([^{}]+?)\s*\}')
TOPIC_RUNS = [
    ('BERTopic', 'bertopic_16_topics_reproduced'),
    ('LDA (K tốt nhất)', 'lda_cleanser_bertopic_corpus'),
    ('LDA (K=16)', 'lda_cleanser_bertopic_corpus_k16'),
]
FILE_SOURCES = {
    '24_shap_aspect_summary': IPA_ROOT / 'xai_aspect_summary.csv',
    '25_ipa_overall': IPA_ROOT / 'ipa_attributes.csv',
    '22_rating_regression_oof': IPA_ROOT / 'model_comparison_oof.csv',
    '30_ipa_platform': PLATFORM_ROOT / 'ipa_all_groups.csv',
    '31_ipa_brand': BRAND_ROOT / 'ipa_all_groups.csv',
    '32_ipa_platform_summary': PLATFORM_ROOT / 'group_run_summary.csv',
    '33_ipa_brand_summary': BRAND_ROOT / 'group_run_summary.csv',
    '34_ipa_mean_median_comparison': CUTOFF_ROOT / 'ipa_mean_median_comparison.csv',
    '35_ipa_cutoff_comparison': CUTOFF_ROOT / 'ipa_mean_median_thresholds.csv',
}


def read(path):
    return pd.read_csv(path, encoding='utf-8-sig', keep_default_na=False)


def canonical_groups(series):
    names = series.astype(str).str.normalize('NFKC').str.split().str.join(' ')
    counts = pd.DataFrame({'key': names.str.casefold(), 'name': names}).value_counts().reset_index(name='n')
    preferred = (counts.sort_values(['key', 'n', 'name'], ascending=[True, False, True])
                 .drop_duplicates('key').set_index('key')['name'].to_dict())
    preferred.update({'lazada': 'Lazada', 'shopee': 'Shopee', 'tiki': 'Tiki'})
    return names.str.casefold().map(preferred)


@lru_cache(maxsize=1)
def reviews():
    data = read(REVIEW_PATH)
    valid = data.cleaned_comment.str.strip().ne('') & data.label.str.strip().ne('')
    data = data.loc[valid].copy().reset_index(drop=True)
    data['Platform'] = canonical_groups(data.platform)
    data['Brand'] = canonical_groups(data.brand)
    data['Whitespace_Tokens'] = data.cleaned_comment.str.split().str.len()
    data['rating'] = pd.to_numeric(data.rating, errors='raise')
    return data


@lru_cache(maxsize=1)
def label_matrix():
    matrix = np.zeros((len(reviews()), 11, 3), dtype=bool)
    for index, value in enumerate(reviews().label):
        for aspect, sentiment in set(PATTERN.findall(value)):
            if aspect in ASPECTS and sentiment in SENTIMENT_ORDER:
                matrix[index, ASPECTS.index(aspect), SENTIMENT_ORDER.index(sentiment)] = True
    return matrix


def sentiment_counts():
    matrix = label_matrix()
    result = pd.DataFrame(matrix.sum(axis=0), columns=SENTIMENT_ORDER)
    result.insert(0, 'Aspect', ASPECTS)
    result['Mentioned_Reviews'] = matrix.any(axis=2).sum(axis=0)
    result['Label_Pairs'] = matrix.sum(axis=(0, 2))
    return result


def platform_overview():
    result = reviews().groupby('Platform').agg(
        Reviews=('label', 'size'), Products=('product_url', 'nunique'), Brands=('Brand', 'nunique')
    ).reset_index()
    result['Share_Pct'] = 100 * result.Reviews / len(reviews())
    return result


def length_summary():
    data = reviews()
    rows = []
    for name, values in [('Toàn bộ', data)] + [(p, data.loc[data.Platform.eq(p)]) for p in ['Lazada', 'Shopee', 'Tiki']]:
        lengths = values.Whitespace_Tokens
        rows.append({'Group': name, 'Mean': lengths.mean(), 'P50': lengths.median(),
                     'P95': lengths.quantile(.95), 'Max': lengths.max(), 'N': len(lengths)})
    return pd.DataFrame(rows)


def split_counts():
    return pd.DataFrame([{'Split': split, 'N': len(read(PROJECT / f'train-models/data/{split}_cleanser.csv'))}
                         for split in ['train', 'val', 'test']])


def dataset_summary():
    return {'raw_reviews': len(read(PROJECT / 'crawl-data/data/combined_reviews.csv')),
            'valid_reviews': len(reviews()),
            'unique_texts': len(read(PROJECT / 'crawl-data/data/unique_texts/unique_texts.csv')),
            'split_counts': split_counts().set_index('Split').N.to_dict()}


def absa_metrics():
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


def aspect_comparison():
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


def topic_metrics():
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
    path = TOPIC_ROOT / 'bertopic_16_topics_reproduced'
    counts = read(path / 'topic_frequency.csv').set_index('Topic')
    words = read(path / 'topic_keyword_weights.csv')
    return pd.DataFrame([{'Topic': topic, 'Count': int(counts.loc[topic, 'Count']),
                          'Keywords': ', '.join(part.sort_values('Rank').head(6).Word.str.replace('_', ' ', regex=False))}
                         for topic, part in words.loc[words.Topic.ge(0)].groupby('Topic', sort=True)])


def load(name):
    if name in FILE_SOURCES:
        result = read(FILE_SOURCES[name])
        if name == '22_rating_regression_oof' and result.PRED25.max() > 1:
            # Historical outputs use percentages; the current metric returns a fraction.
            result['PRED25'] = result.PRED25 / 100
        return result
    derived = {
        'review_observations': reviews,
        '02_platform_overview': platform_overview,
        '04_length_summary': length_summary,
        '05_rating_counts': lambda: pd.crosstab(reviews().Platform, reviews().rating)
            .reindex(columns=[1, 2, 3, 4, 5], fill_value=0).rename(columns=str).rename_axis('Platform').reset_index(),
        '06_aspect_sentiment_counts': sentiment_counts,
        '10_split_counts': split_counts,
        '11_absa_model_metrics': absa_metrics,
        '15_leading_architecture_aspect_comparison': aspect_comparison,
        'topic_model_metrics': topic_metrics,
        'bertopic_topics': bertopic_topics,
    }
    if name not in derived:
        raise KeyError(f'Unknown selected research source: {name}')
    return derived[name]().copy()


def ordered(frame, aspects=ASPECTS):
    return frame.set_index('Aspect').reindex(aspects).reset_index()


def number(value, digits=2):
    return f'{float(value):,.{digits}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')


def input_files():
    files = {REVIEW_PATH, PROJECT / 'crawl-data/data/combined_reviews.csv',
             PROJECT / 'crawl-data/data/unique_texts/unique_texts.csv', HISTORY_PATH,
             DATA / 'literature_reference_rows.json', *FILE_SOURCES.values()}
    files.update(EVALUATIONS.glob('*.txt'))
    files.update(PROJECT / f'train-models/data/{split}_cleanser.csv' for split in ['train', 'val', 'test'])
    for label, folder in TOPIC_RUNS:
        path = TOPIC_ROOT / folder
        files.add(path / ('single_run_params.json' if label == 'BERTopic' else 'best_params.json'))
        files.add(path / 'evaluation_metrics.json')
    files.update(TOPIC_ROOT / 'bertopic_16_topics_reproduced' / name
                 for name in ['topic_frequency.csv', 'topic_keyword_weights.csv'])
    return sorted(files)
