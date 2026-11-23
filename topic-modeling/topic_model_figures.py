"""Shared publication figures and provenance for local topic-model runs."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import textwrap

import numpy as np
import pandas as pd


COLORS = ['#d95f02', '#0072a8', '#cc79a7', '#e69f00', '#56b4e9',
          '#009e73', '#d9d72b', '#b84b00']


def corpus_hash(texts) -> str:
    return hashlib.sha256(json.dumps(list(texts), ensure_ascii=False,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def save_figure(fig, path: Path):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix('.png'), dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    plt.close(fig)


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def plot_keywords(words: pd.DataFrame, output: Path, *, title: str,
                  weight_label: str, columns=4, topn=10):
    """One panel per topic; bars are keyword weights, not review frequencies."""
    from matplotlib.ticker import MaxNLocator
    plt = pyplot()
    topic_ids = sorted(int(t) for t in words['Topic'].unique() if int(t) >= 0)
    if not topic_ids:
        return
    rows = math.ceil(len(topic_ids) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(17, 3.15 * rows + .7), squeeze=False)
    for index, (ax, topic_id) in enumerate(zip(axes.flat, topic_ids)):
        terms = words.loc[words['Topic'].eq(topic_id)].sort_values('Rank').head(topn)
        labels = ['\n'.join(textwrap.wrap(str(w), 25, break_long_words=True)) for w in terms['Word']]
        ax.barh(np.arange(len(terms)), terms['Weight'], color=COLORS[index % len(COLORS)], height=.75)
        ax.set_yticks(np.arange(len(terms)), labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f'Topic {topic_id}', fontsize=11, pad=6)
        ax.set_xlim(left=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        ax.tick_params(axis='x', labelsize=8, length=0)
        ax.tick_params(axis='y', length=0)
        ax.grid(axis='x', alpha=.13)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in list(axes.flat)[len(topic_ids):]:
        ax.set_visible(False)
    fig.suptitle(title, fontsize=16, y=.995)
    fig.supxlabel(weight_label, fontsize=11)
    fig.tight_layout(rect=(0, .025, 1, .977), h_pad=1.6, w_pad=2.0)
    save_figure(fig, output)


def plot_frequencies(counts: pd.DataFrame, output: Path, *, title: str, note: str,
                     unassigned_label='Outliers (-1)'):
    plt = pyplot()
    counts = counts.sort_values('Topic').copy()
    labels = [f'Topic {int(t)}' if t >= 0 else unassigned_label for t in counts['Topic']]
    colors = ['#94a3b8' if t < 0 else COLORS[int(t) % len(COLORS)] for t in counts['Topic']]
    fig, ax = plt.subplots(figsize=(max(11, .62 * len(counts)), 5.7))
    bars = ax.bar(labels, counts['Count'], color=colors, width=.76)
    ax.bar_label(bars, labels=[f'{int(n):,}' for n in counts['Count']], fontsize=8, padding=3)
    ax.set_ylabel('Number of unique reviews')
    ax.set_title(title, pad=17)
    ax.tick_params(axis='x', rotation=55, labelsize=9)
    ax.set_ylim(0, max(counts['Count'].max() * 1.13, 1))
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    fig.text(.07, .01, note, fontsize=9, color='#475569')
    fig.tight_layout(rect=(0, .035, 1, 1))
    save_figure(fig, output)


def export_lda_figures(output: Path, n_topics: int, label: str, topn=10):
    words = pd.read_csv(output / 'lda_best_topic_words.csv', encoding='utf-8-sig')
    counts = pd.read_csv(output / 'lda_topic_distribution.csv', encoding='utf-8-sig').set_index('Topic')
    indices = ([-1] if -1 in counts.index else []) + list(range(n_topics))
    counts = counts.reindex(indices, fill_value=0).reset_index()
    total = int(counts['Count'].sum())
    empty = int(counts.loc[counts['Topic'].eq(-1), 'Count'].sum())
    plot_keywords(words, output / 'lda_topic_keyword_weights',
                  title=f'{label}: top {topn} keywords per topic (K={n_topics}; {total:,} reviews)',
                  weight_label='Word probability', topn=topn)
    note = f'{total:,} unique reviews. Dominant topic per review; LDA retains topic mixtures.'
    if empty:
        note += f' {empty} reviews have no vocabulary tokens and remain unassigned.'
    plot_frequencies(counts, output / 'lda_topic_frequency',
                     title=f'{label}: dominant-topic review frequency (K={n_topics})', note=note,
                     unassigned_label='No tokens (-1)')


def plot_metrics_table(table: pd.DataFrame, output: Path, *, title: str, notes: list[str]):
    plt = pyplot()
    note_lines = [line for note in notes for line in textwrap.wrap(note, 135)]
    height = 2.15 + .21 * len(note_lines)
    with plt.rc_context({'font.family': 'DejaVu Serif'}):
        fig, ax = plt.subplots(figsize=(12.4, height))
        ax.axis('off')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=17)
        widths = [0.29] + [0.71 / (len(table.columns) - 1)] * (len(table.columns) - 1)
        rendered = ax.table(cellText=table.values, colLabels=table.columns,
                            colWidths=widths, cellLoc='left', colLoc='left',
                            bbox=[0, .10, 1, .78])
        rendered.auto_set_font_size(False)
        rendered.set_fontsize(11)
        for (row, col), cell in rendered.get_celld().items():
            cell.visible_edges = ''
            cell.PAD = .035
            if row == 0:
                cell.set_text_props(fontweight='bold')
                cell.visible_edges = 'TB'
                cell.set_linewidth(.8)
            elif row == len(table):
                cell.visible_edges = 'B'
                cell.set_linewidth(.8)
        bottom = .08 + .07 * len(note_lines)
        fig.subplots_adjust(left=.04, right=.98, top=.78, bottom=min(bottom, .46))
        for index, line in enumerate(note_lines):
            fig.text(.04, .035 + .21 / height * (len(note_lines) - index - 1),
                     line, fontsize=9, va='bottom')
        save_figure(fig, output)


def export_bertopic_figures(model, docs: list[str], topic_assignments, cv, output: Path,
                           *, coherence_sample=0):
    words = []
    for topic in sorted(set(topic_assignments) - {-1}):
        terms = [(word, float(weight)) for word, weight in model.get_topic(topic)
                 if word and weight > 0][:10]
        words.extend({'Topic': int(topic), 'Rank': rank, 'Word': word, 'Weight': weight}
                     for rank, (word, weight) in enumerate(terms, 1))
    words = pd.DataFrame(words, columns=['Topic', 'Rank', 'Word', 'Weight'])
    words.to_csv(output / 'topic_keyword_weights.csv', index=False, encoding='utf-8-sig')
    counts = (pd.Series(topic_assignments).value_counts().sort_index()
              .rename_axis('Topic').reset_index(name='Count'))
    counts['Percentage'] = 100 * counts['Count'] / len(docs)
    counts.to_csv(output / 'topic_frequency.csv', index=False, encoding='utf-8-sig')
    metadata = {
        'model': 'BERTopic', 'coherence_cv': cv,
        'n_topics': int((counts['Topic'] >= 0).sum()), 'n_documents': len(docs),
        'n_outliers': int(sum(t == -1 for t in topic_assignments)),
        'top_n_words': 10, 'coherence_sample': coherence_sample,
        'ordered_corpus_sha256': corpus_hash(docs),
        'keyword_weight': 'c-TF-IDF; not a frequency or a probability',
        'frequency_unit': 'unique reviews; duplicate_count is not a weight',
    }
    (output / 'evaluation_metrics.json').write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    plot_keywords(words, output / 'topic_keyword_weights', title='BERTopic: top 10 keywords per topic',
                  weight_label='c-TF-IDF weight')
    plot_frequencies(counts, output / 'topic_frequency', title='BERTopic: review frequency by topic',
                     note='Each unique review counts once. Topic -1 contains unassigned reviews.')
    table = pd.DataFrame({'Metric': ['Topic Coherence (C_v)', 'Topics Found', 'Total comments'],
                          'BERTopic': [f'{cv:.4f}' if cv is not None else 'N/A',
                                       f'{metadata["n_topics"]} topics', f'{len(docs):,}']})
    table.to_csv(output / 'evaluation_table.csv', index=False, encoding='utf-8-sig')
    plot_metrics_table(table, output / 'evaluation_table', title='Topic modeling evaluation metrics',
                       notes=['C_v computed via Gensim CoherenceModel on per-topic top-10 keyword sets.',
                              'Topics Found excludes -1; Total comments includes unassigned reviews.'])
