"""Tune LDA at a specified K, reusing the original LDA training/selection logic.

Python/PyCharm Run defaults to K=16. No K search is performed.
Examples: python lda_fixed_k_cleanser.py --num-topics 16
          python lda_fixed_k_cleanser.py --num-topics 12 --grid-passes 10 20
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import pandas as pd

import lda_cleanser_aspects as lda
from topic_model_figures import corpus_hash, export_lda_figures

DEFAULT_K = 16


def build_parser():
    parser = argparse.ArgumentParser(description='Tune LDA hyperparameters at a fixed K')
    parser.add_argument('--num-topics', '--k', type=int, default=DEFAULT_K)
    parser.add_argument('--input', default=lda.DEFAULT_INPUT)
    parser.add_argument('--output-dir')
    parser.add_argument('--reference-dir', default=str(lda.BASE_DIR / 'data' / 'lda_cleanser'))
    parser.add_argument('--text-column', default='cleaned_comment')
    parser.add_argument('--match-bertopic', nargs='?', const=lda.DEFAULT_BERTOPIC_REFERENCE,
                        help='Use the exact ordered BERTopic corpus, retaining empty BOW rows.')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--min-df', type=int, default=3)
    parser.add_argument('--max-df', type=float, default=.85)
    parser.add_argument('--ngram-max', type=int, choices=[1, 2], default=2)
    parser.add_argument('--remove-brand-tokens', action='store_true')
    parser.add_argument('--passes', type=int, default=20)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--grid-alpha', type=lda.parse_prior, nargs='+', default=lda.DEFAULT_GRID_ALPHA)
    parser.add_argument('--grid-eta', type=lda.parse_prior, nargs='+', default=lda.DEFAULT_GRID_ETA)
    parser.add_argument('--grid-passes', type=int, nargs='+')
    parser.add_argument('--grid-iterations', type=int, nargs='+')
    parser.add_argument('--top-n-words', type=int, default=10)
    parser.add_argument('--representative-n', type=int, default=3)
    parser.add_argument('--skip-perplexity', action='store_true')
    parser.add_argument('--skip-figures', action='store_true')
    parser.add_argument('--check-input', action='store_true')
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (args.num_topics < 1 or args.workers < 1 or args.min_df < 1 or not 0 < args.max_df <= 1
            or args.top_n_words < 2 or args.representative_n < 1):
        parser.error('Invalid K, vocabulary thresholds or output settings.')
    if any(n < 1 for n in [args.passes, args.iterations] +
           (args.grid_passes or []) + (args.grid_iterations or [])):
        parser.error('passes and iterations must be positive.')
    if 'asymmetric' in args.grid_eta:
        parser.error('Gensim eta does not support asymmetric.')
    corpus_metadata = {}
    if args.match_bertopic:
        df, column, corpus_metadata = lda.load_bertopic_corpus(Path(args.match_bertopic), args.text_column)
        source = Path(corpus_metadata['input'])
    else:
        source = lda.resolve_input(args.input)
        df, column = lda.load_dataset(source, args.text_column)
    candidates = lda.parameter_candidates(args, args.num_topics)
    if not candidates:
        parser.error('No valid hyperparameter candidates for the requested K.')
    print(f'[*] Fixed K={args.num_topics}; {len(candidates)} tuning trials; no K search.', flush=True)
    if args.check_input:
        print(f'[*] Input text rows: {len(df):,}; column: {column}')
        return
    from gensim import corpora
    df, tokens, vectorizer = lda.build_tokens(
        df, min_df=args.min_df, max_df=args.max_df,
        ngram_max=args.ngram_max, remove_brand_tokens=args.remove_brand_tokens,
        preserve_empty=bool(args.match_bertopic))
    dictionary = corpora.Dictionary(tokens)
    if min(len(df), len(dictionary)) < args.num_topics:
        raise ValueError('K exceeds the number of documents or vocabulary size.')
    corpus = [dictionary.doc2bow(words) for words in tokens]
    default_name = (f'lda_cleanser_bertopic_corpus_k{args.num_topics}' if args.match_bertopic
                    else f'lda_cleanser_k{args.num_topics}')
    output = lda.resolve_output(args.output_dir or str(lda.BASE_DIR / 'data' / default_name))
    output.mkdir(parents=True, exist_ok=True)
    reference_folder = Path(args.match_bertopic or args.reference_dir)
    reference_file = reference_folder / 'lda_document_topics.csv'
    reference_matches = True if args.match_bertopic else None
    if reference_file.is_file() and not args.match_bertopic:
        reference = pd.read_csv(reference_file, encoding='utf-8-sig')
        reference_matches = (df[column].tolist() == reference[column].tolist()
                             and df['text_group_id'].tolist() == reference['text_group_id'].tolist())
        if not reference_matches:
            raise ValueError('Training corpus differs from the reference LDA run. '
                             'Specify a matching --input or a different --reference-dir.')
    config = {
        'search_strategy': 'fixed_k_hyperparameters', 'fixed_k': args.num_topics,
        **corpus_metadata, 'workers': args.workers,
        'input': str(source.resolve()), 'text_column': column,
        'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'ordered_corpus_sha256': corpus_hash(df[column].tolist()),
        'reference_dir': str(reference_folder.resolve()),
        'reference_corpus_matches': True if args.match_bertopic else reference_matches,
        'valid_documents': len(df), 'vocabulary_size': len(dictionary),
        'candidates': candidates, 'fixed_params': lda.FIXED_PARAMS,
        'top_n_words': args.top_n_words,
        'vectorizer': {'min_df': args.min_df, 'max_df': args.max_df,
                       'ngram_max': args.ngram_max, 'remove_brand_tokens': args.remove_brand_tokens,
                       'stop_words': sorted(lda.TOPIC_STOPWORDS)},
        'selection': 'Maximum finite C_v at fixed K; first trial wins ties.',
        'package_versions': {name: importlib.metadata.version(name)
                             for name in ('gensim', 'numpy', 'scikit-learn')},
    }
    lda.write_json(output / 'run_config.json', config)
    model, params, score, trials = lda.run_grid_search(
        candidates, corpus, dictionary, tokens, output,
        topn=args.top_n_words, skip_perplexity=args.skip_perplexity,
        stage='fixed_k_hyperparameter_search', workers=args.workers)
    winner = trials.loc[trials['Status'].eq('ok')].sort_values('Coherence_Cv', ascending=False,
                                                           kind='stable').iloc[0]
    lda.write_json(output / 'best_params.json', {
        'params': params, 'fixed_params': lda.FIXED_PARAMS,
        'coherence_cv': score, 'trial': int(winner['Trial']),
        'stage': 'fixed_k_hyperparameter_search',
    })
    lda.export_best(model, dictionary, corpus, df, output, args.top_n_words, args.representative_n,
                    mark_empty_unassigned=bool(args.match_bertopic))
    lda.write_json(output / 'evaluation_metrics.json', {
        'model': 'LDA (Matched K)', 'n_topics': args.num_topics, 'n_documents': len(df),
        'coherence_cv': score, 'top_n_words': args.top_n_words,
        'ordered_corpus_sha256': config['ordered_corpus_sha256'],
        'reference_corpus_matches': True if args.match_bertopic else reference_matches,
        'n_empty_bow': sum(not bow for bow in corpus), **corpus_metadata,
    })
    if not args.skip_figures:
        export_lda_figures(output, args.num_topics, 'LDA (Matched K)', topn=args.top_n_words)
    (output / 'lda_training_summary.txt').write_text(
        f'Fixed K: {args.num_topics}\nValid LDA documents: {len(df)}\nVocabulary: {len(dictionary)}\n'
        f'Trials: {len(trials)}\nBest params: {params}\nBest C_v: {score}\n'
        f'Reference corpus matches: {reference_matches}\n', encoding='utf-8')
    print(f'\n[+] Completed fixed K={args.num_topics}; C_v={score:.6f}; outputs: {output}', flush=True)


if __name__ == '__main__':
    main()
