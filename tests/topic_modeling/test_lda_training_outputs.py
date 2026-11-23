"""Test fixed-K tuning, exact training corpora and exported LDA assignments."""
import json
import tempfile
import unittest

import sys
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[2] / 'topic-modeling'
sys.path.append(str(TOPIC_DIR))

import pandas as pd
import lda_cleanser_aspects as lda
import lda_fixed_k_cleanser as fixed
from topic_model_figures import corpus_hash


class FixedKTests(unittest.TestCase):
    def test_default_grid_tunes_six_models_only_at_k16(self):
        args = fixed.build_parser().parse_args([])
        candidates = lda.parameter_candidates(args, args.num_topics)
        self.assertEqual(len(candidates), 6)
        self.assertEqual({row['num_topics'] for row in candidates}, {16})
        self.assertEqual({row['passes'] for row in candidates}, {20})
        self.assertEqual({row['iterations'] for row in candidates}, {200})

    def test_requested_k_and_custom_training_grid_are_preserved(self):
        args = fixed.build_parser().parse_args(['--k', '7', '--grid-passes', '10', '20',
                                              '--grid-iterations', '100', '200'])
        candidates = lda.parameter_candidates(args, args.num_topics)
        self.assertEqual(len(candidates), 24)
        self.assertEqual({row['num_topics'] for row in candidates}, {7})


class ExactCorpusTests(unittest.TestCase):
    def test_parallel_trials_match_serial_scores_and_selection(self):
        import numpy as np
        from gensim import corpora
        tokens = [['clean', 'skin', 'soft'], ['clean', 'soft'], ['skin', 'soft'],
                  ['smell', 'soap', 'clean'], ['smell', 'soap'], ['skin', 'soap']]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(row) for row in tokens]
        candidates = [{'num_topics': k, 'alpha': 'symmetric', 'eta': 'auto',
                       'passes': 2, 'iterations': 10} for k in (2, 3)]
        with tempfile.TemporaryDirectory() as serial_dir, tempfile.TemporaryDirectory() as parallel_dir:
            serial = lda.run_grid_search(candidates, corpus, dictionary, tokens, Path(serial_dir),
                                         topn=3, skip_perplexity=False)
            parallel = lda.run_grid_search(candidates, corpus, dictionary, tokens, Path(parallel_dir),
                                           topn=3, skip_perplexity=False, workers=2)
            self.assertEqual(serial[1:3], parallel[1:3])
            np.testing.assert_array_equal(serial[0].get_topics(), parallel[0].get_topics())
            np.testing.assert_array_equal(serial[3]['Coherence_Cv'], parallel[3]['Coherence_Cv'])

    def test_exact_mode_retains_rows_without_vocabulary_evidence(self):
        frame = pd.DataFrame({'_lda_text': ['clean soft skin', 'clean soft skin', 'rare'],
                              'text_group_id': ['a', 'b', 'c']})
        legacy, _, _ = lda.build_tokens(frame, min_df=2, max_df=1., ngram_max=1,
                                        remove_brand_tokens=False)
        exact, tokens, _ = lda.build_tokens(frame, min_df=2, max_df=1., ngram_max=1,
                                           remove_brand_tokens=False, preserve_empty=True)
        self.assertEqual(len(legacy), 2)
        self.assertEqual(exact['text_group_id'].tolist(), ['a', 'b', 'c'])
        self.assertEqual(tokens[-1], [])

    def test_bertopic_reference_requires_recorded_text_order(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            frame = pd.DataFrame({'text_group_id': ['b', 'a'], 'cleaned_comment': ['text b', 'text a'],
                                  'Topic': [0, -1]})
            frame.to_csv(path / 'reviews_with_topics.csv', index=False)
            (path / 'evaluation_metrics.json').write_text(json.dumps({
                'n_documents': 2, 'ordered_corpus_sha256': corpus_hash(['text b', 'text a'])}))
            loaded, _, metadata = lda.load_bertopic_corpus(path, 'cleaned_comment')
            self.assertEqual(loaded['text_group_id'].tolist(), ['b', 'a'])
            self.assertNotIn('Topic', loaded)
            self.assertTrue(metadata['reference_corpus_matches'])
            frame.iloc[::-1].to_csv(path / 'reviews_with_topics.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'order'):
                lda.load_bertopic_corpus(path, 'cleaned_comment')

    def test_empty_document_is_unassigned_in_export_and_counts(self):
        from gensim import corpora
        from gensim.models import LdaModel
        tokens = [['clean', 'skin'], ['skin', 'soft'], []]
        dictionary = corpora.Dictionary(tokens)
        corpus = [dictionary.doc2bow(row) for row in tokens]
        model = LdaModel(corpus, id2word=dictionary, num_topics=2, passes=1, iterations=5, random_state=42)
        frame = pd.DataFrame({'_lda_text': ['clean skin', 'skin soft', 'rare'],
                              'cleaned_comment': ['clean skin', 'skin soft', 'rare']})
        with tempfile.TemporaryDirectory() as folder:
            lda.export_best(model, dictionary, corpus, frame, Path(folder), 2, 1,
                            mark_empty_unassigned=True)
            exported = pd.read_csv(Path(folder) / 'lda_document_topics.csv')
            counts = pd.read_csv(Path(folder) / 'lda_topic_distribution.csv')
            self.assertEqual(len(exported), 3)
            self.assertEqual(exported.iloc[-1]['LDA_Topic'], -1)
            self.assertTrue(pd.isna(exported.iloc[-1]['LDA_Topic_Probability']))
            self.assertEqual(counts['Count'].sum(), 3)
            self.assertEqual(counts.loc[counts.Topic.eq(-1), 'Count'].iloc[0], 1)


if __name__ == '__main__':
    unittest.main()
