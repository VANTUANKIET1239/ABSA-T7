"""Test review-based BERTopic input, grid selection and export behavior."""
import contextlib
import io
import json
import os
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import sys
from pathlib import Path

TOPIC_DIR = Path(__file__).resolve().parents[2] / 'topic-modeling'
sys.path.append(str(TOPIC_DIR))

import pandas as pd
import bertopic_cleanser_aspects as script


class InputPathTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.project = Path(self.directory.name).resolve()
        self.topic_dir = self.project / 'topic-modeling'
        self.topic_dir.mkdir()
        self.source = self.project / script.DEFAULT_INPUT
        self.source.parent.mkdir(parents=True)
        self.source.write_text('comment,cleaned_comment\nreview example,review_example\n', encoding='utf-8')

    def test_default_input_resolves_from_project_script_and_unrelated_cwd(self):
        for cwd in [self.project, self.topic_dir, self.project / 'unrelated']:
            with self.subTest(cwd=cwd), patch.object(script, 'BASE_DIR', self.topic_dir), patch.object(
                script, 'PROJECT_DIR', self.project,
            ), patch.object(script.Path, 'cwd', return_value=cwd):
                self.assertEqual(script.resolve_input_path(script.DEFAULT_INPUT), self.source)

    def test_explicit_absolute_input_is_preserved(self):
        self.assertEqual(script.resolve_input_path(str(self.source)), self.source)

    def test_explicit_relative_input_prefers_cwd(self):
        custom = self.topic_dir / 'custom.csv'
        custom.write_text('comment\nreview example\n', encoding='utf-8')
        with patch.object(script.Path, 'cwd', return_value=self.topic_dir):
            self.assertEqual(script.resolve_input_path('custom.csv'), custom)

    def test_missing_input_lists_checked_paths_and_input_option(self):
        with patch.object(script, 'BASE_DIR', self.topic_dir), patch.object(
            script, 'PROJECT_DIR', self.project,
        ), patch.object(script.Path, 'cwd', return_value=self.project), self.assertRaises(FileNotFoundError) as error:
            script.resolve_input_path('missing.csv')
        message = str(error.exception)
        self.assertIn(str(self.project / 'missing.csv'), message)
        self.assertIn(str(self.topic_dir / 'missing.csv'), message)
        self.assertIn('--input', message)


class UniqueTextInputTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'unique_texts.csv'
        self.data = pd.DataFrame({
            'text_group_id': ['txt_first', 'txt_second', 'txt_short'],
            'cleaned_comment': ['cực_kì hài_lòng', 'không_kích_ứng dưỡng_ẩm', 'ok'],
            'duplicate_count': [2447, 3, 12],
            'representative_source_row': [1, 2, 3],
            'representative_comment': ['Cực kì hài lòng!', 'Không kích ứng, dưỡng ẩm tốt.', 'ok'],
        })
        self.data.to_csv(self.source, index=False, encoding='utf-8-sig')

    def test_explicit_representative_column_keeps_ids_without_expanding_counts(self):
        data, embedding_col, topic_col = script.read_dataset(self.source, 'representative_comment', 'cleaned_comment')
        self.assertEqual((embedding_col, topic_col), ('representative_comment', 'cleaned_comment'))
        self.assertEqual(len(data), 2)
        self.assertEqual(data['text_group_id'].tolist(), ['txt_first', 'txt_second'])
        self.assertEqual(data['duplicate_count'].tolist(), [2447, 3])
        self.assertEqual(data['_embedding_text'].tolist(),
                         self.data['representative_comment'][:2].tolist())
        self.assertEqual(data['_topic_text'].tolist(), self.data['cleaned_comment'][:2].tolist())

    def test_cleaned_only_input_uses_readable_embedding_fallback(self):
        self.data.drop(columns=['representative_comment']).to_csv(self.source, index=False, encoding='utf-8-sig')
        data, embedding_col, topic_col = script.read_dataset(self.source, 'normalized_comment', 'topic_comment')
        self.assertEqual((embedding_col, topic_col), ('cleaned_comment', 'cleaned_comment'))
        self.assertEqual(data['_embedding_text'].tolist(), ['cực kì hài lòng', 'không kích ứng dưỡng ẩm'])
        self.assertEqual(data['_topic_text'].tolist(), self.data['cleaned_comment'][:2].tolist())

    def test_export_keeps_mapping_keys_and_representative_raw_comments(self):
        data, _, _ = script.read_dataset(self.source, 'normalized_comment', 'topic_comment')
        model = Mock()
        model.get_topic_info.return_value = pd.DataFrame({'Topic': [0], 'Count': [2], 'Name': ['test']})
        model.get_topic.return_value = [('dịu_nhẹ', 0.5), ('', 0.00001)]
        model.get_representative_docs.return_value = [data['_topic_text'].iloc[1]]
        output = self.root / 'output'
        script.export_outputs(data, data['_topic_text'].tolist(), model, [0, 0], None, output, 0.5)
        saved = pd.read_csv(output / 'reviews_with_topics.csv', encoding='utf-8-sig')
        self.assertEqual(saved['text_group_id'].tolist(), ['txt_first', 'txt_second'])
        self.assertEqual(saved['duplicate_count'].tolist(), [2447, 3])
        candidates = pd.read_csv(output / 'aspect_candidates.csv', encoding='utf-8-sig')
        self.assertEqual(candidates.loc[0, 'Representative_Review_1'], self.data.loc[1, 'representative_comment'])
        self.assertEqual(candidates.loc[0, 'Keywords'], 'dịu_nhẹ')
        self.assertIn('Counts refer to unique texts', (output / 'training_summary.txt').read_text(encoding='utf-8'))


class GridSearchTests(unittest.TestCase):
    def setUp(self):
        self.output = tempfile.TemporaryDirectory()
        self.addCleanup(self.output.cleanup)
        self.out = Path(self.output.name)
        self.docs = ['review example'] * 100
        self.embeddings = object()
        self.args = types.SimpleNamespace(
            single_run=False, coherence_sample=0,
            grid_min_cluster_sizes=[15, 30, 50], grid_min_samples=[3, 5, 10],
            grid_n_neighbors=[10, 20, 30], grid_min_df=None, grid_max_df=None,
        )
        redirect = contextlib.redirect_stdout(io.StringIO())
        redirect.__enter__()
        self.addCleanup(redirect.__exit__, None, None, None)

    def fake_model(self, topics=None):
        model = Mock()
        model.fit_transform.return_value = (topics if topics is not None else [0] * 90 + [-1] * 10, None)
        model.get_topic.side_effect = lambda topic: [(f'word_{topic}', 0.5), (f'other_{topic}', 0.3)]
        return model

    def search(self, candidates):
        return script.grid_search(candidates, object(), self.embeddings, self.docs, self.args, self.out)

    def test_default_grid_contains_baseline_once_without_mutating_it(self):
        original = script.BASELINE_PARAMS.copy()
        candidates = script.parameter_candidates(self.args, original, 100)
        self.assertEqual(len(candidates), 27)
        self.assertEqual(candidates[0], original)
        self.assertEqual(sum(item == original for item in candidates), 1)
        self.assertEqual(script.BASELINE_PARAMS, original)

    def test_custom_grid_keeps_baseline_and_deduplicates_effective_neighbors(self):
        self.args.grid_min_cluster_sizes = [30]
        self.args.grid_min_samples = [5]
        self.args.grid_n_neighbors = [100, 200]
        self.args.grid_min_df = [1, 3]
        candidates = script.parameter_candidates(self.args, script.BASELINE_PARAMS, 100)
        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0], script.BASELINE_PARAMS)
        self.assertTrue(all(item['n_neighbors'] == 99 for item in candidates[1:]))

    def test_selects_maximum_finite_score_skips_failures_and_keeps_first_tie(self):
        candidates = [{**script.BASELINE_PARAMS, 'min_cluster_size': size}
                      for size in [30, 15, 20, 25, 35, 40, 45]]
        models = [self.fake_model() for _ in range(7)]
        models[2].fit_transform.side_effect = ValueError('empty vocabulary')
        with patch.object(script, 'create_model', side_effect=models), patch.object(
            script, 'coherence_cv', side_effect=[0.4, 0.8, float('nan'), 0.8, None, float('inf')],
        ):
            best = self.search(candidates)
        self.assertIs(best[0], models[1])
        self.assertEqual(best[3], 0.8)
        for model in models:
            model.fit_transform.assert_called_once_with(self.docs, embeddings=self.embeddings)
        selected = json.loads((self.out / 'best_params.json').read_text(encoding='utf-8'))
        self.assertEqual(selected['params']['min_cluster_size'], 15)
        self.assertEqual(selected['trial'], 2)
        self.assertAlmostEqual(selected['outlier_rate'], 0.1)
        rows = pd.read_csv(self.out / 'grid_search_results.csv')
        self.assertEqual(rows['status'].tolist(), [
            'ok', 'ok', 'failed', 'invalid_coherence', 'ok', 'invalid_coherence', 'invalid_coherence',
        ])
        self.assertIn('empty vocabulary', rows.loc[2, 'error'])

    def test_rejects_invalid_params_before_fit_and_logs_them(self):
        candidates = [{**script.BASELINE_PARAMS, 'min_cluster_size': 101}, script.BASELINE_PARAMS]
        with patch.object(script, 'create_model', return_value=self.fake_model()) as create, patch.object(
            script, 'coherence_cv', return_value=0.5,
        ):
            self.search(candidates)
        create.assert_called_once()
        rows = pd.read_csv(self.out / 'grid_search_results.csv')
        self.assertEqual(rows['status'].tolist(), ['failed', 'ok'])

    def test_all_invalid_scores_raise_without_best_params(self):
        with patch.object(script, 'create_model', return_value=self.fake_model()), patch.object(
            script, 'coherence_cv', return_value=None,
        ), self.assertRaisesRegex(RuntimeError, 'C_v'):
            self.search([script.BASELINE_PARAMS])
        self.assertFalse((self.out / 'best_params.json').exists())
        self.assertTrue((self.out / 'grid_search_results.csv').exists())

    def test_main_saves_baseline_and_exports_exact_winning_fit(self):
        data = pd.DataFrame({
            'normalized_comment': self.docs, 'topic_comment': self.docs,
            '_embedding_text': self.docs, '_topic_text': self.docs,
        })
        models = [self.fake_model([topic] * 100) for topic in range(4)]
        encoder = Mock()
        encoder.encode.return_value = self.embeddings
        encoder_class = Mock(return_value=encoder)
        dependencies = {
            'sentence_transformers': types.SimpleNamespace(SentenceTransformer=encoder_class),
            'gensim': types.ModuleType('gensim'),
            'gensim.models': types.SimpleNamespace(CoherenceModel=Mock()),
        }
        argv = ['bertopic_cleanser_aspects.py', '--grid-search', '--skip-figures', '--device', 'cpu', '--output-dir', str(self.out),
                '--grid-min-cluster-sizes', '15', '50', '--grid-min-samples', '5', '--grid-n-neighbors', '20']
        with patch('sys.argv', argv), patch.dict('sys.modules', dependencies), patch.object(
            script, 'read_dataset', return_value=(data, 'normalized_comment', 'topic_comment'),
        ), patch.object(script, 'create_model', side_effect=models), patch.object(
            script, 'coherence_cv', side_effect=[0.3, 0.9, 0.5, 0.4],
        ), patch.object(script, 'resolve_embedding_device', return_value='cpu'), patch.object(
            script, 'export_outputs',
        ) as export:
            models[1].get_topic_info.return_value = pd.DataFrame({'Topic': [1], 'Count': [100], 'Name': ['best']})
            models[1].get_topics.return_value = {1: [('sạch', 0.5)]}
            models[1].get_topic.side_effect = lambda topic: [('sạch', 0.5), ('dịu_nhẹ', 0.3)]
            script.main()
        encoder.encode.assert_called_once()
        self.assertIs(export.call_args.args[2], models[1])
        self.assertEqual(export.call_args.args[3], [1] * 100)
        baseline = json.loads((self.out / 'baseline_params.json').read_text(encoding='utf-8'))
        self.assertEqual(baseline['params'], script.BASELINE_PARAMS)
        self.assertEqual(baseline['model_configuration']['umap']['n_components'], 5)
        self.assertEqual(baseline['batch_size'], 32)
        selected = json.loads((self.out / 'best_params.json').read_text(encoding='utf-8'))
        self.assertEqual(selected['c_v'], 0.9)
        self.assertEqual(len(json.loads((self.out / 'run_config.json').read_text(encoding='utf-8'))['candidates']), 4)

    def test_cli_rejects_skip_coherence_for_grid_search(self):
        with patch('sys.argv', ['bertopic_cleanser_aspects.py', '--grid-search', '--skip-coherence']), patch.object(
            script, 'read_dataset',
        ) as read, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            script.main()
        self.assertEqual(error.exception.code, 2)
        read.assert_not_called()


if __name__ == '__main__':
    unittest.main()
