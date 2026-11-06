"""Kiểm tra đường dẫn, schema unique và Grid Search bằng mô hình giả."""
import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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

    def test_unique_schema_selects_representative_and_keeps_ids_without_expanding_counts(self):
        data, embedding_col, topic_col = script.read_dataset(self.source, 'normalized_comment', 'topic_comment')
        self.assertEqual((embedding_col, topic_col), ('representative_comment', 'cleaned_comment'))
        self.assertEqual(len(data), 2)
        self.assertEqual(data['text_group_id'].tolist(), ['txt_first', 'txt_second'])
        self.assertEqual(data['duplicate_count'].tolist(), [2447, 3])
        self.assertEqual(data['_embedding_text'].tolist(),
                         [text.lower() for text in self.data['representative_comment'][:2]])
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


class TopicNormalizationTests(unittest.TestCase):
    def analyzer(self):
        from sklearn.feature_extraction.text import CountVectorizer
        return CountVectorizer(**script.model_configuration(script.BASELINE_PARAMS)['vectorizer']).build_analyzer()

    def test_restores_contextual_accents_without_losing_existing_segmentation(self):
        self.assertEqual(
            script.normalize_topic_text('san pham rat tot, dong_goi can than, dung tich 150ml; rat da!'),
            'sản_phẩm rất tốt đóng_gói cẩn_thận dung_tích 150ml rát_da',
        )
        self.assertEqual(script.normalize_review_accents('KHÔNG_kích_ứng, rat tot!'),
                         'không_kích_ứng, rất tốt!')

    def test_keeps_ambiguous_words_and_foreign_ingredients(self):
        text = 'màng co có mùi chua; dung tích 50ml; on sale; tranexamic acid'
        normalized = script.normalize_topic_text(text)
        for phrase in ['màng_co', 'mùi_chua', 'dung_tích', 'on sale', 'tranexamic_acid']:
            self.assertIn(phrase, normalized)

    def test_protects_sour_smell_before_capacity_when_punctuation_is_missing(self):
        self.assertEqual(script.normalize_topic_text('có mùi chua dung tich 50ml mang co lop nen'),
                         'có mùi_chua dung_tích 50ml màng_co lớp_nền')
        self.assertEqual(script.normalize_topic_text('chua dung san pham'), 'chưa_dùng sản_phẩm')

    def test_repairs_contextual_words_and_preserves_template_aspect_values(self):
        text = 'san sale, rat chat luong; khong co huong lieu; kinh_nghiem su dung: rat da'
        normalized = script.normalize_topic_text(text)
        self.assertEqual(normalized, 'săn sale rất chất_lượng không có hương_liệu kinh_nghiệm_sử_dụng rát_da')
        keywords = set(self.analyzer()(normalized))
        self.assertTrue({'sale', 'hương_liệu', 'rát_da'} <= keywords)
        self.assertNotIn('kinh_nghiệm_sử_dụng', keywords)

    def test_generic_sentiment_and_deferred_reviews_do_not_become_keywords(self):
        text = 'cực kì hài lòng ok rất tốt sẽ tiếp tục ủng hộ chưa dùng chưa sử dụng'
        self.assertEqual(self.analyzer()(script.normalize_topic_text(text)), [])

    def test_keeps_negation_and_aspect_terms(self):
        text = 'không sạch chưa sạch không kích ứng rát da khô căng không hương liệu sale giá rẻ chai'
        tokens = set(self.analyzer()(script.normalize_topic_text(text)))
        self.assertTrue({'không_sạch', 'chưa', 'sạch', 'không_kích_ứng', 'rát_da',
                         'khô_căng', 'không_hương_liệu', 'sale', 'giá_rẻ', 'chai'} <= tokens)

    def test_normalization_is_idempotent_for_segmented_text_and_unicode(self):
        import unicodedata
        text = unicodedata.normalize('NFD', 'san phẩm hài_lòng không_kích_ứng đóng_gói can than')
        normalized = script.normalize_topic_text(text)
        self.assertEqual(script.normalize_topic_text(normalized), normalized)
        self.assertNotRegex(normalized, r'[.,:!?]')

    def test_input_normalization_preserves_original_columns_and_mapping(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.csv'
            original = pd.DataFrame({
                'text_group_id': ['txt_original'], 'duplicate_count': [50],
                'representative_comment': ['San pham rat tot, khong kich ung!'],
                'cleaned_comment': ['san pham rat tot khong_kich_ung'],
            })
            original.to_csv(path, index=False, encoding='utf-8-sig')
            data, _, _ = script.read_dataset(path, 'normalized_comment', 'topic_comment')
            pd.testing.assert_frame_equal(data[original.columns], original)
            self.assertEqual(data.loc[0, '_embedding_text'], 'sản phẩm rất tốt, không kích ứng!')
            self.assertEqual(data.loc[0, '_topic_text'], 'sản_phẩm rất tốt không_kích_ứng')
            pd.testing.assert_frame_equal(pd.read_csv(path, encoding='utf-8-sig'), original)


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
        models = [self.fake_model([0] * 100), self.fake_model([1] * 100), self.fake_model([2] * 100)]
        encoder = Mock()
        encoder.encode.return_value = self.embeddings
        encoder_class = Mock(return_value=encoder)
        dependencies = {
            'sentence_transformers': types.SimpleNamespace(SentenceTransformer=encoder_class),
            'gensim': types.ModuleType('gensim'),
            'gensim.models': types.SimpleNamespace(CoherenceModel=Mock()),
        }
        argv = ['bertopic_cleanser_aspects.py', '--text-unit', 'review', '--output-dir', str(self.out),
                '--grid-min-cluster-sizes', '15', '50', '--grid-min-samples', '5', '--grid-n-neighbors', '20']
        with patch('sys.argv', argv), patch.dict('sys.modules', dependencies), patch.object(
            script, 'read_dataset', return_value=(data, 'normalized_comment', 'topic_comment'),
        ), patch.object(script, 'create_model', side_effect=models), patch.object(
            script, 'coherence_cv', side_effect=[0.3, 0.9, 0.5],
        ), patch.object(script, 'export_outputs') as export:
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
        self.assertEqual(len(json.loads((self.out / 'run_config.json').read_text(encoding='utf-8'))['candidates']), 3)

    def test_cli_rejects_skip_coherence_for_grid_search(self):
        with patch('sys.argv', ['bertopic_cleanser_aspects.py', '--skip-coherence']), patch.object(
            script, 'read_dataset',
        ) as read, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            script.main()
        self.assertEqual(error.exception.code, 2)
        read.assert_not_called()

    def test_balanced_selection_penalizes_a_dominant_topic_despite_higher_cv(self):
        models = [self.fake_model([0] * 90 + [1] * 10), self.fake_model([0] * 50 + [1] * 50)]
        with patch.object(script, 'create_model', side_effect=models), patch.object(
            script, 'coherence_cv', side_effect=[0.9, 0.65],
        ):
            best = self.search([script.BASELINE_PARAMS, {**script.BASELINE_PARAMS, 'min_cluster_size': 15}])
        self.assertIs(best[0], models[1])
        rows = pd.read_csv(self.out / 'grid_search_results.csv')
        self.assertEqual(rows.largest_topic_share.tolist(), [0.9, 0.5])
        self.assertGreater(rows.selection_score.iloc[1], rows.selection_score.iloc[0])
        saved = json.loads((self.out / 'best_params.json').read_text(encoding='utf-8'))
        self.assertEqual(saved['selection']['metric'], 'balanced')
        self.assertEqual(saved['c_v'], 0.65)

    def test_cv_selection_keeps_legacy_ranking_and_still_exports_quality_metrics(self):
        self.args.selection_metric = 'c_v'
        models = [self.fake_model([0] * 90 + [1] * 10), self.fake_model([0] * 50 + [1] * 50)]
        with patch.object(script, 'create_model', side_effect=models), patch.object(
            script, 'coherence_cv', side_effect=[0.9, 0.65],
        ):
            best = self.search([script.BASELINE_PARAMS, {**script.BASELINE_PARAMS, 'min_cluster_size': 15}])
        self.assertIs(best[0], models[0])
        rows = pd.read_csv(self.out / 'grid_search_results.csv')
        self.assertIn('topic_diversity', rows)
        self.assertIn('keyword_overlap_mean', rows)
        self.assertEqual(rows.selection_score.tolist(), [0.9, 0.65])


class TopicQualityTests(unittest.TestCase):
    def test_metrics_use_explicit_denominators_and_ignore_outlier_keywords(self):
        model = Mock()
        model.get_topic.side_effect = lambda topic: {
            0: [('sạch', .5), ('thơm', .4), ('', .01), ('ignored', 0)],
            1: [('thơm', .5), ('mịn', .4), ('mịn', .2)],
            -1: [('noise', 1)],
        }[topic]
        metrics = script.topic_quality_metrics(model, [0, 0, 1, -1])
        self.assertEqual(metrics['n_topics'], 2)
        self.assertEqual(metrics['outlier_rate'], .25)
        self.assertEqual(metrics['largest_topic_share'], .5)
        self.assertAlmostEqual(metrics['largest_topic_share_assigned'], 2 / 3)
        self.assertEqual(metrics['topic_diversity'], .75)
        self.assertAlmostEqual(metrics['keyword_overlap_mean'], 1 / 3)
        self.assertAlmostEqual(metrics['keyword_overlap_max'], 1 / 3)
        self.assertEqual(metrics['valid_topic_ratio'], 1)
        self.assertAlmostEqual(script.selection_score(.6, metrics, 'balanced'), .075)
        self.assertEqual(script.selection_score(.6, metrics, 'c_v'), .6)

    def test_empty_and_single_cluster_outputs_have_no_artificial_quality_score(self):
        model = Mock()
        model.get_topic.return_value = []
        metrics = script.topic_quality_metrics(model, [-1, -1])
        self.assertEqual(metrics['topic_diversity'], 0)
        self.assertEqual(metrics['valid_topic_ratio'], 0)
        self.assertEqual(script.selection_score(.9, metrics, 'balanced'), 0)
        model.get_topic.return_value = [('sạch', .5), ('mịn', .4)]
        metrics = script.topic_quality_metrics(model, [0, 0])
        self.assertEqual(script.selection_score(.9, metrics, 'balanced'), 0)

    def test_topics_without_representative_keywords_reduce_valid_topic_ratio(self):
        model = Mock()
        model.get_topic.side_effect = lambda topic: [('sạch', .5), ('mịn', .4)] if topic == 0 else []
        metrics = script.topic_quality_metrics(model, [0, 1])
        self.assertEqual(metrics['valid_topic_ratio'], .5)


class TextUnitTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.path = self.root / 'input.csv'
        self.source = pd.DataFrame({
            'text_group_id': ['txt_a', 'txt_b', 'txt_generic', 'txt_empty'],
            'duplicate_count': [100, 3, 10, 1],
            'representative_comment': ['Rửa sạch, không khô da. Giao hàng nhanh!',
                                       'Rửa sạch. Mùi hơi nồng; Rửa sạch.', 'Ok. Cực kì hài lòng.', ''],
            'cleaned_comment': ['không dùng cột này'] * 4,
        })
        self.source.to_csv(self.path, index=False, encoding='utf-8-sig')

    def prepare(self):
        return script.read_unit_dataset(self.path, 'clause', None, 2)

    def test_clause_split_keeps_decimal_and_negation_scope(self):
        self.assertEqual(script.split_text_units('pH 5.5, rửa sạch nhưng mùi hơi nồng. Giao hàng nhanh!'),
                         ['pH 5.5', 'rửa sạch', 'nhưng mùi hơi nồng', 'Giao hàng nhanh'])
        self.assertEqual(script.split_text_units('Không chứa cồn, hương liệu, paraben; da mềm.'),
                         ['Không chứa cồn, hương liệu, paraben', 'da mềm'])
        self.assertIn('không_gây_bít_tắc', script.normalize_topic_text('không gây bít tắc'))

    def test_sentence_mode_preserves_commas_and_contrast_within_a_sentence(self):
        text = 'Rửa sạch, không khô da nhưng mùi hơi nồng. Giao hàng nhanh!'
        self.assertEqual(script.split_text_units(text, 'sentence'),
                         ['Rửa sạch, không khô da nhưng mùi hơi nồng', 'Giao hàng nhanh'])

    def test_field_labels_create_boundaries_before_punctuation_removal(self):
        text = 'Làm sạch:sạch da Hương thơm:dịu nhẹ Kinh nghiệm sử dụng:không khô da'
        self.assertEqual(script.split_text_units(text),
                         ['Làm sạch:sạch da', 'Hương thơm:dịu nhẹ', 'Kinh nghiệm sử dụng:không khô da'])

    def test_raw_unit_teencode_normalization_keeps_negation_and_service_terms(self):
        self.source.loc[0, 'representative_comment'] = 'ko gây bít tắc. Shop giao nhanh.'
        self.source.to_csv(self.path, index=False, encoding='utf-8-sig')
        units, _, _, _ = self.prepare()
        self.assertIn('không_gây_bít_tắc', units['_topic_text'].tolist())
        self.assertIn('cửa hàng giao nhanh', units['_embedding_text'].tolist())
        self.assertEqual(script.split_text_units('k gây khô, rát da'), ['k gây khô, rát da'])

    def test_deduplicates_units_but_preserves_every_source_occurrence_and_original_input(self):
        before = self.path.read_bytes()
        units, column, mapping, source = self.prepare()
        self.assertEqual(column, 'representative_comment')
        self.assertEqual(len(units), 4)
        self.assertEqual(len(mapping), 8)
        self.assertEqual(mapping.unit_status.eq('kept').sum(), 6)
        clean = units.loc[units.unit_text.eq('Rửa sạch')].iloc[0]
        self.assertEqual(clean.unit_occurrences, 3)
        self.assertEqual(clean.source_review_count, 2)
        self.assertEqual(clean.duplicate_count, 100)
        self.assertEqual(len(mapping.loc[mapping.unit_id.eq(clean.unit_id)]), 3)
        self.assertTrue(units.unit_id.is_unique)
        self.assertNotIn('không dùng cột này', units['_topic_text'].tolist())
        pd.testing.assert_frame_equal(source.fillna(''), self.source)
        self.assertEqual(before, self.path.read_bytes())

    def test_export_maps_multiple_topics_to_reviews_and_keeps_excluded_reviews(self):
        units, _, mapping, source = self.prepare()
        model = Mock()
        model.get_topic_info.return_value = pd.DataFrame({'Topic': list(range(4)), 'Count': [1] * 4})
        model.get_topic.side_effect = lambda topic: [(f'word_{topic}', .5), (f'other_{topic}', .3)]
        model.get_representative_docs.side_effect = lambda topic: [units['_topic_text'].iloc[topic]]
        output = self.root / 'output'
        script.export_outputs(units, units['_topic_text'].tolist(), model, list(range(4)), [.8] * 4,
                              output, .6, unit_mapping=mapping, source_reviews=source, text_unit='clause')
        reviews = pd.read_csv(output / 'reviews_with_topics.csv', encoding='utf-8-sig')
        self.assertEqual(len(reviews), 4)
        self.assertEqual(reviews.Topics.map(json.loads).tolist(), [[0, 1, 2], [0, 3], [], []])
        self.assertEqual(reviews.duplicate_count.tolist(), [100, 3, 10, 1])
        self.assertEqual(reviews.Unit_Count.tolist(), [3, 3, 0, 0])
        linked = pd.read_csv(output / 'review_unit_mapping.csv', encoding='utf-8-sig')
        self.assertEqual(len(linked), 8)
        self.assertTrue(linked.loc[linked.unit_status.ne('kept'), 'Topic'].isna().all())
        info = pd.read_csv(output / 'topic_info.csv', encoding='utf-8-sig')
        self.assertEqual(info.loc[info.Topic.eq(0), 'Source_Review_Count'].iloc[0], 2)
        candidates = pd.read_csv(output / 'aspect_candidates.csv', encoding='utf-8-sig')
        self.assertEqual(candidates.loc[0, 'Representative_Unit_1'], 'Rửa sạch')
        metrics = json.loads((output / 'evaluation_metrics.json').read_text(encoding='utf-8'))
        self.assertEqual(metrics['largest_topic_share'], .25)
        self.assertTrue((output / 'topic_keyword_overlap.csv').is_file())

    def test_prepare_only_exports_units_without_loading_or_fitting_models(self):
        output = self.root / 'prepared'
        argv = ['bertopic_cleanser_aspects.py', '--input', str(self.path), '--prepare-only',
                '--output-dir', str(output)]
        with patch('sys.argv', argv), patch.object(script, 'create_model') as create:
            script.main()
        create.assert_not_called()
        self.assertTrue((output / 'text_units.csv').is_file())
        self.assertTrue((output / 'review_unit_mapping.csv').is_file())
        summary = json.loads((output / 'unit_preparation_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(summary['input_reviews'], 4)
        self.assertEqual(summary['training_unique_units'], 4)


if __name__ == '__main__':
    unittest.main()
