import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

SOURCE = Path(__file__).with_name('lda_cleanser_aspects.py')
SPEC = importlib.util.spec_from_file_location('lda_cleanser_aspects', SOURCE)
lda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(lda)


def grid_args(**overrides):
    values = dict(min_k=1, max_k=20, k_step=1,
                  grid_alpha=lda.DEFAULT_GRID_ALPHA, grid_eta=lda.DEFAULT_GRID_ETA,
                  grid_passes=None, grid_iterations=None, passes=20, iterations=200,
                  k_alpha='symmetric', k_eta='auto', top_n_words=2, skip_perplexity=True)
    values.update(overrides)
    return SimpleNamespace(**values)


class LDAGridTests(unittest.TestCase):
    def test_defaults_cover_every_k_and_original_model(self):
        candidates = lda.k_candidates(grid_args())
        self.assertEqual(len(candidates), 20)
        self.assertEqual({p['num_topics'] for p in candidates}, set(range(1, 21)))
        self.assertTrue(all(p['alpha'] == 'symmetric' and p['eta'] == 'auto'
                            and p['passes'] == 20 and p['iterations'] == 200
                            for p in candidates))
        self.assertIn(lda.BASELINE_PARAMS, candidates)

    def test_hyperparameters_only_use_selected_k(self):
        candidates = lda.parameter_candidates(grid_args(), 7)
        self.assertEqual(len(candidates), 6)
        self.assertEqual({p['num_topics'] for p in candidates}, {7})
        one_topic = lda.parameter_candidates(grid_args(), 1)
        self.assertEqual(len(one_topic), 4)
        self.assertFalse(any(p['alpha'] == 'auto' for p in one_topic))

    def test_custom_grid_crosses_all_dimensions_and_removes_duplicates(self):
        candidates = lda.parameter_candidates(grid_args(
            max_k=2, grid_alpha=[0.1, 'auto', 0.1], grid_eta=['auto', 0.2],
            grid_passes=[1, 2, 1], grid_iterations=[10, 20]), 2)
        self.assertEqual(len(candidates), 16)
        self.assertEqual({p['num_topics'] for p in candidates}, {2})
        self.assertEqual({p['passes'] for p in candidates}, {1, 2})
        self.assertEqual({p['iterations'] for p in candidates}, {10, 20})

    def test_prior_validation(self):
        self.assertEqual(lda.parse_prior('0.1'), 0.1)
        self.assertEqual(lda.parse_prior('auto'), 'auto')
        for value in ('0', '-1', 'nan', 'inf', 'typo'):
            with self.subTest(value=value), self.assertRaises(argparse.ArgumentTypeError):
                lda.parse_prior(value)

    def test_selection_ignores_failed_nan_and_keeps_first_tie(self):
        candidates = lda.k_candidates(grid_args(
            max_k=5, grid_alpha=['symmetric'], grid_eta=['auto']))
        models = {}

        def train(**kwargs):
            k = kwargs['num_topics']
            if k == 1:
                raise ValueError('Invalid trial')
            models[k] = SimpleNamespace(
                alpha=[1.0], eta=[1.0], get_topics=lambda: [[0.6, 0.4]],
                num_topics=k, show_topic=lambda *a, **kw: [('clean', 0.6), ('skin', 0.4)],
                log_perplexity=lambda *a: -3.0)
            return models[k]

        with tempfile.TemporaryDirectory() as folder, \
                patch('gensim.models.LdaModel', side_effect=train), \
                patch('gensim.models.CoherenceModel') as coherence:
            coherence.return_value.get_coherence.side_effect = [float('nan'), 0.4, 0.9, 0.9]
            model, params, score, results = lda.run_grid_search(
                candidates, [], {}, [], Path(folder), topn=2, skip_perplexity=False)
            self.assertIs(model, models[4])
            self.assertEqual(params['num_topics'], 4)
            self.assertEqual(score, 0.9)
            self.assertEqual(results['Status'].tolist(),
                             ['failed', 'invalid_coherence', 'ok', 'ok', 'ok'])
            saved = pd.read_csv(Path(folder) / 'lda_grid_search_results.csv')
            self.assertEqual(len(saved), 5)
            self.assertEqual(saved.loc[2, 'Log_Perplexity'], -3.0)

    def test_all_failed_trials_leave_results_for_inspection(self):
        candidates = lda.parameter_candidates(grid_args(
            max_k=1, grid_alpha=['symmetric'], grid_eta=['auto']), 1)
        with tempfile.TemporaryDirectory() as folder, \
                patch('gensim.models.LdaModel', side_effect=ValueError('bad prior')):
            with self.assertRaisesRegex(RuntimeError, 'C_v'):
                lda.run_grid_search(candidates, [], {}, [], Path(folder),
                                    topn=2, skip_perplexity=True)
            saved = pd.read_csv(Path(folder) / 'lda_grid_search_results.csv')
            self.assertEqual(saved.loc[0, 'Status'], 'failed')

    def test_nonfinite_model_is_rejected_before_coherence(self):
        candidates = lda.parameter_candidates(grid_args(
            max_k=1, grid_alpha=['symmetric'], grid_eta=['auto']), 1)
        model = SimpleNamespace(alpha=[float('nan')], eta=[1.0],
                                get_topics=lambda: [[0.5, 0.5]])
        with tempfile.TemporaryDirectory() as folder, \
                patch('gensim.models.LdaModel', return_value=model), \
                patch('gensim.models.CoherenceModel') as coherence:
            with self.assertRaises(RuntimeError):
                lda.run_grid_search(candidates, [], {}, [], Path(folder),
                                    topn=2, skip_perplexity=True)
            coherence.assert_not_called()

    def test_two_stages_lock_k_chosen_with_fixed_parameters(self):
        trained = []
        current_score = [None]

        def train(**kwargs):
            trained.append(kwargs.copy())
            k = kwargs['num_topics']
            if kwargs['alpha'] == 'symmetric' and kwargs['eta'] == 'auto':
                score = {1: 0.4, 2: 0.8, 3: 0.7}[k]
            else:
                score = 0.92 if k == 2 else 0.99
            current_score[0] = score
            return SimpleNamespace(
                alpha=[1.0], eta=[1.0], get_topics=lambda: [[0.6, 0.4]],
                num_topics=k, show_topic=lambda *a, **kw: [('clean', 0.6), ('skin', 0.4)])

        with tempfile.TemporaryDirectory() as folder, \
                patch('gensim.models.LdaModel', side_effect=train), \
                patch('gensim.models.CoherenceModel') as coherence:
            coherence.return_value.get_coherence.side_effect = lambda: current_score[0]
            config = {'hyperparameter_search': {}}
            _, params, score, results, selected = lda.run_two_stage_search(
                grid_args(max_k=3), [], {}, [], Path(folder), config)
            self.assertEqual(selected['selected_k'], 2)
            self.assertEqual(selected['coherence_cv'], 0.8)
            self.assertEqual(params['num_topics'], 2)
            self.assertEqual(score, 0.92)
            self.assertEqual([p['num_topics'] for p in trained[:3]], [1, 2, 3])
            self.assertTrue(all(p['num_topics'] == 2 for p in trained[3:]))
            self.assertEqual(results.groupby('Stage').size().to_dict(),
                             {'k_search': 3, 'hyperparameter_search': 6})
            self.assertEqual(len(pd.read_csv(Path(folder) / 'lda_grid_search_results.csv')), 9)
            self.assertEqual(len(pd.read_csv(Path(folder) / 'lda_k_coherence_results.csv')), 3)
            self.assertEqual(len(pd.read_csv(Path(folder) / 'lda_hyperparameter_results.csv')), 6)
            self.assertEqual(config['hyperparameter_search']['selected_k'], 2)

    def test_stage_two_keeps_stage_one_winner_as_reference(self):
        calls = []
        current_score = [None]

        def train(**kwargs):
            calls.append(kwargs.copy())
            current_score[0] = 0.8 if kwargs['passes'] == 20 else 0.3
            return SimpleNamespace(
                alpha=[1.0], eta=[1.0], get_topics=lambda: [[0.6, 0.4]],
                num_topics=2, show_topic=lambda *a, **kw: [('clean', 0.6), ('skin', 0.4)])

        args = grid_args(min_k=2, max_k=2, grid_alpha=['symmetric'],
                         grid_eta=['auto'], grid_passes=[1])
        with tempfile.TemporaryDirectory() as folder, \
                patch('gensim.models.LdaModel', side_effect=train), \
                patch('gensim.models.CoherenceModel') as coherence:
            coherence.return_value.get_coherence.side_effect = lambda: current_score[0]
            _, params, score, _, _ = lda.run_two_stage_search(
                args, [], {}, [], Path(folder), {'hyperparameter_search': {}})
            self.assertEqual(params['passes'], 20)
            self.assertEqual(score, 0.8)
            self.assertEqual([p['passes'] for p in calls], [20, 20, 1])

    def test_failed_k_search_does_not_start_hyperparameter_search(self):
        with tempfile.TemporaryDirectory() as folder, \
                patch('gensim.models.LdaModel', side_effect=ValueError('bad model')) as train:
            with self.assertRaises(RuntimeError):
                lda.run_two_stage_search(grid_args(max_k=2), [], {}, [], Path(folder),
                                         {'hyperparameter_search': {}})
            self.assertEqual(train.call_count, 2)
            self.assertFalse((Path(folder) / 'lda_hyperparameter_results.csv').exists())


if __name__ == '__main__':
    unittest.main()
