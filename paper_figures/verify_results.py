"""Validate selected outputs, scientific values, and retained original files."""
import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))
from build_all import FIGURE_STEMS
from build_tables import TABLE_STEMS
from common.config import ASPECTS, PROJECT
from common.data_utils import dataset_summary, label_matrix, load, reviews


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for extension in ['png', 'pdf']:
        actual = {p.stem for p in (BASE / 'output' / extension).glob('*.' + extension)}
        assert actual == set(FIGURE_STEMS), (extension, actual)
        for stem in actual:
            path = BASE / 'output' / extension / (stem + '.' + extension)
            assert path.stat().st_size > 5000, path
            if extension == 'pdf':
                assert path.read_bytes().startswith(b'%PDF-'), path
    assert {p.stem for p in (BASE / 'tables').glob('*.csv')} == set(TABLE_STEMS)
    meta = dataset_summary()
    assert (meta['raw_reviews'], meta['valid_reviews'], meta['unique_texts']) == (16822, 16292, 12690)
    assert list(meta['split_counts'].values()) == [8882, 1904, 1904]
    assert int(label_matrix().sum()) == 43495
    topics = load('topic_model_metrics')
    assert topics.Topics.tolist() == [16, 12, 16]
    assert topics.Reviews.tolist() == [12741] * 3
    np.testing.assert_allclose(topics.Native_Cv, [.5757062354607918, .6663524880839993, .6447784548214361])
    assert len(load('bertopic_topics')) == 16
    metrics = load('11_absa_model_metrics')
    focal = metrics.loc[metrics.Model.eq('birnn') & metrics.Scenario.eq('focal_loss')].iloc[0]
    np.testing.assert_allclose([focal.F1_Macro, focal.Weighted_F1, focal.Accuracy], [70.01, 88.44, 83.95])
    reg = load('22_rating_regression_oof').set_index('Model').loc['XGBoost']
    np.testing.assert_allclose([reg.RMSE, reg.MBRE, reg.MIBRE, reg.PRED25],
                               [.692109989083627, .1975839901150215, .11243210358083092, .8652712987969555])
    # Recompute performance for all 90 IPA cells from the original reference labels.
    matrix = label_matrix()
    data = reviews()
    for name, column in [('25_ipa_overall', None), ('30_ipa_platform', 'Platform'), ('31_ipa_brand', 'Brand')]:
        frame = load(name)
        groups = frame.groupby(column, sort=False) if column else [('All', frame)]
        for group, sub in groups:
            mask = np.ones(len(data), dtype=bool) if column is None else data[column].eq(group).to_numpy()
            np.testing.assert_allclose(sub.Importance_Pct.sum(), 100, atol=1e-8)
            for row in sub.itertuples():
                labels = matrix[mask, ASPECTS.index(row.Aspect)]
                single = labels.sum(axis=1) == 1
                counts = labels[single].sum(axis=0)
                assert tuple(counts) == (row.N_Positive, row.N_Negative, row.N_Neutral)
                np.testing.assert_allclose(100 * (counts[0] + .5 * counts[2]) / counts.sum(), row.Performance_0_100)
    compare = load('34_ipa_mean_median_comparison')
    assert set(compare.loc[compare.IPA_Quadrant_Mean.ne(compare.IPA_Quadrant_Median), 'Aspect']) == {ASPECTS[1], ASPECTS[2], ASPECTS[9]}

    protected = json.loads((BASE / 'data/protected_sources.json').read_text(encoding='utf-8'))
    for name, digest in protected['sha256'].items():
        assert sha(PROJECT / name) == digest, f'Retained source changed: {name}'
    for name in protected.get('retired_visuals_sha256', {}):
        assert not (PROJECT / name).exists(), f'Retired visual was recreated: {name}'
    for name, expected in protected['checkpoints'].items():
        status = (PROJECT / name).stat()
        assert {'size': status.st_size, 'mtime_ns': status.st_mtime_ns} == expected, name
    nodes = ast.parse((PROJECT / 'Business/ipa_shap_cleanser.py').read_text(encoding='utf-8')).body
    checked_functions = set()
    for node in nodes:
        if isinstance(node, ast.FunctionDef) and node.name in protected.get('business_algorithms_ast_sha256', {}):
            assert hashlib.sha256(ast.dump(node).encode()).hexdigest() == protected['business_algorithms_ast_sha256'][node.name], node.name
            checked_functions.add(node.name)
    assert checked_functions == set(protected.get('business_algorithms_ast_sha256', {}))
    manifest = json.loads((BASE / 'data/manifest.json').read_text(encoding='utf-8'))
    for name, digest in manifest['inputs_sha256'].items():
        assert sha(PROJECT / name) == digest, name
    report = {'figures': 10, 'tables': 13, 'ipa_cells_verified': 90,
              'retained_files_verified': len(protected['sha256']),
              'checkpoints_verified': len(protected['checkpoints']), 'status': 'passed'}
    (BASE / 'data/validation_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False))


if __name__ == '__main__':
    main()
