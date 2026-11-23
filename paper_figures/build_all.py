"""Recreate the selected research figures/tables using saved experiment results."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'scripts'))
os.environ.setdefault('MPLCONFIGDIR', str(BASE / '.cache/matplotlib'))
FIGURE_STEMS = [
    'fig01_review_length', 'fig02_rating_distribution', 'fig03_sentiment_distribution',
    'fig04_birnn_focal_loss', 'fig05_shap_importance_frequency', 'fig06_ipa_overall',
    'fig07_ipa_platforms', 'fig08_ipa_brands', 'fig09_ipa_comparison_heatmap', 'fig10_ipa_sensitivity',
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tables-only', action='store_true')
    group.add_argument('--figures-only', action='store_true')
    args = parser.parse_args()
    if not args.tables_only:
        for stem in FIGURE_STEMS:
            runpy.run_path(str(BASE / 'scripts' / (stem + '.py')), run_name='__main__')
    if not args.figures_only:
        runpy.run_path(str(BASE / 'scripts/build_tables.py'), run_name='__main__')
    from common.data_utils import input_files
    sources = {p.relative_to(BASE.parent).as_posix(): sha(p) for p in input_files()}
    manifest = {'inputs_sha256': sources, 'figures': FIGURE_STEMS,
                'notes': ['Sources are read directly; no training or inference is run.',
                          'Literature comparison rows were supplied in user screenshots.',
                          'PRED(25) is exported as a fraction; historical percentages are converted.']}
    (BASE / 'data/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
