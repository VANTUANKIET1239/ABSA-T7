"""Run the original cleanser IPA independently for the five most-reviewed brands.

Run this file directly with Python/PyCharm. Keep ipa_by_platform_cleanser.py and
ipa_shap_cleanser.py beside it. The shared runner reuses the original algorithms.

The top five are selected by review count AFTER the original numeric/required-label
cleaning. Names differing only in Unicode formatting, whitespace or letter case
are merged. Each CSV review row counts once; the source data is never changed.
Selection is deterministic: descending valid review count, then brand name.
An ineligible selected brand is reported as skipped, not replaced by brand #6.

Each brand has its own 10-fold CV, final XGBoost, native Tree SHAP, original IPA
calculation and three PNG/PDF figures. Combined comparison tables and heatmaps
are saved in the output root. IPA importance/cutoffs are relative to each brand.
The shared pipeline uses 10 IPA aspects, excludes General from model inputs,
and retains reviews mentioning only General.
"""
from pathlib import Path

if __package__:
    from .ipa_by_platform_cleanser import run_grouped_ipa
else:
    from ipa_by_platform_cleanser import run_grouped_ipa


TOP_N_BRANDS = 5
IPA_CUT_METHOD = "mean"
OUTPUT_DIR = Path(__file__).resolve().parent / "ipa_top5_brands_results_cleanser"


def main():
    run_grouped_ipa("brand", OUTPUT_DIR, top_n=TOP_N_BRANDS, cut_method=IPA_CUT_METHOD)


if __name__ == "__main__":
    main()
