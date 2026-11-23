"""Compute IPA independently for the five most-reviewed brands.

Reuse ipa_by_platform_cleanser.run_grouped_ipa and the canonical IPA algorithms.
Write combined IPA and run-summary CSVs only; selected plots live in paper_figures.
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
