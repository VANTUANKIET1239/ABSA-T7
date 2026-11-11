"""Facial cleanser IPA: compare mean/median cutoffs and signed SHAP distributions.

Run this file directly, including with PyCharm Run. Keep IPA-M.py beside it.
The original module supplies preprocessing, models, 10-fold CV, SHAP, performance,
quadrant classification and existing figures without modifying its source.

Two distinct comparisons:
1. IPA mean vs median changes ONLY the two quadrant cutoffs. Aspect importance
   remains the original global mean absolute SHAP; performance is unchanged.
2. Positive/negative distribution figures compare mean and median SIGNED SHAP
   on reviews where that sentiment feature equals 1, matching calculate_xai.
   Mixed reviews can activate both features, following the original encoding.

Additional outputs (PNG at 300 dpi plus PDF for every figure):
    ipa_mean/ipa_attributes.csv, ipa_median/ipa_attributes.csv
    ipa_mean_median_comparison.csv, ipa_mean_median_thresholds.csv
    ipa_mean_median_comparison.png/.pdf
    shap_active_mean_median_summary.csv
    shap_positive_distribution_mean_median.png/.pdf
    shap_negative_distribution_mean_median.png/.pdf

The model is fitted once for SHAP; both IPA methods use those same predictions,
SHAP values and aspect performance scores. All outputs use a separate folder.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


BASE_SCRIPT = Path(__file__).resolve().with_name("IPA-M.py")
OUTPUT_DIR = Path(__file__).resolve().parent / "customer_satisfaction_xai_ipa_mean_median_results_cleanser"
QUADRANT_COLORS = {
    "Concentrate Here": "#c2410c",
    "Keep Up the Good Work": "#15803d",
    "Low Priority": "#64748b",
    "Possible Overkill": "#2563eb",
    "Insufficient Data": "#a3a3a3",
}


def load_original_pipeline():
    """Reuse the original functions instead of maintaining a second algorithm."""
    spec = importlib.util.spec_from_file_location("cleanser_ipa_original", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the original IPA pipeline: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def calculate_both_ipa_methods(pipeline, df, xai):
    """Call the unchanged calculate_ipa for each cutoff; restore its configuration."""
    previous_method, previous_output = pipeline.IPA_CUT_METHOD, pipeline.OUTPUT_DIR
    tables = {}
    try:
        for method in ("mean", "median"):
            pipeline.IPA_CUT_METHOD = method
            pipeline.OUTPUT_DIR = previous_output / f"ipa_{method}"
            pipeline.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
            tables[method] = pipeline.calculate_ipa(df, xai)
    finally:
        pipeline.IPA_CUT_METHOD = previous_method
        pipeline.OUTPUT_DIR = previous_output

    keys = ["Aspect", "Importance_Pct", "Performance_0_100",
            "N_Performance_Valid", "N_Mixed_Excluded"]
    mean_table, median_table = tables["mean"], tables["median"]
    pd.testing.assert_frame_equal(mean_table[keys], median_table[keys])
    comparison = mean_table[keys].copy()
    for method, table in tables.items():
        comparison[f"IPA_Quadrant_{method.title()}"] = table["IPA_Quadrant"]
        comparison[f"Importance_Cut_{method.title()}"] = table["Importance_Cut"]
        comparison[f"Performance_Cut_{method.title()}"] = table["Performance_Cut"]
    comparison["Quadrant_Changed"] = (
        comparison["IPA_Quadrant_Mean"] != comparison["IPA_Quadrant_Median"]
    )
    comparison.to_csv(previous_output / "ipa_mean_median_comparison.csv",
                      index=False, encoding="utf-8-sig")

    threshold_rows = []
    for method, table in tables.items():
        counts = table["IPA_Quadrant"].value_counts()
        threshold_rows.append({
            "Cut_Method": method,
            "Importance_Cut": float(table["Importance_Cut"].iloc[0]),
            "Performance_Cut": float(table["Performance_Cut"].iloc[0]),
            "N_Aspects": len(table),
            **{quadrant: int(counts.get(quadrant, 0)) for quadrant in QUADRANT_COLORS},
        })
    thresholds = pd.DataFrame(threshold_rows)
    thresholds.to_csv(previous_output / "ipa_mean_median_thresholds.csv",
                      index=False, encoding="utf-8-sig")
    changed = comparison.loc[comparison["Quadrant_Changed"]]
    print("\n=== IPA MEAN VS MEDIAN CUTOFFS: SAME IMPORTANCE AND PERFORMANCE ===")
    print(thresholds.round(4).to_string(index=False))
    print(comparison[["Aspect", "Importance_Pct", "Performance_0_100",
                      "IPA_Quadrant_Mean", "IPA_Quadrant_Median",
                      "Quadrant_Changed"]].round(4).to_string(index=False))
    print(f"Changed quadrants: {len(changed)}/{len(comparison)} aspects.")
    return mean_table, median_table, comparison


def readable_aspect(aspect):
    import re
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", aspect)


def plot_ipa_cutoff_comparison(pipeline, mean_table, median_table, comparison):
    """Same coordinates/axis limits in both panels; only cutoffs and colors differ."""
    valid = comparison.dropna(subset=["Performance_0_100", "Importance_Pct"])
    cuts_p = [float(table["Performance_Cut"].iloc[0]) for table in (mean_table, median_table)]
    cuts_i = [float(table["Importance_Cut"].iloc[0]) for table in (mean_table, median_table)]
    x_min = max(0, min(valid["Performance_0_100"].min(), *cuts_p) - 6)
    x_max = min(100, max(valid["Performance_0_100"].max(), *cuts_p) + 6)
    y_max = max(valid["Importance_Pct"].max(), *cuts_i) * 1.22
    changed = set(comparison.loc[comparison["Quadrant_Changed"], "Aspect"])
    offsets = {"General": (6, 7), "Packaging": (6, -15), "ShopService": (-7, 11),
               "SkinCompatibilitySafety": (-7, -15), "Fragrance": (-7, 8),
               "TextureFoaming": (-7, 8)}
    # Spread the dense lower-right cluster over separate label rows.
    cluster_label_rows = {"HydrationSkinComfort": .22, "Delivery": .18,
                          "CleansingEffectiveness": .14, "PriceValue": .10,
                          "IngredientsFormulation": .06}
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), sharex=True, sharey=True)
    for ax, method, table in zip(axes, ("Mean", "Median"), (mean_table, median_table)):
        i_cut = float(table["Importance_Cut"].iloc[0])
        p_cut = float(table["Performance_Cut"].iloc[0])
        ax.axhline(i_cut, color="#475569", ls="--", linewidth=1)
        ax.axvline(p_cut, color="#475569", ls="--", linewidth=1)
        for rank, row in table.reset_index(drop=True).iterrows():
            if not (np.isfinite(row["Performance_0_100"]) and np.isfinite(row["Importance_Pct"])):
                continue
            is_changed = row["Aspect"] in changed
            ax.scatter(row["Performance_0_100"], row["Importance_Pct"],
                       c=QUADRANT_COLORS[row["IPA_Quadrant"]], s=100,
                       edgecolors="black" if is_changed else "white",
                       linewidths=1.4 if is_changed else .8, zorder=3)
            label = readable_aspect(row["Aspect"]) + (" *" if is_changed else "")
            point = (row["Performance_0_100"], row["Importance_Pct"])
            if row["Aspect"] in cluster_label_rows:
                ax.annotate(label, point,
                            xytext=(x_max - .5, y_max * cluster_label_rows[row["Aspect"]]),
                            textcoords="data", ha="right", va="center", fontsize=8,
                            arrowprops=dict(arrowstyle="-", color="#94a3b8", lw=.7),
                            bbox=dict(facecolor="white", alpha=.9, edgecolor="none", pad=.8))
            else:
                dx, dy = offsets.get(row["Aspect"], (6, 7))
                ax.annotate(label, point,
                            xytext=(dx, dy), textcoords="offset points", fontsize=8,
                            ha="left" if dx > 0 else "right",
                            bbox=dict(facecolor="white", alpha=.8, edgecolor="none", pad=.8))
        ax.set_title(f"{method} cutoffs: importance={i_cut:.3f}%, performance={p_cut:.3f}")
        ax.set_xlabel("Sentiment-derived performance (0-100; mentioned reviews)")
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(0, y_max)
        ax.grid(alpha=.12)
    axes[0].set_ylabel("Relative mean absolute SHAP importance (%)")
    handles = [Line2D([], [], marker="o", linestyle="none", color=color,
                      label=quadrant, markersize=7)
               for quadrant, color in QUADRANT_COLORS.items() if quadrant != "Insufficient Data"]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(.5, .055), fontsize=9)
    fig.suptitle("Facial cleanser IPA: mean vs median cutoff comparison", fontsize=15, y=.98)
    fig.text(.5, .025, "* Black outline: aspect changes quadrant. Importance and performance are identical in both panels.",
             ha="center", fontsize=9, color="#475569")
    fig.tight_layout(rect=(0, .13, 1, .94))
    pipeline.save_figure(fig, "ipa_mean_median_comparison")


def summarize_active_shap(X, shap_values, aspects):
    """Actual signed SHAP samples, matching the original active-state mean/median."""
    if shap_values.shape != X.shape:
        raise ValueError(f"SHAP shape {shap_values.shape} does not match feature shape {X.shape}")
    rows, samples = [], {}
    for aspect in aspects:
        for sentiment, suffix in (("positive", "Pos"), ("negative", "Neg")):
            feature = f"{aspect}_{suffix}"
            active = X[feature].to_numpy() == 1
            values = np.asarray(shap_values[active, X.columns.get_loc(feature)], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite active SHAP values for {feature}")
            samples[(aspect, sentiment)] = values
            row = {"Aspect": aspect, "Sentiment": sentiment, "Feature": feature,
                   "N_Active": len(values)}
            if len(values):
                row.update({"Mean_Signed_SHAP": float(np.mean(values)),
                            "Median_Signed_SHAP": float(np.median(values)),
                            "Std_Signed_SHAP": float(np.std(values)),
                            "Min": float(np.min(values)), "P05": float(np.percentile(values, 5)),
                            "P25": float(np.percentile(values, 25)), "P75": float(np.percentile(values, 75)),
                            "P95": float(np.percentile(values, 95)), "Max": float(np.max(values))})
            else:
                row.update({key: np.nan for key in ["Mean_Signed_SHAP", "Median_Signed_SHAP",
                           "Std_Signed_SHAP", "Min", "P05", "P25", "P75", "P95", "Max"]})
            row["Mean_Minus_Median"] = row["Mean_Signed_SHAP"] - row["Median_Signed_SHAP"]
            rows.append(row)
    return pd.DataFrame(rows), samples


def plot_shap_mean_median_distributions(pipeline, X, shap_values, xai):
    """Two full-sample violin/box figures: positive and negative active features."""
    aspects = xai["Aspect"].tolist()
    summary, samples = summarize_active_shap(X, shap_values, aspects)
    # Check against the existing XAI computation, without replacing its results.
    indexed = xai.set_index("Aspect")
    for row in summary.itertuples():
        suffix = "Pos" if row.Sentiment == "positive" else "Neg"
        original = indexed.loc[row.Aspect]
        assert row.N_Active == int(original[f"N_{row.Sentiment.title()}_Active"])
        assert np.isclose(row.Mean_Signed_SHAP, original[f"Mean_SHAP_{suffix}_Active"],
                          atol=1e-7, rtol=1e-6, equal_nan=True)
        assert np.isclose(row.Median_Signed_SHAP, original[f"Median_SHAP_{suffix}_Active"],
                          atol=1e-7, rtol=1e-6, equal_nan=True)
    summary.to_csv(pipeline.OUTPUT_DIR / "shap_active_mean_median_summary.csv",
                   index=False, encoding="utf-8-sig")

    for sentiment, color in (("positive", "#a5c7e5"), ("negative", "#d7b1c4")):
        fig, ax = plt.subplots(figsize=(13.5, max(8, .72 * len(aspects) + 2.0)))
        subset = summary[summary["Sentiment"] == sentiment].set_index("Aspect").loc[aspects]
        available = [(rank, samples[(aspect, sentiment)]) for rank, aspect in enumerate(aspects)
                     if len(samples[(aspect, sentiment)])]
        varying = [(rank, values) for rank, values in available
                   if len(values) > 1 and float(np.ptp(values)) > 0]
        if varying:
            violin = ax.violinplot([v for _, v in varying], positions=[r for r, _ in varying],
                                   orientation="horizontal", widths=.70, showmeans=False,
                                   showmedians=False, showextrema=False)
            for body in violin["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor("#64748b")
                body.set_alpha(.6)
        if available:
            ax.boxplot([v for _, v in available], positions=[r for r, _ in available],
                       orientation="horizontal", widths=.20, whis=(5, 95), showfliers=False,
                       patch_artist=True, manage_ticks=False,
                       boxprops=dict(facecolor="white", edgecolor="#64748b", alpha=.8),
                       medianprops=dict(color="#ea580c", linewidth=1.6),
                       whiskerprops=dict(color="#64748b"), capprops=dict(color="#64748b"))
        positions = np.arange(len(aspects))
        ax.scatter(subset["Mean_Signed_SHAP"], positions, c="#1d4ed8", marker="D",
                   s=42, edgecolors="white", linewidths=.5, label="Mean signed SHAP", zorder=5)
        ax.scatter(subset["Median_Signed_SHAP"], positions, c="#ea580c", marker="s",
                   s=35, edgecolors="white", linewidths=.5, label="Median signed SHAP", zorder=5)
        labels = [f"{readable_aspect(aspect)}  (n={int(subset.loc[aspect, 'N_Active']):,})"
                  for aspect in aspects]
        ax.set_yticks(positions, labels)
        ax.set_ylim(-.7, len(aspects) - .3)
        ax.invert_yaxis()
        ax.axvline(0, color="#334155", linewidth=1, ls="--", alpha=.7)
        ax.set_xlabel("Signed Tree SHAP contribution (rating units)")
        ax.set_title(f"{sentiment.title()} sentiment: SHAP distributions with mean and median", pad=16)
        ax.legend(loc="best", framealpha=.95)
        ax.grid(axis="x", alpha=.15)
        for rank, aspect in enumerate(aspects):
            if not len(samples[(aspect, sentiment)]):
                ax.text(.02, rank, "No active reviews", transform=ax.get_yaxis_transform(),
                        va="center", fontsize=9, color="#64748b")
        fig.text(.02, .042, "Violin: full active-review sample. Box: 25th-75th percentiles; whiskers: 5th-95th. No SHAP values are trimmed.",
                 fontsize=9, color="#475569")
        fig.text(.02, .024, "Violin widths are normalized within each aspect; n gives the active-review sample size.",
                 fontsize=9, color="#475569")
        fig.text(.02, .006, "Mean/median here summarize signed SHAP within an active sentiment feature; they do not replace IPA importance or its cutoffs.",
                 fontsize=8.5, color="#475569")
        fig.tight_layout(rect=(0, .075, 1, 1))
        pipeline.save_figure(fig, f"shap_{sentiment}_distribution_mean_median")
    print("\n=== ACTIVE-STATE SIGNED SHAP: MEAN VS MEDIAN ===")
    print(summary[["Aspect", "Sentiment", "N_Active", "Mean_Signed_SHAP",
                   "Median_Signed_SHAP", "Mean_Minus_Median"]].round(5).to_string(index=False))
    return summary


def main():
    pipeline = load_original_pipeline()
    pipeline.OUTPUT_DIR = OUTPUT_DIR
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, raw_features, y, sentiment_cols = pipeline.preprocess_data(pipeline.DATA_PATH)
    print(f"Loaded {len(df)} valid reviews ({len(sentiment_cols)} sentiment features).")

    # Original prediction/XAI/IPA pipeline: same models, settings and calculations.
    pipeline.evaluate_cv(raw_features, y, sentiment_cols)
    X, model, scaler, shap_values, base_values = pipeline.fit_xgboost_for_explanation(raw_features, y, sentiment_cols)
    xai_results = pipeline.calculate_xai(X, shap_values)
    ipa = pipeline.calculate_ipa(df, xai_results)
    pipeline.plot_importance(ipa)
    pipeline.plot_reward_penalty(ipa)
    pipeline.plot_ipa(ipa)
    pipeline.plot_xai_shap_summary(X, shap_values)
    pipeline.plot_xai_local_waterfall(X, y, shap_values, base_values, model)
    pipeline.plot_ipa_penalty_bubble(ipa)

    # Add comparisons using the exact same fitted model and SHAP matrix.
    mean_ipa, median_ipa, comparison = calculate_both_ipa_methods(pipeline, df, xai_results)
    plot_ipa_cutoff_comparison(pipeline, mean_ipa, median_ipa, comparison)
    plot_shap_mean_median_distributions(pipeline, X, shap_values, xai_results)

    low_support = ipa.loc[ipa["Low_Mention_Support"], "Aspect"].tolist()
    weak_negative = ipa.loc[ipa["Weak_Negative_Evidence"], "Aspect"].tolist()
    if low_support:
        print(f"WARNING: low performance sample support (<{pipeline.MIN_MENTIONS_FOR_RELIABLE_INTERPRETATION}): {low_support}")
    if weak_negative:
        print(f"WARNING: few negative labels (<{pipeline.MIN_NEGATIVE_FOR_REWARD_PENALTY}): {weak_negative}; "
              "interpret median negative SHAP cautiously.")
    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")
    print("XAI describes model predictions, not causal product effects.")
    print("Reference ABSA labels are used; predicted-PhoBERT end-to-end validity is not assessed here.")


if __name__ == "__main__":
    main()
