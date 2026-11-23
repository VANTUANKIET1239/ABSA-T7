"""Run the original cleanser IPA separately for every platform.

Run this file directly with Python/PyCharm. Keep ipa_shap_cleanser.py beside it.
The shared preprocessing, 10 IPA aspects, sentiment encoding, regression models,
10-fold CV, XGBoost Tree SHAP, aspect performance and IPA classification are reused.
General is excluded from the model inputs and IPA; General-only reviews are kept.

Each group gets an independent fitted model, SHAP baseline and IPA cutoffs.
Importance percentages are normalized within each group. Review counts count CSV
rows after the original required-column cleaning, without deduplicating texts.
Names are grouped after Unicode/whitespace normalization and case folding; the
source CSV is never modified. Missing group names are recorded and excluded.

Outputs: per-group original CV/XAI/IPA CSVs and three original PNG/PDF figures;
combined IPA/CV tables, a group execution summary, quadrant comparison table,
and importance/performance heatmaps. This module also supplies the shared runner
used by ipa_top5_brands_cleanser.py, avoiding duplicated analysis logic.
"""
from __future__ import annotations

from pathlib import Path
import re
import traceback
import unicodedata

import numpy as np
import pandas as pd

if __package__:
    from . import ipa_shap_cleanser as pipeline
else:
    import ipa_shap_cleanser as pipeline


OUTPUT_DIR = Path(__file__).resolve().parent / "ipa_by_platform_results_cleanser"
IPA_CUT_METHOD = "mean"
PLATFORM_DISPLAY_NAMES = {"tiki": "Tiki", "shopee": "Shopee", "lazada": "Lazada"}


def select_groups(df, group_column, top_n=None, display_names=None):
    """Rank groups by valid review rows, merging only formatting/case variants."""
    if group_column not in df.columns:
        raise ValueError(f"Missing grouping column: {group_column}")
    if top_n is not None and (not isinstance(top_n, int) or top_n < 1):
        raise ValueError("top_n must be a positive integer or None")
    raw_names = (df[group_column].astype("string").str.normalize("NFKC")
                 .str.strip().str.replace(r"\s+", " ", regex=True))
    keys = raw_names.str.casefold().fillna("")
    valid = keys.ne("")
    aliases = (pd.DataFrame({"Group_Key": keys[valid], "Source_Name": raw_names[valid]})
               .groupby(["Group_Key", "Source_Name"], as_index=False)
               .size().rename(columns={"size": "N_Valid_Reviews"}))
    if aliases.empty:
        raise ValueError(f"No valid non-empty {group_column} names remain")
    preferred = (aliases.sort_values(["Group_Key", "N_Valid_Reviews", "Source_Name"],
                                    ascending=[True, False, True], kind="stable")
                 .drop_duplicates("Group_Key").set_index("Group_Key")["Source_Name"].to_dict())
    preferred.update(display_names or {})
    counts = keys[valid].value_counts().rename_axis("Group_Key").reset_index(name="N_Valid_Reviews")
    counts["Group"] = counts["Group_Key"].map(preferred)
    counts = counts.sort_values(["N_Valid_Reviews", "Group"], ascending=[False, True],
                                kind="stable").reset_index(drop=True)
    counts.insert(0, "Rank", np.arange(1, len(counts) + 1))
    counts["Selected"] = True if top_n is None else counts["Rank"].le(top_n)
    aliases["Group"] = aliases["Group_Key"].map(preferred)
    return keys, counts, aliases


def group_output_directory(output_dir, rank, label):
    """Readable, collision-free directory name with a ranking prefix."""
    ascii_name = unicodedata.normalize("NFKD", label).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_name.casefold()).strip("_") or "group"
    output = Path(output_dir).resolve()
    path = output / f"{rank:02d}_{slug}"
    if not path.resolve().is_relative_to(output):
        raise ValueError(f"Group output escaped its root: {label}")
    return path


def plot_group_heatmap(combined, group_label, group_order, value_column, output_stem, title):
    """Compare aligned aspect rows; mark sparse evidence without imputing values."""
    matrix = combined.pivot(index="Aspect", columns=group_label, values=value_column)
    matrix = matrix.reindex(index=pipeline.ASPECTS, columns=group_order)
    support = combined.pivot(index="Aspect", columns=group_label, values="N_Performance_Valid")
    support = support.reindex(index=pipeline.ASPECTS, columns=group_order)
    group_counts = combined.groupby(group_label)["Review_N"].first()
    values = matrix.to_numpy(dtype=float)
    performance = value_column == "Performance_0_100"
    vmax = 100 if performance else max(1, float(np.nanmax(values)))
    fig, ax = pipeline.plt.subplots(figsize=(max(9, 1.65 * len(group_order) + 4), 8.6))
    cmap = pipeline.plt.get_cmap("YlGnBu" if performance else "Blues").copy()
    cmap.set_bad("#e2e8f0")
    im = ax.imshow(np.ma.masked_invalid(values), aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(group_order)),
                  [f"{name}\nn={int(group_counts[name]):,}" for name in group_order])
    names = [re.sub(r"(?<=[a-z])(?=[A-Z])", " ", aspect) for aspect in pipeline.ASPECTS]
    ax.set_yticks(range(len(names)), names)
    for row in range(len(names)):
        for col in range(len(group_order)):
            value = values[row, col]
            sparse = support.iloc[row, col] < pipeline.MIN_MENTIONS_FOR_RELIABLE_INTERPRETATION
            text = (f"{value:.1f}" if performance else f"{value:.2f}") if np.isfinite(value) else "NA"
            if sparse and np.isfinite(value):
                text += "*"
            ax.text(col, row, text, ha="center", va="center", fontsize=10,
                    color="white" if np.isfinite(value) and value / vmax > .62 else "#0f172a")
    ax.set_title(title, pad=15)
    color_label = "Sentiment-derived performance (0-100)" if performance else "Within-group mean absolute SHAP importance (%)"
    fig.colorbar(im, ax=ax, pad=.025, label=color_label)
    missing_note = "NA: no valid sentiment mentions." if performance else "NA: unavailable importance."
    fig.text(.02, .035,
             f"* Fewer than {pipeline.MIN_MENTIONS_FOR_RELIABLE_INTERPRETATION} performance-valid aspect mentions. {missing_note}",
             fontsize=8.5, color="#475569")
    fig.text(.02, .012, "Each group has its own fitted model and IPA cutoffs; importance is normalized within group.",
             fontsize=8.5, color="#475569")
    fig.tight_layout(rect=(0, .065, 1, 1))
    pipeline.save_figure(fig, output_stem)


def run_grouped_ipa(group_column, output_dir, *, top_n=None, display_names=None, cut_method="mean"):
    """Apply the original pipeline independently to the selected review groups."""
    if cut_method not in {"mean", "median"}:
        raise ValueError("cut_method must be 'mean' or 'median'")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    group_label = "Platform" if group_column == "platform" else "Brand" if group_column == "brand" else "Group"
    original_output = pipeline.OUTPUT_DIR
    original_cut = pipeline.IPA_CUT_METHOD
    original_save = pipeline.save_figure
    records, ipa_tables, cv_tables = [], [], []
    try:
        pipeline.OUTPUT_DIR = output_dir
        pipeline.IPA_CUT_METHOD = cut_method
        df, raw_features, y, sentiment_cols = pipeline.preprocess_data(pipeline.DATA_PATH)
        keys, counts, aliases = select_groups(df, group_column, top_n, display_names)
        counts.to_csv(output_dir / "group_review_counts.csv", index=False, encoding="utf-8-sig")
        aliases.to_csv(output_dir / "group_name_aliases.csv", index=False, encoding="utf-8-sig")
        selected = counts.loc[counts["Selected"]].copy()
        # Two validation rows per fold are needed for the original R2 calculation.
        minimum_rows = 2 * pipeline.N_SPLITS
        pd.DataFrame([{
            "Grouping_Column": group_column,
            "Valid_Reviews_Before_Group_Filter": len(df),
            "Missing_Group_Reviews": int(keys.eq("").sum()),
            "Selected_Groups": len(selected),
            "Selected_Reviews": int(selected["N_Valid_Reviews"].sum()),
            "Minimum_Rows_For_CV": minimum_rows,
            "IPA_Cut_Method": cut_method,
        }]).to_csv(output_dir / "group_selection_summary.csv", index=False, encoding="utf-8-sig")
        print(f"\n=== IPA BY {group_label.upper()}: {len(selected)} SELECTED GROUPS ===", flush=True)
        print(selected[["Rank", "Group", "N_Valid_Reviews"]].to_string(index=False), flush=True)

        for row in selected.itertuples(index=False):
            mask = keys.eq(row.Group_Key)
            group_df = df.loc[mask].reset_index(drop=True)
            group_features = raw_features.loc[mask].reset_index(drop=True)
            group_y = y.loc[mask].reset_index(drop=True)
            folder = group_output_directory(output_dir, row.Rank, row.Group)
            record = {group_label: row.Group, "Group_Key": row.Group_Key, "Rank": row.Rank,
                      "Review_N": len(group_df), "Status": "pending", "Reason": "",
                      "Output_Subdirectory": folder.name}
            records.append(record)
            valid_aspects = sum(group_df[f"{aspect}_State"].isin(["Positive", "Neutral", "Negative"]).any()
                                for aspect in pipeline.ASPECTS)
            if len(group_df) < minimum_rows:
                record.update(Status="skipped", Reason=f"Need >= {minimum_rows} rows for {pipeline.N_SPLITS}-fold CV")
            elif group_y.nunique() < 2:
                record.update(Status="skipped", Reason="Rating is constant; no varying satisfaction target")
            elif valid_aspects < 2:
                record.update(Status="skipped", Reason="Fewer than two aspects have valid sentiment mentions")
            else:
                folder.mkdir(parents=True, exist_ok=True)
                pipeline.OUTPUT_DIR = folder

                def save_group_figure(fig, stem, label=row.Group):
                    fig.suptitle(f"{group_label}: {label}", fontsize=13, y=1.035)
                    original_save(fig, stem)

                pipeline.save_figure = save_group_figure
                print(f"\n--- {group_label}: {row.Group} | {len(group_df):,} reviews | original {pipeline.N_SPLITS}-fold CV ---",
                      flush=True)
                try:
                    cv = pipeline.evaluate_cv(group_features, group_y, sentiment_cols)
                    X, model, scaler, shap_values, base_values = pipeline.fit_xgboost_for_explanation(
                        group_features, group_y, sentiment_cols)
                    xai = pipeline.calculate_xai(X, shap_values)
                    ipa = pipeline.calculate_ipa(group_df, xai)
                    pipeline.plot_importance(ipa)
                    pipeline.plot_reward_penalty(ipa)
                    pipeline.plot_ipa(ipa)
                    tagged_ipa = ipa.copy()
                    tagged_ipa.insert(0, group_label, row.Group)
                    tagged_ipa.insert(1, "Group_Rank", row.Rank)
                    ipa_tables.append(tagged_ipa)
                    tagged_cv = cv.copy()
                    tagged_cv.insert(0, group_label, row.Group)
                    cv_tables.append(tagged_cv)
                    xgb_metrics = cv.set_index("Model").loc["XGBoost"]
                    record.update(
                        Status="done", N_Valid_Aspects=valid_aspects,
                        Importance_Cut=float(ipa["Importance_Cut"].iloc[0]),
                        Performance_Cut=float(ipa["Performance_Cut"].iloc[0]),
                        Cut_Method=cut_method, Low_Support_Aspects=int(ipa["Low_Mention_Support"].sum()),
                        Weak_Negative_Aspects=int(ipa["Weak_Negative_Evidence"].sum()),
                        XGBoost_RMSE=float(xgb_metrics["RMSE"]),
                        XGBoost_MAE=float(xgb_metrics["MAE"]), XGBoost_R2=float(xgb_metrics["R2"]),
                    )
                    print(f"Finished {row.Group}; low-support aspects: {record['Low_Support_Aspects']}; "
                          f"aspects with few negative mentions: {record['Weak_Negative_Aspects']}", flush=True)
                except Exception as error:
                    record.update(Status="failed", Reason=f"{type(error).__name__}: {error}")
                    traceback.print_exc()
                finally:
                    pipeline.save_figure = original_save
                    pipeline.OUTPUT_DIR = output_dir
            pd.DataFrame(records).to_csv(output_dir / "group_run_summary.csv", index=False, encoding="utf-8-sig")

        if ipa_tables:
            combined = pd.concat(ipa_tables, ignore_index=True)
            combined.to_csv(output_dir / "ipa_all_groups.csv", index=False, encoding="utf-8-sig")
            pd.concat(cv_tables, ignore_index=True).to_csv(output_dir / "model_comparison_all_groups.csv",
                                                          index=False, encoding="utf-8-sig")
            order = [record[group_label] for record in records if record["Status"] == "done"]
            quadrants = combined.pivot(index="Aspect", columns=group_label, values="IPA_Quadrant")
            quadrants.reindex(index=pipeline.ASPECTS, columns=order).to_csv(
                output_dir / "ipa_quadrants_comparison.csv", encoding="utf-8-sig")
            plot_group_heatmap(combined, group_label, order, "Importance_Pct", "aspect_importance_by_group",
                               f"Facial cleanser: SHAP importance by {group_label.lower()}")
            plot_group_heatmap(combined, group_label, order, "Performance_0_100", "aspect_performance_by_group",
                               f"Facial cleanser: sentiment performance by {group_label.lower()}")
        print("\n=== GROUP EXECUTION SUMMARY ===", flush=True)
        summary = pd.DataFrame(records)
        print(summary.to_string(index=False), flush=True)
        print(f"\nOutputs: {output_dir}", flush=True)
        print("Groups use independent models and relative, within-group importance/cutoffs.", flush=True)
        failures = [record[group_label] for record in records if record["Status"] == "failed"]
        if failures:
            raise RuntimeError(f"IPA failed for groups: {failures}; see group_run_summary.csv")
        if not ipa_tables:
            raise ValueError("No group was eligible for IPA; see group_run_summary.csv")
        return summary
    finally:
        pipeline.OUTPUT_DIR = original_output
        pipeline.IPA_CUT_METHOD = original_cut
        pipeline.save_figure = original_save


def main():
    run_grouped_ipa("platform", OUTPUT_DIR, display_names=PLATFORM_DISPLAY_NAMES,
                    cut_method=IPA_CUT_METHOD)


if __name__ == "__main__":
    main()
