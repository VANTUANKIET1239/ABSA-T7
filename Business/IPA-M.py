"""Facial cleanser reviews: Customer Satisfaction Prediction -> Explainable AI -> IPA.

Adapted from the user's SHAP-Kano/interaction pipeline.

Data requirements
-----------------
CSV fields: price, discount, rating, label
ABSA label example:
    {CleansingEffectiveness#positive};{SkinCompatibilitySafety#negative}
Other columns (product_name, brand, cleaned_comment, ...) are retained but not used in modeling.

Outputs
-------
model_comparison_oof.csv, model_cv_fold_metrics.csv, model_cv_mean_std.csv,
xai_feature_importance_all.csv, xai_aspect_summary.csv, ipa_attributes.csv,
ipa_matrix.png/.pdf, aspect_importance.png/.pdf, reward_penalty.png/.pdf,
xai_shap_summary.png/.pdf, xai_local_waterfall.png/.pdf,
ipa_penalty_bubble.png/.pdf, xai_local_explanation.csv,
modeling_data_summary.csv

Requires: pip install numpy pandas matplotlib scikit-learn xgboost
XGBoost supplies Tree SHAP natively (no extra shap dependency required).

Interpretation notes
--------------------
1. Input sentiment features come from the CSV's 'label' column (reference labels),
   not automatically from a trained PhoBERT model.
2. Model-derived SHAP importance is NOT a causal effect or stated importance.
3. Performance reflects sentiment among reviews MENTIONING a specific aspect.
4. The IPA cutoffs are relative to the 11 analyzed aspects, not absolute standards.
5. The final XGBoost model is deliberately chosen for native Tree SHAP;
   do not call it the best predictive model if CV results show otherwise.
"""
from __future__ import annotations

from pathlib import Path
import re
import warnings

import matplotlib
matplotlib.use("Agg")  # Allow chart creation on headless machines.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import xgboost as xgb
from xgboost import XGBRegressor

# ============================================================
# 1. CONFIG
# ============================================================
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "crawl-data" / "data" / "combined_reviews_brand_filled_language_filtered.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "customer_satisfaction_xai_ipa_results_cleanser"
RANDOM_STATE = 42
N_SPLITS = 10
IPA_CUT_METHOD = "mean"  # "mean" or "median"
MIN_MENTIONS_FOR_RELIABLE_INTERPRETATION = 30
MIN_NEGATIVE_FOR_REWARD_PENALTY = 20
# Additional research figures
XAI_SCATTER_MAX_REVIEWS = 1800  # Scatter sample only; summary statistics use all data.
XAI_SCATTER_TOP_FEATURES = 12
LOCAL_WATERFALL_REVIEW_INDEX = 0  # Zero-based index; a fixed, reproducible example.
LOCAL_WATERFALL_MAX_FEATURES = 10

ASPECTS = [
    "CleansingEffectiveness",
    "HydrationSkinComfort",
    "SkinCompatibilitySafety",
    "TextureFoaming",
    "Fragrance",
    "IngredientsFormulation",
    "Packaging",
    "PriceValue",
    "ShopService",
    "Delivery",
    "General",
]
NUMERIC_COLS = ["price", "discount"]
SCALED_COLS = ["price_scaled", "discount_scaled"]
LABEL_PATTERN = re.compile(r"\{\s*([^#{}]+?)\s*#\s*([^{}]+?)\s*\}")
SENTIMENTS = {"positive", "negative", "neutral"}


def extract_aspects(label_str: object) -> list[tuple[str, str]]:
    if pd.isna(label_str):
        return []
    return LABEL_PATTERN.findall(str(label_str))


def parse_price(value: object) -> float:
    """Parse VND prices such as 129000, 129000.0, 129.000, 1.250.000.

    Grouping separators must appear in 3-digit groups; otherwise '.' / ',' is
    treated as a decimal separator. Check source conventions for unusual data.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).replace("₫", "").replace("VND", "").replace("đ", "")
    s = re.sub(r"\s+", "", s).strip()
    if not s:
        return np.nan
    # Decimal .0 from a CSV-exported floating-point column.
    if re.fullmatch(r"[+-]?\d+\.0+", s):
        return float(s)
    if re.fullmatch(r"[+-]?\d{1,3}(?:[.,]\d{3})+", s):
        return float(s.replace(".", "").replace(",", ""))
    if re.fullmatch(r"[+-]?\d+(?:[.,]\d+)?", s):
        return float(s.replace(",", "."))
    # Combined grouping + decimals: 1.250.000,00 or 1,250,000.00.
    if re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{3})+,\d{1,2}", s):
        return float(s.replace(".", "").replace(",", "."))
    if re.fullmatch(r"[+-]?\d{1,3}(?:,\d{3})+\.\d{1,2}", s):
        return float(s.replace(",", ""))
    return np.nan


def parse_discount(value: object) -> float:
    """Interpret 15%, 15 as 0.15, but 0.15 as an existing fraction.

    Without '%', the convention 1 == 100% is assumed; verify if 1 means 1%.
    """
    if pd.isna(value):
        return np.nan
    s = str(value).strip().replace(" ", "")
    percent_mark = "%" in s
    s = s.replace("%", "").replace(",", ".")
    try:
        number = float(s)
    except ValueError:
        return np.nan
    return number / 100 if percent_mark or abs(number) > 1 else number


def preprocess_data(path: Path):
    # dtype='string' protects the original thousands separators in prices.
    df = pd.read_csv(path, encoding="utf-8", dtype={"price": "string", "discount": "string"})
    required = {"price", "discount", "rating", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    raw_count = len(df)
    df["price"] = df["price"].map(parse_price).astype(float)
    df["discount"] = df["discount"].map(parse_discount).astype(float)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["price", "discount", "rating", "label"]).reset_index(drop=True)
    if len(df) < N_SPLITS:
        raise ValueError(f"Need >= {N_SPLITS} valid reviews for {N_SPLITS}-fold CV")

    # Preserve the original encoding for the 22 cleanser sentiment features:
    # Positive and Negative = binary; Neutral and None = both zero.
    # For IPA, retain four sentiment states per aspect separately.
    # A review with conflicting sentiments for the same aspect is marked MIXED,
    # retained for the original predictive features but excluded from that aspect's
    # sentiment-derived performance score (no arbitrary sentiment assignment).
    unknown_labels: set[tuple[str, str]] = set()
    label_issues = 0
    for aspect in ASPECTS:
        df[f"{aspect}_Pos"] = np.zeros(len(df), dtype=np.int8)
        df[f"{aspect}_Neg"] = np.zeros(len(df), dtype=np.int8)
        df[f"{aspect}_State"] = "None"

    for row_id, raw in enumerate(df["label"]):
        state_map: dict[str, set[str]] = {}
        for aspect, sentiment in extract_aspects(raw):
            aspect, sentiment = aspect.strip(), sentiment.strip().lower()
            if aspect not in ASPECTS or sentiment not in SENTIMENTS:
                unknown_labels.add((aspect, sentiment))
                continue
            state_map.setdefault(aspect, set()).add(sentiment)
        for aspect, sentiments in state_map.items():
            if "positive" in sentiments:
                df.at[row_id, f"{aspect}_Pos"] = 1
            if "negative" in sentiments:
                df.at[row_id, f"{aspect}_Neg"] = 1
            if len(sentiments) > 1:
                df.at[row_id, f"{aspect}_State"] = "Mixed"
                label_issues += 1
            else:
                df.at[row_id, f"{aspect}_State"] = next(iter(sentiments)).capitalize()

    if unknown_labels:
        print(f"WARNING: unrecognized aspect/sentiment label pairs: {sorted(unknown_labels)[:15]}")
    if label_issues:
        print(f"WARNING: {label_issues} conflicting aspect-review labels marked Mixed for IPA performance.")

    sentiment_cols = [f"{a}_{pol}" for a in ASPECTS for pol in ("Pos", "Neg")]
    raw_features = df[NUMERIC_COLS + sentiment_cols].copy()
    target = df["rating"].astype(float)
    overview = pd.DataFrame([
        {"Metric": "Input rows", "Value": raw_count},
        {"Metric": "Rows after numeric and required-label cleaning", "Value": len(df)},
        {"Metric": "Excluded rows", "Value": raw_count - len(df)},
        {"Metric": "Conflicting aspect-review label assignments", "Value": label_issues},
        {"Metric": "Features", "Value": len(raw_features.columns)},
    ])
    overview.to_csv(OUTPUT_DIR / "modeling_data_summary.csv", index=False, encoding="utf-8-sig")
    return df, raw_features, target, sentiment_cols


# ============================================================
# 2. CUSTOMER SATISFACTION PREDICTION: 10-fold CV
# ============================================================

def make_scaled_features(train_df, val_df, sentiment_cols):
    """Fit the numeric scaler on the training fold only, avoiding CV leakage."""
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[NUMERIC_COLS])
    val_num = scaler.transform(val_df[NUMERIC_COLS])
    names = SCALED_COLS + sentiment_cols
    X_train = pd.DataFrame(
        np.column_stack([train_num, train_df[sentiment_cols].to_numpy()]),
        columns=names, index=train_df.index
    )
    X_val = pd.DataFrame(
        np.column_stack([val_num, val_df[sentiment_cols].to_numpy()]),
        columns=names, index=val_df.index
    )
    return X_train, X_val, scaler


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """Preserve the original regression metrics alongside RMSE, MAE, and R2."""
    actual = np.asarray(y_true, dtype=float)
    predicted = np.asarray(y_pred, dtype=float)
    errors = np.abs(actual - predicted)
    valid_mape = actual > 0
    valid_ratio = (actual > 0) & (predicted > 0)
    return {
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "MAE": float(mean_absolute_error(actual, predicted)),
        "R2": float(r2_score(actual, predicted)),
        "MAPE": float(np.mean(errors[valid_mape] / actual[valid_mape])) if valid_mape.any() else np.nan,
        "PRED25": float(np.mean(errors[valid_mape] / actual[valid_mape] <= 0.25) * 100) if valid_mape.any() else np.nan,
        "MBRE": float(np.mean(errors[valid_ratio] / np.minimum(actual[valid_ratio], predicted[valid_ratio]))) if valid_ratio.any() else np.nan,
        "MIBRE": float(np.mean(errors[valid_ratio] / np.maximum(actual[valid_ratio], predicted[valid_ratio]))) if valid_ratio.any() else np.nan,
    }


def make_models() -> dict[str, object]:
    return {
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.4,
            objective="reg:squarederror", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


def evaluate_cv(raw_features, y, sentiment_cols) -> pd.DataFrame:
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_rows, oof_rows = [], []
    for model_name, base_model in make_models().items():
        oof_predictions = np.full(len(y), np.nan)
        for fold, (train_ids, val_ids) in enumerate(kf.split(raw_features), 1):
            raw_train = raw_features.iloc[train_ids]
            raw_val = raw_features.iloc[val_ids]
            X_train, X_val, _ = make_scaled_features(raw_train, raw_val, sentiment_cols)
            model = clone(base_model)
            model.fit(X_train, y.iloc[train_ids])
            prediction = model.predict(X_val)
            oof_predictions[val_ids] = prediction
            fold_rows.append({
                "Model": model_name,
                "Fold": fold,
                **regression_metrics(y.iloc[val_ids], prediction),
            })
        oof_rows.append({
            "Model": model_name,
            **regression_metrics(y, oof_predictions),
        })

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows).sort_values("RMSE", ascending=True)
    summary = fold_df.groupby("Model", as_index=False).agg(**{
        f"{metric}_{stat.capitalize()}": (metric, stat)
        for metric in ("RMSE", "MAE", "R2", "MAPE", "PRED25", "MBRE", "MIBRE")
        for stat in ("mean", "std")
    })
    oof_df.to_csv(OUTPUT_DIR / "model_comparison_oof.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUTPUT_DIR / "model_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT_DIR / "model_cv_mean_std.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== CUSTOMER SATISFACTION PREDICTION: {N_SPLITS}-FOLD CV ===")
    print(oof_df.round(4).to_string(index=False))
    winner = str(oof_df.iloc[0]["Model"])
    if winner != "XGBoost":
        warnings.warn(
            f"{winner} has the lowest out-of-fold RMSE, not XGBoost. "
            "Following the original framework, XGBoost is still used for native Tree SHAP. "
            "Do not claim XGBoost is the best-performing model.", stacklevel=2,
        )
    return oof_df


# ============================================================
# 3. EXPLAINABLE AI: NATIVE XGBOOST TREE SHAP
# ============================================================

def fit_xgboost_for_explanation(raw_features, y, sentiment_cols):
    # Fit on full dataset AFTER CV; resulting SHAP describes this final fitted
    # explanatory model. CV metrics above are not in-sample predictions.
    X, _, scaler = make_scaled_features(raw_features, raw_features, sentiment_cols)
    model = make_models()["XGBoost"]
    model.fit(X, y)
    matrix = xgb.DMatrix(X, feature_names=X.columns.tolist())
    contributions = np.asarray(model.get_booster().predict(
        matrix, pred_contribs=True, strict_shape=False,
    ))
    shap_values = contributions[:, :-1]  # Last column is bias/base value.
    base_values = contributions[:, -1]
    predicted = model.predict(X)
    if not np.allclose(shap_values.sum(axis=1) + base_values, predicted, atol=1e-3):
        raise AssertionError("XGBoost native SHAP contributions do not sum to predictions")
    print(f"\n=== EXPLAINABLE AI: {len(X)} REVIEWS, {X.shape[1]} FEATURES ===")
    print(f"SHAP additivity verified; baseline = {np.mean(base_values):.4f}")
    return X, model, scaler, shap_values, base_values


def calculate_xai(X: pd.DataFrame, shap_values: np.ndarray) -> pd.DataFrame:
    """Global mean |SHAP| across all reviews + active-state median signed SHAP.

    Two raw Pos/Neg features represent each aspect.
    Primary aspect importance = E[|SHAP_Pos| + |SHAP_Neg|].
    Alternative diagnostic = E[|SHAP_Pos + SHAP_Neg|].
    Note: these are feature aggregation choices, not exact Shapley group values.
    """
    feature_importance = pd.DataFrame({
        "Feature": X.columns.tolist(),
        "Mean_Abs_SHAP": np.abs(shap_values).mean(axis=0),
        "Mean_Signed_SHAP": shap_values.mean(axis=0),
    }).sort_values("Mean_Abs_SHAP", ascending=False)
    feature_importance["Model_Wide_Importance_Pct"] = (
        100 * feature_importance["Mean_Abs_SHAP"] /
        feature_importance["Mean_Abs_SHAP"].sum()
    )
    feature_importance.to_csv(
        OUTPUT_DIR / "xai_feature_importance_all.csv", index=False, encoding="utf-8-sig"
    )

    rows = []
    for aspect in ASPECTS:
        pos_col, neg_col = f"{aspect}_Pos", f"{aspect}_Neg"
        pos_values = shap_values[:, X.columns.get_loc(pos_col)]
        neg_values = shap_values[:, X.columns.get_loc(neg_col)]
        pos_active = X[pos_col].to_numpy() == 1
        neg_active = X[neg_col].to_numpy() == 1
        # Median signed SHAP
        pos_med = float(np.median(pos_values[pos_active])) if pos_active.any() else np.nan
        neg_med = float(np.median(neg_values[neg_active])) if neg_active.any() else np.nan

        # Mean signed SHAP - NEW
        pos_mean = float(np.mean(pos_values[pos_active])) if pos_active.any() else np.nan
        neg_mean = float(np.mean(neg_values[neg_active])) if neg_active.any() else np.nan
        rows.append({
            "Aspect": aspect,

            # Global Importance
            "Importance_Mean_Abs_SHAP": float(
                np.mean(np.abs(pos_values) + np.abs(neg_values))
            ),
            "Alternative_Grouped_Mean_Abs_SHAP": float(
                np.mean(np.abs(pos_values + neg_values))
            ),

            # Median Signed SHAP - OLD
            "Median_SHAP_Pos_Active": pos_med,
            "Median_SHAP_Neg_Active": neg_med,

            # Mean Signed SHAP - NEW
            "Mean_SHAP_Pos_Active": pos_mean,
            "Mean_SHAP_Neg_Active": neg_mean,

            # Reward / Penalty based on Median
            "Reward_Magnitude": max(0, pos_med) if np.isfinite(pos_med) else np.nan,
            "Penalty_Magnitude": max(0, -neg_med) if np.isfinite(neg_med) else np.nan,

            # Reward / Penalty based on Mean - NEW
            "Mean_Reward_Magnitude": max(0, pos_mean) if np.isfinite(pos_mean) else np.nan,
            "Mean_Penalty_Magnitude": max(0, -neg_mean) if np.isfinite(neg_mean) else np.nan,

            "N_Positive_Active": int(pos_active.sum()),
            "N_Negative_Active": int(neg_active.sum()),
        })
    result = pd.DataFrame(rows)
    total_aspect = result["Importance_Mean_Abs_SHAP"].sum()
    if total_aspect <= 0:
        raise ValueError("All aspect SHAP values are zero; cannot build IPA importance scores")
    result["Importance_Pct"] = 100 * result["Importance_Mean_Abs_SHAP"] / total_aspect
    # Optional diagnostic normalized based on the alternative grouping method.
    grouped_total = result["Alternative_Grouped_Mean_Abs_SHAP"].sum()
    result["Alternative_Importance_Pct"] = (
        100 * result["Alternative_Grouped_Mean_Abs_SHAP"] / grouped_total
        if grouped_total > 0 else np.nan
    )
    result["Penalty_Minus_Reward"] = result["Penalty_Magnitude"] - result["Reward_Magnitude"]
    result["Weak_Negative_Evidence"] = result["N_Negative_Active"] < MIN_NEGATIVE_FOR_REWARD_PENALTY
    result = result.sort_values("Importance_Pct", ascending=False).reset_index(drop=True)
    result.to_csv(OUTPUT_DIR / "xai_aspect_summary.csv", index=False, encoding="utf-8-sig")
    print("\n=== XAI ATTRIBUTE IMPORTANCE AND SENTIMENT CONTRIBUTIONS ===")
    print(result[["Aspect", "Importance_Pct", "Median_SHAP_Pos_Active",
                  "Median_SHAP_Neg_Active", "N_Negative_Active"]].round(4).to_string(index=False))
    return result


# ============================================================
# 4. IPA: SENTIMENT-DERIVED PERFORMANCE + SHAP IMPORTANCE
# ============================================================

def calculate_aspect_performance(df: pd.DataFrame) -> pd.DataFrame:
    """P_a = 50 * (1 + mean(s)) for aspect mentions only.

    s = +1 (Positive), 0 (Neutral), -1 (Negative).
    None is excluded (aspect not mentioned), Mixed is excluded from score.
    Equivalent: Performance = 100 * (N_Pos + 0.5*N_Neutral) / N_Valid.
    """
    rows = []
    for aspect in ASPECTS:
        counts = df[f"{aspect}_State"].value_counts()
        n_pos = int(counts.get("Positive", 0))
        n_neu = int(counts.get("Neutral", 0))
        n_neg = int(counts.get("Negative", 0))
        n_mixed = int(counts.get("Mixed", 0))
        n_none = int(counts.get("None", 0))
        n_valid = n_pos + n_neu + n_neg
        score = (100 * (n_pos + 0.5 * n_neu) / n_valid) if n_valid else np.nan
        rows.append({
            "Aspect": aspect,
            "N_Positive": n_pos,
            "N_Neutral": n_neu,
            "N_Negative": n_neg,
            "N_Mixed_Excluded": n_mixed,
            "N_None": n_none,
            "N_Mentioned": n_valid + n_mixed,
            "N_Performance_Valid": n_valid,
            "Performance_0_100": score,
            "Negative_Rate_Pct": 100 * n_neg / n_valid if n_valid else np.nan,
            "Low_Mention_Support": n_valid < MIN_MENTIONS_FOR_RELIABLE_INTERPRETATION,
        })
    return pd.DataFrame(rows)


def quadrant_of(importance, performance, importance_cut, performance_cut):
    if not (np.isfinite(importance) and np.isfinite(performance)):
        return "Insufficient Data"
    high_i = importance >= importance_cut
    high_p = performance >= performance_cut
    if high_i and not high_p:
        return "Concentrate Here"
    if high_i and high_p:
        return "Keep Up the Good Work"
    if not high_i and not high_p:
        return "Low Priority"
    return "Possible Overkill"


def managerial_interpretation(quadrant):
    """Proposed management heuristic, not a validated optimization policy."""
    return {
        "Concentrate Here": "Prioritize diagnosing and improving low-rated high-importance attributes",
        "Keep Up the Good Work": "Maintain quality and monitor performance",
        "Low Priority": "Monitor complaints and investigate selectively",
        "Possible Overkill": "Maintain appropriate standards; reassess incremental investment",
        "Insufficient Data": "Collect additional valid feedback before prioritizing",
    }[quadrant]


def calculate_ipa(df: pd.DataFrame, xai: pd.DataFrame) -> pd.DataFrame:
    performance = calculate_aspect_performance(df)
    ipa = xai.merge(performance, on="Aspect", how="left", validate="one_to_one")
    valid = ipa["Performance_0_100"].notna() & ipa["Importance_Pct"].notna()
    if valid.sum() < 2:
        raise ValueError("Need at least 2 aspects with valid Performance and Importance")

    if IPA_CUT_METHOD == "mean":
        importance_cut = float(ipa.loc[valid, "Importance_Pct"].mean())
        performance_cut = float(ipa.loc[valid, "Performance_0_100"].mean())
    elif IPA_CUT_METHOD == "median":
        importance_cut = float(ipa.loc[valid, "Importance_Pct"].median())
        performance_cut = float(ipa.loc[valid, "Performance_0_100"].median())
    else:
        raise ValueError("IPA_CUT_METHOD must be 'mean' or 'median'")

    ipa["IPA_Quadrant"] = ipa.apply(
        lambda r: quadrant_of(
            r["Importance_Pct"], r["Performance_0_100"],
            importance_cut, performance_cut,
        ), axis=1,
    )
    ipa["Managerial_Interpretation"] = ipa["IPA_Quadrant"].map(managerial_interpretation)
    ipa["Importance_Cut"] = importance_cut
    ipa["Performance_Cut"] = performance_cut
    ipa["Cut_Method"] = IPA_CUT_METHOD
    ipa["Review_N"] = len(df)

    ipa.to_csv(OUTPUT_DIR / "ipa_attributes.csv", index=False, encoding="utf-8-sig")
    print(f"\n=== IPA ({IPA_CUT_METHOD.upper()} CUTS: importance={importance_cut:.3f}%, "
          f"performance={performance_cut:.3f}/100) ===")
    print(ipa[["Aspect", "Importance_Pct", "Performance_0_100", "IPA_Quadrant",
               "N_Performance_Valid", "N_Mixed_Excluded"]].round(3).to_string(index=False))
    print("\nNOTE: IPA performance = sentiment in mentioned aspects; importance = model-derived SHAP.")
    return ipa


# ============================================================
# 5. RESEARCH-READY FIGURES (300 DPI PNG + VECTOR PDF)
# ============================================================

def save_figure(fig, stem):
    fig.savefig(OUTPUT_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_importance(ipa):
    d = ipa.sort_values("Importance_Pct", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.barh(d["Aspect"].str.replace("_", " ", regex=False), d["Importance_Pct"])
    ax.set_xlabel("Relative aspect importance (% of the 11 aspect scores)")
    ax.set_title("Global Attribute Importance from Mean Absolute SHAP")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    save_figure(fig, "aspect_importance")


def plot_reward_penalty(ipa):
    d = ipa.sort_values("Penalty_Magnitude", ascending=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.barh(y - 0.2, d["Median_SHAP_Pos_Active"], height=0.36, label="Positive (median signed SHAP)")
    ax.barh(y + 0.2, d["Median_SHAP_Neg_Active"], height=0.36, label="Negative (median signed SHAP)")
    ax.set_yticks(y, d["Aspect"].str.replace("_", " ", regex=False))
    ax.axvline(0, linewidth=0.9)
    ax.set_xlabel("Conditional median feature contribution (rating units)")
    ax.set_title("Positive and Negative SHAP Contributions")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()
    save_figure(fig, "reward_penalty")


def plot_ipa(ipa):
    colors = {
        "Concentrate Here": "#c2410c",
        "Keep Up the Good Work": "#15803d",
        "Low Priority": "#64748b",
        "Possible Overkill": "#2563eb",
        "Insufficient Data": "#a3a3a3",
    }
    fig, ax = plt.subplots(figsize=(11.8, 8))
    x_cut, y_cut = ipa["Performance_Cut"].iloc[0], ipa["Importance_Cut"].iloc[0]
    ax.axvline(x_cut, linestyle="--", alpha=0.7)
    ax.axhline(y_cut, linestyle="--", alpha=0.7)
    for quadrant, group in ipa.groupby("IPA_Quadrant"):
        ax.scatter(
            group["Performance_0_100"], group["Importance_Pct"],
            s=130, label=quadrant, color=colors[quadrant],
            edgecolors="white", linewidths=0.7, zorder=3,
        )
    # Fixed label offsets and white backgrounds help with dense aspect clusters.
    offsets = [(6, 8), (6, -16), (-7, 11), (-7, -15), (6, 17), (-7, 19), (6, -21), (-7, -22)]
    for idx, row in ipa.reset_index(drop=True).iterrows():
        if not np.isfinite(row["Performance_0_100"]):
            continue
        dx, dy = offsets[idx % len(offsets)]
        ax.annotate(
            row["Aspect"].replace("_", " "),
            (row["Performance_0_100"], row["Importance_Pct"]),
            xytext=(dx, dy), textcoords="offset points",
            fontsize=8.5, ha="left" if dx > 0 else "right",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1),
        )
    ax.set_xlabel("Sentiment-derived performance (0–100; aspect mentions only)")
    ax.set_ylabel("Relative mean absolute SHAP importance (%)")
    ax.set_title(f"Importance–Performance Analysis (cutoffs: {IPA_CUT_METHOD})")
    # Zoom to observed performance to distinguish clusters; never imply the
    # visible range is the full 0–100 scale.
    valid_x = ipa["Performance_0_100"].dropna()
    x_min = max(0, min(float(valid_x.min()), x_cut) - 7)
    x_max = min(100, max(float(valid_x.max()), x_cut) + 7)
    if x_max - x_min < 14:
        x_min = max(0, x_min - 7)
        x_max = min(100, x_max + 7)
    ax.set_xlim(x_min, x_max)
    ymax = max(ipa["Importance_Pct"].max(), y_cut)
    ax.set_ylim(0, max(15, ymax * 1.30))
    ax.text(0.01, 0.99, "Concentrate Here", ha="left", va="top", transform=ax.transAxes, fontsize=8.5)
    ax.text(0.99, 0.99, "Keep Up the Good Work", ha="right", va="top", transform=ax.transAxes, fontsize=8.5)
    ax.text(0.01, 0.02, "Low Priority", ha="left", va="bottom", transform=ax.transAxes, fontsize=8.5)
    ax.text(0.99, 0.02, "Possible Overkill", ha="right", va="bottom", transform=ax.transAxes, fontsize=8.5)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8)
    ax.grid(alpha=0.11)
    fig.tight_layout()
    save_figure(fig, "ipa_matrix")



def plot_xai_shap_summary(X: pd.DataFrame, shap_values: np.ndarray):
    """Beeswarm-style signed Tree SHAP dot plot; does not require `shap`.

    Points may be randomly subsampled for visual legibility, but the top features
    are ranked by mean absolute SHAP over ALL reviews. Feature colors are
    normalized separately for each feature (0=low, 1=high); do not compare
    color shades as absolute values across differently scaled variables.
    """
    if shap_values.shape != X.shape:
        raise ValueError(f"SHAP matrix {shap_values.shape} mismatches X {X.shape}")
    top_indices = np.argsort(np.abs(shap_values).mean(axis=0))[::-1][:XAI_SCATTER_TOP_FEATURES]
    rng = np.random.default_rng(RANDOM_STATE)
    sample_size = min(len(X), XAI_SCATTER_MAX_REVIEWS)
    sample_ids = np.sort(rng.choice(len(X), size=sample_size, replace=False))

    fig, ax = plt.subplots(figsize=(11, max(5.5, .47 * len(top_indices) + 1.3)))
    for rank, feature_idx in enumerate(top_indices):
        values = X.iloc[sample_ids, feature_idx].to_numpy(dtype=float)
        contributions = shap_values[sample_ids, feature_idx]
        finite = np.isfinite(contributions) & np.isfinite(values)
        values, contributions = values[finite], contributions[finite]
        if len(values) == 0:
            continue
        lo, hi = np.percentile(values, [5, 95])
        colors = np.clip((values - lo) / (hi - lo), 0, 1) if hi > lo else np.full(len(values), 0.5)
        y = np.full(len(values), rank, dtype=float) + rng.uniform(-0.26, 0.26, len(values))
        ax.scatter(contributions, y, c=colors, cmap="coolwarm", vmin=0, vmax=1,
                   s=11, alpha=.58, edgecolors="none", rasterized=True)

    labels = [X.columns[i].replace("_", " ") for i in top_indices]
    ax.set_yticks(np.arange(len(top_indices)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=1.0, alpha=.7)
    ax.set_xlabel("Signed Tree SHAP value (rating units)")
    ax.set_title("XAI: Feature Contributions Across Reviews")
    ax.grid(axis="x", alpha=.12)
    # Color encodes within-feature relative value: 0/1 for binary sentiment flags.
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=plt.Normalize(0, 1))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, pad=.02, shrink=.85, label="Relative feature value (low → high)")
    fig.text(.12, .015, f"Dots: {sample_size:,} reviews (random reproducible sample); feature ranking: all {len(X):,} reviews.",
             fontsize=8, color="#555555")
    fig.tight_layout(rect=(0, .03, 1, 1))
    save_figure(fig, "xai_shap_summary")


def plot_xai_local_waterfall(
    X: pd.DataFrame, y: pd.Series, shap_values: np.ndarray, base_values: np.ndarray,
    model: XGBRegressor,
):
    """Explain one selected in-sample prediction by additive native SHAP values.

    The plot is LOCAL, not median SHAP and not global mean-|SHAP|. The residual
    contributions of hidden features are combined into one additive 'Other' bar.
    """
    row_index = LOCAL_WATERFALL_REVIEW_INDEX
    if not 0 <= row_index < len(X):
        raise ValueError(f"LOCAL_WATERFALL_REVIEW_INDEX={row_index} outside 0..{len(X)-1}")
    phi = np.asarray(shap_values[row_index], dtype=float)
    base = float(base_values[row_index])
    actual = float(y.iloc[row_index])
    predicted = float(model.predict(X.iloc[[row_index]])[0])
    if not np.isclose(base + phi.sum(), predicted, atol=1e-3):
        raise AssertionError("Local waterfall sum does not reconstruct model prediction")

    order = np.argsort(np.abs(phi))[::-1]
    top_count = min(LOCAL_WATERFALL_MAX_FEATURES, len(order))
    top = order[:top_count]
    hidden = order[top_count:]
    # Start with the grouped smaller contributions, then progress to the largest.
    bars = [("Other features", float(phi[hidden].sum()))] if len(hidden) else []
    for col_index in top[::-1]:
        v = float(X.iloc[row_index, col_index])
        fmt_value = str(int(v)) if X.columns[col_index].endswith(("_Pos", "_Neg")) else f"{v:.2f}"
        bars.append((f"{X.columns[col_index].replace('_', ' ')} = {fmt_value}", float(phi[col_index])))

    fig, ax = plt.subplots(figsize=(11.5, max(6, 0.48 * len(bars) + 2)))
    running = base
    all_pos = [base, predicted]
    for rank, (label, delta) in enumerate(bars):
        next_running = running + delta
        left = min(running, next_running)
        ax.barh(rank, abs(delta), left=left, height=.65,
                color="#d81b60" if delta >= 0 else "#0284c7",
                edgecolor="white", linewidth=.7, zorder=3)
        ax.plot([running, running], [rank - .38, rank + .38], color="#555555", lw=.6, alpha=.5)
        ax.text(next_running, rank, f" {delta:+.2f} " if delta >= 0 else f"{delta:+.2f} ",
                fontsize=8, va="center", ha="left" if delta >= 0 else "right")
        all_pos.extend([running, next_running])
        running = next_running
    ax.set_yticks(range(len(bars)), [name for name, _ in bars])
    ax.axvline(base, ls="--", color="#64748b", alpha=.7, linewidth=1.1)
    ax.axvline(predicted, ls=":", color="black", alpha=.9, linewidth=1.25)
    ax.set_xlabel("Predicted star rating (model output)")
    ax.set_title(f"XAI Waterfall — Review #{row_index} (actual={actual:.1f}, predicted={predicted:.2f})")
    fig.text(.12, .025, f"Base value: {base:.3f}   |   SHAP sum: {phi.sum():+.3f}   |   Prediction: {predicted:.3f}",
             fontsize=9, color="#475569")
    span = max(all_pos) - min(all_pos)
    pad = max(.18, span * .13)
    ax.set_xlim(min(all_pos) - pad, max(all_pos) + pad)
    ax.grid(axis="x", alpha=.10)
    fig.tight_layout(rect=(0, .045, 1, 1))
    save_figure(fig, "xai_local_waterfall")

    pd.DataFrame({
        "Review_Index": row_index,
        "Feature": X.columns,
        "Feature_Value": X.iloc[row_index].values,
        "Signed_SHAP": phi,
        "Baseline": base,
        "Predicted_Rating": predicted,
        "Actual_Rating": actual,
    }).to_csv(OUTPUT_DIR / "xai_local_explanation.csv", index=False, encoding="utf-8-sig")


def plot_ipa_penalty_bubble(ipa: pd.DataFrame):
    """IPA coordinates with bubble area representing median negative SHAP magnitude.

    Bubble size is supplementary model-based information, not an IPA axis.
    A warning mark identifies attributes with too few negative reviews.
    """
    palette = {
        "Concentrate Here": "#c2410c",
        "Keep Up the Good Work": "#15803d",
        "Low Priority": "#64748b",
        "Possible Overkill": "#2563eb",
        "Insufficient Data": "#a3a3a3",
    }
    d = ipa.dropna(subset=["Performance_0_100", "Importance_Pct"]).copy()
    if d.empty:
        return
    penalties = d["Penalty_Magnitude"].clip(lower=0).fillna(0).to_numpy(dtype=float)
    max_penalty = float(penalties.max()) if len(penalties) else 0
    sizes = 90 + (650 * penalties / max_penalty if max_penalty else np.zeros_like(penalties))

    fig, ax = plt.subplots(figsize=(12, 7.7))
    i_cut = float(d["Importance_Cut"].iloc[0])
    p_cut = float(d["Performance_Cut"].iloc[0])
    ax.axhline(i_cut, ls="--", color="#64748b", alpha=.65)
    ax.axvline(p_cut, ls="--", color="#64748b", alpha=.65)
    for (_, item), size in zip(d.iterrows(), sizes):
        color = palette[item["IPA_Quadrant"]]
        ax.scatter(item["Performance_0_100"], item["Importance_Pct"], s=size,
                   c=color, edgecolor="white", alpha=.63, linewidth=1, zorder=3)
        label = item["Aspect"].replace("_", " ")
        if bool(item["Weak_Negative_Evidence"]):
            label += "*"
        ax.annotate(label, (item["Performance_0_100"], item["Importance_Pct"]),
                    xytext=(5, 7), textcoords="offset points", fontsize=8.3,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.72, pad=.7))
    ax.set_xlabel("Sentiment-derived performance (0–100; mentioned reviews)")
    ax.set_ylabel("Relative mean absolute SHAP importance (%)")
    ax.set_title("IPA with Negative SHAP Penalty (Bubble Area)")
    x0 = min(d["Performance_0_100"].min(), p_cut)
    x1 = max(d["Performance_0_100"].max(), p_cut)
    xpad = max(4, float(x1 - x0) * .25)
    ax.set_xlim(max(0, x0 - xpad), min(100, x1 + xpad))
    y0 = min(d["Importance_Pct"].min(), i_cut)
    y1 = max(d["Importance_Pct"].max(), i_cut)
    ypad = max(2, float(y1 - y0) * .28)
    ax.set_ylim(max(0, y0 - ypad), y1 + ypad)
    # Bubble size samples are in rating-unit negative median SHAP magnitudes.
    if max_penalty > 0:
        reference = sorted(set([round(max_penalty * f, 2) for f in (.25, .5, 1.)]))
        handles = [ax.scatter([], [], s=90 + 650 * r / max_penalty, color="#64748b",
                              edgecolor="white", alpha=.45) for r in reference]
        ax.legend(handles, [f"Penalty {r:.2f}" for r in reference],
                  title="Bubble area (conditional median)", loc="upper right",
                  fontsize=8, title_fontsize=8, framealpha=.94)
    fig.text(.12, .015,
             f"* Fewer than {MIN_NEGATIVE_FOR_REWARD_PENALTY} negative mentions; penalty estimate is uncertain."
             " Bubble sizes do not represent sample counts or causality.",
             fontsize=8, color="#555555")
    ax.grid(alpha=.10)
    fig.tight_layout(rect=(0, .035, 1, 1))
    save_figure(fig, "ipa_penalty_bubble")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, raw_features, y, sentiment_cols = preprocess_data(DATA_PATH)
    print(f"Loaded {len(df)} valid reviews ({len(sentiment_cols)} sentiment features).")

    # Stage 1 - Customer satisfaction prediction (10-fold CV + full fitted model)
    evaluate_cv(raw_features, y, sentiment_cols)
    X, model, scaler, shap_values, base_values = fit_xgboost_for_explanation(raw_features, y, sentiment_cols)

    # Stage 2 - XAI (global importance plus median signed sentiment contributions)
    xai_results = calculate_xai(X, shap_values)

    # Stage 3 - IPA (aspect sentiment performance + model-derived importance)
    ipa = calculate_ipa(df, xai_results)
    plot_importance(ipa)
    plot_reward_penalty(ipa)
    plot_ipa(ipa)
    # Additional complementary figures: distribution, local explanation, IPA bubble.
    plot_xai_shap_summary(X, shap_values)
    plot_xai_local_waterfall(X, y, shap_values, base_values, model)
    plot_ipa_penalty_bubble(ipa)

    low_support = ipa.loc[ipa["Low_Mention_Support"], "Aspect"].tolist()
    weak_negative = ipa.loc[ipa["Weak_Negative_Evidence"], "Aspect"].tolist()
    if low_support:
        print(f"WARNING: low performance sample support (<{MIN_MENTIONS_FOR_RELIABLE_INTERPRETATION}): {low_support}")
    if weak_negative:
        print(f"WARNING: few negative labels (<{MIN_NEGATIVE_FOR_REWARD_PENALTY}): {weak_negative}; "
              "interpret median negative SHAP cautiously.")
    print(f"\nAll outputs saved in: {OUTPUT_DIR.resolve()}")
    print("XAI describes model predictions, not causal product effects.")
    print("Reference ABSA labels are used; predicted-PhoBERT end-to-end validity is not assessed here.")


if __name__ == "__main__":
    main()
