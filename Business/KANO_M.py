"""Facial cleanser reviews: 10-fold regression CV, original SHAP-Kano, SHAP interaction analysis.

Requires: pip install pandas numpy scikit-learn xgboost matplotlib
No dependency on the external `shap` Python package: XGBoost computes Tree SHAP natively.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from xgboost import XGBRegressor

# ============================================================
# 1. CONFIG
# ============================================================
DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "crawl-data" / "data" / "combined_reviews_brand_filled_language_filtered.csv"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "shap_interaction_results_cleanser"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42
N_SPLITS = 10
INDIFFERENT_THRESHOLD = 0.10
MIN_PAIR_SUPPORT = 20    # Minimum reviews with both sentiment features active
TOP_N_PAIRS = 20
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


def extract_aspects(label_str):
    if pd.isna(label_str):
        return []
    return re.findall(r"\{\s*([^#{}]+?)\s*#\s*([^{}]+?)\s*\}", str(label_str))


def preprocess_data(path):
    df = pd.read_csv(path, encoding="utf-8")
    required = {"price", "discount", "rating", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    # Same currency convention as original code: '.' and ',' are grouping marks.
    df["price"] = pd.to_numeric(
        df["price"].astype(str).str.replace("₫", "", regex=False)
        .str.replace(".", "", regex=False)
        .str.replace(",", "", regex=False).str.strip(),
        errors="coerce"
    )
    df["discount"] = pd.to_numeric(
        df["discount"].astype(str).str.replace("%", "", regex=False).str.strip(),
        errors="coerce"
    ) / 100
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["price", "discount", "rating", "label"]).reset_index(drop=True)
    if len(df) < N_SPLITS:
        raise ValueError("Too few valid rows for cross-validation")

    for aspect in ASPECTS:
        df[f"{aspect}_Pos"] = 0
        df[f"{aspect}_Neg"] = 0

    # Populate indicator columns from the cleanser ABSA label column.
    # Neutral labels retain the original behavior: no Pos/Neg indicator is activated.
    for i, raw in enumerate(df["label"]):
        for aspect, sentiment in extract_aspects(raw):
            aspect, sentiment = aspect.strip(), sentiment.strip().lower()
            if aspect not in ASPECTS:
                continue
            if sentiment == "positive":
                df.at[i, f"{aspect}_Pos"] = 1
            elif sentiment == "negative":
                df.at[i, f"{aspect}_Neg"] = 1

    sentiment_cols = [f"{aspect}_{pol}" for aspect in ASPECTS for pol in ("Pos", "Neg")]
    raw_features = df[NUMERIC_COLS + sentiment_cols].copy()
    target = df["rating"].astype(float)
    return df, raw_features, target, sentiment_cols


def make_scaled_features(train_df, val_df, sentiment_cols):
    """Fit scaler on CV train only to avoid fold-level leakage."""
    scaler = StandardScaler()
    train_num = scaler.fit_transform(train_df[NUMERIC_COLS])
    val_num = scaler.transform(val_df[NUMERIC_COLS])

    col_names = SCALED_COLS + sentiment_cols
    train_x = pd.DataFrame(
        np.column_stack([train_num, train_df[sentiment_cols].to_numpy()]),
        columns=col_names, index=train_df.index
    )
    val_x = pd.DataFrame(
        np.column_stack([val_num, val_df[sentiment_cols].to_numpy()]),
        columns=col_names, index=val_df.index
    )
    return train_x, val_x, scaler


def regression_metrics(y_true, y_pred):
    true, pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    abs_err = np.abs(true - pred)
    # Positive star ratings expected; exclude invalid denominators.
    valid_mape = true > 0
    valid_mbre = (true > 0) & (pred > 0)
    result = {
        "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
        "MAE": float(mean_absolute_error(true, pred)),
        "R2": float(r2_score(true, pred)),
        "MAPE": float(np.mean(abs_err[valid_mape] / true[valid_mape])) if valid_mape.any() else np.nan,
        "PRED25": float(np.mean(abs_err[valid_mape] / true[valid_mape] <= 0.25) * 100)
        if valid_mape.any() else np.nan,
        "MBRE": float(np.mean(abs_err[valid_mbre] / np.minimum(true[valid_mbre], pred[valid_mbre])))
        if valid_mbre.any() else np.nan,
        "MIBRE": float(np.mean(abs_err[valid_mbre] / np.maximum(true[valid_mbre], pred[valid_mbre])))
        if valid_mbre.any() else np.nan
    }
    return result


def evaluate_cv(raw_features, y, sentiment_cols):
    models = {
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.4,
            random_state=RANDOM_STATE, n_jobs=-1,
            objective="reg:squarederror"
        ),
    }
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    pooled_rows, fold_rows = [], []
    for model_name, base_model in models.items():
        out_of_fold = np.full(len(y), np.nan)
        for fold, (train_idx, val_idx) in enumerate(kf.split(raw_features), 1):
            raw_train = raw_features.iloc[train_idx]
            raw_val = raw_features.iloc[val_idx]
            X_train, X_val, _ = make_scaled_features(raw_train, raw_val, sentiment_cols)
            model = clone(base_model)
            model.fit(X_train, y.iloc[train_idx])
            predicted = model.predict(X_val)
            out_of_fold[val_idx] = predicted
            fold_rows.append({"Model": model_name, "Fold": fold,
                              **regression_metrics(y.iloc[val_idx], predicted)})
        pooled_rows.append({"Model": model_name, **regression_metrics(y, out_of_fold)})

    pooled_df, fold_df = pd.DataFrame(pooled_rows), pd.DataFrame(fold_rows)
    pooled_df.to_csv(OUTPUT_DIR / "model_comparison_oof.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUTPUT_DIR / "model_cv_fold_metrics.csv", index=False, encoding="utf-8-sig")
    summary = fold_df.groupby("Model").agg({
        metric: ["mean", "std"] for metric in ("RMSE", "MAE", "R2", "MAPE", "PRED25", "MBRE", "MIBRE")
    })
    summary.to_csv(OUTPUT_DIR / "model_cv_mean_std.csv", encoding="utf-8-sig")
    print("\n=== 10-fold OOF evaluation ===")
    print(pooled_df.round(4).to_string(index=False))
    return pooled_df


def kano_classification(pos, neg, threshold=INDIFFERENT_THRESHOLD):
    if not np.isfinite(pos) or not np.isfinite(neg):
        return "Insufficient data"
    if abs(pos) < threshold and abs(neg) < threshold:
        return "Indifferent (I)"
    if pos > 0 and neg < 0:
        return "Performance (O)"
    if pos > 0 and neg >= 0:
        return "Excitement (E)"
    if pos <= 0 and neg < 0:
        return "Must-be (M)"
    return "Reverse (R)"


def kano_summary(X, shap_values):
    rows = []
    for aspect in ASPECTS:
        p, n = f"{aspect}_Pos", f"{aspect}_Neg"
        mask_p = X[p].to_numpy() == 1
        mask_n = X[n].to_numpy() == 1
        med_p = float(np.median(shap_values[mask_p, X.columns.get_loc(p)])) if mask_p.any() else np.nan
        med_n = float(np.median(shap_values[mask_n, X.columns.get_loc(n)])) if mask_n.any() else np.nan
        rows.append({"Aspect": aspect, "N_Positive": int(mask_p.sum()), "N_Negative": int(mask_n.sum()),
                     "SHAP_Pos_Median": med_p, "SHAP_Neg_Median": med_n,
                     "Kano": kano_classification(med_p, med_n)})
    result = pd.DataFrame(rows)
    result.to_csv(OUTPUT_DIR / "original_kano.csv", index=False, encoding="utf-8-sig")
    print("\n=== Original SHAP-Kano ===")
    print(result.round(4).to_string(index=False))
    return result


def interaction_analysis(X, y, interaction_values, sentiment_cols):
    """Native XGBoost off-diagonal SHAP interaction values are symmetric.

    A pair's combined interaction contribution = 2 * matrix[i,j] for i != j.
    The matrix diagonal is main effect (excluded).
    """
    n, p, q = interaction_values.shape
    if p != X.shape[1] or q != X.shape[1]:
        raise ValueError(f"Interaction matrix shape mismatch: {interaction_values.shape} vs {X.shape}")
    columns = X.columns.tolist()
    rows = []
    for i in range(p):
        for j in range(i + 1, p):
            a, b = columns[i], columns[j]
            pair_total = 2.0 * interaction_values[:, i, j]
            both_mask = (X[a].to_numpy() == 1) & (X[b].to_numpy() == 1) if (a in sentiment_cols and b in sentiment_cols) else None
            both_count = int(both_mask.sum()) if both_mask is not None else np.nan
            both_mean = float(np.mean(pair_total[both_mask])) if both_mask is not None and both_count else np.nan
            both_abs = float(np.mean(np.abs(pair_total[both_mask]))) if both_mask is not None and both_count else np.nan
            rows.append({
                "Feature_1": a, "Feature_2": b,
                "Is_Aspect_Pair": (a in sentiment_cols and b in sentiment_cols),
                "Mean_Abs_Interaction_All": float(np.mean(np.abs(pair_total))),
                "Mean_Signed_Interaction_All": float(np.mean(pair_total)),
                "N_Both_Active": both_count,
                "Mean_Interaction_Both_Active": both_mean,
                "Mean_Abs_Interaction_Both_Active": both_abs,
                "Enough_Both_Active": bool(both_count >= MIN_PAIR_SUPPORT) if both_mask is not None else False,
            })
    pair_df = pd.DataFrame(rows).sort_values("Mean_Abs_Interaction_All", ascending=False)
    pair_df.to_csv(OUTPUT_DIR / "all_feature_interactions.csv", index=False, encoding="utf-8-sig")
    aspect_pairs = pair_df[pair_df["Is_Aspect_Pair"]].copy()
    aspect_pairs.to_csv(OUTPUT_DIR / "aspect_pair_interactions.csv", index=False, encoding="utf-8-sig")

    # Cleanser annotations can include both positive and negative for the same aspect.
    # Keep the original pair analysis, including pairs of features from the same aspect.
    print("\n=== Top 15 aspect-feature interactions (all reviews) ===")
    print(aspect_pairs.head(15)[[
        "Feature_1", "Feature_2", "Mean_Abs_Interaction_All",
        "N_Both_Active", "Mean_Interaction_Both_Active"
    ]].round(5).to_string(index=False))

    strong_supported = aspect_pairs[aspect_pairs["Enough_Both_Active"]].copy()
    strong_supported.to_csv(OUTPUT_DIR / "aspect_pairs_with_support.csv", index=False, encoding="utf-8-sig")
    print(f"\nPairs with >= {MIN_PAIR_SUPPORT} jointly-active reviews: {len(strong_supported)}")

    # Global heatmap: mean absolute combined pair interactions.
    mean_abs_matrix = np.mean(np.abs(interaction_values[:, :, :]), axis=0) * 2.0
    np.fill_diagonal(mean_abs_matrix, np.nan)  # diagonal is main effect
    aspect_indices = [columns.index(s) for s in sentiment_cols]
    heat = mean_abs_matrix[np.ix_(aspect_indices, aspect_indices)]
    fig, ax = plt.subplots(figsize=(15, 12))
    im = ax.imshow(heat, cmap="viridis", interpolation="nearest")
    ax.set_xticks(range(len(sentiment_cols)), sentiment_cols, rotation=90)
    ax.set_yticks(range(len(sentiment_cols)), sentiment_cols)
    ax.set_title("Mean Absolute Pairwise SHAP Interaction (Aspect Features)")
    fig.colorbar(im, ax=ax, label="Mean |2 x SHAP interaction| (rating units)")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "shap_interaction_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    plot_top = aspect_pairs.head(TOP_N_PAIRS).iloc[::-1]
    fig, ax = plt.subplots(figsize=(12, max(7, len(plot_top) * .42)))
    labels = plot_top["Feature_1"] + "  ×  " + plot_top["Feature_2"]
    ax.barh(labels, plot_top["Mean_Abs_Interaction_All"])
    ax.set_xlabel("Mean absolute pairwise SHAP interaction (rating units)")
    ax.set_title(f"Top {min(TOP_N_PAIRS, len(aspect_pairs))} Pairwise Aspect Interactions")
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "top_shap_interaction_pairs.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Row-level values for top 5 globally ranked aspect pairs: inspect sign by review.
    detail = pd.DataFrame({"Rating": y.to_numpy()})
    for _, pair in aspect_pairs.head(5).iterrows():
        a, b = pair["Feature_1"], pair["Feature_2"]
        i, j = columns.index(a), columns.index(b)
        prefix = f"{a}__X__{b}"
        detail[f"{prefix}__pair_interaction"] = 2.0 * interaction_values[:, i, j]
        detail[f"{prefix}__both_active"] = ((X[a] == 1) & (X[b] == 1)).to_numpy()
    detail.to_csv(OUTPUT_DIR / "top5_interactions_per_review.csv", index=False, encoding="utf-8-sig")
    return pair_df, aspect_pairs


# ============================================================
# RUN PIPELINE
# ============================================================
if __name__ == "__main__":
    df, raw_features, y, sentiment_cols = preprocess_data(DATA_PATH)
    print(f"Valid reviews: {len(df)}; aspect features: {len(sentiment_cols)}")
    evaluate_cv(raw_features, y, sentiment_cols)

    # Final explanatory model trained on all available data.
    X, _, final_scaler = make_scaled_features(raw_features, raw_features, sentiment_cols)
    model = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.4,
        random_state=RANDOM_STATE, n_jobs=-1,
        objective="reg:squarederror"
    )
    model.fit(X, y)
    matrix = xgb.DMatrix(X, feature_names=X.columns.tolist())
    booster = model.get_booster()

    # Native XGBoost Tree SHAP: works around some XGBoost 3.x / SHAP compatibility bugs.
    contributions = booster.predict(matrix, pred_contribs=True, strict_shape=False)
    shap_values = np.asarray(contributions[:, :-1])  # final column is baseline
    baseline = contributions[:, -1]
    check_pred = model.predict(X)
    assert np.allclose(shap_values.sum(axis=1) + baseline, check_pred, atol=1e-3)
    print(f"SHAP contributions: {shap_values.shape}")

    kano_summary(X, shap_values)

    print("Calculating SHAP pairwise interaction values...")
    full_interactions = booster.predict(matrix, pred_interactions=True, strict_shape=False)
    full_interactions = np.asarray(full_interactions)
    interaction_values = full_interactions[:, :-1, :-1]  # remove bias row and column
    assert np.allclose(interaction_values.sum(axis=2), shap_values, atol=1e-3)
    print(f"SHAP interactions: {interaction_values.shape}")

    interaction_analysis(X, y, interaction_values, sentiment_cols)
    print(f"\nOutputs saved to {OUTPUT_DIR.resolve()}")
    print("Note: SHAP interactions explain model predictions; they are not causal effects.")
