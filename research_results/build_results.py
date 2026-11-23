"""Build local, reproducible tables/charts for the draft cleanser research report.

Run with .venv-bertopic/Scripts/python.exe -B -X utf8 research_results/build_results.py
No training, remote APIs, or changes to source datasets/checkpoints are performed.
"""
from __future__ import annotations

import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import csv
import hashlib
import importlib.util
import io
import json
import re
import sys
import unicodedata
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TABLES = HERE / "tables"
FIGURES = HERE / "figures"
CACHE = HERE / ".cache"
ASPECTS = ["CleansingEffectiveness", "HydrationSkinComfort", "SkinCompatibilitySafety",
           "TextureFoaming", "Fragrance", "IngredientsFormulation", "Packaging",
           "PriceValue", "ShopService", "Delivery", "General"]
NAMES = ["Làm sạch / kiểm soát dầu", "Ẩm / cảm giác sau rửa", "Tương thích / an toàn",
         "Kết cấu / tạo bọt", "Mùi hương", "Thành phần / công thức", "Bao bì / đóng gói",
         "Giá / giá trị", "Dịch vụ shop", "Giao hàng", "Đánh giá chung"]
LABELS = dict(zip(ASPECTS, NAMES))
CODES = {a: f"A{i + 1}" for i, a in enumerate(ASPECTS[:10])}
POLS = ["positive", "negative", "neutral"]  # Same order as the evaluator.
COLORS = {"positive": "#178477", "neutral": "#d8a23d", "negative": "#c15c64"}
NAVY = "#17314b"
BLUE = "#397ea1"
QUADRANTS = ["Concentrate Here", "Keep Up the Good Work", "Low Priority", "Possible Overkill"]
QCOLORS = ["#c45f51", "#25836e", "#98a5b1", "#4e85b1"]
QSHORT = {"Concentrate Here": "Ưu tiên", "Keep Up the Good Work": "Duy trì",
          "Low Priority": "Theo dõi", "Possible Overkill": "Cân nhắc nguồn lực"}
PATTERN = re.compile(r"\{\s*([^#{}]+?)\s*#\s*([^{}]+?)\s*\}")
MANIFEST: dict = {"generated_at": datetime.now().astimezone().isoformat(), "inputs": {}, "figures": [],
                  "tables": [], "notes": [], "validation": {}}


def progress(message):
    print(message, flush=True)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def source(relative):
    path = ROOT / relative
    MANIFEST["inputs"][relative] = sha(path)
    return path


def frame(relative, **kwargs):
    return pd.read_csv(source(relative), encoding="utf-8-sig", **kwargs)


def table(name, data):
    data.to_csv(TABLES / f"{name}.csv", index=False, encoding="utf-8-sig")
    MANIFEST["tables"].append(name)
    return data


def figure(name, fig):
    fig.savefig(FIGURES / f"{name}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    MANIFEST["figures"].append(name)


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.titlesize": 13, "axes.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": NAVY, "text.color": NAVY,
                         "xtick.color": NAVY, "ytick.color": NAVY,
                         "figure.constrained_layout.use": True,
                         "pdf.fonttype": 42})


def normalize(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).split())


def pairs(value):
    return [(a.strip(), s.strip().lower()) for a, s in PATTERN.findall(str(value))
            if a.strip() in ASPECTS and s.strip().lower() in POLS]


def labels_matrix(data):
    matrix = np.zeros((len(data), 11, 3), dtype=bool)
    for i, raw in enumerate(data.label):
        for aspect, sentiment in set(pairs(raw)):
            matrix[i, ASPECTS.index(aspect), POLS.index(sentiment)] = True
    return matrix


def canonical_groups(series):
    names = series.fillna("").map(normalize)
    keys = names.str.casefold()
    counts = pd.DataFrame({"key": keys, "name": names}).value_counts().reset_index(name="n")
    preferred = (counts.sort_values(["key", "n", "name"], ascending=[True, False, True])
                 .drop_duplicates("key").set_index("key")["name"].to_dict())
    preferred.update({"tiki": "Tiki", "shopee": "Shopee", "lazada": "Lazada"})
    return keys.map(preferred)


def group_sentiment_overall(grouped):
    """Describe all three sentiments by group, with label-pair denominators."""
    result = grouped.groupby(["Group_Type", "Group"], as_index=False).agg(
        Review_N=("Review_N", "first"), Positive=("Positive", "sum"),
        Negative=("Negative", "sum"), Neutral=("Neutral", "sum"))
    result["Label_Pairs"] = result[["Positive", "Negative", "Neutral"]].sum(axis=1)
    for sentiment in ["Positive", "Negative", "Neutral"]:
        result[f"{sentiment}_Pct"] = 100 * result[sentiment] / result.Label_Pairs
    table("08_group_sentiment_overall", result)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), gridspec_kw={"width_ratios": [.85, 1.15]})
    for ax, kind in zip(axes, ["Platform", "Brand"]):
        values = result.loc[result.Group_Type.eq(kind)].sort_values("Review_N", ascending=False)
        left = np.zeros(len(values))
        for sentiment, color, label in [("Positive", COLORS["positive"], "Tích cực"),
                                        ("Negative", COLORS["negative"], "Tiêu cực"),
                                        ("Neutral", COLORS["neutral"], "Trung tính")]:
            share = values[f"{sentiment}_Pct"].to_numpy()
            ax.barh(np.arange(len(values)), share, left=left, color=color, label=label)
            for i, v in enumerate(share):
                if v >= 10:
                    ax.text(left[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", fontsize=10,
                            color=NAVY if sentiment == "Neutral" else "white")
            left += share
        ax.set_yticks(np.arange(len(values)), [f"{x.Group} (n={x.Review_N:,})" for x in values.itertuples()])
        ax.invert_yaxis()
        ax.set_xlim(0, 100)
        ax.set_xlabel("Tỷ lệ trên tổng cặp nhãn trong nhóm (%)")
        ax.set_title("Cơ cấu sentiment theo nền tảng" if kind == "Platform" else "Cơ cấu sentiment theo top 5 brand")
    axes[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, -.26), frameon=False, fontsize=10)
    figure("07b_group_sentiment_overall", fig)


def preprocessing_history():
    with zipfile.ZipFile(source("json_cleanup_backup.zip")) as z:
        def get(name):
            return json.loads(z.read(name).decode("utf-8-sig"))
        relevance = get("crawl-data/analysis_output/relevance_filter/removal_summary_20261104_021037_737088.json")
        language = get("crawl-data/analysis_output/language_audit/removal_summary_20261104_030132_600523.json")
        seller = get("crawl-data/analysis_output/seller_reply_cleanup/summary.json")
        noise = get("data_cleanup_log.json")
        unlabeled = get("unlabeled_cleanup_log.json")
    with zipfile.ZipFile(source("data_cleanup_backup.zip")) as z:
        def records(name):
            return list(csv.DictReader(io.StringIO(z.read(name).decode("utf-8-sig"), newline="")))
        early = records("crawl-data/data/combined_reviews_brand_filled.before_preprocessing_20261104_013222_692151.csv")
        unresolved = records("crawl-data/analysis_output/brand_enrichment/unresolved_products.csv")
        before_relevance = records("crawl-data/data/combined_reviews_brand_filled.before_relevance_removal_20261104_021037_737088.csv")
        before_seller = records("crawl-data/data/combined_reviews_brand_filled.before_seller_reply_cleanup_20261104_024256_121778.csv")
    raw = frame("crawl-data/data/combined_reviews.csv")
    early_removed = len(raw) - len(early)
    kept_urls = {r["product_url"] for r in early}
    excluded_products = [r for r in unresolved if r["product_url"] not in kept_urls]
    assert sum(int(r["affected_review_count"]) for r in excluded_products) == early_removed
    extra = relevance["rows_after"] - len(before_seller)
    key = "crawl-data\\data\\combined_reviews_brand_filled_language_filtered.csv"
    rows = [
        ("Thương hiệu chưa xác minh (11 sản phẩm)", len(raw), early_removed, len(early), "Đối chiếu snapshot"),
        ("Ngoài chủ đề / nội dung rác đã duyệt", relevance["rows_before"], relevance["rows_removed"], relevance["rows_after"], "Nhật ký xóa"),
        ("Review tiếng Nga kèm phản hồi shop", relevance["rows_after"], extra, len(before_seller), "Đối chiếu snapshot"),
        ("Chỉ có phản hồi người bán", seller["rows_before"], seller["seller_only_rows_removed"], seller["rows_after"], "Nhật ký xóa"),
        ("Ngôn ngữ khác / trộn / chưa rõ", language["rows_before"], language["rows_removed"], language["rows_after"], "Nhật ký xóa"),
        ("40 câu nhiễu và 3 câu mơ hồ", language["rows_after"], noise["removed_records"][key], noise["remaining_records"][key], "Nhật ký xóa"),
        ("Các nội dung còn nhãn rỗng", noise["remaining_records"][key], unlabeled["removed_records"][key], unlabeled["remaining_records"][key], "17 text duy nhất / 19 review"),
    ]
    result = table("01_preprocessing_stages", pd.DataFrame(rows, columns=["Stage", "Before", "Removed", "After", "Evidence"]))
    assert all(result.Before - result.Removed == result.After)
    assert result.Before.iloc[1:].tolist() == result.After.iloc[:-1].tolist()
    fig, ax = plt.subplots(figsize=(11.4, 3.7))
    x = np.arange(len(result) + 1)
    y = [len(raw)] + result.After.tolist()
    ax.plot(x, y, marker="o", linewidth=2.3, color=BLUE)
    for xx, yy in zip(x, y):
        ax.annotate(f"{yy:,}", (xx, yy), xytext=(0, 10), textcoords="offset points", ha="center", weight="bold")
    ax.set_xticks(x, ["Thu thập", "Thương\nhiệu", "Nội dung", "Rà soát\nbổ sung", "Phản hồi\nshop", "Ngôn ngữ", "Nhiễu sau\ngán nhãn", "Nhãn\nrỗng"])
    ax.set_ylim(min(y) - 100, max(y) + 120)
    ax.set_ylabel("Số review còn lại")
    ax.set_title("Các mốc dữ liệu được khôi phục từ nhật ký và bản sao lưu")
    ax.grid(axis="y", alpha=.15)
    figure("01_preprocessing_flow", fig)
    MANIFEST["preprocessing"] = {"raw_reviews": len(raw), "removed": int(result.Removed.sum()),
                                  "final_source_reviews": int(result.After.iloc[-1]),
                                  "seller_replies_stripped_kept": seller["replies_stripped_customer_kept"],
                                  "empty_templates_in_relevance_removal": 13}


def dataset_and_sentiments():
    source_df = frame("crawl-data/data/combined_reviews_brand_filled_language_filtered.csv")
    data = source_df.loc[source_df.cleaned_comment.fillna("").str.strip().ne("") & source_df.label.fillna("").str.strip().ne("")].copy().reset_index(drop=True)
    data["Platform"] = canonical_groups(data.platform)
    data["Brand"] = canonical_groups(data.brand)
    data["Whitespace_Tokens"] = data.cleaned_comment.str.split().str.len()
    data["Characters"] = data.cleaned_comment.str.len()
    mat = labels_matrix(data)
    mentioned = mat.any(axis=2)
    overview = data.groupby("Platform").agg(Reviews=("label", "size"), Products=("product_url", "nunique"),
                                            Brands=("Brand", "nunique"), Median_Whitespace_Tokens=("Whitespace_Tokens", "median"),
                                            Mean_Rating=("rating", "mean")).reset_index()
    overview["Share_Pct"] = 100 * overview.Reviews / len(data)
    table("02_platform_overview", overview)
    brand_counts = data.Brand.value_counts().rename_axis("Brand").reset_index(name="Reviews")
    table("03_brand_counts", brand_counts)
    top5 = brand_counts.head(5).Brand.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.8), gridspec_kw={"width_ratios": [.8, 1.4]})
    d = overview.sort_values("Reviews")
    axes[0].barh(d.Platform, d.Reviews, color=["#7da8b8", BLUE, NAVY])
    for i, row in enumerate(d.itertuples()): axes[0].text(row.Reviews + 70, i, f"{row.Reviews:,}\n({row.Share_Pct:.1f}%)", va="center", fontsize=10)
    axes[0].set_xlim(0, d.Reviews.max() * 1.27)
    axes[0].set_title("Review theo nền tảng")
    axes[0].set_xlabel("Số review hợp lệ")
    d = brand_counts.head(10).iloc[::-1]
    axes[1].barh(d.Brand, d.Reviews, color=BLUE)
    for i, row in enumerate(d.itertuples()):axes[1].text(row.Reviews + 15, i, f"{row.Reviews:,}", va="center", fontsize=10)
    axes[1].set_xlim(0, d.Reviews.max() * 1.15)
    axes[1].set_title("10 thương hiệu có nhiều review nhất")
    axes[1].set_xlabel("Số review hợp lệ")
    figure("02_platform_and_top_brands", fig)
    lengths = []
    for name, sub in [("Toàn bộ", data)] + [(g, data.loc[data.Platform.eq(g)]) for g in overview.Platform]:
        lengths.append({"Group": name, "N": len(sub), "Mean": sub.Whitespace_Tokens.mean(), "P50": sub.Whitespace_Tokens.quantile(.5),
                        "P95": sub.Whitespace_Tokens.quantile(.95), "P99": sub.Whitespace_Tokens.quantile(.99), "Max": sub.Whitespace_Tokens.max()})
    table("04_length_summary", pd.DataFrame(lengths))
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
    axes[0].hist(data.Whitespace_Tokens, bins=np.arange(0, min(500, data.Whitespace_Tokens.max()) + 5, 5), color=BLUE, edgecolor="white")
    axes[0].set_xlim(0, min(250, data.Whitespace_Tokens.quantile(.995)))
    axes[0].set_title("Phân phối độ dài cleaned_comment")
    axes[0].set_xlabel("Đơn vị cách nhau bằng khoảng trắng")
    axes[0].set_ylabel("Số review")
    groups = overview.Platform.tolist()
    axes[1].boxplot([data.loc[data.Platform.eq(g), "Whitespace_Tokens"] for g in groups], tick_labels=groups,
                    showfliers=False, patch_artist=True, boxprops={"facecolor": "#b3cbd7"})
    axes[1].set_title("So sánh độ dài giữa các nền tảng")
    axes[1].set_ylabel("Đơn vị cách nhau bằng khoảng trắng")
    figure("03_review_lengths", fig)
    ratings = pd.crosstab(data.Platform, pd.to_numeric(data.rating)).reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    table("05_rating_counts", ratings.rename_axis("Platform").reset_index())
    fig, ax = plt.subplots(figsize=(10.8, 3.7))
    bottom = np.zeros(len(ratings))
    for star, color in zip([1, 2, 3, 4, 5], ["#b64e58", "#d58e76", "#d6c38c", "#82afa3", "#247f72"]):
        values = 100 * ratings[star].to_numpy() / ratings.sum(axis=1).to_numpy()
        ax.barh(ratings.index, values, left=bottom, label=f"{star} sao", color=color)
        for i, v in enumerate(values):
            if v >= 5:ax.text(bottom[i] + v / 2, i, f"{v:.1f}%", ha="center", va="center", color="white" if star in [1, 5] else NAVY, fontsize=10)
        bottom += values
    ax.set_xlabel("Tỷ lệ review trong từng nền tảng (%)")
    ax.set_xlim(0, 100)
    ax.legend(ncol=5, loc="upper center", bbox_to_anchor=(.5, 1.22), frameon=False)
    figure("04_rating_by_platform", fig)
    counts = mat.sum(axis=0)
    stats = pd.DataFrame(counts, columns=POLS)
    stats.insert(0, "Aspect", ASPECTS)
    stats["Mentioned_Reviews"] = mentioned.sum(axis=0)
    stats["Label_Pairs"] = counts.sum(axis=1)
    stats["Multiple_Sentiment_Reviews"] = (mat.sum(axis=2) > 1).sum(axis=0)
    stats["Negative_Review_Rate_Pct"] = 100 * stats.negative / stats.Mentioned_Reviews
    table("06_aspect_sentiment_counts", stats)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.1), gridspec_kw={"width_ratios": [1, 1.15]})
    im = axes[0].imshow(counts, cmap="Blues", aspect="auto")
    axes[0].set_xticks([0, 1, 2], ["Tích cực", "Tiêu cực", "Trung tính"])
    axes[0].set_yticks(range(11), NAMES)
    for i in range(11):
        for j in range(3):axes[0].text(j, i, f"{counts[i,j]:,}", ha="center", va="center", fontsize=9, color="white" if counts[i,j] > counts.max() * .52 else NAVY)
    axes[0].set_title("Số cặp nhãn aspect - sentiment")
    left = np.zeros(11)
    totals = counts.sum(axis=1)
    for j, pol in enumerate(POLS):
        v = 100 * counts[:, j] / totals
        axes[1].barh(range(11), v, left=left, color=COLORS[pol], label={"positive":"Tích cực", "negative":"Tiêu cực", "neutral":"Trung tính"}[pol])
        for i, val in enumerate(v):
            if val >= 9:axes[1].text(left[i] + val / 2, i, f"{val:.0f}%", va="center", ha="center", fontsize=9, color="white" if pol != "neutral" else NAVY)
        left += v
    axes[1].set_yticks(range(11), [""] * 11)
    axes[1].invert_yaxis()
    axes[1].set_xlim(0, 100)
    axes[1].set_title("Cơ cấu trên tổng cặp nhãn của aspect")
    axes[1].set_xlabel("Tỷ lệ cặp nhãn (%)")
    axes[1].legend(ncol=3, loc="lower center", bbox_to_anchor=(.5, -.22), fontsize=9, frameon=False)
    figure("05_sentiment_counts_and_shares", fig)
    order = np.argsort(stats.Negative_Review_Rate_Pct.to_numpy())
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharey=True)
    axes[0].barh(np.arange(11), stats.negative.to_numpy()[order], color=COLORS["negative"])
    axes[0].set_yticks(range(11), [NAMES[i] for i in order])
    axes[0].set_title("Số review có nhãn tiêu cực")
    axes[0].set_xlabel("Số review")
    axes[1].barh(np.arange(11), stats.Negative_Review_Rate_Pct.to_numpy()[order], color="#d29c6f")
    for j, i in enumerate(order):axes[1].text(stats.Negative_Review_Rate_Pct.iloc[i] + .4, j, f"{stats.Negative_Review_Rate_Pct.iloc[i]:.1f}% | n={stats.Mentioned_Reviews.iloc[i]:,}", va="center", fontsize=9)
    axes[1].set_xlim(0, stats.Negative_Review_Rate_Pct.max() * 1.65)
    axes[1].set_title("Tỷ lệ tiêu cực trong review đề cập")
    axes[1].set_xlabel("Review có nhãn negative / review đề cập (%)")
    figure("06_negative_counts_and_rates", fig)
    group_rows = []
    for column, groups in [("Platform", overview.Platform.tolist()), ("Brand", top5)]:
        for group in groups:
            mask = data[column].eq(group).to_numpy()
            group_counts = mat[mask].sum(axis=0)
            n = mentioned[mask].sum(axis=0)
            for i, a in enumerate(ASPECTS):
                group_rows.append({"Group_Type":column,"Group":group,"Aspect":a,"Review_N":int(mask.sum()),"Mentioned":int(n[i]),
                                   "Positive":int(group_counts[i,0]),"Negative":int(group_counts[i,1]),"Neutral":int(group_counts[i,2]),
                                   "Negative_Rate_Pct":100*group_counts[i,1]/n[i] if n[i] else np.nan})
    grouped = table("07_sentiment_by_group", pd.DataFrame(group_rows))
    group_sentiment_overall(grouped)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.2), gridspec_kw={"width_ratios":[.9,1.25]}, sharey=True)
    for ax, kind, groups in zip(axes, ["Platform","Brand"], [overview.Platform.tolist(),top5]):
        subset=grouped.loc[grouped.Group_Type.eq(kind)]
        m=subset.pivot(index="Aspect",columns="Group",values="Negative_Rate_Pct").reindex(index=ASPECTS,columns=groups)
        n=subset.pivot(index="Aspect",columns="Group",values="Mentioned").reindex(index=ASPECTS,columns=groups)
        im=ax.imshow(m,cmap="YlOrRd",vmin=0,vmax=50,aspect="auto")
        for i in range(11):
            for j in range(len(groups)):
                ax.text(j,i,f"{m.iloc[i,j]:.1f}" + ("*" if n.iloc[i,j]<30 else ""),ha="center",va="center",fontsize=9,color="white" if m.iloc[i,j]>32 else NAVY)
        ax.set_xticks(range(len(groups)),groups,rotation=25,ha="right")
        ax.set_yticks(range(11),NAMES)
        ax.set_title("Tỷ lệ tiêu cực theo sàn (%)" if kind=="Platform" else "Tỷ lệ tiêu cực theo top 5 brand (%)")
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.7,label="Review có negative / review đề cập (%)")
    figure("07_negative_rate_by_group",fig)
    for kind,groups in [("Platform",overview.Platform.tolist()),("Brand",top5)]:
        sub=grouped.loc[grouped.Group_Type.eq(kind)].copy()
        for pol in ["Positive","Negative","Neutral"]:sub[f"{pol}_Pair_Share_Pct"]=100*sub[pol]/sub[["Positive","Negative","Neutral"]].sum(axis=1)
        table(f"08_{kind.lower()}_sentiment_shares",sub)
    MANIFEST["dataset"]={"source_rows":len(source_df),"valid_reviews":len(data),"empty_text_rows":int(source_df.cleaned_comment.fillna("").str.strip().eq("").sum()),
                          "brands":int(data.Brand.nunique()),"products":int(data.product_url.nunique()),"platforms":overview.Platform.tolist(),
                          "top5_brands":top5,"label_pairs":int(mat.sum()),"sentiment_totals":dict(zip(POLS,mat.sum(axis=(0,1)).astype(int).tolist())),
                          "multi_sentiment_reviews":int((mat.sum(axis=2)>1).any(axis=1).sum())}
    return data,mat


def token_lengths(data):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained("vinai/phobert-base",local_files_only=True)
    unique=frame("Labeling/data/unique_texts_gemini_labeled.csv")
    splits={name:frame(f"train-models/data/{name}_cleanser.csv") for name in ["train","val","test"]}
    all_texts=list(dict.fromkeys(data.cleaned_comment.astype(str).tolist()+unique.cleaned_comment.astype(str).tolist()))
    lengths={}
    for start in range(0,len(all_texts),512):
        batch=all_texts[start:start+512]
        ids=tokenizer(batch,add_special_tokens=True,truncation=False,padding=False,verbose=False)["input_ids"]
        lengths.update(zip(batch,map(len,ids)))
    all_data={"Review nguồn":data,"Text duy nhất":unique,**splits}
    rows=[]
    for name,df in all_data.items():
        values=df.cleaned_comment.astype(str).map(lengths).to_numpy()
        rows.append({"Dataset":name,"N":len(values),"Mean":values.mean(),"P50":np.quantile(values,.5),"P95":np.quantile(values,.95),
                     "P99":np.quantile(values,.99),"Max":values.max(),"Over_128":int((values>128).sum()),"Over_128_Pct":100*(values>128).mean()})
    stats=table("09_phobert_token_lengths",pd.DataFrame(rows))
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.3))
    for ax,name in zip(axes,["Review nguồn","train"]):
        values=all_data[name].cleaned_comment.astype(str).map(lengths)
        ax.hist(values,bins=np.arange(0,values.max()+10,10),color=BLUE,edgecolor="white")
        ax.axvline(128,color=COLORS["negative"],ls="--",lw=1.8,label="MAX_LEN = 128")
        ax.set_xlim(0,min(450,values.max()+10))
        ax.set_title("Token PhoBERT: toàn bộ review" if name=="Review nguồn" else "Token PhoBERT: tập train")
        ax.set_xlabel("Token, gồm token đặc biệt, trước khi cắt")
        ax.set_ylabel("Số review / text")
        ax.legend(frameon=False)
    figure("08_phobert_token_distribution",fig)
    table("10_split_counts",pd.DataFrame([{"Split":name,"Rows":len(df)} for name,df in splits.items()]))
    MANIFEST["dataset"]["unique_texts"]=len(unique)
    MANIFEST["dataset"]["split_counts"]={name:len(df) for name,df in splits.items()}
    MANIFEST["dataset"]["tokenizer"]="vinai/phobert-base; local cache; add_special_tokens=True; truncation=False"
    assert sum(len(df) for df in splits.values())==len(unique)
    sets=[set(df.cleaned_comment) for df in splits.values()]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    return splits,tokenizer


def model_tables():
    models=[];aspects=[];details=[]
    for path in sorted((ROOT/"evaluation/results/cleanser_models_comparison/outputs").glob("*.txt")):
        source(str(path.relative_to(ROOT)).replace("\\","/"))
        text=path.read_text(encoding="utf-8-sig")
        architecture=re.search(r"^Model: (\w+)",text,re.M).group(1)
        scenario=re.search(r"^Scenario: (\w+)",text,re.M).group(1)
        for line in text.splitlines():
            cells=[c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells)==11 and cells[0] in ["ACD","SPC","ACD+SPC"]:
                models.append({"Model":architecture,"Scenario":scenario,"Task":cells[0],"Precision_Macro":float(cells[2])*100,
                               "Recall_Macro":float(cells[5])*100,"F1_Macro":float(cells[8])*100,"F1_Micro":float(cells[7])*100})
            elif len(cells)==8 and cells[0] in ASPECTS:
                aspects.append({"Model":architecture,"Scenario":scenario,"Aspect":cells[0],"ACD_P":float(cells[1]),"ACD_R":float(cells[2]),
                                "ACD_F1":float(cells[3]),"Joint_P":float(cells[4]),"Joint_R":float(cells[5]),"Joint_F1":float(cells[6]),"Support":int(cells[7])})
            elif len(cells)==5 and "#" in cells[0] and cells[0].split("#")[0] in ASPECTS:
                aspect,pol=cells[0].split("#")
                details.append({"Model":architecture,"Scenario":scenario,"Aspect":aspect,"Sentiment":pol,
                                "Precision":float(cells[1]),"Recall":float(cells[2]),"F1":float(cells[3]),"Support":int(cells[4])})
    models=table("11_absa_model_metrics",pd.DataFrame(models))
    aspects=table("12_absa_aspect_metrics",pd.DataFrame(aspects))
    details=table("13_absa_detailed_metrics",pd.DataFrame(details))
    assert len(models)==36 and len(aspects)==132 and len(details)==396
    rank=models.loc[models.Task.eq("ACD+SPC")].sort_values("F1_Macro",ascending=False)
    table("14_absa_ranking",rank)
    best=rank.iloc[0]
    MANIFEST["absa"]={"leading_test_model":best.Model,"leading_test_scenario":best.Scenario,"joint_f1_pct":best.F1_Macro,
                      "rank_definition":"Ranking of existing test outputs; not a new validation-based selection or significance claim."}
    architecture_order=["bilstm","multitask","bigru","birnn"]
    fig,ax=plt.subplots(figsize=(10.4,4.6))
    for j,(scenario,label,color) in enumerate(zip(["baseline","class_weights","focal_loss"],["CE","CE có trọng số","Focal Loss"],["#a5b8c4",BLUE,"#2a8877"])):
        vals=rank.loc[rank.Scenario.eq(scenario)].set_index("Model").loc[architecture_order,"F1_Macro"]
        bars=ax.bar(np.arange(4)+(j-1)*.24,vals,width=.22,label=label,color=color)
        ax.bar_label(bars,fmt="%.2f",fontsize=9,padding=3)
    ax.set_xticks(range(4),["PhoBERT + BiLSTM","PhoBERT MultiTask","PhoBERT + BiGRU","PhoBERT + BiRNN"])
    ax.set_ylim(0,80)
    ax.set_ylabel("Joint macro-F1 (%)")
    ax.legend(ncol=3,frameon=False,loc="upper left")
    ax.grid(axis="y",alpha=.15)
    figure("09_absa_architecture_loss",fig)
    best_aspects=aspects.loc[aspects.Model.eq(best.Model)].pivot(index="Aspect",columns="Scenario",values="Joint_F1").reindex(ASPECTS)
    best_aspects["Support"]=aspects.loc[aspects.Model.eq(best.Model)&aspects.Scenario.eq(best.Scenario)].set_index("Aspect").Support
    best_aspects["Delta_Focal_CE"]=best_aspects.focal_loss-best_aspects.baseline
    table("15_leading_architecture_aspect_comparison",best_aspects.rename_axis("Aspect").reset_index())
    fig,ax=plt.subplots(figsize=(10.6,5.2))
    y=np.arange(11)
    ax.hlines(y,best_aspects.baseline,best_aspects.focal_loss,color="#b1bdc5",lw=2)
    ax.scatter(best_aspects.baseline,y,s=48,color="#899eaf",label="CE")
    ax.scatter(best_aspects.focal_loss,y,s=48,color="#25836e",label="Focal Loss")
    ax.set_yticks(y,NAMES)
    ax.invert_yaxis()
    ax.set_xlim(25,100)
    ax.set_xlabel("Joint macro-F1 theo aspect (%)")
    ax.legend(frameon=False,ncol=2,loc="upper center",bbox_to_anchor=(.5,1.10))
    ax.grid(axis="x",alpha=.15)
    figure("10_absa_aspect_focal_vs_ce",fig)
    leading=details.loc[details.Model.eq(best.Model)&details.Scenario.eq(best.Scenario)]
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.2))
    for ax,column,title,cmap in zip(axes,["F1","Support"],["F1 của 33 cặp aspect - sentiment (%)","Support tương ứng trong tập test"],["YlGnBu","Blues"]):
        values=leading.pivot(index="Aspect",columns="Sentiment",values=column).reindex(index=ASPECTS,columns=POLS)
        im=ax.imshow(values,cmap=cmap,aspect="auto",vmin=0,vmax=100 if column=="F1" else None)
        for i in range(11):
            for j in range(3):
                value=values.iloc[i,j]
                ax.text(j,i,f"{value:.1f}" if column=="F1" else f"{int(value)}",ha="center",va="center",fontsize=9,
                        color="white" if value>(65 if column=="F1" else values.to_numpy().max()*.55) else NAVY)
        ax.set_xticks(range(3),["Tích cực","Tiêu cực","Trung tính"])
        ax.set_yticks(range(11),NAMES if column=="F1" else [""]*11)
        ax.set_title(title)
    figure("11_absa_sentiment_f1_support",fig)
    return models,aspects,details


def inference_and_errors(splits,tokenizer):
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import f1_score,confusion_matrix
    torch.set_num_threads(4)
    spec=importlib.util.spec_from_file_location("cleanser_evaluation_report",source("evaluation/evaluation_2_cleanser.py"))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    test=splits["test"].reset_index(drop=True)
    label_lists=[[f"{a}#{s}" for a,s in pairs(raw)] for raw in test.label]
    dataset=module.ABSADataset(test.cleaned_comment.tolist(),label_lists,tokenizer,max_len=128)
    loader=DataLoader(dataset,batch_size=32,shuffle=False)
    checkpoints={"birnn_focal_loss":("PhoBERT_BiRNN","best_model_birnn_cleanser_focal_loss_VI.pt"),
                 "birnn_baseline":("PhoBERT_BiRNN","best_model_birnn_cleanser_baseline_VI.pt")}
    hashes={key:sha(ROOT/"train-models/Models/Cleanser"/name) for key,(_,name) in checkpoints.items()}
    signature={"checkpoints":hashes,"test":sha(ROOT/"train-models/data/test_cleanser.csv"),"evaluator":sha(ROOT/"evaluation/evaluation_2_cleanser.py")}
    cache_meta=CACHE/"absa_predictions_signature.json"
    cache_npz=CACHE/"absa_predictions.npz"
    if cache_meta.exists() and cache_npz.exists() and json.loads(cache_meta.read_text())==signature:
        cached=np.load(cache_npz)
        results={name:cached[name] for name in checkpoints}
        true=cached["true"]
        progress("Reusing verified local inference cache.")
    else:
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        results={};true=None
        for name,(cls,filename) in checkpoints.items():
            progress(f"Running local inference: {name} ({device})")
            model=getattr(module,cls)().to(device)
            weights=torch.load(ROOT/"train-models/Models/Cleanser"/filename,map_location="cpu",weights_only=True)
            model.load_state_dict(weights,strict=True)
            del weights
            model.eval();pred=[];gold=[]
            with torch.inference_mode():
                for batch in loader:
                    logits=model(input_ids=batch["input_ids"].to(device),attention_mask=batch["attention_mask"].to(device))
                    pred.append(logits.argmax(dim=2).cpu().numpy())
                    gold.append(batch["targets"].numpy())
            results[name]=np.vstack(pred)
            found_true=np.vstack(gold)
            if true is None:true=found_true
            else:np.testing.assert_array_equal(true,found_true)
            del model
            if device.type=="cuda":torch.cuda.empty_cache()
        np.savez_compressed(cache_npz,true=true,**results)
        cache_meta.write_text(json.dumps(signature,indent=2),encoding="utf-8")
    def binary(values):
        return np.stack([values[:,i]==j for i in range(11) for j in range(3)],axis=1)
    true_bin=binary(true)
    metrics=[];per_row={}
    for name,pred in results.items():
        pred_bin=binary(pred)
        score=100*f1_score(true_bin,pred_bin,average="macro",zero_division=0)
        expected=MANIFEST["absa"]["joint_f1_pct"] if name=="birnn_focal_loss" else 64.40
        assert abs(score-expected)<.02,(name,score,expected)
        per_row[name]=np.stack([(true_bin&pred_bin).astype(np.int16),((~true_bin)&pred_bin).astype(np.int16),(true_bin&(~pred_bin)).astype(np.int16)],axis=2)
        metrics.append({"Model":name,"Joint_F1_Pct":score})
    rng=np.random.default_rng(42)
    weights=rng.multinomial(len(test),np.full(len(test),1/len(test)),size=1000).astype(np.float64)
    bootstrap={}
    for row in metrics:
        name=row["Model"]
        counts=(weights@per_row[name].reshape(len(test),-1)).reshape(1000,33,3)
        den=2*counts[:,:,0]+counts[:,:,1]+counts[:,:,2]
        scores=np.divide(2*counts[:,:,0],den,out=np.zeros_like(den),where=den>0).mean(axis=1)*100
        bootstrap[name]=scores
        row["CI95_Lower"],row["CI95_Upper"]=np.quantile(scores,[.025,.975])
    delta=bootstrap["birnn_focal_loss"]-bootstrap["birnn_baseline"]
    table("16_paired_bootstrap",pd.DataFrame(metrics))
    MANIFEST["absa"]["paired_bootstrap"]={"replicates":1000,"difference_focal_minus_ce_pp":metrics[0]["Joint_F1_Pct"]-metrics[1]["Joint_F1_Pct"],
                                            "ci95_lower":float(np.quantile(delta,.025)),"ci95_upper":float(np.quantile(delta,.975)),
                                            "scope":"Paired resampling of 1904 test reviews conditional on fixed checkpoints; not multi-seed training variance."}
    pred=results["birnn_focal_loss"]
    cm=confusion_matrix(true.ravel(),pred.ravel(),labels=[0,1,2,3])
    table("17_best_model_confusion_counts",pd.DataFrame(cm,columns=["Pred_Positive","Pred_Negative","Pred_Neutral","Pred_None"]).assign(True_State=["positive","negative","neutral","None"]))
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.5))
    for ax,values,title,percent in [(axes[0],cm,"Ma trận nhầm lẫn: số vị trí aspect",False),
                                    (axes[1],100*cm/cm.sum(axis=1,keepdims=True),"Chuẩn hóa theo trạng thái thật (%)",True)]:
        ax.imshow(values,cmap="Blues",vmin=0,vmax=100 if percent else None)
        for i in range(4):
            for j in range(4):ax.text(j,i,f"{values[i,j]:.1f}" if percent else f"{int(values[i,j]):,}",ha="center",va="center",fontsize=11,color="white" if values[i,j]>(60 if percent else values.max()*.5) else NAVY)
        ax.set_xticks(range(4),["Positive","Negative","Neutral","None"])
        ax.set_yticks(range(4),["Positive","Negative","Neutral","None"])
        ax.set_xlabel("Dự đoán");ax.set_ylabel("Tham chiếu");ax.set_title(title)
    figure("12_best_model_confusion",fig)
    records=[];error_types=Counter()
    for i,j in np.argwhere(pred!=true):
        t,p=int(true[i,j]),int(pred[i,j])
        kind="Bỏ sót aspect" if p==3 else "Phát hiện dư aspect" if t==3 else "Nhầm sentiment"
        error_types[kind]+=1
        records.append({"Test_Row_1based":int(i+1),"Aspect":ASPECTS[j],"Gold":(POLS+["None"])[t],"Predicted":(POLS+["None"])[p],
                        "Error_Type":kind,"Review":str(test.cleaned_comment.iloc[i]),"Token_Length":len(tokenizer.encode(str(test.cleaned_comment.iloc[i]),add_special_tokens=True,truncation=False))})
    errors=table("18_best_model_errors",pd.DataFrame(records))
    chosen=[];used=set()
    for kind in ["Nhầm sentiment","Bỏ sót aspect","Phát hiện dư aspect"]:
        candidates=errors.loc[errors.Error_Type.eq(kind)&errors.Review.str.len().between(35,200)].sort_values(["Token_Length","Test_Row_1based"])
        for row in candidates.to_dict("records"):
            if row["Test_Row_1based"] not in used:
                chosen.append(row);used.add(row["Test_Row_1based"])
            if sum(x["Error_Type"]==kind for x in chosen)>=2:break
    table("19_error_examples",pd.DataFrame(chosen))
    table("20_error_type_counts",pd.DataFrame([{"Error_Type":k,"Count":v} for k,v in error_types.items()]))
    prediction_rows=test[["cleaned_comment","label"]].copy()
    prediction_rows.insert(0,"Test_Row_1based",np.arange(1,len(test)+1))
    prediction_rows["Predicted_Labels"]=[";".join(f"{{{ASPECTS[j]}#{POLS[p]}}}" for j,p in enumerate(row) if p<3) for row in pred]
    table("21_best_model_predictions",prediction_rows)
    MANIFEST["absa"]["inference_verified"]=True
    MANIFEST["absa"]["checkpoints"]=hashes


def load_pipeline():
    spec=importlib.util.spec_from_file_location("cleanser_ipa_for_report",source("Business/ipa_shap_cleanser.py"))
    p=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(p)
    p.OUTPUT_DIR=TABLES
    return p


def shap_and_ipa(data,mat):
    import xgboost as xgb
    p=load_pipeline()
    source("Business/kano_shap_cleanser.py")
    regression=frame("Business/customer_satisfaction_xai_ipa_results_cleanser/model_comparison_oof.csv")
    table("22_rating_regression_oof",regression)
    folds=frame("Business/customer_satisfaction_xai_ipa_results_cleanser/model_cv_fold_metrics.csv")
    table("23_rating_regression_folds",folds)
    df,raw,y,sentiment_cols=p.preprocess_data(p.DATA_PATH)
    assert len(df)==len(data) and raw.shape[1]==22 and len(sentiment_cols)==20
    progress("Fitting the canonical IPA explanatory model and recalculating SHAP interactions.")
    X,model,scaler,shap_values,base=p.fit_xgboost_for_explanation(raw,y,sentiment_cols)
    xai=p.calculate_xai(X,shap_values)
    ipa=p.calculate_ipa(df,xai)
    table("24_shap_aspect_summary",xai)
    table("25_ipa_overall",ipa)
    old=frame("Business/customer_satisfaction_xai_ipa_results_cleanser/ipa_attributes.csv")
    np.testing.assert_allclose(ipa.set_index("Aspect").loc[p.ASPECTS,"Importance_Pct"],old.set_index("Aspect").loc[p.ASPECTS,"Importance_Pct"],rtol=2e-5,atol=1e-5)
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.0),gridspec_kw={"width_ratios":[1.15,1]})
    order=xai.sort_values("Importance_Pct")
    axes[0].barh([LABELS[a] for a in order.Aspect],order.Importance_Pct,color=BLUE)
    for i,v in enumerate(order.Importance_Pct):axes[0].text(v+.25,i,f"{v:.2f}%",va="center",fontsize=9)
    axes[0].set_xlim(0,order.Importance_Pct.max()*1.22);axes[0].set_xlabel("Importance trong 10 aspect (%)")
    axes[0].set_title("Mức độ quan trọng từ mean |SHAP|")
    mentions=mat.any(axis=2).sum(axis=0)
    for a in p.ASPECTS:
        xx=100*mentions[ASPECTS.index(a)]/len(data)
        yy=float(xai.set_index("Aspect").loc[a,"Importance_Pct"])
        axes[1].scatter(xx,yy,s=70,color=BLUE)
        axes[1].annotate(CODES[a],(xx,yy),xytext=(5,5),textcoords="offset points",fontsize=10)
    axes[1].set_xlabel("Review đề cập aspect / toàn bộ review (%)")
    axes[1].set_ylabel("Importance (%)");axes[1].set_title("Tần suất đề cập và importance")
    axes[1].grid(alpha=.15)
    figure("13_shap_importance_and_frequency",fig)
    indexed=xai.set_index("Aspect").loc[p.ASPECTS]
    fig,axes=plt.subplots(1,2,figsize=(11.4,5.0),sharey=True)
    for ax,suffix,title in zip(axes,["Pos","Neg"],["Đóng góp khi nhãn positive được bật","Đóng góp khi nhãn negative được bật"]):
        yy=np.arange(10)
        ax.barh(yy-.17,indexed[f"Mean_SHAP_{suffix}_Active"],height=.3,color="#81a9bc",label="Mean")
        ax.barh(yy+.17,indexed[f"Median_SHAP_{suffix}_Active"],height=.3,color="#ca8d70",label="Median")
        ax.axvline(0,color="#556575",lw=.8)
        ax.set_yticks(yy,[LABELS[a] for a in p.ASPECTS])
        ax.set_title(title);ax.set_xlabel("Signed SHAP (đơn vị rating)");ax.legend(frameon=False)
    axes[0].invert_yaxis()
    figure("14_shap_signed_mean_median",fig)
    matrix=xgb.DMatrix(X,feature_names=X.columns.tolist())
    iv=np.asarray(model.get_booster().predict(matrix,pred_interactions=True))[:,:-1,:-1]
    np.testing.assert_allclose(iv.sum(axis=2),shap_values,atol=1e-3)
    rows=[]
    for i in range(len(X.columns)):
        for j in range(i+1,len(X.columns)):
            a,b=X.columns[i],X.columns[j]
            both=(X[a].to_numpy()==1)&(X[b].to_numpy()==1)
            is_aspect=a in sentiment_cols and b in sentiment_cols
            values=2*iv[:,i,j]
            rows.append({"Feature_1":a,"Feature_2":b,"Is_Aspect_Pair":is_aspect,
                         "Different_Aspects":is_aspect and a.rsplit("_",1)[0]!=b.rsplit("_",1)[0],
                         "Mean_Abs_Interaction_All":float(np.abs(values).mean()),"Mean_Signed_Interaction_All":float(values.mean()),
                         "N_Both_Active":int(both.sum()) if is_aspect else np.nan})
    interactions=table("26_canonical_shap_interactions",pd.DataFrame(rows).sort_values("Mean_Abs_Interaction_All",ascending=False))
    cross=table("27_supported_cross_aspect_interactions",interactions.loc[interactions.Different_Aspects&interactions.N_Both_Active.ge(20)])
    top=cross.head(10).iloc[::-1]
    def feature_name(s):
        a,pol=s.rsplit("_",1)
        return f"{CODES[a]} ({'+' if pol=='Pos' else '-'})"
    fig,ax=plt.subplots(figsize=(10.6,4.9))
    names=[f"{feature_name(r.Feature_1)} × {feature_name(r.Feature_2)}" for r in top.itertuples()]
    ax.barh(names,top.Mean_Abs_Interaction_All,color=BLUE)
    for i,row in enumerate(top.itertuples()):ax.text(row.Mean_Abs_Interaction_All+.00012,i,f"n={int(row.N_Both_Active)}",va="center",fontsize=9)
    ax.set_xlim(0,top.Mean_Abs_Interaction_All.max()*1.24)
    ax.set_xlabel("Mean |2 × SHAP interaction| trên toàn bộ review")
    ax.set_title("10 cặp khác aspect có tương tác lớn nhất và ít nhất 20 review đồng đề cập")
    figure("15_top_cross_aspect_interactions",fig)
    table("28_top_interaction_labels",cross.head(10).assign(Label=lambda t:t.apply(lambda r:f'{feature_name(r.Feature_1)} × {feature_name(r.Feature_2)}',axis=1)))
    pred=model.predict(X)
    selected=[]
    for rating in [1,5]:
        ids=np.flatnonzero((y.to_numpy()==rating)&(df.cleaned_comment.astype(str).str.len().to_numpy()<250))
        candidate=ids[np.argmin(np.abs(pred[ids]-rating))]
        selected.append(int(candidate))
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.9))
    examples=[]
    for ax,i in zip(axes,selected):
        order=np.argsort(np.abs(shap_values[i]))[::-1]
        picked=order[:6];rest=order[6:]
        vals=np.r_[shap_values[i,picked],shap_values[i,rest].sum()]
        names=[feature_name(X.columns[j]) if X.columns[j] in sentiment_cols else ("Giá" if X.columns[j]=="price_scaled" else "Giảm giá") for j in picked]+["Các biến còn lại"]
        current=float(base[i])
        for j,(v,name) in enumerate(zip(vals,names)):
            ax.barh(j,abs(v),left=min(current,current+v),color=COLORS["positive"] if v>=0 else COLORS["negative"],height=.55)
            ax.text(current+v,j,f" {v:+.2f}",va="center",fontsize=9)
            current+=v
        ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
        ax.axvline(base[i],ls="--",color="#8c9ba4",lw=.8)
        ax.set_title(f"Rating thật {int(y.iloc[i])}; dự đoán {pred[i]:.2f}")
        ax.set_xlabel(f"Rating dự đoán; baseline = {base[i]:.2f}")
        examples.append({"Valid_Row_0based":i,"Actual_Rating":int(y.iloc[i]),"Predicted_Rating":float(pred[i]),"Baseline":float(base[i]),"Review":str(df.cleaned_comment.iloc[i])})
        assert abs(current-pred[i])<1e-3
    figure("16_local_rating_explanations",fig)
    table("29_local_rating_examples",pd.DataFrame(examples))
    platforms=frame("Business/ipa_by_platform_results_cleanser/ipa_all_groups.csv")
    brands=frame("Business/ipa_top5_brands_results_cleanser/ipa_all_groups.csv")
    table("30_ipa_platform",platforms);table("31_ipa_brand",brands)
    psummary=frame("Business/ipa_by_platform_results_cleanser/group_run_summary.csv")
    bsummary=frame("Business/ipa_top5_brands_results_cleanser/group_run_summary.csv")
    table("32_ipa_platform_summary",psummary);table("33_ipa_brand_summary",bsummary)
    comparison=frame("Business/customer_satisfaction_xai_ipa_mean_median_results_cleanser/ipa_mean_median_comparison.csv")
    table("34_ipa_mean_median_comparison",comparison)
    thresholds=frame("Business/customer_satisfaction_xai_ipa_mean_median_results_cleanser/ipa_mean_median_thresholds.csv")
    table("35_ipa_cutoff_comparison",thresholds)
    def ipa_axes(ax,values,title):
        for row in values.itertuples():
            ax.scatter(row.Performance_0_100,row.Importance_Pct,s=65,c=QCOLORS[QUADRANTS.index(row.IPA_Quadrant)],edgecolors="white",zorder=3)
            ax.annotate(CODES[row.Aspect],(row.Performance_0_100,row.Importance_Pct),xytext=(5,6),textcoords="offset points",fontsize=9)
        ax.axhline(values.Importance_Cut.iloc[0],ls="--",color="#7b8a97",lw=1)
        ax.axvline(values.Performance_Cut.iloc[0],ls="--",color="#7b8a97",lw=1)
        ax.set_xlim(max(0,values.Performance_0_100.min()-9),min(100,values.Performance_0_100.max()+9))
        ax.set_ylim(0,max(values.Importance_Pct.max(),values.Importance_Cut.iloc[0])*1.28)
        ax.set_title(title,fontsize=12)
        ax.set_xlabel("Performance (0-100)");ax.set_ylabel("Importance (%)");ax.grid(alpha=.1)
        p.arrange_ipa_labels(ax.figure,ax=ax)
    fig,ax=plt.subplots(figsize=(8.5,5.0))
    ipa_axes(ax,ipa,"IPA tổng thể - ngưỡng mean")
    figure("17_ipa_overall",fig)
    for group_col,values,summary,stem in [("Platform",platforms,psummary,"18_ipa_platform_matrices"),("Brand",brands,bsummary,"19_ipa_brand_matrices")]:
        groups=summary[group_col].tolist()
        cols=3;rs=1 if len(groups)==3 else 2
        fig,axes=plt.subplots(rs,cols,figsize=(11.4,4.4 if rs==1 else 7.4),squeeze=False)
        for ax,group in zip(axes.flat,groups):ipa_axes(ax,values.loc[values[group_col].eq(group)],group)
        for ax in list(axes.flat)[len(groups):]:ax.set_visible(False)
        figure(stem,fig)
        for group in groups:
            fig,ax=plt.subplots(figsize=(8.3,5))
            ipa_axes(ax,values.loc[values[group_col].eq(group)],group)
            slug=re.sub(r"[^a-z0-9]+","_",group.casefold()).strip("_")
            figure(f"ipa_{group_col.lower()}_{slug}",fig)
    from matplotlib.colors import ListedColormap
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.8),gridspec_kw={"width_ratios":[.85,1.2]},sharey=True)
    for ax,col,values,summary in zip(axes,["Platform","Brand"],[platforms,brands],[psummary,bsummary]):
        groups=summary[col].tolist()
        pivot=values.pivot(index="Aspect",columns=col,values="IPA_Quadrant").reindex(index=p.ASPECTS,columns=groups)
        numeric=pivot.map(lambda q:QUADRANTS.index(q))
        ax.imshow(numeric,cmap=ListedColormap(QCOLORS),vmin=-.5,vmax=3.5,aspect="auto")
        for i in range(10):
            for j in range(len(groups)):
                q=pivot.iloc[i,j]
                short=["Ưu tiên","Duy trì","Theo dõi","Cân nhắc"][QUADRANTS.index(q)]
                ax.text(j,i,short,ha="center",va="center",color="white" if q!="Low Priority" else NAVY,fontsize=9)
        ax.set_xticks(range(len(groups)),groups,rotation=25,ha="right")
        ax.set_yticks(range(10),[LABELS[a] for a in p.ASPECTS]);ax.set_title("Theo nền tảng" if col=="Platform" else "Theo top 5 thương hiệu")
    figure("20_ipa_quadrant_comparison",fig)
    fig,axes=plt.subplots(1,2,figsize=(11.4,4.6))
    for ax,method in zip(axes,["Mean","Median"]):
        values=ipa.copy()
        c=comparison.set_index("Aspect").loc[values.Aspect]
        values["IPA_Quadrant"]=c[f"IPA_Quadrant_{method}"].to_numpy()
        values["Importance_Cut"]=c[f"Importance_Cut_{method}"].to_numpy()
        values["Performance_Cut"]=c[f"Performance_Cut_{method}"].to_numpy()
        ipa_axes(ax,values,f"Ngưỡng {method.lower()}")
    figure("21_ipa_mean_median",fig)
    priority=ipa[["Aspect","IPA_Quadrant"]].rename(columns={"IPA_Quadrant":"Overall"}).set_index("Aspect")
    for col,values in [("Platform",platforms),("Brand",brands)]:
        priority=priority.join(values.pivot(index="Aspect",columns=col,values="IPA_Quadrant"))
    table("36_ipa_priority_comparison",priority.reindex(p.ASPECTS).reset_index())
    MANIFEST["ipa"]={"aspects":len(p.ASPECTS),"features":X.shape[1],"importance_cut":float(ipa.Importance_Cut.iloc[0]),
                      "performance_cut":float(ipa.Performance_Cut.iloc[0]),"quadrant_changes_mean_median":int(comparison.Quadrant_Changed.sum()),
                      "interaction_model":"Canonical IPA preprocessing + same XGBoost settings, recalculated locally for this report.",
                      "shap_additivity_verified":True,"interaction_additivity_verified":True}
    MANIFEST["validation"]["ipa_matches_existing_results"]=True
    MANIFEST["notes"] += ["ABSA statistics retain 11 aspects; SHAP/IPA exclude General and retain General-only reviews.",
                           "Raw label-pair counts allow multiple sentiments per aspect; classification targets keep the last sentiment, as in the existing evaluator.",
                           "Descriptive negative rates include any negative label among mentioned reviews. IPA performance excludes multi-sentiment aspect-review cases.",
                           "Regression CV metrics are existing ordinary 10-fold OOF results, not grouped CV. SHAP explains the full-data explanatory model.",
                           "SHAP interactions here use canonical IPA preprocessing, avoiding the separate legacy Kano price/discount parsing.",
                           "No external APIs were called. No new ABSA training or multi-seed experiments were conducted."]


def main():
    for directory in [TABLES,FIGURES,CACHE]:directory.mkdir(parents=True,exist_ok=True)
    setup_style()
    progress("1/6 Recovering preprocessing history.");preprocessing_history()
    progress("2/6 Computing descriptive and sentiment statistics.");data,mat=dataset_and_sentiments()
    progress("3/6 Counting untruncated PhoBERT tokens from local cache.");splits,tokenizer=token_lengths(data)
    progress("4/6 Parsing all 12 existing ABSA evaluation outputs.");model_tables()
    progress("5/6 Verifying leading-checkpoint predictions and collecting actual errors.");inference_and_errors(splits,tokenizer)
    progress("6/6 Building canonical SHAP, interactions and IPA comparisons.");shap_and_ipa(data,mat)
    unchanged=[]
    for relative,digest in MANIFEST["inputs"].items():
        assert sha(ROOT/relative)==digest,f"Source changed during report build: {relative}"
        unchanged.append(relative)
    MANIFEST["validation"]["source_files_unchanged"]=len(unchanged)
    MANIFEST["validation"]["figure_count"]=len(MANIFEST["figures"])
    MANIFEST["validation"]["table_count"]=len(MANIFEST["tables"])
    (HERE/"provenance.json").write_text(json.dumps(MANIFEST,ensure_ascii=False,indent=2),encoding="utf-8")
    progress(f"DONE: {len(MANIFEST['tables'])} tables, {len(MANIFEST['figures'])} figures. Sources unchanged.")


if __name__=="__main__":main()
