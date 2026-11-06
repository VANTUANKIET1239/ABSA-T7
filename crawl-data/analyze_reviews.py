"""Export review statistics as CSV tables, PNG charts, and an HTML report."""

from __future__ import annotations

import argparse
from datetime import date
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HERE = Path(__file__).resolve().parent
UNKNOWN = "Unknown / missing"
COLORS = ["#2563eb", "#0d9488", "#ea580c", "#9333ea", "#db2777"]


def prepare_data(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    # Preserve IDs, literal strings, and blanks instead of guessing numeric types.
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"platform", "brand", "rating", "date", "comment", "product_url"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")
    if raw.empty:
        raise ValueError("The input CSV contains no reviews.")

    df = raw.copy()
    platform = df["platform"].str.strip()
    df["platform"] = platform.str.casefold().map(
        {"shopee": "Shopee", "tiki": "Tiki", "lazada": "Lazada"}
    ).fillna(platform.mask(platform.eq(""), UNKNOWN))

    # Merge spelling differences in case/whitespace only; do not infer absent brands.
    brands = df["brand"].str.replace(r"\s+", " ", regex=True).str.strip()
    brand_keys = brands.str.casefold()
    brand_labels = brands[brands.ne("")].groupby(brand_keys).agg(
        lambda values: values.value_counts().index[0]
    )
    df["brand"] = brand_keys.map(brand_labels).fillna(UNKNOWN)

    numeric_rating = pd.to_numeric(df["rating"], errors="coerce")
    df["rating"] = numeric_rating.where(numeric_rating.isin([1, 2, 3, 4, 5]))
    df["parsed_date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")
    # Characters count Unicode code points (including spaces and punctuation).
    df["text_chars"] = df["comment"].str.len()
    # Whitespace-delimited tokens, not linguistic Vietnamese word segmentation.
    df["text_tokens"] = df["comment"].str.split().str.len()
    df["empty_comment"] = df["comment"].str.strip().eq("")
    df["product_key"] = df["product_url"].str.strip().replace("", pd.NA)

    stored = pd.to_numeric(
        raw.get("review_text_length", pd.Series("", index=raw.index)), errors="coerce"
    )
    quality = {
        "records": len(df),
        "platforms": df["platform"].nunique(),
        "known_brands": df.loc[df["brand"].ne(UNKNOWN), "brand"].nunique(),
        "missing_brand_records": int(df["brand"].eq(UNKNOWN).sum()),
        "exact_duplicate_rows": int(raw.duplicated().sum()),
        "repeated_comment_records_after_first": int(
            df.loc[~df["empty_comment"], "comment"].duplicated().sum()
        ),
        "empty_comments": int(df["empty_comment"].sum()),
        "invalid_or_missing_ratings": int(df["rating"].isna().sum()),
        "unparseable_or_missing_dates": int(df["parsed_date"].isna().sum()),
        "future_dated_records": int(df["parsed_date"].dt.normalize().gt(pd.Timestamp(date.today())).sum()),
        "missing_or_invalid_stored_lengths": int(stored.isna().sum()),
        "stored_length_mismatches": int((stored.notna() & stored.ne(df["text_chars"])).sum()),
    }
    return raw, df, quality


def grouped_summary(df: pd.DataFrame, key: str) -> pd.DataFrame:
    summary = df.groupby(key).agg(
        records=("comment", "size"),
        unique_product_urls=("product_key", "nunique"),
        valid_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        mean_text_chars=("text_chars", "mean"),
        median_text_chars=("text_chars", "median"),
        mean_text_tokens=("text_tokens", "mean"),
        missing_brands=("brand", lambda s: s.eq(UNKNOWN).sum()),
        empty_comments=("empty_comment", "sum"),
    )
    summary["share_pct"] = summary["records"] / len(df) * 100
    high = df["rating"].ge(4).groupby(df[key]).sum()
    summary["rating_4_or_5_pct"] = high.div(summary["valid_ratings"].replace(0, float("nan"))) * 100
    summary["missing_brand_pct"] = summary["missing_brands"] / summary["records"] * 100
    return summary.sort_values("records", ascending=False)


def make_tables(raw: pd.DataFrame, df: pd.DataFrame, quality: dict) -> dict[str, pd.DataFrame]:
    missing = raw.apply(lambda s: s.str.strip().eq("").sum()).to_frame("missing_records")
    missing["missing_pct"] = missing["missing_records"] / len(raw) * 100
    missing.index.name = "column"
    missing_platform = raw.assign(platform=df["platform"]).groupby("platform").agg(
        {c: lambda s: s.str.strip().eq("").mean() * 100 for c in raw.columns if c != "platform"}
    ).T
    missing_platform.index.name = "column"
    ratings = pd.crosstab(df["platform"], df["rating"]).reindex(columns=[1, 2, 3, 4, 5], fill_value=0)
    ratings.columns = [f"{int(c)}_star" for c in ratings.columns]
    monthly = pd.crosstab(df["parsed_date"].dt.to_period("M"), df["platform"])
    if not monthly.empty:
        monthly = monthly.reindex(pd.period_range(monthly.index.min(), monthly.index.max(), freq="M"), fill_value=0)
    monthly.index.name = "month"
    length_stats = df.groupby("platform")[["text_chars", "text_tokens"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    length_stats.columns = [f"{metric}_{stat}" for metric, stat in length_stats.columns]
    overall_lengths = df[["text_chars", "text_tokens"]].describe(
        percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    )
    brand_platform = pd.crosstab(df["brand"], df["platform"])
    brand_platform["total"] = brand_platform.sum(axis=1)
    return {
        "overview": pd.Series(quality, name="value").rename_axis("metric").to_frame(),
        "platform_summary": grouped_summary(df, "platform"),
        "brand_summary": grouped_summary(df, "brand"),
        "brand_by_platform": brand_platform.sort_values("total", ascending=False),
        "rating_by_platform": ratings,
        "length_by_platform": length_stats,
        "length_overall": overall_lengths,
        "missing_values": missing.sort_values("missing_records", ascending=False),
        "missing_pct_by_platform": missing_platform,
        "monthly_reviews": monthly,
        "unparsed_dates": df.loc[df["parsed_date"].isna()].groupby(
            ["platform", "date"]
        ).size().rename("records").to_frame(),
    }


def make_charts(df: pd.DataFrame, tables: dict, output: Path, top_n: int) -> list[tuple[str, str]]:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    charts = []

    def save(fig, name, caption):
        fig.tight_layout()
        fig.savefig(output / name, dpi=160, bbox_inches="tight")
        plt.close(fig)
        charts.append((name, caption))

    platforms = tables["platform_summary"]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(platforms.index, platforms["records"], color=COLORS[:len(platforms)])
    ax.bar_label(bars, labels=[f"{n:,}\n({n / len(df):.1%})" for n in platforms["records"]], padding=4)
    ax.set(title="Review records by platform", ylabel="Records", ylim=(0, platforms["records"].max() * 1.2))
    save(fig, "01_platform_counts.png", "All records, including repeated review text.")

    brands = tables["brand_summary"].drop(index=UNKNOWN, errors="ignore").head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, max(5, len(brands) * 0.35)))
    bars = ax.barh(brands.index, brands["records"], color=COLORS[1])
    ax.bar_label(bars, fmt="%.0f", padding=3)
    ax.set(title=f"Top {min(top_n, len(brands))} known brands by review count", xlabel="Records")
    ax.margins(x=0.18)
    save(fig, "02_top_brands.png", "Missing brands excluded. Case and repeated whitespace normalized; no brands inferred.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    counts = df["rating"].value_counts().reindex([1, 2, 3, 4, 5], fill_value=0)
    bars = axes[0].bar(range(1, 6), counts, color=COLORS[0])
    axes[0].bar_label(bars, fmt="%.0f", padding=3)
    axes[0].set(title="Overall rating distribution", xlabel="Stars", ylabel="Records", xticks=range(1, 6))
    axes[0].margins(y=0.15)
    rating_table = tables["rating_by_platform"]
    proportions = rating_table.div(rating_table.sum(axis=1).replace(0, float("nan")), axis=0).mul(100)
    proportions.plot.barh(stacked=True, ax=axes[1], color=["#dc2626", "#f97316", "#eab308", "#2dd4bf", "#0d9488"])
    axes[1].set(title="Rating mix by platform", xlabel="Percent of valid ratings", ylabel="", xlim=(0, 100))
    axes[1].legend(title="Rating", loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    save(fig, "03_rating_distribution.png", "Rating mix uses valid 1–5 ratings only. Collection quotas can affect these distributions.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    limit = max(1, float(df["text_chars"].quantile(0.99)))
    for i, platform in enumerate(platforms.index):
        values = df.loc[df["platform"].eq(platform), "text_chars"]
        axes[0].hist(values[values.le(limit)], bins=35, range=(0, limit), histtype="step", linewidth=2,
                     label=platform, color=COLORS[i % len(COLORS)])
    axes[0].set(title="Review character lengths (through overall P99)", xlabel="Characters", ylabel="Records")
    axes[0].legend()
    axes[1].boxplot([df.loc[df["platform"].eq(p), "text_chars"] for p in platforms.index],
                    tick_labels=platforms.index, showfliers=False)
    axes[1].set(title="Character lengths by platform", ylabel="Characters")
    save(fig, "04_review_lengths.png", "Lengths recalculated from comment text. Histogram omits values above P99; boxplot hides outliers. Tables include all records.")

    fig, ax = plt.subplots(figsize=(11, 5))
    monthly = tables["monthly_reviews"]
    if not monthly.empty:
        monthly.set_axis(monthly.index.to_timestamp()).plot(
            ax=ax, color=[COLORS[list(platforms.index).index(p) % len(COLORS)] for p in monthly.columns]
        )
    else:
        ax.text(0.5, 0.5, "No parseable review dates", ha="center", transform=ax.transAxes)
    ax.set(title="Collected reviews by review month", xlabel="Review date", ylabel="Records")
    save(fig, "05_monthly_reviews.png", "Uses recorded review dates, not crawl dates. Months between first and last dates include zeros; unparseable dates excluded.")

    fig, ax = plt.subplots(figsize=(10, 7))
    missing = tables["missing_pct_by_platform"]
    heat = ax.imshow(missing.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=100)
    ax.set(xticks=range(len(missing.columns)), xticklabels=missing.columns,
           yticks=range(len(missing)), yticklabels=missing.index,
           title="Missing source fields by platform")
    for row in range(len(missing)):
        for col in range(len(missing.columns)):
            value = missing.iloc[row, col]
            ax.text(col, row, f"{value:.0f}%", ha="center", va="center",
                    color="white" if value > 65 else "black", fontsize=8)
    fig.colorbar(heat, ax=ax, label="Missing (%)")
    save(fig, "06_missing_data.png", "Blank or whitespace-only source fields. Many fields are specific to one platform.")
    return charts


def write_report(df, quality, tables, charts, output, source):
    platform_table = tables["platform_summary"].round(2).to_html(border=0)
    brand_table = tables["brand_summary"].head(20).round(2).to_html(border=0)
    quality_table = tables["overview"].to_html(border=0)
    figures = "\n".join(f'<figure><img src="{escape(name)}" alt="{escape(name)}"><figcaption>{escape(caption)}</figcaption></figure>' for name, caption in charts)
    first, last = df["parsed_date"].min(), df["parsed_date"].max()
    dates = "Unavailable" if pd.isna(first) else f"{first:%Y-%m-%d} to {last:%Y-%m-%d}"
    html = f'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Review dataset overview</title>
<style>body{{font:16px/1.6 system-ui,sans-serif;color:#172033;background:#f3f6fa;margin:0}}main{{max-width:1200px;margin:auto;padding:32px}}h1,h2{{line-height:1.2}}section,figure{{background:white;padding:24px;margin:24px 0;border-radius:12px}}img{{max-width:100%;height:auto}}figcaption{{color:#536176}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;font-size:13px;white-space:nowrap}}th,td{{padding:8px 12px;border-bottom:1px solid #dce3ed;text-align:right}}th:first-child{{text-align:left}}</style>
<main><h1>Review dataset overview</h1>
<p>Source: {escape(source.name)} · Generated {date.today().isoformat()}</p>
<section><h2>{len(df):,} reviews across {quality['platforms']} platforms</h2>
<p>{quality['known_brands']:,} known brands; {quality['missing_brand_records']:,} records lack a brand.
Mean rating: {df['rating'].mean():.2f}/5. Median review length: {df['text_chars'].median():,.0f} characters.
Review dates: {dates}.</p></section>
<section><h2>How to interpret these results</h2><ul>
<li>These are collected review records. The Tiki crawler uses rating quotas, so rating shares should not be interpreted as representative market satisfaction.</li>
<li>All rows are retained. Repeated text and duplicate rows are counted in the quality table, not removed.</li>
<li>Platform names and brand case/whitespace variants are normalized in the analysis only. Missing brands remain unknown, and related brand names are not inferred or merged.</li>
<li>Character length includes whitespace and punctuation. Token length counts whitespace-separated tokens; it is not Vietnamese linguistic word segmentation.</li>
<li>Product counts use nonblank product URLs within each group. They do not identify the same product across platforms.</li>
<li>Shares use all records; rating percentages use valid ratings within each group. Source missingness is measured before calculated fields are added.</li>
<li>Dates are parsed without timezone conversion. {quality['unparseable_or_missing_dates']:,} missing or unparseable dates are excluded from the monthly chart and listed in unparsed_dates.csv. Relative dates need a crawl date to resolve. Future-date checks use the computer's current date.</li>
<li>Prices and discounts are omitted from comparisons because listing size, currency, and discount units have not been validated.</li>
</ul></section>
<section><h2>Platform summary</h2><div class="scroll">{platform_table}</div></section>
{figures}
<section><h2>Largest brand groups (including missing)</h2><div class="scroll">{brand_table}</div></section>
<section><h2>Data quality</h2><div class="scroll">{quality_table}</div></section>
<p>Complete summary tables are saved alongside this report as UTF-8 CSV files.</p></main></html>'''
    (output / "report.html").write_text(html, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "data" / "combined_reviews.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE / "analysis_output")
    parser.add_argument("--top-brands", type=int, default=15)
    args = parser.parse_args()
    if args.top_brands < 1:
        parser.error("--top-brands must be at least 1")
    raw, df, quality = prepare_data(args.input)
    tables = make_tables(raw, df, quality)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(args.output_dir / f"{name}.csv", encoding="utf-8-sig")
    charts = make_charts(df, tables, args.output_dir, args.top_brands)
    write_report(df, quality, tables, charts, args.output_dir, args.input)
    print(f"Analyzed {len(df):,} records.")
    print(tables["platform_summary"][["records", "share_pct", "mean_rating", "median_text_chars"]].round(2).to_string())
    print(f"Report: {(args.output_dir / 'report.html').resolve()}")


if __name__ == "__main__":
    main()
