# Review analysis

From the project root, install the dependencies in your Python environment and run:

```powershell
python -m pip install -r crawl-data/requirements-analysis.txt
python crawl-data/analyze_reviews.py
```

The default input is [combined_reviews.csv](data/combined_reviews.csv). Paths are
resolved relative to the script, so it also works when launched from another folder.

For another file or output location:

```powershell
python crawl-data/analyze_reviews.py --input "path/to/reviews.csv" --output-dir "path/to/results" --top-brands 20
```

Open [the HTML report](analysis_output/report.html) in a browser. Keep its six PNG
images in the same folder when sharing the report. Re-running the script replaces
its generated output files and leaves the input unchanged.

The output folder contains:

- `platform_summary.csv`: record counts, dataset shares, product URL counts,
  rating summaries, review lengths, and missing-brand rates.
- `brand_summary.csv` and `brand_by_platform.csv`: all brand groups, including
  missing brands. Case and whitespace variants are combined, without guessing
  brands from product names.
- `rating_by_platform.csv`: counts for each star rating by platform.
- `length_overall.csv` and `length_by_platform.csv`: character and token counts,
  means, standard deviations, extrema, quartiles, and P90/P95/P99.
- `monthly_reviews.csv`: review counts by month and platform.
- `unparsed_dates.csv`: date strings that could not be assigned to a month,
  counted by platform. Relative dates such as "3 weeks ago" require a crawl date.
- `missing_values.csv` and `missing_pct_by_platform.csv`: missing source fields.
- `overview.csv`: dataset size and data-quality checks, including exact duplicate
  rows, repeated comments, invalid ratings/dates, and stored-length mismatches.
- Six PNG charts and `report.html`: the visual overview and interpretation notes.

Character lengths are recalculated for every platform and include spaces and
punctuation. Tokens are whitespace-separated units, not Vietnamese linguistic
words. No reviews are removed. Blank brands stay unknown. Product URL counts do
not match products across platforms. Tiki's crawler uses rating quotas, so these
results describe the collected dataset rather than a representative market sample.
