"""Vẽ phân phối độ dài cleaned_comment của bộ unique text."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / 'data' / 'unique_texts' / 'unique_texts.csv'
DEFAULT_OUTPUT = HERE / 'analysis_output' / 'unique_texts'


def summarize(lengths: pd.Series) -> dict:
    return {
        'count': int(lengths.size), 'min': int(lengths.min()),
        'mean': float(lengths.mean()), 'median': float(lengths.median()),
        'p90': float(lengths.quantile(.90)), 'p95': float(lengths.quantile(.95)),
        'p99': float(lengths.quantile(.99)), 'max': int(lengths.max()),
    }


def plot_distribution(input_path: Path, output_dir: Path, column: str = 'cleaned_comment'):
    df = pd.read_csv(input_path, encoding='utf-8-sig', dtype=str, keep_default_na=False)
    if column not in df:
        raise ValueError(f'Không có cột {column!r}. Các cột hiện có: {list(df.columns)}')
    if df.empty:
        raise ValueError('CSV không có dòng dữ liệu.')
    text = df[column].str.strip()
    tokens = text.str.split().str.len()
    characters = text.str.len()
    summary = {
        'input': str(input_path.resolve()), 'column': column,
        'unique_text_count': len(df), 'empty_text_count': int(text.eq('').sum()),
        'weighting': 'Each CSV row counted once; duplicate_count is not used.',
        'token_definition': 'Whitespace-delimited tokens; underscores stay inside tokens.',
        'character_definition': 'Unicode code points including spaces, after trimming.',
        'tokens': summarize(tokens), 'characters': summarize(characters),
    }

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 11,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.titleweight': 'bold', 'axes.labelcolor': '#334155',
        'xtick.color': '#475569', 'ytick.color': '#475569',
        'figure.facecolor': '#ffffff', 'axes.facecolor': '#ffffff',
        'svg.fonttype': 'none',
    })
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.8))
    fig.subplots_adjust(left=.065, right=.975, bottom=.23, top=.77, wspace=.20)
    fig.text(.065, .94, 'PHÂN PHỐI ĐỘ DÀI COMMENT', fontsize=21,
             fontweight='bold', color='#0f172a')
    fig.text(.065, .88,
             f'{len(df):,} text duy nhất  |  Cột {column}  |  Mỗi text được tính một lần',
             fontsize=12, color='#475569')

    for ax, values, key, title, width, color in (
        (axes[0], tokens, 'tokens', 'Độ dài theo token', 5, '#2563eb'),
        (axes[1], characters, 'characters', 'Độ dài theo ký tự', 25, '#0d9488'),
    ):
        stats = summary[key]
        stop = max(width, int(np.ceil(values.max() / width)) * width)
        bins = np.arange(0, stop + width, width)
        heights, _, _ = ax.hist(values, bins=bins, color=color, edgecolor='white',
                                linewidth=.6, alpha=.88, zorder=3)
        if int(heights.sum()) != len(df):
            raise AssertionError('Histogram phải chứa toàn bộ các dòng CSV.')
        for metric, label, line_color, style in (
            ('median', 'Trung vị', '#0f172a', '--'),
            ('p95', 'P95', '#ea580c', '-.'),
        ):
            ax.axvline(stats[metric], color=line_color, linestyle=style,
                       linewidth=1.7, zorder=4,
                       label=f'{label}: {stats[metric]:g}')
        ax.set(title=title, xlabel='Số token' if key == 'tokens' else 'Số ký tự',
               ylabel='Số text', xlim=(0, stop))
        ax.set_ylim(0, max(heights) * 1.16)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{int(x):,}'))
        ax.grid(axis='y', color='#e2e8f0', linewidth=.8, zorder=0)
        ax.legend(loc='upper right', frameon=False, fontsize=10)
        ax.text(.97, .72,
                f'Trung bình: {stats["mean"]:.1f}\n'
                f'P99: {stats["p99"]:.1f}\n'
                f'Nhỏ nhất: {stats["min"]}\nLớn nhất: {stats["max"]}',
                transform=ax.transAxes, ha='right', va='top', fontsize=10,
                color='#475569', linespacing=1.7)

    fig.text(.065, .12,
             'Token được đếm bằng khoảng trắng: “làm_sạch” được tính là 1 token. '
             'Đây không phải số subword token của PhoBERT.',
             fontsize=10, color='#475569')
    fig.text(.065, .075,
             'Hiển thị toàn bộ dải độ dài, không cắt đuôi dài. '
             'Ký tự bao gồm khoảng trắng và dấu gạch dưới; không dùng trọng số duplicate_count.',
             fontsize=10, color='#475569')
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / 'cleaned_comment_length_distribution.png'
    fig.savefig(png, dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    (output_dir / 'comment_length_summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    return png, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--column', default='cleaned_comment')
    args = parser.parse_args()
    png, summary = plot_distribution(args.input, args.output_dir, args.column)
    print(json.dumps({'chart': str(png.resolve()), 'summary': summary}, ensure_ascii=False))


if __name__ == '__main__':
    main()
