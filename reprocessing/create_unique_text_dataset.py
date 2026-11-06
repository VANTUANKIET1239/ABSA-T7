"""Tạo bộ text duy nhất và ánh xạ về review nguồn, giữ nguyên CSV gốc."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
import unicodedata
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_DIR / 'crawl-data/data/combined_reviews_brand_filled_language_filtered.csv'
DEFAULT_OUTPUT = PROJECT_DIR / 'crawl-data/data/unique_texts'
UNIQUE_COLUMNS = [
    'text_group_id', 'cleaned_comment', 'duplicate_count',
    'representative_source_row', 'representative_comment',
]
MAPPING_COLUMNS = ['source_row', 'text_group_id', 'text_status']


def normalize_text_key(text: str) -> str:
    """Giữ nguyên nội dung từ, dấu tiếng Việt, chữ số, phủ định và _ ghép từ."""
    return ' '.join(unicodedata.normalize('NFC', text).lower().split())


def text_group_id(text: str) -> str:
    return 'txt_' + hashlib.sha256(text.encode('utf-8')).hexdigest()


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not columns or any(not name for name in columns) or len(set(columns)) != len(columns):
        raise ValueError('Tên cột CSV phải khác rỗng và không trùng nhau.')
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError('CSV có dòng không khớp số cột.')
    return columns, rows


def write_csv(path: Path, columns: list[str], rows: list[dict]):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def create_unique_text_dataset(input_file: str | Path, output_dir: str | Path,
                               text_column: str = 'cleaned_comment') -> dict:
    source = Path(input_file).expanduser().resolve()
    out = Path(output_dir).expanduser().resolve()
    names = ['unique_texts.csv', 'review_text_mapping.csv', 'README.md', 'deduplication_summary.json']
    if source in [out / name for name in names]:
        raise ValueError('Không được ghi kết quả đè lên CSV nguồn.')
    original_bytes = source.read_bytes()
    columns, source_rows = read_csv(source)
    if text_column not in columns:
        raise ValueError(f'Không có cột text nguồn: {text_column!r}')
    if set(columns) & set(MAPPING_COLUMNS):
        raise ValueError('CSV nguồn đã có cột source_row/text_group_id/text_status; cần tránh trùng tên.')

    groups = {}
    mapping = []
    for source_row, row in enumerate(source_rows, start=1):
        text = normalize_text_key(row[text_column])
        group_id = text_group_id(text) if text else ''
        if text:
            if text not in groups:
                groups[text] = {
                    'text_group_id': group_id, 'cleaned_comment': text, 'duplicate_count': 0,
                    'representative_source_row': source_row,
                    'representative_comment': row.get('comment', ''),
                }
            groups[text]['duplicate_count'] += 1
        mapping.append({
            'source_row': source_row, 'text_group_id': group_id,
            'text_status': 'mapped' if text else 'empty_text', **row,
        })

    unique_rows = list(groups.values())
    counts = Counter(row['text_group_id'] for row in mapping if row['text_group_id'])
    if len(counts) != len(unique_rows):
        raise RuntimeError('ID nhóm text bị trùng ngoài dự kiến.')

    summary = {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': str(source), 'source_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'text_column': text_column,
        'normalization': ['Unicode NFC', 'lowercase', 'collapse whitespace', 'strip whitespace'],
        'representative_rule': 'First occurrence in source order; its comment is an example only.',
        'source_row_definition': '1-based data-record index, excluding header; not a physical CSV line number.',
        'input_rows': len(source_rows),
        'nonempty_text_rows': sum(counts.values()),
        'empty_text_rows': len(source_rows) - sum(counts.values()),
        'unique_texts': len(unique_rows),
        'duplicate_excess_rows': sum(counts.values()) - len(unique_rows),
        'repeated_text_groups': sum(count > 1 for count in counts.values()),
        'output_dir': str(out),
    }
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.unique_texts_', dir=out) as directory:
        stage = Path(directory)
        write_csv(stage / names[0], UNIQUE_COLUMNS, unique_rows)
        write_csv(stage / names[1], [*MAPPING_COLUMNS, *columns], mapping)
        saved_columns, saved_unique = read_csv(stage / names[0])
        mapping_columns, saved_mapping = read_csv(stage / names[1])
        if saved_columns != UNIQUE_COLUMNS or mapping_columns != [*MAPPING_COLUMNS, *columns]:
            raise RuntimeError('Sai tên hoặc thứ tự cột sau khi ghi CSV.')
        if len(saved_mapping) != len(source_rows) or len(saved_unique) != len(unique_rows):
            raise RuntimeError('Sai số dòng sau khi ghi CSV.')
        for original, saved, row_number in zip(source_rows, saved_mapping, range(1, len(source_rows) + 1)):
            if any(saved[column] != original[column] for column in columns):
                raise RuntimeError('Metadata hoặc text gốc trong bảng ánh xạ bị thay đổi.')
            text = normalize_text_key(original[text_column])
            expected_id = text_group_id(text) if text else ''
            if (saved['source_row'] != str(row_number) or saved['text_group_id'] != expected_id
                    or saved['text_status'] != ('mapped' if text else 'empty_text')):
                raise RuntimeError('Ánh xạ về dòng nguồn không chính xác.')
        expected_unique = [{key: str(value) for key, value in row.items()} for row in unique_rows]
        if saved_unique != expected_unique:
            raise RuntimeError('Text đại diện hoặc số lần xuất hiện không khớp sau khi ghi CSV.')
        if len({row['cleaned_comment'] for row in saved_unique}) != len(saved_unique):
            raise RuntimeError('Bộ text duy nhất vẫn có nội dung trùng.')
        if any(int(row['duplicate_count']) != counts[row['text_group_id']] for row in saved_unique):
            raise RuntimeError('duplicate_count không khớp bảng ánh xạ.')
        if source.read_bytes() != original_bytes:
            raise RuntimeError('CSV nguồn vừa thay đổi; dừng xuất kết quả.')

        summary['output_sha256'] = {
            name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in names[:2]
        }
        (stage / names[2]).write_text(
            '# Bộ text duy nhất\n\n'
            f'Nguồn: `{source.name}`, cột `{text_column}`.\n\n'
            f'- Review nguồn: {summary["input_rows"]:,}.\n'
            f'- Text khác rỗng: {summary["nonempty_text_rows"]:,}.\n'
            f'- Nội dung duy nhất: {summary["unique_texts"]:,}.\n'
            f'- Bản lặp dư: {summary["duplicate_excess_rows"]:,}.\n'
            f'- Text rỗng: {summary["empty_text_rows"]:,}, giữ trong bảng ánh xạ với trạng thái `empty_text`.\n\n'
            '## Cách tạo nhóm\n\n'
            'Chuẩn hóa Unicode NFC, chữ thường và khoảng trắng trên cột nguồn. '
            'Giữ nguyên dấu tiếng Việt, phủ định, chữ số và dấu `_`. '
            'Không gộp theo độ giống ngữ nghĩa. Không lọc câu ngắn/khen chung, '
            'không gán nhãn aspect/sentiment và không chia train/validation/test.\n\n'
            '## Các file\n\n'
            '- [unique_texts.csv](unique_texts.csv): mỗi nội dung có một dòng. '
            '`text_group_id` là SHA-256 của text chuẩn hóa, có tiền tố `txt_`. '
            '`duplicate_count` tính cả bản đại diện. `cleaned_comment` chứa text chuẩn hóa dùng làm khóa nhóm.\n'
            '- `representative_source_row` là vị trí bản ghi dữ liệu đầu tiên của nhóm, tính từ 1, '
            'không tính header; không phải số dòng vật lý vì một review có thể chứa xuống dòng. '
            '`representative_comment` là ví dụ comment gốc của bản ghi đó.\n'
            '- [review_text_mapping.csv](review_text_mapping.csv): giữ mọi bản ghi, thứ tự, '
            'giá trị nguồn và metadata; thêm `source_row`, `text_group_id`, `text_status`. '
            'Dùng `text_group_id` để nối kết quả trên bộ unique về toàn bộ review.\n'
            '- [deduplication_summary.json](deduplication_summary.json): thống kê, quy tắc '
            'và SHA-256 để xác định chính xác CSV nguồn/các CSV đầu ra.\n\n'
            'Một nhóm có thể chứa review từ nhiều thương hiệu, sàn hoặc rating. '
            'Metadata của một bản đại diện không được coi là metadata chung của nhóm; '
            'xem bảng ánh xạ khi phân tích theo các thuộc tính đó.\n\n'
            'Tạo lại từ thư mục gốc dự án:\n\n'
            '```powershell\npython reprocessing/create_unique_text_dataset.py\n```\n',
            encoding='utf-8',
        )
        (stage / names[3]).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
        # Copy sang out để kế thừa ACL; di chuyển từ thư mục tạm sẽ giữ quyền riêng trên Windows.
        # Xuất manifest cuối cùng để đối chiếu hash hai CSV.
        for name in names:
            publish_path = out / f'.{name}.{uuid.uuid4().hex}.tmp'
            try:
                shutil.copyfile(stage / name, publish_path)
                os.replace(publish_path, out / name)
            finally:
                publish_path.unlink(missing_ok=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Tạo bộ text duy nhất và ánh xạ review gốc')
    parser.add_argument('--input', default=str(DEFAULT_INPUT))
    parser.add_argument('--output-dir', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--text-column', default='cleaned_comment')
    args = parser.parse_args()
    summary = create_unique_text_dataset(args.input, args.output_dir, args.text_column)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
