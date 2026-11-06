import csv
import hashlib
import json
import os
import tempfile
import unicodedata
import unittest
from pathlib import Path

from create_unique_text_dataset import create_unique_text_dataset, normalize_text_key, read_csv


class UniqueTextDatasetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / 'source.csv'
        self.output = self.root / 'unique'
        self.columns = ['comment', 'cleaned_comment', 'review_id', 'brand', 'rating']

    def write_source(self, rows, columns=None):
        with self.source.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=columns or self.columns)
            writer.writeheader()
            writer.writerows(rows)

    def row(self, text, review_id='0001', **extra):
        return {'comment': 'Ví dụ gốc, có dấu câu!\nDòng thứ hai.', 'cleaned_comment': text,
                'review_id': review_id, 'brand': 'NA', 'rating': '5', **extra}

    def test_groups_unicode_case_whitespace_and_preserves_every_source_record(self):
        rows = [self.row('CỰC_KÌ  HÀI_LÒNG', '001'),
                self.row(unicodedata.normalize('NFD', ' cực_kì hài_lòng '), '002', brand='Other', rating='4'),
                self.row('không_kích_ứng', '003'), self.row('kích_ứng', '004'),
                self.row('ph 5', '005'), self.row('ph 6', '006'), self.row('ok', '007')]
        self.write_source(rows)
        original = self.source.read_bytes()
        summary = create_unique_text_dataset(self.source, self.output)
        _, unique = read_csv(self.output / 'unique_texts.csv')
        mapping_columns, mapping = read_csv(self.output / 'review_text_mapping.csv')
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(len(unique), 6)
        self.assertEqual(unique[0]['cleaned_comment'], 'cực_kì hài_lòng')
        self.assertEqual(unique[0]['duplicate_count'], '2')
        self.assertEqual(unique[0]['representative_source_row'], '1')
        self.assertEqual(unique[0]['representative_comment'], rows[0]['comment'])
        self.assertEqual(mapping[0]['text_group_id'], mapping[1]['text_group_id'])
        self.assertNotEqual(mapping[2]['text_group_id'], mapping[3]['text_group_id'])
        self.assertNotEqual(mapping[4]['text_group_id'], mapping[5]['text_group_id'])
        self.assertEqual(mapping_columns, ['source_row', 'text_group_id', 'text_status', *self.columns])
        self.assertEqual([{key: row[key] for key in self.columns} for row in mapping], rows)
        self.assertEqual([row['source_row'] for row in mapping], list(map(str, range(1, 8))))
        self.assertEqual(summary['input_rows'], 7)
        self.assertEqual(summary['duplicate_excess_rows'], 1)

    def test_empty_texts_are_traceable_without_becoming_a_unique_example(self):
        self.write_source([self.row(''), self.row(' \t\n'), self.row('hài_lòng')])
        summary = create_unique_text_dataset(self.source, self.output)
        _, unique = read_csv(self.output / 'unique_texts.csv')
        _, mapping = read_csv(self.output / 'review_text_mapping.csv')
        self.assertEqual(len(unique), 1)
        self.assertEqual(len(mapping), 3)
        self.assertEqual([row['text_status'] for row in mapping], ['empty_text', 'empty_text', 'mapped'])
        self.assertEqual([row['text_group_id'] for row in mapping[:2]], ['', ''])
        self.assertEqual(summary['empty_text_rows'], 2)
        self.assertEqual(sum(int(row['duplicate_count']) for row in unique), summary['nonempty_text_rows'])

    def test_group_ids_are_stable_when_source_order_changes(self):
        rows = [self.row('hài_lòng'), self.row('không_khô_căng')]
        self.write_source(rows)
        create_unique_text_dataset(self.source, self.output)
        _, first = read_csv(self.output / 'unique_texts.csv')
        self.write_source(list(reversed(rows)))
        create_unique_text_dataset(self.source, self.output)
        _, second = read_csv(self.output / 'unique_texts.csv')
        self.assertEqual({r['cleaned_comment']: r['text_group_id'] for r in first},
                         {r['cleaned_comment']: r['text_group_id'] for r in second})

    def test_manifest_hashes_match_saved_files_and_source(self):
        self.write_source([self.row('hài_lòng'), self.row('hài_lòng')])
        create_unique_text_dataset(self.source, self.output)
        manifest = json.loads((self.output / 'deduplication_summary.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['source_sha256'], hashlib.sha256(self.source.read_bytes()).hexdigest())
        for name, expected in manifest['output_sha256'].items():
            self.assertEqual(hashlib.sha256((self.output / name).read_bytes()).hexdigest(), expected)
        self.assertEqual(len(list(self.output.iterdir())), 4)

    @unittest.skipUnless(os.name == 'nt', 'Windows ACL regression')
    def test_published_files_inherit_access_instead_of_private_staging_acl(self):
        import ctypes
        from ctypes import wintypes

        self.write_source([self.row('hài_lòng')])
        create_unique_text_dataset(self.source, self.output)
        api = ctypes.WinDLL('advapi32', use_last_error=True)
        read_security = api.GetFileSecurityW
        read_security.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.LPVOID,
                                 wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)]
        read_security.restype = wintypes.BOOL
        read_control = api.GetSecurityDescriptorControl
        read_control.argtypes = [wintypes.LPVOID, ctypes.POINTER(wintypes.WORD),
                                ctypes.POINTER(wintypes.DWORD)]
        read_control.restype = wintypes.BOOL
        for path in self.output.iterdir():
            with self.subTest(file=path.name):
                needed = wintypes.DWORD()
                read_security(str(path), 4, None, 0, ctypes.byref(needed))  # DACL_SECURITY_INFORMATION
                descriptor = ctypes.create_string_buffer(needed.value)
                if not read_security(str(path), 4, descriptor, needed.value, ctypes.byref(needed)):
                    raise ctypes.WinError(ctypes.get_last_error())
                control, revision = wintypes.WORD(), wintypes.DWORD()
                if not read_control(descriptor, ctypes.byref(control), ctypes.byref(revision)):
                    raise ctypes.WinError(ctypes.get_last_error())
                self.assertFalse(control.value & 0x1000, 'File kept private ACL (SE_DACL_PROTECTED)')

    def test_missing_column_and_malformed_csv_fail_without_outputs(self):
        for content in ['comment\ntext\n', 'comment,cleaned_comment\ntext,cleaned,extra\n']:
            with self.subTest(content=content):
                self.source.write_text(content, encoding='utf-8')
                with self.assertRaises(ValueError):
                    create_unique_text_dataset(self.source, self.output)
                self.assertFalse(self.output.exists())

    def test_normalization_preserves_negation_accents_digits_and_word_joiners(self):
        self.assertEqual(normalize_text_key('  KHÔNG_KÍCH_ỨNG\n ph 5  '), 'không_kích_ứng ph 5')


if __name__ == '__main__':
    unittest.main()
