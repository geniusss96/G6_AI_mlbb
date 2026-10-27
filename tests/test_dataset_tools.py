import contextlib
import io
import tempfile
from pathlib import Path
import unittest
from PIL import Image
import yaml
from training.dataset_tools import audit, prepare, read_labels
from training.train_v4 import main as train


class TestDatasetTools(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'raw'
        self.source.mkdir()
        self.spec = self.source / 'data.yaml'
        self.spec.write_text(yaml.safe_dump({'path': 'Z:/old/computer', 'train': 'train/images',
            'val': 'valid/images', 'test': 'test/images', 'nc': 2, 'names': ['hp_self', 'hp_enemy']}))
        self.sample('train', 'train', (10, 20, 30))
        self.sample('valid', 'val', (40, 50, 60))
        self.sample('test', 'test', (70, 80, 90))

    def sample(self, split, name, color, label='0 0.5 0.5 0.2 0.2\n', suffix='.png'):
        base = self.source / split
        (base / 'images').mkdir(parents=True, exist_ok=True)
        (base / 'labels').mkdir(exist_ok=True)
        Image.new('RGB', (16, 16), color).save(base / 'images' / (name + suffix))
        (base / 'labels' / (name + '.txt')).write_text(label)

    def test_relocation_preserves_ids_and_raw_files(self):
        before = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        report, _ = audit(self.spec)
        self.assertEqual(len(report['path_warnings']), 3)
        output = self.root / 'clean'
        result = prepare(self.spec, output)
        self.assertEqual(result['prepared_images'], {'test': 1, 'val': 1, 'train': 1})
        clean, _ = audit(output / 'data.yaml')
        self.assertEqual(clean['names'], ['hp_self', 'hp_enemy'])
        self.assertEqual(clean['issues'], [])
        self.assertEqual(clean['path_warnings'], [])
        after = {p.relative_to(self.source): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        self.assertEqual(before, after)
        with self.assertRaises(ValueError):
            prepare(self.spec, output)

    def test_pixel_duplicates_preserve_holdout_split(self):
        # Same decoded pixels, different file format and filename.
        self.sample('train', 'leak', (70, 80, 90), suffix='.bmp')
        report, _ = audit(self.spec)
        self.assertEqual(report['cross_split_duplicate_groups'], 1)
        result = prepare(self.spec, self.root / 'clean')
        self.assertEqual(result['excluded_images'], 1)
        self.assertEqual(result['prepared_images']['test'], 1)
        self.assertEqual(result['prepared_images']['train'], 1)

    def test_conflicting_duplicate_annotations_exclude_entire_group(self):
        self.sample('train', 'a', (100, 110, 120))
        self.sample('valid', 'b', (100, 110, 120), '1 0.5 0.5 0.2 0.2\n')
        result = prepare(self.spec, self.root / 'clean')
        self.assertEqual(len(result['conflicting_label_groups']), 1)
        self.assertEqual(result['excluded_images'], 2)

    def test_invalid_sample_is_not_partially_kept(self):
        self.sample('train', 'bad', (1, 2, 3), '0 0.5 0.5 0.2 0.2\n9 0.5 0.5 0.2 0.2\n')
        self.sample('train', 'missing', (4, 5, 6))
        (self.source / 'train/labels/missing.txt').unlink()
        (self.source / 'train/labels/orphan.txt').write_text('')
        report = prepare(self.spec, self.root / 'clean')
        self.assertEqual(len(report['issues']), 3)
        self.assertEqual(report['excluded_images'], 2)
        self.assertEqual(report['prepared_images']['train'], 1)

    def test_label_validation_and_explicit_background(self):
        label = self.root / 'label.txt'
        for text in ('0 nan 0.5 0.2 0.2', '0 0.5 0.5 0 0.2', '0 0.1 0.1 1 1', '0 0.5 0.5', '0.5 0.5 0.5 0.2 0.2'):
            with self.subTest(text=text):
                label.write_text(text)
                with self.assertRaises(ValueError):
                    read_labels(label, 2)
        label.write_text('')
        self.assertEqual(read_labels(label, 2), [])

    def test_training_dry_run_requires_clean_data_and_no_model(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                train(['--data', str(self.spec), '--dry-run'])
            output = self.root / 'clean'
            prepare(self.spec, output)
            self.assertEqual(train(['--data', str(output / 'data.yaml'), '--dry-run']), 0)
