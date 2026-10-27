"""Audit YOLO detection datasets and prepare a clean, split-preserving copy.

Raw files and class IDs are never modified. No guessed mappings between datasets.
Run: python -m training.dataset_tools --help
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

from PIL import Image, ImageOps
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / 'datasets/raw/MLBB_Combined_v4/data.yaml'
DEFAULT_OUTPUT = ROOT / 'datasets/prepared/v4_clean'
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
SPLITS = ('train', 'val', 'test')


def read_spec(yaml_path):
    yaml_path = Path(yaml_path).resolve()
    spec = yaml.safe_load(yaml_path.read_text(encoding='utf-8-sig'))
    names = spec['names']
    if isinstance(names, dict):
        if set(names) != set(range(len(names))):
            raise ValueError('Class IDs in names must be contiguous integers from zero')
        names = [names[i] for i in range(len(names))]
    if not names or len(set(names)) != len(names):
        raise ValueError('Class names must be nonempty and unique')
    if spec.get('nc', len(names)) != len(names):
        raise ValueError('nc does not match names')
    return yaml_path, spec, names


def split_path(yaml_path, spec, split):
    """Use declared paths when valid; recover relocated standard YOLO exports."""
    value = spec.get(split)
    if not value:
        return None, None
    if not isinstance(value, str):
        raise ValueError('Only directory-based detection datasets are supported')
    base = Path(spec.get('path', '.'))
    if not base.is_absolute():
        base = yaml_path.parent / base
    declared = (base / value).resolve()
    if declared.is_dir():
        return declared, None
    local = yaml_path.parent / ('valid' if split == 'val' else split) / 'images'
    if local.is_dir():
        return local, f'{split}: declared path unavailable; using local {local.parent.name}/images'
    raise ValueError(f'{split}: image directory not found for {yaml_path.name}')


def read_labels(path, nc):
    """An invalid box invalidates the whole sample, never just its annotation."""
    rows = []
    for number, line in enumerate(path.read_text(encoding='utf-8-sig').splitlines(), 1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f'line {number}: expected class cx cy width height')
        cid = int(parts[0])
        box = tuple(float(x) for x in parts[1:])
        if not 0 <= cid < nc:
            raise ValueError(f'line {number}: class ID {cid} outside 0..{nc-1}')
        if not all(math.isfinite(x) for x in box):
            raise ValueError(f'line {number}: nonfinite box')
        x, y, w, h = box
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            raise ValueError(f'line {number}: box must be normalized with positive size')
        # Roboflow exports may have harmless decimal rounding at the border.
        if min(x-w/2, y-h/2) < -1e-4 or max(x+w/2, y+h/2) > 1.0001:
            raise ValueError(f'line {number}: box extends outside image')
        rows.append((cid, *box))
    return rows


def audit(yaml_path):
    yaml_path, spec, names = read_spec(yaml_path)
    records, issues, warnings = [], [], []
    counts = {s: Counter() for s in SPLITS}
    images = {}
    groups = defaultdict(list)
    for split in SPLITS:
        directory, warning = split_path(yaml_path, spec, split)
        if warning:
            warnings.append(warning)
        if directory is None:
            images[split] = 0
            continue
        paths = sorted(p for p in directory.rglob('*') if p.suffix.lower() in IMAGE_SUFFIXES)
        images[split] = len(paths)
        if not paths:
            issues.append({'split': split, 'error': 'empty image directory'})
        expected_labels = set()
        for path in paths:
            label = directory.parent / 'labels' / path.relative_to(directory).with_suffix('.txt')
            expected_labels.add(label)
            record = {'split': split, 'image': path, 'label': label, 'rows': [], 'valid': False}
            records.append(record)
            try:
                if not label.is_file():
                    raise ValueError('missing label; background images require an empty label file')
                rows = read_labels(label, len(names))
                with Image.open(path) as image:
                    image.load()
                    rgb = ImageOps.exif_transpose(image).convert('RGB')
                    digest = hashlib.sha256(f'{rgb.size}'.encode() + rgb.tobytes()).hexdigest()
                record.update(rows=rows, valid=True, digest=digest)
                counts[split].update(names[row[0]] for row in rows)
                groups[digest].append(record)
            except (OSError, ValueError) as error:
                issues.append({'split': split, 'image': path.name, 'error': str(error)})
        for label in sorted((directory.parent / 'labels').rglob('*.txt')):
            if label not in expected_labels:
                issues.append({'split': split, 'label': label.name, 'error': 'orphan label'})
    duplicate_groups, conflicting_groups = [], []
    for group in groups.values():
        if len(group) < 2:
            continue
        summary = [{'split': r['split'], 'image': r['image'].name} for r in group]
        duplicate_groups.append(summary)
        signatures = {tuple(sorted(r['rows'])) for r in group}
        if len(signatures) > 1:
            conflicting_groups.append(summary)
            for record in group:
                record['valid'] = False
                record['conflict'] = True
    report = {
        'dataset': yaml_path.parent.name, 'names': names, 'images': images,
        'boxes_by_class': {s: dict(counts[s]) for s in SPLITS},
        'issues': issues, 'path_warnings': warnings,
        'duplicate_groups': duplicate_groups,
        'cross_split_duplicate_groups': sum(len({r['split'] for r in g}) > 1 for g in duplicate_groups),
        'conflicting_label_groups': conflicting_groups,
    }
    return report, records


def prepare(yaml_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'Output already exists: {output}; choose a new directory')
    report, records = audit(yaml_path)
    # Preserve held-out evaluation images first; never move them into training.
    order = {'test': 0, 'val': 1, 'train': 2}
    selected, seen = [], set()
    for record in sorted(records, key=lambda r: (order[r['split']], str(r['image']))):
        if record['valid'] and record['digest'] not in seen:
            selected.append(record)
            seen.add(record['digest'])
    split_counts = Counter(r['split'] for r in selected)
    if not split_counts['train'] or not split_counts['val']:
        raise ValueError('No usable train/val split remains; inspect audit before training')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=output.name + '-', dir=output.parent))
    try:
        manifest = []
        for split in SPLITS:
            (staging / split / 'images').mkdir(parents=True)
            (staging / split / 'labels').mkdir()
        for index, record in enumerate(selected):
            split = record['split']
            stem = f'{index:06d}_{record["digest"][:12]}'
            destination = staging / split / 'images' / (stem + record['image'].suffix.lower())
            shutil.copy2(record['image'], destination)
            label = staging / split / 'labels' / (stem + '.txt')
            label.write_text(''.join(' '.join(map(str, row)) + '\n' for row in record['rows']), encoding='utf-8')
            manifest.append({'split': split, 'source_image': record['image'].name,
                             'output_image': destination.name, 'pixel_sha256': record['digest']})
        spec = {'path': str(output), 'train': 'train/images', 'val': 'val/images',
                'nc': len(report['names']), 'names': report['names']}
        if split_counts['test']:
            spec['test'] = 'test/images'
        (staging / 'data.yaml').write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding='utf-8')
        report['prepared_images'] = dict(split_counts)
        report['excluded_images'] = len(records) - len(selected)
        report['preparation_policy'] = 'Reject invalid/conflicting samples; pixel deduplicate; test > val > train; preserve class IDs'
        (staging / 'audit.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        (staging / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        staging.rename(output)
    except Exception:
        shutil.rmtree(staging)
        raise
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'prepare'])
    parser.add_argument('--data', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='New directory for prepare')
    parser.add_argument('--report', type=Path, help='Optional JSON audit report')
    args = parser.parse_args(argv)
    try:
        report = prepare(args.data, args.output) if args.command == 'prepare' else audit(args.data)[0]
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
        print(json.dumps({k: v for k, v in report.items() if k not in ('duplicate_groups', 'conflicting_label_groups', 'boxes_by_class', 'issues')}, indent=2, ensure_ascii=False))
        print(f"Issues: {len(report['issues'])}; conflicting groups: {len(report['conflicting_label_groups'])}")
        if args.command == 'audit':
            return 2 if report['issues'] or report['duplicate_groups'] else 0
        print(f'Prepared: {args.output.resolve() / "data.yaml"}')
        return 0
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError) as error:
        parser.exit(1, f'Error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
