#!/usr/bin/env python3
"""CUAS hard-negative ingestion, preprocessing, metadata splitting.
Run from project root: python acoustic/prepare_v4.py {urban,audioset,manifest}
"""
import argparse
import csv
import hashlib
import os
import random
import shutil
import subprocess
import tempfile
import wave
from collections import Counter, defaultdict
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / 'data' / 'raw'
CACHE = ROOT / 'data' / 'cache' / 'hard_negatives_16k'
OUT = ROOT / 'data' / 'v4_hard_negatives.csv'
FIELDS = ['file_path', 'label', 'source', 'category', 'group_id', 'split', 'duration_sec', 'notes']
BASE_URL = 'https://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/'
KEYWORDS = {
    'hvac_fan': ('air conditioning', 'air conditioner', 'fan', 'ventilation'),
    'vehicle': ('car', 'truck', 'motorcycle', 'traffic noise', 'bus', 'vehicle'),
    'engine': ('engine', 'lawn mower', 'chainsaw', 'leaf blower', 'generator'),
    'industrial': ('power tool', 'drill', 'jackhammer', 'mechanical', 'machinery'),
    'aircraft': ('aircraft', 'helicopter', 'airplane'),
    'environment': ('wind', 'rain', 'bird', 'insect'),
}
EXCLUDED = ('drone', 'quadcopter', 'unmanned aerial')


def check_wav(path):
    try:
        with wave.open(str(path), 'rb') as wf:
            if (wf.getframerate(), wf.getnchannels(), wf.getsampwidth(), wf.getcomptype()) != (16000, 1, 2, 'NONE'):
                return 0.0
            return wf.getnframes() / 16000
    except (OSError, EOFError, wave.Error):
        return 0.0


def convert(source, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix('.partial.wav')
    try:
        subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y',
                        '-i', str(source), '-vn', '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(tmp)],
                       check=True, timeout=120)
        if check_wav(tmp) < 0.5:
            return False
        tmp.replace(dest)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False
    finally:
        tmp.unlink(missing_ok=True)


def urban(_args):
    archive = RAW / 'UrbanSound8K' / 'UrbanSound8K.tar.gz'
    if not archive.exists():
        raise SystemExit(f'Missing {archive}. Download with curl command in README.')
    root = RAW / 'UrbanSound8K' / 'UrbanSound8K'
    metadata = root / 'metadata' / 'UrbanSound8K.csv'
    if not metadata.exists():
        print('Extracting official archive...', flush=True)
        subprocess.run(['tar', '-xzf', str(archive), '-C', str(archive.parent)], check=True)
    assert metadata.exists(), f'Missing metadata: {metadata}'
    counts = Counter()
    entries = list(csv.DictReader(metadata.open(newline='', encoding='utf-8-sig')))
    for i, r in enumerate(entries, 1):
        src = root / 'audio' / f"fold{r['fold']}" / r['slice_file_name']
        dst = CACHE / 'urbansound' / f"fold{r['fold']}" / r['slice_file_name']
        if not src.exists():
            counts['missing'] += 1
            continue
        if check_wav(dst) < 0.5:
            if not convert(src, dst):
                counts['failed_or_short'] += 1
                continue
        counts['ready'] += 1
        if i % 1000 == 0:
            print('Urban:', i, '/', len(entries), dict(counts), flush=True)
    print('UrbanSound result:', dict(counts), flush=True)


def download_metadata():
    md = RAW / 'AudioSet' / 'metadata'
    md.mkdir(parents=True, exist_ok=True)
    for name in ('class_labels_indices.csv', 'balanced_train_segments.csv', 'unbalanced_train_segments.csv'):
        p = md / name
        if not p.exists():
            print('Getting', name, flush=True)
            urlretrieve(BASE_URL + name, p)
    return md


def audioset(args):
    md = download_metadata()
    names = {r['mid']: r['display_name'].lower() for r in csv.DictReader((md / 'class_labels_indices.csv').open())}
    mapping = defaultdict(list)
    for mid, name in names.items():
        for cat, phrases in KEYWORDS.items():
            if any(p == name or p in name for p in phrases):
                mapping[cat].append((mid, name))
    print('Selected AudioSet class mappings:')
    for cat, hits in mapping.items():
        print(cat, ':', ', '.join(f'{name} [{mid}]' for mid, name in hits))
    candidates = defaultdict(dict)
    for filename in ('balanced_train_segments.csv', 'unbalanced_train_segments.csv'):
        with (md / filename).open(newline='') as f:
            for line in f:
                if line.startswith('#'):
                    continue
                row = next(csv.reader([line], skipinitialspace=True))
                if len(row) != 4:
                    continue
                vid, start, end, mids_str = row
                labels = mids_str.split(',')
                if any(any(k in names.get(mid, '') for k in EXCLUDED) for mid in labels):
                    continue
                found = [cat for cat, pairs in mapping.items()
                         if any(mid in {p[0] for p in pairs} for mid in labels)]
                if not found:
                    continue
                cat = found[0]
                key = f'{vid}_{int(round(float(start)*1000))}'
                candidates[cat][key] = (vid, float(start), float(end), cat, key)
    rng = random.Random(42)
    for cat in candidates:
        values = list(candidates[cat].values())
        rng.shuffle(values)
        candidates[cat] = values
    print('Candidates:', {c: len(v) for c, v in candidates.items()}, flush=True)
    categories = sorted(candidates)
    if not categories:
        raise SystemExit('No AudioSet categories matched. Check mappings.')
    success = Counter()
    errors = Counter()
    # Interleave category selection. Skip already cached clips, so repeated runs extend the dataset.
    indexes = {c: 0 for c in categories}
    attempted = 0
    while sum(success.values()) < args.target and attempted < args.max_attempts:
        pending = [c for c in categories if indexes[c] < len(candidates[c])]
        if not pending:
            break
        cat = min(pending, key=lambda c: (success[c], indexes[c]))
        vid, start, end, _, key = candidates[cat][indexes[cat]]
        indexes[cat] += 1
        dst = CACHE / 'audioset' / cat / f'{key}.wav'
        if check_wav(dst) >= 0.5:
            success[cat] += 1
            continue
        attempted += 1
        with tempfile.TemporaryDirectory(prefix='audioset_') as temp:
            template = str(Path(temp) / '%(id)s.%(ext)s')
            cmd = ['yt-dlp', '--no-playlist', '--no-progress', '--quiet',
                   '--download-sections', f'*{start:.3f}-{end:.3f}',
                   '--force-keyframes-at-cuts', '-f', 'bestaudio/best',
                   '--socket-timeout', '15', '--retries', '1', '-o', template,
                   f'https://www.youtube.com/watch?v={vid}']
            try:
                proc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=90)
                outputs = [p for p in Path(temp).iterdir() if p.is_file() and p.stat().st_size > 0]
                if proc.returncode == 0 and outputs and convert(outputs[0], dst):
                    success[cat] += 1
                else:
                    errors[cat] += 1
            except subprocess.TimeoutExpired:
                errors[cat] += 1
        if attempted % 50 == 0:
            print(f'AudioSet attempts={attempted} saved={sum(success.values())} failed={sum(errors.values())}', flush=True)
    print('AudioSet successful (includes already cached):', dict(success))
    print('AudioSet failures:', dict(errors))
    print('Ready WAV count:', len(list((CACHE / 'audioset').glob('*/*.wav'))))


def deterministic_split(key, train=0.9):
    num = int(hashlib.sha256(key.encode()).hexdigest()[:12], 16) / 16**12
    return 'train' if num < train else 'val'


def manifest(_args):
    result = []
    urbanroot = RAW / 'UrbanSound8K' / 'UrbanSound8K'
    csvfile = urbanroot / 'metadata' / 'UrbanSound8K.csv'
    if csvfile.exists():
        urban_rows = list(csv.DictReader(
            csvfile.open(newline='', encoding='utf-8-sig')
        ))
        fsid_splits = defaultdict(set)
        for r in urban_rows:
            fold = int(r['fold'])
            split = 'train' if fold <= 8 else ('val' if fold == 9 else 'test')
            fsid_splits[r['fsID']].add(split)

        conflicts = {
            fsid for fsid, splits in fsid_splits.items()
            if len(splits) > 1
        }
        print('Excluded cross-split UrbanSound fsIDs:', sorted(conflicts))

        for r in urban_rows:
            if r['fsID'] in conflicts:
                continue
            wav = CACHE / 'urbansound' / f"fold{r['fold']}" / r['slice_file_name']
            duration = check_wav(wav)
            if duration < 0.5:
                continue
            fold = int(r['fold'])
            split = 'train' if fold <= 8 else ('val' if fold == 9 else 'test')
            result.append(dict(file_path=str(wav.relative_to(ROOT)), label=0, source='urbansound',
                               category=r['class'], group_id=f"urban_fsid_{r['fsID']}",
                               split=split, duration_sec=f'{duration:.4f}', notes=f'official_fold={fold}'))
    # AudioSet file names encode their video id and start time; avoid overlapping video across splits.
    for wav in sorted((CACHE / 'audioset').glob('*/*.wav')):
        duration = check_wav(wav)
        if duration < 0.5:
            continue
        vid = wav.stem.rsplit('_', 1)[0]
        result.append(dict(file_path=str(wav.relative_to(ROOT)), label=0, source='audioset',
                           category=wav.parent.name, group_id=f'audioset_{vid}',
                           split=deterministic_split(vid), duration_sec=f'{duration:.4f}',
                           notes='weak AudioSet segment label; manual audit recommended'))
    if not result:
        raise SystemExit('No usable hard-negative WAV files. Run urban and/or audioset first.')
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(sorted(result, key=lambda r: (r['source'], r['split'], r['group_id'], r['file_path'])))
    groups = defaultdict(set)
    for r in result:
        groups[r['group_id']].add(r['split'])
    overlap = {k:v for k,v in groups.items() if len(v) > 1}
    if overlap:
        raise SystemExit(f'LEAKAGE: {list(overlap.items())[:10]}')
    print('Hard-negative manifest:', OUT)
    print('Counts:', dict(Counter((r['source'], r['split']) for r in result)))
    print('Categories:', dict(Counter(r['category'] for r in result)))
    print('PASS: source-group-disjoint hard-negative splits')


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('urban')
    audio = sub.add_parser('audioset')
    audio.add_argument('--target', type=int, default=3000, help='target usable clips from this run/cached set')
    audio.add_argument('--max-attempts', type=int, default=10000)
    sub.add_parser('manifest')
    args = p.parse_args()
    {'urban': urban, 'audioset': audioset, 'manifest': manifest}[args.command](args)


if __name__ == '__main__':
    main()
