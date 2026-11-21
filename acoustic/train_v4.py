#!/usr/bin/env python3
"""Train full-DADS + v3 + AudioSet/UrbanSound v4 with group-heldout validation."""
import argparse
import csv
import random
import sys
import wave
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'acoustic'))
from preprocessing import load_manifest, create_window_index, DroneAudio
from dads_dataset import DADSAudio
from model import DroneDetectionModel
from training_augmentation import TrainingAugmentation


class HardNegativeAudio(Dataset):
    def __init__(self, records, training):
        self.records = records
        self.training = training

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        rec = self.records[index]
        path = ROOT / rec['file_path']
        with wave.open(str(path), 'rb') as wav:
            if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth(), wav.getcomptype()) != (16000, 1, 2, 'NONE'):
                raise ValueError(f'Unexpected audio format: {path}')
            samples = array('h')
            samples.frombytes(wav.readframes(wav.getnframes()))
            if sys.byteorder != 'little':
                samples.byteswap()
        x = torch.tensor(samples, dtype=torch.float32).div_(32768.0)
        if x.numel() > 8000:
            start = random.randrange(x.numel() - 8000 + 1) if self.training else (x.numel() - 8000) // 2
            x = x[start:start + 8000]
        elif x.numel() < 8000:
            x = torch.nn.functional.pad(x, (0, 8000 - x.numel()))
        return {'audio': x, 'label': torch.tensor(0, dtype=torch.long),
                'group_id': rec['group_id'], 'source': rec['source']}


def assemble():
    # Preserve the old dataset splits; no re-splitting v3 by window.
    windows = create_window_index(load_manifest())
    local_train = DroneAudio(windows, split='train')
    local_val = DroneAudio(windows, split='val')
    dads_train = DADSAudio(training=True)  # all 180,320 indexed records

    neg_path = ROOT / 'data' / 'v4_hard_negatives.csv'
    if not neg_path.exists():
        raise SystemExit('Missing data/v4_hard_negatives.csv: run prepare_v4.py manifest')
    with neg_path.open(newline='') as f:
        negatives = list(csv.DictReader(f))
    new_train = [r for r in negatives if r['split'] == 'train']
    new_val = [r for r in negatives if r['split'] == 'val']
    if not new_train or not new_val:
        raise SystemExit('Hard negatives must contain train AND val examples')
    assert all(r['label'] == '0' for r in negatives)
    assert not ({r['group_id'] for r in new_train} & {r['group_id'] for r in new_val})
    # Note Urban fold10 remains reserved for test and is excluded here.

    train = ConcatDataset([local_train, dads_train, HardNegativeAudio(new_train, True)])
    val = ConcatDataset([local_val, HardNegativeAudio(new_val, False)])

    # Raw source/category metadata for each indexed example.
    metadata = ([(int(w['label']), w['source'], w['source']) for w in local_train.windows]
                + [(int(r['label']), 'dads', 'dads') for r in dads_train.records]
                + [(0, r['source'], r['category']) for r in new_train])
    assert len(metadata) == len(train)
    print('Full DADS retained:', len(dads_train), '| v3 windows:', len(local_train),
          '| hard negative train:', len(new_train), '| hard negative val:', len(new_val))
    print('Validation:', len(val), '| Urban fold10 reserved for test')
    return train, val, metadata


def make_sampler(metadata, samples, seed):
    # 50/50 label, source-weighted within label, category-balanced within source.
    priors = {
        0: {'audioset': .35, 'urbansound': .25, 'dads': .20, 'lab': .10, 'librispeech': .10},
        1: {'dads': .50, 'lab': .30, 'mlsp': .20},
    }
    counts = Counter(metadata)
    by_src = defaultdict(set)
    for y, src, category in metadata:
        by_src[(y, src)].add(category)
    active = {}
    for y in (0, 1):
        sources = {src for label, src, cat in metadata if label == y}
        missing = sources - priors[y].keys()
        if missing:
            raise ValueError(
                f'Missing sampler weights for label {y}: {sorted(missing)}'
            )

        total = sum(priors[y][s] for s in sources)
        if total <= 0:
            raise ValueError(
                f'Invalid sampler weight total for label {y}'
            )

        active[y] = {s: priors[y][s] / total for s in sources}
    weights = []
    for y, src, cat in metadata:
        w = .5 * active[y][src] / len(by_src[(y, src)]) / counts[(y, src, cat)]
        weights.append(w)
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double),
                                   num_samples=samples, replacement=True,
                                   generator=torch.Generator().manual_seed(seed))
    print('Sampler source probabilities:', active)
    return sampler


@torch.no_grad()
def calculate_stats(features, dataset, sampler, count, device):
    # Compute normalization from actual source-weighted training audio only.
    features.eval().to(device)
    original_mean = features.mel_mean.detach().to(device)
    original_std = features.mel_std.detach().to(device)
    n = 0
    total = None
    total_sq = None
    for i, idx in enumerate(sampler):
        if i >= count:
            break
        x = dataset[idx]['audio'].unsqueeze(0).to(device)
        raw = features.compute_log_mel(x)
        if raw.ndim != 3 or raw.shape[1] != 80:
            raise ValueError(f'Unexpected mel features shape: {tuple(raw.shape)}')
        s = raw.double().sum(dim=(0, 2))
        ss = raw.double().square().sum(dim=(0, 2))
        total = s if total is None else total + s
        total_sq = ss if total_sq is None else total_sq + ss
        n += raw.shape[0] * raw.shape[2]
        if (i+1) % 256 == 0:
            print('Normalization windows:', i+1, flush=True)
    mean = total / n
    std = (total_sq / n - mean.square()).clamp(min=1e-6).sqrt()
    return mean.float().cpu(), std.float().cpu()


def epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    lossfn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    correct = tp = fp = tn = fn = count = 0
    for batch in loader:
        x = batch['audio'].to(device)
        y = batch['label'].to(device).float()
        lengths = torch.full((x.shape[0],), x.shape[1], device=device, dtype=torch.long)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(x, lengths).reshape(-1)
            loss = lossfn(logits, y)
            if training:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
        pred = (logits >= 0).long()
        yy = y.long()
        tp += int(((pred == 1) & (yy == 1)).sum())
        fp += int(((pred == 1) & (yy == 0)).sum())
        tn += int(((pred == 0) & (yy == 0)).sum())
        fn += int(((pred == 0) & (yy == 1)).sum())
        bs = x.shape[0]
        total_loss += loss.item() * bs
        count += bs
    return dict(loss=round(total_loss / count, 5), accuracy=round((tp+tn)/count, 5),
                recall=round(tp/max(1,tp+fn), 5), fpr=round(fp/max(1,fp+tn), 5),
                tp=tp, fp=fp, tn=tn, fn=fn)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--samples-per-epoch', type=int, default=20000)
    p.add_argument('--stats-windows', type=int, default=1024)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true',
                   help='Resume from checkpoints/v4_full/last.pt')
    p.add_argument('--cover-all-dads', action='store_true',
                   help='Visit every DADS record once across all epochs (much slower)')
    args = p.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device, flush=True)
    train, val, metadata = assemble()
    sampler = make_sampler(metadata, args.samples_per_epoch, args.seed)
    model = DroneDetectionModel()
    out = ROOT / 'checkpoints' / 'v4_full'
    out.mkdir(parents=True, exist_ok=True)

    if not args.resume:
        existing = [
            name for name in ('best.pt', 'last.pt', 'mel_stats.pt')
            if (out / name).exists()
        ]
        if existing:
            raise FileExistsError(
                f'Existing full-training artifacts: {existing}. '
                'Use --resume or archive the previous run.'
            )

    stats_path = out / 'mel_stats.pt'

    if args.resume:
        if not stats_path.exists():
            raise FileNotFoundError(
                f'Missing normalization statistics: {stats_path}'
            )

        stats = torch.load(
            stats_path,
            map_location='cpu',
            weights_only=True,
        )

        mean = stats['mel_mean'].reshape(80)
        std = stats['mel_std'].reshape(80)

        print('Reusing saved v4 normalization statistics', flush=True)

    else:
        print(
            'Computing v4 log-mel stats from sampled training audio...',
            flush=True,
        )

        stats_sampler = make_sampler(
            metadata,
            args.stats_windows,
            args.seed + 3,
        )

        mean, std = calculate_stats(
            model.features,
            train,
            stats_sampler,
            args.stats_windows,
            device,
        )

        torch.save(
            dict(
                mel_mean=mean.reshape(1, 80, 1),
                mel_std=std.reshape(1, 80, 1),
                samples=args.stats_windows,
            ),
            stats_path,
        )

    model.features.set_normalization_stats(mean, std)
    model.to(device)

    # Keep normalization statistics based on original training audio.
    # Augment training only; validation remains unchanged.
    train = TrainingAugmentation(
        train,
        metadata,
        mix_probability=0.5,
        min_snr_db=-15.0,
        max_snr_db=15.0,
        min_gain_db=-20.0,
        max_gain_db=20.0,
    )

    print('Training augmentation enabled: gain + SNR mixing')

    if args.cover_all_dads:
        dads_indices = [idx for idx, (y, src, cat) in enumerate(metadata) if src == 'dads']
        random.Random(args.seed).shuffle(dads_indices)
        print('FULL COVERAGE: Each DADS index will appear once across', args.epochs, 'epochs.', flush=True)
    else:
        train_loader = DataLoader(train, batch_size=args.batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = float('inf')
    start_epoch = 0

    if args.resume:
        resume_path = out / 'last.pt'
        if not resume_path.exists():
            raise FileNotFoundError(resume_path)

        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=True,
        )

        if checkpoint.get('epochs') != args.epochs:
            raise ValueError(
                'Resume requires the original --epochs value'
            )
        if checkpoint.get('cover_all_dads') != args.cover_all_dads:
            raise ValueError(
                'Resume requires the original coverage mode'
            )

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Restore stochastic training state.
        # DataLoader uses num_workers=0 in this training script.
        random.setstate(checkpoint['python_rng_state'])
        torch.set_rng_state(checkpoint['torch_rng_state'].cpu())

        if torch.cuda.is_available():
            cuda_states = checkpoint['cuda_rng_states']
            if cuda_states is None:
                raise RuntimeError('Missing CUDA RNG state')
            torch.cuda.set_rng_state_all(cuda_states)

        sampler.generator.set_state(
            checkpoint['sampler_rng_state'].cpu()
        )

        best = checkpoint['best_val_loss']
        start_epoch = checkpoint['epoch']

        print(
            f'Resuming from completed epoch {start_epoch}',
            flush=True
        )

    for i in range(start_epoch, args.epochs):
        print(f'Epoch {i+1}/{args.epochs} starting...', flush=True)
        if args.cover_all_dads:
            block = dads_indices[i * len(dads_indices) // args.epochs:(i+1) * len(dads_indices) // args.epochs]
            positive_dads = sum(metadata[index][0] == 1 for index in block)
            negative_dads = len(block) - positive_dads
            # Include extra local/MLSP positives and enough new negatives for near-50/50.
            extra_pos = max(1, int(0.15 * positive_dads))
            extra_neg = max(1, positive_dads + extra_pos - negative_dads)
            pool_pos = [j for j, (y, source, _) in enumerate(metadata) if y == 1 and source != 'dads']
            pool_neg = [j for j, (y, source, _) in enumerate(metadata) if y == 0 and source != 'dads']
            if not pool_pos or not pool_neg:
                raise RuntimeError('Insufficient non-DADS sources for full-coverage balancing')
            rr = random.Random(args.seed + i + 1000)

            # Balance non-DADS examples by source rather than raw
            # dataset size, so large sources cannot dominate.
            def sample_by_source(pool, count):
                by_source = defaultdict(list)

                for idx in pool:
                    _, source, _ = metadata[idx]
                    by_source[source].append(idx)

                sources = sorted(by_source)

                if not sources:
                    raise RuntimeError("Empty source pool")

                selected = []
                for _ in range(count):
                    source = rr.choice(sources)
                    selected.append(rr.choice(by_source[source]))

                return selected

            indexes = (
                list(block)
                + sample_by_source(pool_pos, extra_pos)
                + sample_by_source(pool_neg, extra_neg)
            )

            rr.shuffle(indexes)
            train_loader = DataLoader(train, batch_size=args.batch_size, sampler=indexes, num_workers=0)
            print('Full-coverage epoch training samples:', len(indexes), 'DADS:', len(block), flush=True)
        tr = epoch(model, train_loader, device, optimizer)
        va = epoch(model, val_loader, device)
        print(f'Epoch {i+1}: train={tr} val={va}', flush=True)
        if va['loss'] < best:
            best = va['loss']
            torch.save(dict(model_state_dict=model.state_dict(), epoch=i+1,
                            val_loss=best, train=tr, val=va, seed=args.seed,
                            manifest='data/v4_hard_negatives.csv',
                            full_dads_records=len(dads_indices) if args.cover_all_dads else sum(1 for _, source, _ in metadata if source == 'dads'), stats='checkpoints/v4_full/mel_stats.pt'),
                       out / 'best.pt')
            print('Saved checkpoints/v4_full/best.pt', flush=True)

        torch.save(
            dict(
                model_state_dict=model.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                epoch=i + 1,
                epochs=args.epochs,
                best_val_loss=best,
                cover_all_dads=args.cover_all_dads,
                seed=args.seed,
                python_rng_state=random.getstate(),
                torch_rng_state=torch.get_rng_state(),
                cuda_rng_states=(
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None
                ),
                sampler_rng_state=sampler.generator.get_state(),
            ),
            out / 'last.pt',
        )
        print('Saved checkpoints/v4_full/last.pt', flush=True)

    print('PASS: v4 training complete. Keep dedicated Q9S/HolyStone holdouts untouched.')


if __name__ == '__main__':
    main()
