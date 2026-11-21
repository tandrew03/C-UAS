
import csv
import random
import sys
import wave
from array import array
from collections import OrderedDict
from io import BytesIO

import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


class DADSAudio(Dataset):
    def __init__(
        self,
        index_path="data/dads_parquet_index_with_durations.csv",
        n_samples=8000,
        training=True,
        cache_groups=2,
    ):
        with open(index_path, newline="") as f:
            records = list(csv.DictReader(f))

        if not records or "n_samples" not in records[0]:
            raise ValueError(
                "DADS index must contain n_samples. "
                "Run acoustic/audit_dads_durations.py first."
            )

        self.records = [
            r for r in records
            if int(r["n_samples"]) >= n_samples
        ]
        self.excluded_short = len(records) - len(self.records)

        self.n_samples = n_samples
        self.training = training
        self.cache_groups = cache_groups

        # Each worker keeps its own small row-group cache.
        self._cache = OrderedDict()

        print(
            f"DADSAudio initialized: {len(self.records)} eligible recordings "
            f"({self.excluded_short} short clips excluded)"
        )

    def __len__(self):
        return len(self.records)

    def _get_row_group(self, path, group_id):
        key = (path, group_id)

        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        parquet = pq.ParquetFile(path)

        table = parquet.read_row_group(
            group_id,
            columns=["audio"],
        )

        self._cache[key] = table.column("audio").to_pylist()

        while len(self._cache) > self.cache_groups:
            self._cache.popitem(last=False)

        return self._cache[key]

    def __getitem__(self, index):
        record = self.records[index]

        path = record["parquet_path"]
        group_id = int(record["row_group"])
        row_index = int(record["row_index"])

        rows = self._get_row_group(path, group_id)
        audio = rows[row_index]

        with wave.open(BytesIO(audio["bytes"]), "rb") as wav:
            if (
                wav.getframerate() != 16000
                or wav.getnchannels() != 1
                or wav.getsampwidth() != 2
                or wav.getcomptype() != "NONE"
            ):
                raise ValueError(
                    f"Unsupported DADS WAV: {audio['path']}"
                )

            raw = wav.readframes(wav.getnframes())

        samples = array("h")
        samples.frombytes(raw)

        if sys.byteorder != "little":
            samples.byteswap()

        waveform = torch.tensor(
            samples, dtype=torch.float32
        ) / 32768.0

        length = waveform.numel()

        if length > self.n_samples:
            if self.training:
                start = random.randint(
                    0, length - self.n_samples
                )
            else:
                start = (length - self.n_samples) // 2

            waveform = waveform[
                start:start + self.n_samples
            ]

        elif length < self.n_samples:
            waveform = torch.nn.functional.pad(
                waveform,
                (0, self.n_samples - length),
            )

        return {
            "audio": waveform,
            "label": torch.tensor(
                int(record["label"]), dtype=torch.long
            ),
            "group_id": f"dads_{index}",
            "source": "dads",
        }
