
import csv
from pathlib import Path
from collections import Counter

import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DADS_ROOT = Path(
    "/mnt/c/Users/andre/.cache/huggingface/hub/"
    "datasets--geronimobasso--drone-audio-detection-samples/"
    "snapshots"
)

OUTPUT = PROJECT_ROOT / "data" / "dads_parquet_index.csv"


def main():
    files = sorted(DADS_ROOT.glob("*/data/*.parquet"))

    if not files:
        raise FileNotFoundError("DADS Parquet shards not found")

    counts = Counter()
    total = 0

    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "parquet_path",
                "row_group",
                "row_index",
                "label",
                "source",
                "split",
            ],
        )
        writer.writeheader()

        for shard_number, path in enumerate(files, 1):
            parquet = pq.ParquetFile(path)
            shard_count = 0

            for group_id in range(parquet.num_row_groups):
                table = parquet.read_row_group(
                    group_id,
                    columns=["label"],
                )

                labels = table.column("label").to_pylist()

                for row_index, label in enumerate(labels):
                    label = int(label)

                    if label not in (0, 1):
                        raise ValueError(
                            f"Unexpected DADS label: {label}"
                        )

                    writer.writerow({
                        "parquet_path": str(path),
                        "row_group": group_id,
                        "row_index": row_index,
                        "label": label,
                        "source": "dads",
                        "split": "train",
                    })

                    counts[label] += 1
                    total += 1
                    shard_count += 1

            print(
                f"Indexed shard {shard_number}/{len(files)} "
                f"| records={shard_count}",
                flush=True,
            )

    print("\nDADS indexing complete")
    print("Total indexed:", total)
    print("Labels:", dict(counts))
    print("Index:", OUTPUT)

    assert total == 180320
    assert counts[0] == 16729
    assert counts[1] == 163591

    print("PASS: Full DADS index created.")


if __name__ == "__main__":
    main()

