"""
Data preparation — Databricks Dolly 15k
Splits the dataset into train / dev / test and converts each example
to the standard Dolly prompt format used across all training scripts.

Usage:
    python src/data/prepare_data.py
    python src/data/prepare_data.py --train 6000 --dev 1000 --test 500
    python src/data/prepare_data.py --downsample 3000
"""

import argparse
import json
import random
from pathlib import Path

from datasets import load_dataset

DATASET_ID  = "databricks/databricks-dolly-15k"
RANDOM_SEED = 42
OUTPUT_DIR  = Path(__file__).resolve().parents[2] / "data" / "splits"

DEFAULT_TRAIN = 12_000
DEFAULT_DEV   =  2_000
DEFAULT_TEST  =  1_000


def to_record(example: dict) -> dict:
    instruction = (example.get("instruction") or "").strip()
    context     = (example.get("context")     or "").strip()
    response    = (example.get("response")    or "").strip()

    if context:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Context:\n{context}\n\n"
            f"### Response:\n"
        )
    else:
        prompt = (
            f"### Instruction:\n{instruction}\n\n"
            f"### Response:\n"
        )

    return {
        "prompt":      prompt,
        "response":    response,
        "instruction": instruction,
        "context":     context,
        "category":    example.get("category", ""),
    }


def save_jsonl(path: Path, records: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"  {path.name:<12} {len(records):>6} examples  →  {path}")


def build_splits(
    n_train: int = DEFAULT_TRAIN,
    n_dev:   int = DEFAULT_DEV,
    n_test:  int = DEFAULT_TEST,
    downsample: int | None = None,
) -> None:

    total = n_train + n_dev + n_test

    print(f"Loading '{DATASET_ID}' from Hugging Face Hub...")
    dataset = load_dataset(DATASET_ID, split="train")
    print(f"  {len(dataset)} examples available")

    if downsample is not None:
        downsample = min(downsample, len(dataset))
        print(f"  Downsampling to {downsample} examples")
        dataset = dataset.shuffle(seed=RANDOM_SEED).select(range(downsample))
        ratio   = downsample / total
        n_train = int(n_train * ratio)
        n_dev   = int(n_dev   * ratio)
        n_test  = downsample - n_train - n_dev
        total   = n_train + n_dev + n_test

    if len(dataset) < total:
        raise ValueError(
            f"Not enough examples: {len(dataset)} available, {total} required."
        )

    records = [to_record(ex) for ex in dataset]
    random.seed(RANDOM_SEED)
    random.shuffle(records)

    train = records[:n_train]
    dev   = records[n_train : n_train + n_dev]
    test  = records[n_train + n_dev : n_train + n_dev + n_test]

    print(f"\nSaving splits to '{OUTPUT_DIR}' ...")
    save_jsonl(OUTPUT_DIR / "train.jsonl", train)
    save_jsonl(OUTPUT_DIR / "dev.jsonl",   dev)
    save_jsonl(OUTPUT_DIR / "test.jsonl",  test)

    print(f"\n  train={len(train)}  dev={len(dev)}  test={len(test)}  "
          f"total={len(train) + len(dev) + len(test)}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Dolly 15k splits")
    parser.add_argument("--train",      type=int, default=DEFAULT_TRAIN)
    parser.add_argument("--dev",        type=int, default=DEFAULT_DEV)
    parser.add_argument("--test",       type=int, default=DEFAULT_TEST)
    parser.add_argument("--downsample", type=int, default=None,
                        help="Cap total dataset size before splitting (e.g. 3000)")
    args = parser.parse_args()

    build_splits(
        n_train    = args.train,
        n_dev      = args.dev,
        n_test     = args.test,
        downsample = args.downsample,
    )
