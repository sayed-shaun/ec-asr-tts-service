"""Download a small labeled Bengali ASR benchmark set for scripts/benchmark.py.

Source: FLEURS (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0),
bn_in test + validation splits — not train, which is a training set.

Every row's audio is written undecoded (Audio(decode=False), so it stays the
file as published) into a single data.tar, plus a ground_truth.json mapping
tar member name -> transcript. benchmark.py reads audio straight out of the
tar, no extraction step.

Requires the `eval` extra: pip install ".[eval]" (datasets, soundfile, tqdm).

Usage (from the repo root):
    python scripts/download_eval_data.py [--data-dir data/eval_fleurs_bn]
"""

import argparse
import io
import json
import tarfile
from pathlib import Path

from datasets import Audio, load_dataset
from tqdm import tqdm

DATASET = "google/fleurs"
CONFIG = "bn_in"
SPLITS = ["test", "validation"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", default="data/eval_fleurs_bn")
    args = parser.parse_args()

    out_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ground_truth = {}
    total_count = 0
    tar_path = out_dir / "data.tar"
    with tarfile.open(tar_path, "w") as tar:
        for split in SPLITS:
            ds = load_dataset(DATASET, CONFIG, split=split, trust_remote_code=True)
            ds = ds.cast_column("audio", Audio(decode=False))

            for count, row in enumerate(
                tqdm(ds, desc=f"{split}", unit="file"), start=1
            ):
                audio = row["audio"]
                ext = Path(audio["path"]).suffix or ".wav"
                fname = f"fleurs_bn_{split}_{count:04d}{ext}"
                data = audio["bytes"]
                info = tarfile.TarInfo(name=fname)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))
                ground_truth[fname] = {
                    "transcription": row["transcription"],
                    "raw_transcription": row["raw_transcription"],
                }
            total_count += count

    (out_dir / "ground_truth.json").write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {total_count} audio files to {tar_path} + ground_truth.json")


if __name__ == "__main__":
    main()
