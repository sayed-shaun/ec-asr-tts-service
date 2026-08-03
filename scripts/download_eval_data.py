"""Download a small labeled Bengali ASR benchmark set for evaluating this
pipeline against real ground-truth transcripts (unlike scripts/benchmark.py's
data/, which has no ground truth at all — see its module docstring).

Source: FLEURS (https://huggingface.co/datasets/google/fleurs, CC-BY-4.0),
Google's small multilingual speech benchmark, via the `datasets` library.
Downloads the full bn_in test + validation splits (a few hundred short
utterances with human transcriptions) — not train, which is a much larger
training set, not an eval set.

Extracts every row's audio (raw bytes, undecoded — Audio(decode=False) so
this gets the original file exactly as published, not a re-encoded copy)
into a single data.tar (one archive member per utterance, rather than
hundreds of loose files) plus a ground_truth.json mapping the tar member
name -> transcript, merged across both splits. scripts/benchmark.py reads
audio straight out of data.tar on the fly, no extraction step needed.

Requires the `eval` extra: pip install ".[eval]" (datasets, soundfile, tqdm).

Usage (from the repo root):
    python scripts/download_eval_data.py [--data-dir data/eval_fleurs_bn]
"""

import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

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

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        print('datasets is required: pip install ".[eval]"', file=sys.stderr)
        sys.exit(1)

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
                # row["id"] is not unique within a split (confirmed: FLEURS
                # validation had 402 rows but only 150 distinct ids, which
                # would silently overwrite 252 members if used as the
                # name) — the loop position is guaranteed unique instead.
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
