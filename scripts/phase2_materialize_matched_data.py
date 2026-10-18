"""Materialize C1 and A1-fixed into a new directory; refuse overwrite.

This script is not a training script. It only creates hardlinks (or copies if hardlinks
are unavailable) and audit metadata in a brand-new phase-2 dataset root.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT / "data" / "phase2_matched_20260919"
DEFAULT_MANIFEST = PROJECT / "reports" / "final_independent_audit" / "phase2" / "c1_repeat_manifest.csv"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_split(source_root: Path, destination_root: Path, prefix: str) -> list[dict]:
    rows = []
    for image in sorted((source_root / "images").iterdir()):
        if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        label = source_root / "labels" / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(label)
        stem = f"{prefix}{image.stem}"
        out_image = destination_root / "images" / f"{stem}{image.suffix.lower()}"
        out_label = destination_root / "labels" / f"{stem}.txt"
        transfer = link_or_copy(image, out_image)
        link_or_copy(label, out_label)
        rows.append({"source_image": str(image.resolve()), "output_image": str(out_image.resolve()), "transfer": transfer})
    return rows


def count_labels(label_dir: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for label in label_dir.glob("*.txt"):
        for raw in label.read_text(encoding="utf-8").splitlines():
            if raw.strip():
                counts[int(float(raw.split()[0]))] += 1
    return counts


def write_yaml(root: Path) -> None:
    (root / "data.yaml").write_text(
        f"path: {root.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\n"
        "names:\n  0: fully_ripened\n  1: green\n  2: half_ripened\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--materialize", action="store_true", help="Required safety gate; otherwise only print the plan")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to reuse or overwrite output: {output}")
    if not args.materialize:
        print(json.dumps({"status": "PLAN_ONLY", "would_create": str(output), "manifest": str(args.manifest.resolve())}, indent=2))
        return

    merchant = PROJECT / "data" / "merchant_clean"
    pooled = PROJECT / "data" / "a_pooled_three_class"
    c1, a1 = output / "c1_merchant_repeat", output / "a1_fixed"
    transfers = {
        "c1_base": copy_split(merchant / "train", c1 / "train", "merchant__"),
        "c1_valid": copy_split(merchant / "valid", c1 / "valid", "merchant__"),
        "a1_train": copy_split(pooled / "train", a1 / "train", ""),
        "a1_valid": copy_split(pooled / "valid", a1 / "valid", ""),
    }
    with args.manifest.open(encoding="utf-8-sig", newline="") as handle:
        repeat_rows = list(csv.DictReader(handle))
    if len(repeat_rows) != 172:
        raise RuntimeError(f"Expected 172 repeat rows, found {len(repeat_rows)}")
    repeat_transfers = []
    for row in repeat_rows:
        image = PROJECT / row["source_image_relative"]
        label = PROJECT / row["source_label_relative"]
        rank = int(row["repeat_rank"])
        stem = f"repeat_{rank:03d}__{image.stem}"
        out_image = c1 / "train" / "images" / f"{stem}{image.suffix.lower()}"
        out_label = c1 / "train" / "labels" / f"{stem}.txt"
        transfer = link_or_copy(image, out_image)
        link_or_copy(label, out_label)
        repeat_transfers.append({"source_image": str(image.resolve()), "output_image": str(out_image.resolve()), "transfer": transfer})
    transfers["c1_repeat"] = repeat_transfers
    write_yaml(c1)
    write_yaml(a1)

    result = {}
    for name, root in (("c1", c1), ("a1_fixed", a1)):
        train_images = [p for p in (root / "train" / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        valid_images = [p for p in (root / "valid" / "images").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES]
        counts = count_labels(root / "train" / "labels")
        if len(train_images) != 2380 or len(valid_images) != 473:
            raise RuntimeError(f"Unexpected materialized counts for {name}: {len(train_images)}, {len(valid_images)}")
        result[name] = {
            "train_images": len(train_images),
            "valid_images": len(valid_images),
            "train_boxes": sum(counts.values()),
            "class_counts": {str(i): counts[i] for i in range(3)},
            "data_yaml": str((root / "data.yaml").resolve()),
        }
    metadata = {
        "output_root": str(output),
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "transfer_counts": {key: dict(Counter(row["transfer"] for row in rows)) for key, rows in transfers.items()},
        "datasets": result,
        "historical_inputs_modified": False,
    }
    (output / "materialization_audit.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
