"""
generate_manifest.py

Walks the dataset/ directory (one subfolder per class) and produces a single
manifest.csv that is the source of truth for every downstream stage
(dataloaders, training, audit logging, reproducibility).

Expected input layout:
    dataset/
    ├── certificate/   *.jpg / *.png / *.jpeg
    ├── form/
    ├── id_card/
    ├── invoice/
    └── resume/

Output: manifest.csv with columns:
    filepath   - relative path from dataset root, forward-slash normalized
    filename   - basename only
    class_name - certificate / form / id_card / invoice / resume
    source     - "real" (RVL-CDIP) or "synthetic" (custom generator)
    split      - train / val / test

Split strategy:
    70/15/15, stratified jointly on (class_name, source) so that no class or
    source is over/under-represented in val/test relative to train. In this
    dataset source is currently a deterministic function of class_name
    (id_card, certificate -> synthetic; invoice, form, resume -> real), so
    stratifying on class_name alone would give an identical split today —
    but stratifying on the combined key keeps this script correct even if a
    class later mixes real + synthetic images (e.g. augmented real data).

Usage:
    python generate_manifest.py --dataset-root /path/to/dataset --out manifest.csv
"""

import argparse
import hashlib
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}

# Folder name -> source. Extend this if a new class is added.
SYNTHETIC_CLASSES = {"id_card", "certificate"}


def infer_source(class_name: str) -> str:
    return "synthetic" if class_name in SYNTHETIC_CLASSES else "real"


def file_hash(path: Path, chunk_size: int = 65536) -> str:
    """Short content hash, used to flag exact-duplicate files within a class."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()[:12]


def scan_dataset(dataset_root: Path) -> pd.DataFrame:
    rows = []
    class_dirs = sorted([d for d in dataset_root.iterdir() if d.is_dir()])

    if not class_dirs:
        sys.exit(f"No class subfolders found under {dataset_root}")

    for class_dir in class_dirs:
        class_name = class_dir.name
        source = infer_source(class_name)
        files = [
            p for p in sorted(class_dir.rglob("*"))
            if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
        ]

        if not files:
            print(f"  WARNING: no image files found in {class_dir}", file=sys.stderr)
            continue

        for p in files:
            rel_path = p.relative_to(dataset_root).as_posix()
            rows.append({
                "filepath": rel_path,
                "filename": p.name,
                "class_name": class_name,
                "source": source,
                "file_hash": file_hash(p),
            })

        print(f"  {class_name:12s} ({source:9s}): {len(files)} images")

    if not rows:
        sys.exit("No images found anywhere under dataset root. Aborting.")

    return pd.DataFrame(rows)


def check_duplicates(df: pd.DataFrame) -> None:
    dupes = df[df.duplicated(subset=["file_hash"], keep=False)]
    if not dupes.empty:
        n_groups = dupes["file_hash"].nunique()
        print(
            f"  WARNING: {len(dupes)} files across {n_groups} hash groups are "
            f"byte-identical to another file in the dataset. This can leak "
            f"train/test if duplicates land on both sides. See duplicates.csv.",
            file=sys.stderr,
        )
        dupes.sort_values("file_hash").to_csv("duplicates.csv", index=False)


def stratified_split(df: pd.DataFrame, val_frac: float, test_frac: float, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["strata"] = df["class_name"] + "__" + df["source"]

    # Guard against strata too small to stratify (need >= 2 per split).
    counts = df["strata"].value_counts()
    too_small = counts[counts < 10]
    if not too_small.empty:
        print(
            f"  WARNING: these class/source groups have <10 images, split may "
            f"be uneven: {too_small.to_dict()}",
            file=sys.stderr,
        )

    train_df, temp_df = train_test_split(
        df,
        test_size=(val_frac + test_frac),
        stratify=df["strata"],
        random_state=seed,
    )
    # Split temp into val/test proportionally
    relative_test_frac = test_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_frac,
        stratify=temp_df["strata"],
        random_state=seed,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    out = pd.concat([train_df, val_df, test_df]).drop(columns=["strata"])
    return out.sort_values(["class_name", "split", "filepath"]).reset_index(drop=True)


def print_summary(df: pd.DataFrame) -> None:
    print("\n=== Split summary (class x split) ===")
    summary = df.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    summary = summary[["train", "val", "test"]] if set(["train", "val", "test"]).issubset(summary.columns) else summary
    print(summary.to_string())

    print("\n=== Split summary (source x split) ===")
    summary2 = df.groupby(["source", "split"]).size().unstack(fill_value=0)
    print(summary2.to_string())

    print(f"\nTotal images: {len(df)}")


def main():
    ap = argparse.ArgumentParser(description="Generate dataset manifest with stratified train/val/test split.")
    ap.add_argument("--dataset-root", type=str, default="dataset", help="Path to dataset/ folder containing per-class subfolders")
    ap.add_argument("--out", type=str, default="manifest.csv", help="Output manifest CSV path")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        sys.exit(f"Dataset root does not exist: {dataset_root}")

    print(f"Scanning {dataset_root} ...")
    df = scan_dataset(dataset_root)

    print("\nChecking for duplicate files...")
    check_duplicates(df)

    print(f"\nSplitting {len(df)} images "
          f"({1 - args.val_frac - args.test_frac:.0%}/{args.val_frac:.0%}/{args.test_frac:.0%}), "
          f"stratified by class+source, seed={args.seed} ...")
    df = stratified_split(df, args.val_frac, args.test_frac, args.seed)

    df.to_csv(args.out, index=False)
    print(f"\nManifest written to {args.out}")

    print_summary(df)


if __name__ == "__main__":
    main()
