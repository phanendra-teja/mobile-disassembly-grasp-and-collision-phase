"""
Explore ACRONYM grasp annotation .h5 files.

Usage:
    pip install h5py numpy
    python explore_acronym.py /path/to/acronym/grasps

This will:
  1. List how many .h5 files are present and parse category info from filenames
  2. Open a sample of files and print their internal HDF5 structure
  3. Report grasp counts, quality score stats, and collision-free ratio per sample
  4. Print an overall category distribution across ALL files (filename-based,
     no mesh needed) so you can start picking your target subset now.
"""

import sys
import glob
import os
from collections import Counter

import h5py
import numpy as np


def parse_filename(path):
    """
    ACRONYM filenames are typically formatted like:
        Category_ModelID_Scale.h5
    e.g. Mug_10f6e09036350e92b3f21f1137c3c347_0.008094298717803466.h5
    Adjust the split logic below if your files look different --
    print a few raw filenames first to confirm the pattern.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split("_")
    if len(parts) >= 3:
        category = parts[0]
        model_id = parts[1]
        scale = parts[-1]
    else:
        category, model_id, scale = stem, None, None
    return category, model_id, scale


def summarize_file(path, verbose=True):
    with h5py.File(path, "r") as f:
        if verbose:
            print(f"\n{'=' * 70}")
            print(f"FILE: {os.path.basename(path)}")
            print(f"{'=' * 70}")

            def walk(name, obj):
                indent = "  " * name.count("/")
                if isinstance(obj, h5py.Dataset):
                    print(f"{indent}{name}  shape={obj.shape}  dtype={obj.dtype}")
                else:
                    print(f"{indent}{name}/")

            f.visititems(walk)

            # print any top-level or grasp-group attributes (metadata)
            print("\n-- attrs --")
            for k, v in f.attrs.items():
                print(f"  {k}: {v}")
            if "grasps" in f and hasattr(f["grasps"], "attrs"):
                for k, v in f["grasps"].attrs.items():
                    print(f"  grasps/{k}: {v}")

        # Try the standard ACRONYM layout. If key names differ, this will
        # raise KeyError -- run once with verbose=True first to see actual
        # dataset paths and adjust the keys below.
        try:
            transforms = f["grasps/transforms"][:]        # (N, 4, 4)
            quality = f["grasps/qualities/flex/object_in_gripper"][:]  # (N,) or similar
        except KeyError:
            transforms, quality = None, None

        try:
            success = f["grasps/qualities/flex/successful"][:]  # binary success/collision-free flag
        except KeyError:
            success = None

        return transforms, quality, success


def main():
    if len(sys.argv) < 2:
        print("Usage: python explore_acronym.py /path/to/acronym/grasps")
        sys.exit(1)

    grasp_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(grasp_dir, "*.h5")))
    print(f"Found {len(files)} .h5 files in {grasp_dir}")

    if not files:
        print("No .h5 files found -- check the path.")
        sys.exit(1)

    # --- 1. Filename-based category distribution across ALL files ---
    categories = Counter()
    for path in files:
        cat, model_id, scale = parse_filename(path)
        categories[cat] += 1

    print(f"\n{len(categories)} unique categories found. Top 30 by count:")
    for cat, count in categories.most_common(30):
        print(f"  {cat:30s} {count}")

    # --- 2. Deep-dive into a small sample of files ---
    sample = files[:3]
    print(f"\n\nDeep-diving into {len(sample)} sample files for internal structure...")

    for path in sample:
        transforms, quality, success = summarize_file(path, verbose=True)

        print("\n-- parsed grasp data --")
        if transforms is not None:
            print(f"  num grasps: {transforms.shape[0]}")
        else:
            print("  Could not find 'grasps/transforms' -- check dataset paths above "
                  "and edit summarize_file() keys accordingly.")

        if quality is not None:
            print(f"  quality: min={quality.min():.4f} max={quality.max():.4f} "
                  f"mean={quality.mean():.4f} std={quality.std():.4f}")
        else:
            print("  Could not find quality dataset -- check paths above.")

        if success is not None:
            success = success.astype(bool)
            ratio = success.mean()
            print(f"  success/collision-free ratio: {ratio:.2%} "
                  f"({success.sum()} / {len(success)})")
        else:
            print("  Could not find success/collision flag dataset -- check paths above.")

    print("\n\nDone. If any 'Could not find' messages appeared above, re-run with "
          "just one file and read the full key listing to find the correct dataset "
          "paths, then update the keys in summarize_file().")


if __name__ == "__main__":
    main()
