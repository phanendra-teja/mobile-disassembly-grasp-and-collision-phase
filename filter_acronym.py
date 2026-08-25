"""
ACRONYM dataset processing: correct quality labeling + phone-adjacent subset filter.

Usage:
    pip install h5py numpy pandas
    python filter_acronym.py /path/to/acronym/grasps

Produces:
    acronym_full_index.csv     - one row per .h5 file: category, model_id, scale,
                                  mass, num_grasps, success_rate, mean_quality
    acronym_category_stats.csv - per-category aggregate stats (count, mean mass, etc.)
    acronym_shortlist.csv      - filtered candidate subset for phone-scale transfer learning

Quality score definition (since raw ACRONYM only has a binary success flag,
not a continuous quality score):
    quality = success * (1 / (1 + closing_linear + closing_angular
                               + shaking_linear + shaking_angular))
    -> 0 if the grasp failed outright
    -> otherwise, higher when the object moved LESS during closing/shaking
       (i.e. a more stable, secure grasp), scaled into (0, 1]
"""

import sys
import glob
import os

import h5py
import numpy as np
import pandas as pd


def parse_filename(path):
    """
    ACRONYM filenames: Category_ModelID_Scale.h5
    Confirmed against real files, e.g.:
        1Shelves_12a64182bbaee7a12b2444829a3507de_0.00914554366969263.h5
    NOTE: some categories start with a digit (e.g. "1Shelves") -- these are
    literal category names in this dataset, not a separate numeric field.
    We split on the LAST two underscores to isolate scale and model_id,
    since category names themselves may contain underscores or digits.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 2)
    if len(parts) == 3:
        category, model_id, scale_str = parts
        try:
            scale = float(scale_str)
        except ValueError:
            scale = None
    else:
        category, model_id, scale = stem, None, None
    return category, model_id, scale


def compute_quality(success, closing_lin, closing_ang, shaking_lin, shaking_ang):
    """Combined per-grasp quality score in (0, 1], 0 for failed grasps."""
    motion_penalty = closing_lin + closing_ang + shaking_lin + shaking_ang
    stability = 1.0 / (1.0 + motion_penalty)
    return success.astype(np.float64) * stability


def read_h5_summary(path):
    """Extract everything needed for indexing + quality computation from one file."""
    with h5py.File(path, "r") as f:
        success = f["grasps/qualities/flex/object_in_gripper"][:]
        closing_lin = f["grasps/qualities/flex/object_motion_during_closing_linear"][:]
        closing_ang = f["grasps/qualities/flex/object_motion_during_closing_angular"][:]
        shaking_lin = f["grasps/qualities/flex/object_motion_during_shaking_linear"][:]
        shaking_ang = f["grasps/qualities/flex/object_motion_during_shaking_angular"][:]

        quality = compute_quality(success, closing_lin, closing_ang, shaking_lin, shaking_ang)

        mass = float(f["object/mass"][()])
        scale = float(f["object/scale"][()])
        obj_file = f["object/file"][()]
        if isinstance(obj_file, bytes):
            obj_file = obj_file.decode("utf-8")

        num_grasps = success.shape[0]
        success_rate = float(success.mean())
        mean_quality = float(quality.mean())
        max_quality = float(quality.max())

    return {
        "mass": mass,
        "scale": scale,
        "object_file": obj_file,
        "num_grasps": num_grasps,
        "success_rate": success_rate,
        "mean_quality": mean_quality,
        "max_quality": max_quality,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python filter_acronym.py /path/to/acronym/grasps")
        sys.exit(1)

    grasp_dir = sys.argv[1]
    files = sorted(glob.glob(os.path.join(grasp_dir, "*.h5")))
    print(f"Found {len(files)} .h5 files in {grasp_dir}")
    if not files:
        sys.exit(1)

    rows = []
    n_errors = 0
    for i, path in enumerate(files):
        category, model_id, scale_from_name = parse_filename(path)
        try:
            summary = read_h5_summary(path)
        except Exception as e:
            n_errors += 1
            if n_errors <= 5:
                print(f"  [WARN] failed to read {os.path.basename(path)}: {e}")
            continue

        rows.append({
            "filename": os.path.basename(path),
            "path": path,
            "category": category,
            "model_id": model_id,
            **summary,
        })

        if (i + 1) % 1000 == 0:
            print(f"  processed {i + 1}/{len(files)}")

    if n_errors:
        print(f"\n{n_errors} files failed to read (see warnings above for first 5).")

    df = pd.DataFrame(rows)
    df.to_csv("acronym_full_index.csv", index=False)
    print(f"\nWrote acronym_full_index.csv ({len(df)} rows)")

    # --- per-category stats ---
    cat_stats = df.groupby("category").agg(
        count=("filename", "count"),
        mean_mass=("mass", "mean"),
        median_mass=("mass", "median"),
        mean_scale=("scale", "mean"),
        mean_success_rate=("success_rate", "mean"),
        mean_quality=("mean_quality", "mean"),
    ).sort_values("count", ascending=False)
    cat_stats.to_csv("acronym_category_stats.csv")
    print(f"Wrote acronym_category_stats.csv ({len(cat_stats)} categories)")

    # --- phone-adjacent shortlist filter ---
    # Adjust these two lists based on what you see in acronym_category_stats.csv --
    # this is a starting point, not a final answer. Cross-check mean_mass / mean_scale
    # per category against real phone component sizes before trusting the filter.
    PHONE_ADJACENT_CATEGORIES = [
        "CellPhone", "Mug", "Bowl", "Book", "Pencil", "DeskLamp", "ToyFigure",
        "Remote", "Camera", "Flashlight", "Stapler", "Calculator", "Wallet",
        "Watch", "USBStick", "PowerStrip", "Router", "Speaker", "Webcam",
        "HardDrive", "Battery", "AAA", "Charger",
    ]

    # NOTE: object/mass in ShapeNetSem is frequently a placeholder/default value,
    # NOT a reliable measured mass -- confirmed by Couch/TV/Desk passing a naive
    # "mass <= 500g" filter in an earlier run. Do NOT use mass as a standalone
    # filter. Use category whitelist as the primary filter, and cross-check
    # candidates against object/scale (which is more trustworthy -- it's the
    # actual geometric scale factor applied to the mesh) as a secondary sanity
    # check, not a fallback that can pull in unrelated categories.
    shortlist = df[df["category"].isin(PHONE_ADJACENT_CATEGORIES)].copy()

    # Flag (don't silently drop) any shortlisted object whose scale looks
    # implausible for a handheld/phone-component-sized item, so you can
    # manually review rather than trust it blindly.
    SCALE_MIN, SCALE_MAX = 0.01, 0.5  # tune after inspecting real values
    shortlist["scale_suspect"] = ~shortlist["scale"].between(SCALE_MIN, SCALE_MAX)
    n_suspect = shortlist["scale_suspect"].sum()
    if n_suspect:
        print(f"\n[WARN] {n_suspect} shortlisted objects have scale outside "
              f"[{SCALE_MIN}, {SCALE_MAX}] -- check these manually, "
              f"see 'scale_suspect' column in acronym_shortlist.csv")

    shortlist.to_csv("acronym_shortlist.csv", index=False)
    print(f"Wrote acronym_shortlist.csv ({len(shortlist)} rows, "
          f"{shortlist['category'].nunique()} categories)")

    print("\nShortlist category breakdown (top 30):")
    print(shortlist["category"].value_counts().head(30).to_string())

    print("\nDone. Inspect acronym_category_stats.csv to refine "
          "PHONE_ADJACENT_CATEGORIES / MASS_MAX_KG, then re-run.")


if __name__ == "__main__":
    main()