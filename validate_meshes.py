"""
Validate that every mesh in the shortlist actually loads correctly via
acronym_tools.load_mesh, BEFORE training. Training crashed on a MemoryError
from one specific .obj file -- rather than catch/skip errors silently
mid-epoch (which would make loss curves unreliable), find and remove the
bad file(s) here, once, up front.

Usage:
    python validate_meshes.py

Produces:
    acronym_shortlist_clean.csv -- same as acronym_shortlist.csv but with
                                    any unloadable meshes removed
    bad_meshes.csv              -- log of what failed and why, for reference
"""

import os
import sys
import traceback

import pandas as pd
import trimesh

sys.path.insert(0, "acronym")  # so acronym_tools is importable without pip install -e . dependency here
from acronym_tools import load_mesh

SHORTLIST_CSV = "acronym_shortlist.csv"
MESH_ROOT = "acronym_mesh_root"
GRASP_DIR = "dataset/grasps"


def main():
    shortlist = pd.read_csv(SHORTLIST_CSV)
    unique = shortlist.drop_duplicates(subset="model_id").reset_index(drop=True)
    print(f"Validating {len(unique)} unique meshes...")

    good_model_ids = []
    bad_rows = []

    for i, row in unique.iterrows():
        h5_path = os.path.join(GRASP_DIR, row["filename"])

        try:
            mesh = load_mesh(h5_path, mesh_root_dir=MESH_ROOT)

            # also confirm it can actually be sampled (catches degenerate
            # meshes, e.g. zero faces, that might load but fail downstream)
            if isinstance(mesh, trimesh.Scene):
                if len(mesh.geometry) == 0:
                    raise ValueError("Scene has no geometry")
                mesh = trimesh.util.concatenate([g for g in mesh.geometry.values()])
            mesh.sample(64)  # small sample count, just a functional check

            good_model_ids.append(row["model_id"])

        except Exception as e:
            bad_rows.append({
                "model_id": row["model_id"],
                "category": row["category"],
                "filename": row["filename"],
                "error_type": type(e).__name__,
                "error_msg": str(e)[:200],
            })
            print(f"[BAD] {row['category']}/{row['model_id']}: {type(e).__name__}: {str(e)[:100]}")

        if (i + 1) % 100 == 0:
            print(f"  checked {i + 1}/{len(unique)}")

    print(f"\nGood: {len(good_model_ids)}, Bad: {len(bad_rows)}")

    clean_shortlist = shortlist[shortlist["model_id"].isin(good_model_ids)]
    clean_shortlist.to_csv("acronym_shortlist_clean.csv", index=False)
    print(f"Wrote acronym_shortlist_clean.csv ({len(clean_shortlist)} rows)")

    if bad_rows:
        pd.DataFrame(bad_rows).to_csv("bad_meshes.csv", index=False)
        print(f"Wrote bad_meshes.csv ({len(bad_rows)} entries) for reference")


if __name__ == "__main__":
    main()