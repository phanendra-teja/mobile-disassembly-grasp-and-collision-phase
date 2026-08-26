"""
acronym_tools.load_mesh() expects meshes organized as:
    <mesh_root>/meshes/<Category>/<model_id>.obj  (+ .mtl)
because it reads the exact relative path from the .h5 file's object/file field
(e.g. "meshes/CellPhone/136ef91c95ca5c2d4b9a4e1a888c5f59.obj").

Our extraction put everything flat under models-OBJ/models/<model_id>.obj.
This script copies (not symlinks, for Windows compatibility) each shortlisted
mesh into the correct nested structure, using the category from our shortlist
CSV, so acronym_visualize_grasps.py can find them via --mesh_root.

Usage:
    python restructure_for_acronym.py
"""

import os
import shutil
import pandas as pd

FLAT_MESH_DIR = "shapenetsem_extracted/ShapeNetSem-backup/models-OBJ/models"
SHORTLIST_CSV = "acronym_shortlist.csv"
OUT_ROOT = "acronym_mesh_root"   # pass this as --mesh_root


def main():
    shortlist = pd.read_csv(SHORTLIST_CSV)
    # one row per unique model_id -- category should be consistent per model_id,
    # but grab the first occurrence in case of any duplicates
    unique = shortlist.drop_duplicates(subset="model_id")[["model_id", "category"]]
    print(f"Restructuring {len(unique)} unique meshes...")

    copied = 0
    missing = 0
    for _, row in unique.iterrows():
        model_id = str(row["model_id"])
        category = str(row["category"])

        dest_dir = os.path.join(OUT_ROOT, "meshes", category)
        os.makedirs(dest_dir, exist_ok=True)

        for ext in [".obj", ".mtl"]:
            src = os.path.join(FLAT_MESH_DIR, f"{model_id}{ext}")
            dst = os.path.join(dest_dir, f"{model_id}{ext}")
            if os.path.exists(src):
                shutil.copy2(src, dst)
                if ext == ".obj":
                    copied += 1
            else:
                if ext == ".obj":
                    missing += 1
                    print(f"[WARN] missing source file: {src}")

    print(f"\nDone. Copied {copied} objects, {missing} missing.")
    print(f"Use: --mesh_root {OUT_ROOT}")


if __name__ == "__main__":
    main()
