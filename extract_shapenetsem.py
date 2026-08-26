"""
Selectively extract only what we need from ShapeNetSem.zip:
  - metadata.csv, categories.synset.csv (and taxonomy.txt if present)
  - OBJ mesh files (+ their .mtl companions) for ONLY the model_ids in
    our filtered ACRONYM shortlist -- not all ~12,000 ShapeNetSem meshes.

This avoids extracting textures, binvox voxelizations, COLLADA duplicates,
and screenshots we don't need, and avoids pulling meshes for the ~8,000+
ACRONYM objects outside our phone-adjacent shortlist.

Usage:
    python extract_shapenetsem.py

Before running: confirm ARCHIVE_PREFIX and MESH_FOLDER_NAME below match
the real internal structure printed by zipstructure.py. Currently set
from partial output -- CONFIRM the full prefix list before trusting this.
"""

import zipfile
import os
import pandas as pd

ZIP_PATH = "shapenetsem_download/ShapeNetSem.zip"
OUT_DIR = "shapenetsem_extracted"

# CONFIRMED from zipstructure.py output:
ARCHIVE_PREFIX = "ShapeNetSem-backup"

# NOT YET CONFIRMED -- update once you paste the full prefix list.
# Likely candidates based on the original ShapeNetSem release naming:
MESH_FOLDER_NAME = "models-OBJ"   # <-- may need correcting

SHORTLIST_CSV = "acronym_shortlist.csv"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shortlist = pd.read_csv(SHORTLIST_CSV)
    model_ids = set(shortlist["model_id"].astype(str))
    print(f"Loaded {len(model_ids)} unique model_ids from shortlist")

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        all_names = z.namelist()

        # --- metadata / category files (always extract, small) ---
        meta_targets = [
            f"{ARCHIVE_PREFIX}/metadata.csv",
            f"{ARCHIVE_PREFIX}/categories.synset.csv",
            f"{ARCHIVE_PREFIX}/taxonomy.txt",
            f"{ARCHIVE_PREFIX}/densities.csv",
            f"{ARCHIVE_PREFIX}/materials.csv",
        ]
        for target in meta_targets:
            if target in all_names:
                z.extract(target, OUT_DIR)
                print(f"Extracted: {target}")
            else:
                print(f"[WARN] not found in archive: {target}")

        # --- mesh files matching our shortlisted model_ids ---
        mesh_prefix = f"{ARCHIVE_PREFIX}/{MESH_FOLDER_NAME}/"
        mesh_entries = [n for n in all_names if n.startswith(mesh_prefix)]
        print(f"\nFound {len(mesh_entries)} total entries under {mesh_prefix}")

        if not mesh_entries:
            print(f"[ERROR] No entries found under '{mesh_prefix}' -- "
                  f"MESH_FOLDER_NAME is likely wrong. Check the full prefix "
                  f"list from zipstructure.py and update this script.")
            return

        matched = 0
        for entry in mesh_entries:
            fname = os.path.basename(entry)
            stem = os.path.splitext(fname)[0]
            if stem in model_ids:
                z.extract(entry, OUT_DIR)
                matched += 1

        print(f"\nExtracted {matched} mesh-related files matching shortlisted model_ids")
        print(f"All files written under: {OUT_DIR}/")


if __name__ == "__main__":
    main()
