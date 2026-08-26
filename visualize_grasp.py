"""
Sanity check: load one shortlisted mesh + its ACRONYM grasp poses together,
visualize a handful of grasps as coordinate frames on the mesh surface.

Usage:
    pip install trimesh numpy pandas pyglet
    python visualize_grasp.py

Adjust MESH_DIR / GRASP_DIR / SHORTLIST_CSV paths below if needed.

What to look for in the viewer window:
  - Does the gripper frame (small axes) sit ON the object surface, not
    floating far away or buried deep inside?
  - Is the frame's scale sensible relative to the object (not a tiny dot
    on a huge mesh, not a giant frame dwarfing the mesh)?
  - Do multiple grasps cluster around plausible "graspable" regions
    (e.g. the body of a mug, not through solid material)?

If any of these look wrong, the likely culprits are:
  - object/scale from the .h5 not being applied to the raw mesh before
    plotting (ACRONYM meshes are often stored unscaled; the grasp
    transforms assume the SCALED mesh)
  - a units mismatch (meters vs another unit)
  - a coordinate-frame convention mismatch between the mesh and grasp data
"""

import os
import h5py
import numpy as np
import pandas as pd
import trimesh

MESH_DIR = "shapenetsem_extracted/ShapeNetSem-backup/models-OBJ/models"
GRASP_DIR = "dataset/grasps"  # your original ACRONYM grasps folder
SHORTLIST_CSV = "acronym_shortlist.csv"

N_GRASPS_TO_SHOW = 15


def load_mesh(model_id):
    obj_path = os.path.join(MESH_DIR, f"{model_id}.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Mesh not found: {obj_path}")
    mesh = trimesh.load(obj_path, force="mesh")
    return mesh


def load_grasps(h5_filename):
    path = os.path.join(GRASP_DIR, h5_filename)
    with h5py.File(path, "r") as f:
        transforms = f["grasps/transforms"][:]          # (N, 4, 4)
        success = f["grasps/qualities/flex/object_in_gripper"][:]
        scale = float(f["object/scale"][()])
    return transforms, success, scale


def make_grasp_axes(transform, size):
    """Small coordinate-axis marker at a grasp pose for visualization."""
    axes = trimesh.creation.axis(origin_size=size * 0.1, axis_length=size)
    axes.apply_transform(transform)
    return axes


def main():
    shortlist = pd.read_csv(SHORTLIST_CSV)

    # Pick one example per a couple of interesting categories to check
    for category in ["CellPhone", "Mug", "Book"]:
        rows = shortlist[shortlist["category"] == category]
        if rows.empty:
            print(f"No shortlist entries for category={category}, skipping")
            continue

        row = rows.iloc[0]
        model_id = row["model_id"]
        h5_filename = row["filename"]

        print(f"\n{'=' * 60}")
        print(f"Category: {category}  |  model_id: {model_id}")
        print(f"grasp file: {h5_filename}")

        try:
            mesh = load_mesh(model_id)
        except FileNotFoundError as e:
            print(f"[SKIP] {e}")
            continue

        transforms, success, scale = load_grasps(h5_filename)

        print(f"Raw mesh bounds (before any scaling): {mesh.bounds}")
        print(f"Raw mesh extents: {mesh.extents}")
        print(f"Raw mesh centroid: {mesh.centroid}")
        print(f"Raw mesh bounding-box center: {mesh.bounding_box.centroid}")
        print(f"object/scale from .h5: {scale}")

        with h5py.File(os.path.join(GRASP_DIR, h5_filename), "r") as f:
            obj_file_field = f["object/file"][()]
            if isinstance(obj_file_field, bytes):
                obj_file_field = obj_file_field.decode("utf-8")
        print(f"object/file field from .h5: {obj_file_field}")

        # Try applying the ACRONYM scale factor to the mesh before plotting --
        # this is the standard ACRONYM convention (grasps are defined relative
        # to the mesh AFTER this scale is applied). If grasps look wrong without
        # this, this is very likely the fix; if they still look wrong WITH it,
        # something else is off (units, coordinate convention).
        mesh_scaled = mesh.copy()
        mesh_scaled.apply_scale(scale)
        print(f"Scaled mesh extents: {mesh_scaled.extents}")
        print(f"Scaled mesh centroid: {mesh_scaled.centroid}")

        # HYPOTHESIS BEING TESTED: ACRONYM may expect the mesh recentered to
        # its bounding-box center (or centroid) at the origin BEFORE scaling,
        # since raw ShapeNetSem meshes can have an arbitrary origin offset.
        # We build a second version with recentering applied, so we can
        # visually compare both against the same grasp poses.
        mesh_recentered = mesh.copy()
        mesh_recentered.apply_translation(-mesh_recentered.bounding_box.centroid)
        mesh_recentered.apply_scale(scale)
        print(f"Recentered+scaled mesh extents: {mesh_recentered.extents}")
        print(f"Recentered+scaled mesh centroid: {mesh_recentered.centroid}")

        # Pick a handful of successful grasps to display (less clutter)
        success_idx = np.where(success == 1)[0]
        if len(success_idx) == 0:
            print("[WARN] no successful grasps in this file, showing random sample instead")
            show_idx = np.random.choice(len(transforms), size=min(N_GRASPS_TO_SHOW, len(transforms)), replace=False)
        else:
            show_idx = np.random.choice(success_idx, size=min(N_GRASPS_TO_SHOW, len(success_idx)), replace=False)

        # marker size relative to object scale so axes are visible but not huge
        marker_size = max(mesh_scaled.extents) * 0.15

        scene = trimesh.Scene()
        mesh_scaled.visual.face_colors = [200, 200, 200, 150]
        scene.add_geometry(mesh_scaled)

        for idx in show_idx:
            axes = make_grasp_axes(transforms[idx], marker_size)
            scene.add_geometry(axes)

        print(f"\n[WINDOW 1 of 2] Scale-only mesh + grasps. "
              f"Showing {len(show_idx)} grasps -- close window to see recentered version.")
        scene.show()

        # Second window: recentered hypothesis, same grasp poses
        scene2 = trimesh.Scene()
        mesh_recentered.visual.face_colors = [200, 150, 150, 150]
        scene2.add_geometry(mesh_recentered)
        for idx in show_idx:
            axes = make_grasp_axes(transforms[idx], marker_size)
            scene2.add_geometry(axes)

        print(f"[WINDOW 2 of 2] Recentered+scaled mesh + SAME grasps. "
              f"Close window to continue to next category.")
        scene2.show()


if __name__ == "__main__":
    main()