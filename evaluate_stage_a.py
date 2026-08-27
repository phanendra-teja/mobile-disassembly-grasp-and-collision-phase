"""
Qualitative evaluation of the trained Stage A grasp ranking model.

For a chosen object:
  1. Load its mesh + full set of candidate grasp poses (from the raw ACRONYM
     .h5, not just the sampled subset used during training).
  2. Run the trained model on ALL candidates for that object.
  3. Rank by predicted score = success_prob * stability.
  4. Visualize the TOP-K predicted grasps overlaid on the mesh, using the
     same official ACRONYM gripper rendering as acronym_visualize_grasps.py
     (not our own axis markers), so the visual is trustworthy.
  5. Also print how many of the top-K predicted grasps were ACTUALLY
     successful in the ground truth -- a concrete, honest sanity number,
     not just "it looks plausible".

Usage:
    python evaluate_stage_a.py --category CellPhone
    python evaluate_stage_a.py --model_id <specific_model_id>
"""

import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import trimesh

sys.path.insert(0, "acronym")
from acronym_tools import load_mesh, load_grasps, create_gripper_marker

from stage_a_model import GraspRankingNet

SHORTLIST_CSV = "acronym_shortlist_clean.csv"
MESH_ROOT = "acronym_mesh_root"
GRASP_DIR = "dataset/grasps"
CHECKPOINT_PATH = "checkpoints/stage_a_best.pth"
NUM_POINTS = 2048
TOP_K = 15


def sample_point_cloud(mesh, num_points):
    if isinstance(mesh, trimesh.Scene):
        # IMPORTANT: use mesh.dump(concatenate=True), NOT
        # trimesh.util.concatenate([g for g in mesh.geometry.values()]).
        # The latter grabs raw sub-meshes directly from mesh.geometry,
        # bypassing the Scene's graph transforms entirely -- which is
        # exactly where load_mesh's apply_scale() lives for Scene objects.
        # Using .geometry.values() directly silently discards scale (and
        # any part positioning), producing coordinates off by ~1000x from
        # the correctly-scaled grasp poses. Confirmed via debug bounds:
        # unscaled mesh spanned ~11-400 units while poses were ~0.05-0.15
        # (meters) -- same object, two different coordinate scales.
        mesh = mesh.dump(concatenate=True)
    points, _ = mesh.sample(num_points, return_index=True)
    return points.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, default=None,
                         help="Pick the first shortlisted object in this category")
    parser.add_argument("--model_id", type=str, default=None,
                         help="Or specify an exact model_id directly")
    args = parser.parse_args()

    shortlist = pd.read_csv(SHORTLIST_CSV).drop_duplicates(subset="model_id").reset_index(drop=True)

    if args.model_id:
        row = shortlist[shortlist["model_id"] == args.model_id].iloc[0]
    elif args.category:
        rows = shortlist[shortlist["category"] == args.category]
        if rows.empty:
            print(f"No shortlisted object in category={args.category}")
            return
        row = rows.iloc[0]
    else:
        row = shortlist.iloc[0]

    model_id = row["model_id"]
    h5_path = os.path.join(GRASP_DIR, row["filename"])
    print(f"Evaluating: {row['category']} / {model_id}")
    print(f"grasp file: {row['filename']}")

    # Load ALL candidate grasps for this object (full 2000, not the
    # training-time subsample) -- this is what real inference would face:
    # rank a large candidate pool, not a pre-filtered set.
    transforms, success = load_grasps(h5_path)
    mesh = load_mesh(h5_path, mesh_root_dir=MESH_ROOT)
    points = sample_point_cloud(mesh, NUM_POINTS)

    # --- run the trained model on all candidates ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GraspRankingNet().to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    points_t = torch.from_numpy(points).unsqueeze(0).to(device)          # (1, N, 3)
    poses_t = torch.from_numpy(transforms.astype(np.float32)).unsqueeze(0).to(device)  # (1, 2000, 4, 4)

    with torch.no_grad():
        success_prob, stability = model(points_t, poses_t)
        score = (success_prob * stability).squeeze(0).cpu().numpy()      # (2000,)

    # --- rank and pick top-K predicted grasps ---
    top_idx = np.argsort(-score)[:TOP_K]

    print(f"\nTop-{TOP_K} predicted grasps:")
    print(f"  predicted scores: min={score[top_idx].min():.3f}, max={score[top_idx].max():.3f}")

    # honest ground-truth check: of the grasps the model ranked highest,
    # how many were actually successful in the physics simulation?
    actual_success = success[top_idx]
    hit_rate = actual_success.mean()
    print(f"  ACTUAL ground-truth success rate among top-{TOP_K}: {hit_rate:.1%} "
          f"({int(actual_success.sum())}/{TOP_K})")

    # compare against random baseline for context
    baseline_rate = success.mean()
    print(f"  (baseline: overall success rate across all {len(success)} candidates "
          f"for this object = {baseline_rate:.1%})")

    # --- visualize using the OFFICIAL gripper geometry, not our own axis markers ---
    scene = trimesh.Scene()
    mesh_vis = mesh.dump(concatenate=True) if isinstance(mesh, trimesh.Scene) else mesh.copy()
    mesh_vis.visual.face_colors = [200, 200, 200, 150]
    scene.add_geometry(mesh_vis, geom_name="object_mesh")

    for rank, idx in enumerate(top_idx):
        color = [0, 255, 0] if success[idx] == 1 else [255, 0, 0]  # RGB, not RGBA -- create_gripper_marker expects 3-value color
        gripper_marker = create_gripper_marker(color=color)
        gripper_marker.apply_transform(transforms[idx])
        # explicit unique geom_name -- without this, trimesh can silently
        # collide/overwrite default auto-generated names across repeated
        # add_geometry() calls, leaving most markers missing from the scene
        # even though no exception is raised.
        scene.add_geometry(gripper_marker, geom_name=f"grasp_{rank}")

    print(f"\nScene now contains {len(scene.geometry)} geometries "
          f"(expected {TOP_K + 1}: 1 mesh + {TOP_K} grippers).")
    if len(scene.geometry) != TOP_K + 1:
        print("[WARN] geometry count mismatch -- some markers were likely "
              "overwritten or failed to add.")

    # Debug: print mesh bounds vs. marker positions so we can see whether
    # markers are actually near the object or positioned far outside camera
    # framing for some reason.
    print(f"\nDEBUG -- mesh_vis bounds: {mesh_vis.bounds}")
    print(f"DEBUG -- mesh_vis extents: {mesh_vis.extents}")
    for rank, idx in enumerate(top_idx[:5]):
        translation = transforms[idx][:3, 3]
        print(f"DEBUG -- grasp {rank} (idx={idx}) translation: {translation}")

    print(f"\nDEBUG -- scene bounds (all geometry combined): {scene.bounds}")
    print(f"DEBUG -- scene extents: {scene.extents}")
    print(f"DEBUG -- scene camera transform:\n{scene.camera_transform}")

    print(f"\nShowing top-{TOP_K} predicted grasps -- GREEN = actually successful "
          f"in ground truth, RED = actually failed (model ranked it high anyway).")
    scene.show()


if __name__ == "__main__":
    main()