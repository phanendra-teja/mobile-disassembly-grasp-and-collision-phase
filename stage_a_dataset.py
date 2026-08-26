"""
Stage A dataloader: (point cloud, grasp pose, quality score) triples
for pretraining the grasp candidate ranking network on the filtered
ACRONYM subset.

Reuses acronym_tools for mesh/grasp loading (validated against the
official visualizer), and our own quality computation from
filter_acronym.py (success + inverse motion magnitude).

Usage:
    from stage_a_dataset import AcronymGraspDataset
    ds = AcronymGraspDataset(
        shortlist_csv="acronym_shortlist.csv",
        mesh_root="acronym_mesh_root",
        grasp_dir="dataset/grasps",
        num_points=2048,
        grasps_per_object=64,
    )
    pc, poses, quality = ds[0]
"""

import os
import sys

import h5py
import numpy as np
import pandas as pd
import torch
import trimesh
from torch.utils.data import Dataset

# acronym_tools must be imported after `pip install -e .` in the cloned
# NVlabs/acronym repo (see restructure_for_acronym.py / earlier setup).
try:
    from acronym_tools import load_mesh, load_grasps
except ImportError:
    raise ImportError(
        "acronym_tools not found. Run `pip install -e .` from inside the "
        "cloned acronym/ repo before using this dataset."
    )


def compute_quality(success, closing_lin, closing_ang, shaking_lin, shaking_ang):
    """Same definition as filter_acronym.py -- kept identical so quality
    labels here match what was reported during dataset filtering."""
    motion_penalty = closing_lin + closing_ang + shaking_lin + shaking_ang
    stability = 1.0 / (1.0 + motion_penalty)
    return success.astype(np.float64) * stability


class AcronymGraspDataset(Dataset):
    """
    One item = one object (mesh + all its grasps), not one grasp --
    this matches how Stage A will actually be used at inference time
    (sample many candidate poses per object, rank them together).

    Returns:
        points:  (num_points, 3) float32 tensor -- sampled point cloud
        poses:   (grasps_per_object, 4, 4) float32 tensor -- grasp transforms
        quality: (grasps_per_object,) float32 tensor -- combined quality score
    """

    def __init__(
        self,
        shortlist_csv,
        mesh_root,
        grasp_dir,
        num_points=2048,
        grasps_per_object=64,
        seed=0,
    ):
        self.shortlist = pd.read_csv(shortlist_csv).drop_duplicates(subset="model_id").reset_index(drop=True)
        self.mesh_root = mesh_root
        self.grasp_dir = grasp_dir
        self.num_points = num_points
        self.grasps_per_object = grasps_per_object
        self.rng = np.random.default_rng(seed)

        # Basic existence check up front so failures surface at construction
        # time, not silently mid-training on a random batch.
        self._validate_files()

    def _validate_files(self):
        missing = []
        for _, row in self.shortlist.iterrows():
            h5_path = os.path.join(self.grasp_dir, row["filename"])
            if not os.path.exists(h5_path):
                missing.append(h5_path)
        if missing:
            print(f"[WARN] {len(missing)} grasp files listed in shortlist "
                  f"are missing on disk. First few:")
            for m in missing[:5]:
                print(f"  {m}")
            print("These rows will raise errors if sampled -- consider "
                  "filtering them out of the shortlist CSV.")

    def __len__(self):
        return len(self.shortlist)

    def _sample_point_cloud(self, mesh):
        """Uniform surface sampling -- matches what the real segmentation
        pipeline's PointNet++ input expects (surface points, not just
        vertices, so density is independent of mesh tessellation).

        trimesh.load() can return a Scene instead of a single Trimesh when
        the .obj has multiple material groups (common in ShapeNetSem) --
        Scene has no .sample(), so merge into one mesh first when needed.
        """
        if isinstance(mesh, trimesh.Scene):
            if len(mesh.geometry) == 0:
                raise ValueError("Loaded Scene has no geometry to sample from.")
            mesh = trimesh.util.concatenate(
                [g for g in mesh.geometry.values()]
            )
        points, _ = mesh.sample(self.num_points, return_index=True)
        return points.astype(np.float32)

    def __getitem__(self, idx):
        row = self.shortlist.iloc[idx]
        h5_path = os.path.join(self.grasp_dir, row["filename"])

        # load_grasps/load_mesh from acronym_tools take a filename string
        # and open the .h5 internally -- do NOT pre-open the file ourselves
        # and pass the h5py.File object (that was the earlier bug: their
        # functions call filename.endswith(...), which only works on a str).
        transforms, success = load_grasps(h5_path)

        # load_grasps doesn't expose the motion-during-closing/shaking
        # fields (only transforms + success), so we still read those
        # ourselves directly for the quality score.
        with h5py.File(h5_path, "r") as f:
            closing_lin = f["grasps/qualities/flex/object_motion_during_closing_linear"][:]
            closing_ang = f["grasps/qualities/flex/object_motion_during_closing_angular"][:]
            shaking_lin = f["grasps/qualities/flex/object_motion_during_shaking_linear"][:]
            shaking_ang = f["grasps/qualities/flex/object_motion_during_shaking_angular"][:]

        quality = compute_quality(success, closing_lin, closing_ang, shaking_lin, shaking_ang)

        # load_mesh handles object/file path resolution + scale internally
        # (validated earlier against the official acronym_visualize_grasps.py).
        mesh = load_mesh(h5_path, mesh_root_dir=self.mesh_root)

        points = self._sample_point_cloud(mesh)

        # Sample a fixed number of grasps per object, biased toward
        # including high-quality ones so rare good grasps aren't
        # drowned out by the majority of low/zero-quality samples.
        n_total = len(quality)
        k = min(self.grasps_per_object, n_total)

        # weight sampling: successful grasps get much higher weight so
        # a batch isn't dominated by failures on objects with low success rate
        weights = quality + 0.01  # small floor so failed grasps aren't impossible to sample
        weights = weights / weights.sum()
        chosen_idx = self.rng.choice(n_total, size=k, replace=False, p=weights)

        sel_poses = transforms[chosen_idx].astype(np.float32)
        sel_quality = quality[chosen_idx].astype(np.float32)

        return (
            torch.from_numpy(points),
            torch.from_numpy(sel_poses),
            torch.from_numpy(sel_quality),
        )


if __name__ == "__main__":
    # Quick smoke test -- run this file directly to sanity-check the
    # dataset loads without errors before wiring it into a training loop.
    ds = AcronymGraspDataset(
        shortlist_csv="acronym_shortlist.csv",
        mesh_root="acronym_mesh_root",
        grasp_dir="dataset/grasps",
        num_points=2048,
        grasps_per_object=64,
    )
    print(f"Dataset size: {len(ds)}")
    points, poses, quality = ds[0]
    print(f"points: {points.shape}, dtype={points.dtype}")
    print(f"poses: {poses.shape}, dtype={poses.dtype}")
    print(f"quality: {quality.shape}, min={quality.min():.3f}, max={quality.max():.3f}")