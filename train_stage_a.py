"""
Stage A training loop: connects AcronymGraspDataset -> GraspRankingNet.

Usage:
    python train_stage_a.py

Adjust paths / hyperparameters in the CONFIG block below before running.
"""

import torch
from torch.utils.data import DataLoader, random_split

from stage_a_dataset import AcronymGraspDataset
from stage_a_model import GraspRankingNet, grasp_ranking_loss

# ---- CONFIG ----
SHORTLIST_CSV = "acronym_shortlist.csv"
MESH_ROOT = "acronym_mesh_root"
GRASP_DIR = "dataset/grasps"

NUM_POINTS = 2048
GRASPS_PER_OBJECT = 64
BATCH_SIZE = 4          # small default -- point cloud encoding is memory-heavy;
                         # raise if your 4050's VRAM allows, lower if you hit OOM
NUM_EPOCHS = 30
LR = 1e-4
VAL_FRACTION = 0.15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/stage_a_best.pth"


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for points, poses, quality in loader:
            points, poses, quality = points.to(device), poses.to(device), quality.to(device)
            success_label = (quality > 0).float()  # quality==0 iff grasp failed
            success_prob, stability = model(points, poses)
            loss, _, _ = grasp_ranking_loss(success_prob, stability, success_label, quality)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    import os
    os.makedirs("checkpoints", exist_ok=True)

    print(f"Using device: {DEVICE}")

    full_dataset = AcronymGraspDataset(
        shortlist_csv=SHORTLIST_CSV,
        mesh_root=MESH_ROOT,
        grasp_dir=GRASP_DIR,
        num_points=NUM_POINTS,
        grasps_per_object=GRASPS_PER_OBJECT,
    )

    val_size = int(len(full_dataset) * VAL_FRACTION)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")

    # num_workers=0 on Windows avoids common multiprocessing pickling issues
    # with h5py file handles -- raise later if you confirm it's stable.
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = GraspRankingNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for points, poses, quality in train_loader:
            points, poses, quality = points.to(DEVICE), poses.to(DEVICE), quality.to(DEVICE)
            success_label = (quality > 0).float()

            optimizer.zero_grad()
            success_prob, stability = model(points, poses)
            loss, success_loss, stability_loss = grasp_ranking_loss(
                success_prob, stability, success_label, quality
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        val_loss = evaluate(model, val_loader, DEVICE)

        print(f"Epoch {epoch:3d}/{NUM_EPOCHS} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"  -> saved new best checkpoint (val_loss={val_loss:.4f})")

    print(f"\nTraining complete. Best val_loss={best_val_loss:.4f}, checkpoint at {CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
