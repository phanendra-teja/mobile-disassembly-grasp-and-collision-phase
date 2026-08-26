"""
Stage A model: grasp candidate ranking network.

Architecture:
  1. PointNet++-style encoder over the object's point cloud -> global
     shape feature vector (shared across all candidate poses for that object,
     computed once per object, not once per pose -- cheaper at inference).
  2. Each candidate pose (4x4 transform) is converted to a compact
     translation + 6D rotation representation (Zhou et al. 2019 -- avoids
     quaternion double-cover and gimbal-lock issues that a raw rotation
     matrix flatten or Euler angles would introduce).
  3. Pose encoding is concatenated with the global shape feature and passed
     through an MLP head with two output branches:
       - success_prob: sigmoid, trained against binary object_in_gripper
       - stability: sigmoid, trained against inverse-motion-magnitude score
  Final rank score at inference = success_prob * stability (matches the
  combined quality label definition used during dataset filtering).

This is a simplified PointNet-style encoder (shared MLP + max-pool), not a
full hierarchical PointNet++ with set abstraction layers -- swap in a real
PointNet++ backbone (e.g. from an existing implementation) once this
simplified version is confirmed to train correctly end-to-end; don't
debug two new things (data pipeline + full PointNet++) at once.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotation_matrix_to_6d(rot_matrix):
    """
    Convert a (..., 3, 3) rotation matrix to the 6D representation from
    'On the Continuity of Rotation Representations in Neural Networks'
    (Zhou et al., CVPR 2019) -- just the first two columns of the matrix,
    flattened. Avoids discontinuities that hurt regression/interpolation
    compared to quaternions or Euler angles.
    """
    return rot_matrix[..., :, :2].reshape(*rot_matrix.shape[:-2], 6)


def pose_to_features(transforms):
    """
    transforms: (..., 4, 4) homogeneous transforms
    returns: (..., 9) feature vector = [translation(3), rotation_6d(6)]
    """
    translation = transforms[..., :3, 3]
    rotation = transforms[..., :3, :3]
    rot6d = rotation_matrix_to_6d(rotation)
    return torch.cat([translation, rot6d], dim=-1)


class PointCloudEncoder(nn.Module):
    """
    Simplified PointNet-style encoder: shared per-point MLP + global max-pool.
    Input: (B, N, 3) point cloud
    Output: (B, feature_dim) global shape feature
    """

    def __init__(self, feature_dim=256):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Linear(3, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, feature_dim),
        )

    def forward(self, points):
        # points: (B, N, 3)
        x = self.mlp1(points)          # (B, N, feature_dim)
        x, _ = torch.max(x, dim=1)     # (B, feature_dim) -- global max-pool
        return x


class GraspRankingNet(nn.Module):
    """
    Full Stage A model: point cloud + candidate poses -> (success_prob, stability)
    per pose.

    Input:
        points: (B, N, 3)
        poses:  (B, K, 4, 4)  -- K candidate poses per object in the batch
    Output:
        success_prob: (B, K) in [0, 1]
        stability:    (B, K) in [0, 1]
    """

    def __init__(self, point_feature_dim=256, pose_feature_dim=9, hidden_dim=256):
        super().__init__()
        self.encoder = PointCloudEncoder(feature_dim=point_feature_dim)

        self.head = nn.Sequential(
            nn.Linear(point_feature_dim + pose_feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.success_head = nn.Linear(hidden_dim, 1)
        self.stability_head = nn.Linear(hidden_dim, 1)

    def forward(self, points, poses):
        B, K = poses.shape[0], poses.shape[1]

        shape_feat = self.encoder(points)              # (B, point_feature_dim)
        pose_feat = pose_to_features(poses)             # (B, K, 9)

        # broadcast shape feature across all K candidate poses for this object
        shape_feat_expanded = shape_feat.unsqueeze(1).expand(-1, K, -1)  # (B, K, point_feature_dim)

        combined = torch.cat([shape_feat_expanded, pose_feat], dim=-1)   # (B, K, point_feature_dim + 9)
        h = self.head(combined)                          # (B, K, hidden_dim)

        success_prob = torch.sigmoid(self.success_head(h)).squeeze(-1)   # (B, K)
        stability = torch.sigmoid(self.stability_head(h)).squeeze(-1)    # (B, K)

        return success_prob, stability

    def rank_score(self, points, poses):
        """Convenience method: returns the final combined ranking score."""
        success_prob, stability = self.forward(points, poses)
        return success_prob * stability


def grasp_ranking_loss(success_prob, stability, success_label, quality_label):
    """
    success_label: (B, K) binary ground truth (object_in_gripper)
    quality_label: (B, K) combined quality score (success * inverse-motion),
                   as computed in filter_acronym.py / stage_a_dataset.py

    Two loss terms:
      - BCE on success probability against the binary label
      - MSE on stability against (quality_label / success_label), i.e. the
        "stability given success" component, masked to only successful
        grasps (stability is undefined/meaningless for failed grasps).
    """
    success_loss = F.binary_cross_entropy(success_prob, success_label.float())

    # avoid divide-by-zero for failed grasps; mask them out of the stability loss
    mask = success_label > 0.5
    if mask.sum() > 0:
        # quality_label already = success * stability, so for successful
        # grasps (success=1), quality_label == stability directly.
        stability_loss = F.mse_loss(stability[mask], quality_label[mask])
    else:
        stability_loss = torch.tensor(0.0, device=success_prob.device)

    total_loss = success_loss + stability_loss
    return total_loss, success_loss, stability_loss


if __name__ == "__main__":
    # Quick smoke test with random data matching the dataset's output shapes
    B, N, K = 4, 2048, 64
    points = torch.randn(B, N, 3)
    poses = torch.eye(4).unsqueeze(0).unsqueeze(0).repeat(B, K, 1, 1)
    poses[:, :, :3, 3] = torch.randn(B, K, 3) * 0.1  # random translations

    model = GraspRankingNet()
    success_prob, stability = model(points, poses)
    print(f"success_prob: {success_prob.shape}, range=({success_prob.min():.3f}, {success_prob.max():.3f})")
    print(f"stability: {stability.shape}, range=({stability.min():.3f}, {stability.max():.3f})")

    # fake labels for loss smoke test
    success_label = (torch.rand(B, K) > 0.3).float()
    quality_label = torch.rand(B, K) * success_label  # 0 where failed
    total, s_loss, st_loss = grasp_ranking_loss(success_prob, stability, success_label, quality_label)
    print(f"total_loss={total.item():.4f}, success_loss={s_loss.item():.4f}, stability_loss={st_loss.item():.4f}")
