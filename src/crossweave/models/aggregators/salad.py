import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/optimal_transport.py
def log_otp_solver(
    log_a, log_b, M, num_iters: int = 20, reg: float = 1.0
) -> torch.Tensor:
    r"""Sinkhorn matrix scaling algorithm for Differentiable Optimal Transport problem.
    This function solves the optimization problem and returns the OT matrix for the given parameters.
    Args:
        log_a : torch.Tensor
            Source weights
        log_b : torch.Tensor
            Target weights
        M : torch.Tensor
            metric cost matrix
        num_iters : int, default=100
            The number of iterations.
        reg : float, default=1.0
            regularization value
    """
    M = M / reg  # regularization

    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)

    for _ in range(num_iters):
        u = log_a - torch.logsumexp(M + v.unsqueeze(1), dim=2).squeeze()
        v = log_b - torch.logsumexp(M + u.unsqueeze(2), dim=1).squeeze()

    return M + u.unsqueeze(2) + v.unsqueeze(1)


# Code adapted from OpenGlue, MIT license
# https://github.com/ucuapps/OpenGlue/blob/main/models/superglue/superglue.py
def get_matching_probs(S, dustbin_score=1.0, num_iters=3, reg=1.0):
    """sinkhorn"""
    batch_size, m, n = S.size()
    # augment scores matrix
    S_aug = torch.empty(batch_size, m + 1, n, dtype=S.dtype, device=S.device)
    S_aug[:, :m, :n] = S
    S_aug[:, m, :] = dustbin_score

    # prepare normalized source and target log-weights
    norm = -torch.tensor(math.log(n + m), device=S.device)
    log_a, log_b = norm.expand(m + 1).contiguous(), norm.expand(n).contiguous()
    log_a[-1] = log_a[-1] + math.log(n - m)
    log_a, log_b = log_a.expand(batch_size, -1), log_b.expand(batch_size, -1)
    log_P = log_otp_solver(log_a, log_b, S_aug, num_iters=num_iters, reg=reg)
    return log_P - norm


# Adapted from DINOv2 SALAD, GPL-3.0 License
# https://github.com/serizba/salad/blob/main/models/aggregators/salad.py
class SALAD(nn.Module):
    # patches and points share the same learned clusters (OT with a dustbin)
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_clusters: int = 64,
        cluster_dim: int = 512,
        dropout_prob: float = 0.3,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim

        dropout = nn.Dropout(dropout_prob) if dropout_prob > 0 else nn.Identity()

        self.cluster_features = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            dropout,
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.cluster_dim),
        )

        self.score = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            dropout,
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.num_clusters),
        )

        self.dustbin = nn.Parameter(torch.tensor(1.0))

    def point_forward(self, point_feat):
        f = self.cluster_features(point_feat)  # [B, N, D_Cluster]
        p = self.score(point_feat).transpose(1, 2)  # [B, N_Clusters, N]

        # Sinkhorn
        p = get_matching_probs(p, self.dustbin, 3)  # [B, N_Clusters+1, N]
        p = torch.exp(p)
        p = p[:, :-1, :]  # [B, N_Clusters, N]

        # Save assignment matrix for visualization
        self.last_point_assignment = p.detach()

        descriptor = torch.einsum("bmn,bnd->bmd", p, f)

        descriptor = F.normalize(descriptor, dim=-1)  # per-cluster
        descriptor = descriptor.flatten(start_dim=1)  # [B, K*D]
        descriptor = F.normalize(descriptor, dim=-1)  # global

        return descriptor  # [B, K * D_cluster]

    def patch_forward(self, patch_feat):
        f = self.cluster_features(patch_feat)
        p = self.score(patch_feat).transpose(1, 2)

        # Sinkhorn
        p = get_matching_probs(p, self.dustbin, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]

        # Save assignment matrix for visualization
        self.last_patch_assignment = p.detach()

        descriptor = torch.einsum("bmn,bnd->bmd", p, f)

        descriptor = F.normalize(descriptor, dim=-1)  # per-cluster
        descriptor = descriptor.flatten(start_dim=1)  # [B, K*D]
        descriptor = F.normalize(descriptor, dim=-1)  # global

        return descriptor  # [B, K * D_cluster]

    def forward(self, point_feat, patch_feat):
        point_desc = self.point_forward(point_feat)
        patch_desc = self.patch_forward(patch_feat)

        return point_desc, patch_desc
