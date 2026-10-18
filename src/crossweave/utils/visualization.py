import math

import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_similarity_matrix(
    sim_matrix: torch.Tensor, rgb: torch.Tensor, point_pixel: torch.Tensor
):
    # sim_matrix (N, N), rgb (B, 3, H, W) ImageNet-normalised, point_pixel (B, 2) or (B, P, 2) as (u, v)
    sim_matrix_np = sim_matrix.detach().cpu().numpy()
    rgb_np = rgb.detach().cpu().numpy()
    point_pixel_np = point_pixel.detach().cpu().numpy()

    B = rgb_np.shape[0]

    grid_cols = math.ceil(math.sqrt(B))
    grid_rows = math.ceil(B / grid_cols)

    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
    rgb_denorm = rgb_np * std + mean
    rgb_denorm = np.clip(rgb_denorm, 0, 1)

    fig = plt.figure(figsize=(4 * grid_cols + 10, max(4 * grid_rows, 8)), dpi=150)

    # images on the left, similarity matrix on the right
    gs = fig.add_gridspec(
        grid_rows, grid_cols + 1, width_ratios=[1] * grid_cols + [max(4, grid_cols * 1.2)]
    )

    for i in range(B):
        r, c = divmod(i, grid_cols)
        ax = fig.add_subplot(gs[r, c])

        img = np.transpose(rgb_denorm[i], (1, 2, 0))
        ax.imshow(img)

        pts = point_pixel_np[i]
        if pts.ndim == 1:
            pts = pts[None, :]  # (1, 2)

        ax.scatter(pts[:, 0], pts[:, 1], c="red", s=20, marker="x", linewidths=1.5)

        ax.axis("off")

    ax_sim = fig.add_subplot(gs[:, -1])
    im = ax_sim.imshow(sim_matrix_np, cmap="turbo", aspect="equal", vmin=-1.0, vmax=1.0)
    ax_sim.set_title("Similarity Matrix")
    fig.colorbar(im, ax=ax_sim, fraction=0.046, pad=0.04)

    fig.subplots_adjust(
        left=0.02, right=0.95, top=0.95, bottom=0.05, wspace=0.05, hspace=0.05
    )

    return fig


def visualize_global_sim_matrix(sim_matrix: torch.Tensor):
    sim_matrix_np = sim_matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)
    im = ax.imshow(sim_matrix_np, cmap="turbo", aspect="equal", vmin=-1.0, vmax=1.0)
    ax.set_title("Global Similarity Matrix (B x B)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def visualise_ot_matrix(ot_matrix: torch.Tensor, title: str):
    ot_matrix_np = ot_matrix.detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    im = ax.imshow(ot_matrix_np, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_ylabel("Clusters (M)")
    ax.set_xlabel("Tokens (N)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig
