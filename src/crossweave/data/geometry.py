import numpy as np
from scipy.spatial import distance
from scipy.spatial.transform import Rotation
import torch


def scale_intrinsics(
    K: np.ndarray, src_wh: tuple[int, int], dst_wh: tuple[int, int]
) -> np.ndarray:
    assert K.shape == (3, 3), K.shape
    sx = dst_wh[0] / src_wh[0]
    sy = dst_wh[1] / src_wh[1]
    return K * np.array([[sx], [sy], [1.0]])


def project_points(
    coord: torch.Tensor,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    hw: tuple[int, int],
    depth: np.ndarray | None = None,
    depth_tol: float = 0.05,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    # coord (N, 3) -> pixels (N, 2) (u, v), valid (N,); valid = in front, in image, not occluded
    assert coord.ndim == 2 and coord.shape[1] == 3, coord.shape
    assert K.shape == (3, 3) and cam_to_world.shape == (4, 4), (
        K.shape,
        cam_to_world.shape,
    )
    H, W = hw
    P = (
        torch.from_numpy(K.astype(np.float64))
        @ torch.linalg.inv(torch.from_numpy(cam_to_world.astype(np.float64)))[:3, :]
    )  # (3, 4)
    P = P.to(device=device, dtype=torch.float32)

    coord = coord.to(device=device, dtype=torch.float32)
    coord_h = torch.cat(
        [coord, torch.ones(coord.shape[0], 1, device=device)], dim=1
    )  # (N, 4)
    pixels = (P @ coord_h.T).T  # (N, 3)
    z = pixels[:, 2]  # (N,) camera-space depth
    pixels = pixels[:, :2] / z[:, None]  # (N, 2)

    valid = (
        (z > 0)
        & (pixels[:, 0] >= 0)
        & (pixels[:, 0] < W)
        & (pixels[:, 1] >= 0)
        & (pixels[:, 1] < H)
    )
    if depth is not None:
        assert depth.shape == (H, W), (depth.shape, hw)
        idx = torch.where(valid)[0]
        u = pixels[idx, 0].long().clamp(0, W - 1)
        v = pixels[idx, 1].long().clamp(0, H - 1)
        surface = torch.as_tensor(depth, dtype=torch.float32, device=device)[v, u]  # (M,)
        valid[idx] &= (surface > 0) & (z[idx] <= surface + depth_tol)
    return pixels.cpu(), valid.cpu()


def patch_correspondences(
    pixels: torch.Tensor, img_size: int, patch_size: int, generator: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    # one uniformly random point per occupied patch: shuffle, stable-sort by patch, keep first.
    assert pixels.ndim == 2 and pixels.shape[1] == 2, pixels.shape
    assert img_size % patch_size == 0, (img_size, patch_size)
    pixels = pixels.cpu()
    n = img_size // patch_size
    col = (pixels[:, 0] // patch_size).clamp(0, n - 1).long()
    row = (pixels[:, 1] // patch_size).clamp(0, n - 1).long()
    patch_idx = row * n + col  # (M,)

    shuffle = torch.randperm(len(patch_idx), generator=generator)  # (M,)
    shuffled = patch_idx[shuffle]
    order = torch.argsort(shuffled, stable=True)
    sorted_idx = shuffled[order]
    first = torch.ones_like(sorted_idx, dtype=torch.bool)
    first[1:] = sorted_idx[1:] != sorted_idx[:-1]
    keep = shuffle[order[first]]  # (P,)
    return patch_idx[keep], keep


def patch_point_pairs(
    coord: torch.Tensor,
    K: np.ndarray,
    cam_to_world: np.ndarray,
    depth: np.ndarray,
    img_size: int,
    patch_size: int,
    generator: torch.Generator,
    depth_tol: float = 0.05,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    # -> point_idx (P,) into coord, patch_idx (P,)
    pixels, valid = project_points(
        coord, K, cam_to_world, (img_size, img_size), depth, depth_tol, device
    )
    valid_idx = torch.where(valid)[0]  # (M,)
    patch_idx, keep = patch_correspondences(
        pixels[valid], img_size, patch_size, generator
    )
    return valid_idx[keep], patch_idx


# --- Eval frame selection
def pose_matrix_to_txyz_quat(pose: np.ndarray) -> np.ndarray:
    rot = pose[:3, :3]
    trans = pose[:3, 3]
    quat = Rotation.from_matrix(rot).as_quat()
    return np.array([trans[0], trans[1], trans[2], quat[0], quat[1], quat[2], quat[3]])


def sample_camera_pos_on_grid(
    pose_data: np.ndarray, num_to_sample: int = 20
) -> list[int]:
    # farthest-point sampling over (K, 7) poses; fixed start (seed 0)
    min_vals = np.min(pose_data, axis=0)
    max_vals = np.max(pose_data, axis=0)
    pose_data_normalised = (pose_data - min_vals) / (max_vals - min_vals)

    rng = np.random.default_rng(0)
    initial_index = rng.integers(len(pose_data_normalised))
    sampled_indices = [initial_index]
    distances = distance.cdist(
        [pose_data_normalised[initial_index]], pose_data_normalised, "euclidean"
    ).flatten()

    for _ in range(num_to_sample - 1):
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        new_distances = distance.cdist(
            [pose_data_normalised[farthest_index]], pose_data_normalised, "euclidean"
        ).flatten()
        distances = np.minimum(distances, new_distances)

    return sampled_indices
