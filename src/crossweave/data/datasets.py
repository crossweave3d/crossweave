import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_cluster import fps

from crossweave.data import scannet, threerscan

DATASETS = {"scannet": scannet, "threerscan": threerscan}
# ImageNet normalisation, same as the old 0-255 form (123.675, ...) / 255
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_model_input(rgb: np.ndarray) -> torch.Tensor:
    # (H, W, 3) uint8 -> (3, H, W) float32
    x = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD


class FrameDataset(Dataset):
    # rgb (3, S, S), point_features (M, 1224), patch_idx (M,) [, coords (M, 3)]
    def __init__(
        self,
        dataset: str,
        root: Path,
        cache_dir: Path,
        split: str = "train",
        img_size: int = 518,
        patch_size: int = 14,
        num_patch_point_pairs: int = 128,
        enable_fps: bool = False,
        fps_ratio: float = 0.125,
        fps_space: str = "feature",
        subsample_points: bool = True,
        return_3d_points: bool = False,
    ):
        if split == "train" and not subsample_points:
            raise ValueError(
                "subsample_points must be True for training (uniform batch sizes)."
            )
        assert fps_space in ("feature", "geometric"), fps_space
        self.module = DATASETS[dataset]
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.img_size = img_size
        self.num_patch_point_pairs = num_patch_point_pairs
        self.enable_fps = enable_fps
        self.fps_ratio = fps_ratio
        self.fps_space = fps_space
        self.subsample_points = subsample_points
        self.return_3d_points = return_3d_points

        self.grid = f"frames_{img_size}_p{patch_size}"
        meta = json.loads((self.cache_dir / f"{split}_{self.grid}.json").read_text())
        assert meta["config"]["min_pairs"] >= num_patch_point_pairs, meta["config"]
        self.samples = [(s["scene_id"], s["frame_id"]) for s in meta["samples"]]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        scene_id, frame_id = self.samples[index]
        rgb = to_model_input(
            self.module.read_rgb(self.root, scene_id, frame_id, self.img_size)
        )

        data = torch.load(
            self.cache_dir / self.grid / scene_id / f"{frame_id}.pt", weights_only=True
        )
        point_features = data["features"].float()  # (P, 1224)
        patch_idx = data["patch_idx"].int()  # (P,)
        point_coords = data["coords"].float()  # (P, 3)

        # new random subset every epoch
        num_points = point_features.shape[0]
        if self.subsample_points and num_points > self.num_patch_point_pairs:
            sample_idx = torch.randperm(num_points)[: self.num_patch_point_pairs]
            point_features = point_features[sample_idx]
            patch_idx = patch_idx[sample_idx]
            point_coords = point_coords[sample_idx]

        if self.enable_fps:
            fps_src = point_features if self.fps_space == "feature" else point_coords
            fps_idx = fps(
                src=fps_src,
                batch=torch.zeros(point_features.shape[0], dtype=torch.long),
                ratio=self.fps_ratio,
                random_start=True,
            )
            point_features = point_features[fps_idx]
            patch_idx = patch_idx[fps_idx]
            point_coords = point_coords[fps_idx]

        if self.return_3d_points:
            return rgb, point_features, patch_idx, point_coords
        return rgb, point_features, patch_idx


class SceneDataset(Dataset):
    # rgbs (k, 3, S, S), point_feat (N, 1224) float16, scene_id
    def __init__(
        self,
        dataset: str,
        root: Path,
        cache_dir: Path,
        num_images: int,
        img_size: int = 518,
        split: str = "val",
    ):
        self.module = DATASETS[dataset]
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.num_images = num_images
        self.img_size = img_size
        scenes = json.loads((self.cache_dir / f"{split}_scene_frames.json").read_text())[
            "scenes"
        ]
        self.samples = sorted(scenes.items())

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        scene_id, order = self.samples[idx]
        frames = order[
            : 2 * self.num_images : 2
        ]  # sample 2k camera-grid poses, keep every 2nd
        rgbs = torch.stack(
            [
                to_model_input(
                    self.module.read_rgb(self.root, scene_id, f, self.img_size)
                )
                for f in frames
            ]
        )
        point_feat = torch.load(
            self.cache_dir / "scenes" / scene_id / "pointcloud.pt", weights_only=True
        )["feat"]
        return rgbs, point_feat, scene_id


class InstanceDataset(Dataset):
    # rgbs (n, 3, S, S), masks (n, S, S), point_feat (n_obj, 1224) float16, scene_id, object_id, nyu40, label
    def __init__(
        self,
        dataset: str,
        root: Path,
        cache_dir: Path,
        num_images: int,
        img_size: int = 518,
        split: str = "val",
    ):
        self.module = DATASETS[dataset]
        self.root = Path(root)
        self.cache_dir = Path(cache_dir)
        self.num_images = num_images
        self.img_size = img_size
        self.instances = json.loads(
            (self.cache_dir / f"{split}_instances.json").read_text()
        )["instances"]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        scene_id, object_id = inst["scene_id"], inst["object_id"]
        frames = [
            frame_id for frame_id, _area in inst["frames"][: self.num_images]
        ]  # most visible first
        rgbs = torch.stack(
            [
                to_model_input(
                    self.module.read_rgb(self.root, scene_id, f, self.img_size)
                )
                for f in frames
            ]
        )
        masks = torch.stack(
            [
                torch.from_numpy(
                    self.module.read_instance_mask(
                        self.root, self.cache_dir, scene_id, f, object_id, self.img_size
                    )
                )
                for f in frames
            ]
        )

        scene_cache = self.cache_dir / "scenes" / scene_id
        vertex_object_id = np.load(scene_cache / "labels.npz")["object_id"]  # (N,)
        rows = torch.from_numpy(np.flatnonzero(vertex_object_id == object_id))
        feat = torch.load(scene_cache / "pointcloud.pt", weights_only=True, mmap=True)[
            "feat"
        ]
        point_feat = feat[
            rows
        ].clone()  # (n_obj, 1224) float16; copy out of the memory map
        assert len(point_feat) == inst["num_vertices"], (scene_id, object_id)
        return rgbs, masks, point_feat, scene_id, object_id, inst["nyu40"], inst["label"]
