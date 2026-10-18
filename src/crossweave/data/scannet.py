from collections.abc import Callable, Iterator
import csv
import json
from pathlib import Path
import struct
from typing import BinaryIO
import zipfile

import cv2
import numpy as np
from PIL import Image
from plyfile import PlyData
import trimesh

from crossweave.data.geometry import scale_intrinsics

SENS_VERSION = 4
SENS_COLOR_JPEG = 2  # colour compression id in the .sens header


def scene_ids(root: Path, split: str) -> list[str]:
    lines = (Path(root) / "splits" / f"scannetv2_{split}.txt").read_text().splitlines()
    return sorted(line.strip() for line in lines if line.strip())


def scene_dir(root: Path, scene_id: str) -> Path:
    return Path(root) / "v2" / "scans" / scene_id


def read_points(root: Path, scene_id: str) -> tuple[np.ndarray, np.ndarray]:
    path = scene_dir(root, scene_id) / f"{scene_id}_vh_clean_2.ply"
    mesh = trimesh.load(path, process=False, force="mesh")
    coord = np.asarray(mesh.vertices, dtype=np.float32)  # (N, 3)
    normal = np.asarray(mesh.vertex_normals, dtype=np.float32)  # (N, 3)
    num_vertices = PlyData.read(str(path))["vertex"].count
    assert coord.shape == normal.shape == (num_vertices, 3), (coord.shape, num_vertices)
    return coord, normal


def read_sens_header(f: BinaryIO) -> dict:
    (version,) = struct.unpack("I", f.read(4))
    assert version == SENS_VERSION, f"unsupported .sens version {version}"
    (strlen,) = struct.unpack("Q", f.read(8))
    f.read(strlen)  # sensor name
    intrinsic_color = np.frombuffer(f.read(64), dtype=np.float32).reshape(4, 4).copy()
    f.read(3 * 64)  # extrinsic_color, intrinsic_depth, extrinsic_depth: unused
    color_compression, _depth_compression = struct.unpack("ii", f.read(8))
    assert color_compression == SENS_COLOR_JPEG, (
        f"colour is not JPEG ({color_compression})"
    )
    color_width, color_height, _depth_width, _depth_height = struct.unpack(
        "IIII", f.read(16)
    )
    f.read(4)  # depth shift: unused
    (num_frames,) = struct.unpack("Q", f.read(8))
    return {
        "intrinsic_color": intrinsic_color,
        "color_width": color_width,
        "color_height": color_height,
        "num_frames": num_frames,
    }


def iter_sens_frames(
    f: BinaryIO, num_frames: int, keep_color: Callable[[int], bool]
) -> Iterator[tuple[int, np.ndarray, bytes | None]]:
    for i in range(num_frames):
        pose_bytes = f.read(64)
        assert len(pose_bytes) == 64, f"truncated .sens at frame {i}"
        camera_to_world = np.frombuffer(pose_bytes, dtype=np.float32).reshape(4, 4).copy()
        f.read(16)  # colour + depth timestamps: unused
        color_size, depth_size = struct.unpack("QQ", f.read(16))
        if keep_color(i):
            color = f.read(color_size)
            assert len(color) == color_size, f"truncated .sens at frame {i}"
        else:
            f.seek(color_size, 1)
            color = None
        f.seek(depth_size, 1)
        yield i, camera_to_world, color


def unpack_scene(root: Path, scene_id: str, split: str, rgb_stride: int = 30) -> dict:
    assert split in ("train", "val"), split
    d = scene_dir(root, scene_id)
    done_file = d / "unpacked.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

    def keep_color(i: int) -> bool:
        return split == "val" or i % rgb_stride == 0

    color_dir = d / "color"
    color_dir.mkdir(exist_ok=True)
    num_color = 0
    with open(d / f"{scene_id}.sens", "rb") as f:
        header = read_sens_header(f)
        num_frames = header["num_frames"]
        poses = np.empty((num_frames, 4, 4), dtype=np.float32)  # (N, 4, 4)
        for i, camera_to_world, color in iter_sens_frames(f, num_frames, keep_color):
            poses[i] = camera_to_world
            if color is not None:
                (color_dir / f"{i}.jpg").write_bytes(color)
                num_color += 1

    np.save(d / "pose.npy", poses)
    np.savetxt(d / "intrinsic_color.txt", header["intrinsic_color"], fmt="%f")

    num_masks = 0
    if split == "val":
        with zipfile.ZipFile(d / f"{scene_id}_2d-instance-filt.zip") as z:
            members = [m for m in z.namelist() if m.endswith(".png")]
            assert all(m.startswith("instance-filt/") for m in members), (
                "unexpected zip layout"
            )
            z.extractall(d, members=members)
            num_masks = len(members)

    info = {
        "scene_id": scene_id,
        "split": split,
        "num_frames": num_frames,
        "color_size": [header["color_width"], header["color_height"]],
        "rgb_stride": 1 if split == "val" else rgb_stride,
        "num_color": num_color,
        "num_nonfinite_poses": int((~np.isfinite(poses).all(axis=(1, 2))).sum()),
        "num_instance_masks": num_masks,
    }
    done_file.write_text(json.dumps(info, indent=2))
    return info


def mesh_path(root: Path, scene_id: str) -> Path:
    return scene_dir(root, scene_id) / f"{scene_id}_vh_clean_2.ply"


def frame_ids(root: Path, scene_id: str) -> list[int]:
    return list(
        range(len(np.load(scene_dir(root, scene_id) / "pose.npy", mmap_mode="r")))
    )


def native_camera(root: Path, scene_id: str) -> tuple[np.ndarray, tuple[int, int]]:
    d = scene_dir(root, scene_id)
    K = np.loadtxt(d / "intrinsic_color.txt")[:3, :3]
    width, height = json.loads((d / "unpacked.json").read_text())["color_size"]
    return K, (width, height)


def pose(root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    return np.array(
        np.load(scene_dir(root, scene_id) / "pose.npy", mmap_mode="r")[frame_id]
    )


def image_intrinsics(K: np.ndarray, wh: tuple[int, int], img_size: int) -> np.ndarray:
    return scale_intrinsics(K, wh, (img_size, img_size))


def image_depth(depth_mm: np.ndarray, img_size: int) -> np.ndarray:
    depth = cv2.resize(depth_mm, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32) / 1000.0


# --- Instance labels and masks


def load_label_map(root: Path) -> dict[str, tuple[int, str]]:
    with open(Path(root) / "v2" / "scannetv2-labels.combined.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    return {
        row["raw_category"].strip().lower(): (
            int(row["nyu40id"]) if row["nyu40id"] else 0,
            row["nyu40class"],
        )
        for row in rows
    }


def vertex_labels(
    root: Path, scene_id: str, label_map: dict[str, tuple[int, str]]
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    d = scene_dir(root, scene_id)
    seg_idx = np.array(
        json.loads((d / f"{scene_id}_vh_clean_2.0.010000.segs.json").read_text())[
            "segIndices"
        ],
        dtype=np.int64,
    )  # (N,) segment id per vertex
    num_vertices = PlyData.read(str(mesh_path(root, scene_id)))["vertex"].count
    assert len(seg_idx) == num_vertices, (len(seg_idx), num_vertices)

    groups = json.loads((d / f"{scene_id}_vh_clean.aggregation.json").read_text())[
        "segGroups"
    ]
    seg_to_object = np.zeros(seg_idx.max() + 1, dtype=np.int32)
    seg_to_nyu40 = np.zeros(seg_idx.max() + 1, dtype=np.uint8)
    names = {}
    for group in groups:
        object_id = group["objectId"] + 1
        nyu40, name = label_map[group["label"].strip().lower()]
        segments = np.array(group["segments"], dtype=np.int64)
        seg_to_object[segments] = object_id
        seg_to_nyu40[segments] = nyu40
        names[object_id] = name
    return seg_to_object[seg_idx], seg_to_nyu40[seg_idx], names


def mask_frame_ids(root: Path, scene_id: str) -> list[int]:
    return sorted(
        int(p.stem) for p in (scene_dir(root, scene_id) / "instance-filt").glob("*.png")
    )


def instance_mask(root: Path, scene_id: str, frame_id: int) -> np.ndarray:
    return np.array(
        Image.open(scene_dir(root, scene_id) / "instance-filt" / f"{frame_id}.png")
    )


def read_rgb(root: Path, scene_id: str, frame_id: int, img_size: int) -> np.ndarray:
    path = scene_dir(root, scene_id) / "color" / f"{frame_id}.jpg"
    bgr = cv2.imread(str(path))
    assert bgr is not None, path
    return cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (img_size, img_size))


def read_instance_mask(
    root: Path,
    cache_dir: Path,
    scene_id: str,
    frame_id: int,
    object_id: int,
    img_size: int,
) -> np.ndarray:
    mask = (instance_mask(root, scene_id, frame_id) == object_id).astype(np.uint8)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.float32)
