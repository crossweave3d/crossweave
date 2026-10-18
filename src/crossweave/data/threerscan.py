import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
from plyfile import PlyData


def scene_ids(root: Path, split: str) -> list[str]:
    lines = (Path(root) / "files" / f"{split}_scans.txt").read_text().splitlines()
    return sorted(line.strip() for line in lines if line.strip())


def scene_dir(root: Path, scan_id: str) -> Path:
    return Path(root) / "data" / scan_id


def unpack_scene(root: Path, scan_id: str) -> dict:
    # extract sequence.zip
    d = scene_dir(root, scan_id)
    seq_dir = d / "sequence"
    with zipfile.ZipFile(d / "sequence.zip") as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        assert all("/" not in m for m in members), f"unexpected zip layout in {scan_id}"
        present = {p.name for p in seq_dir.iterdir()} if seq_dir.exists() else set()
        extracted = not set(members) <= present
        if extracted:
            z.extractall(seq_dir, members=members)
    missing = [m for m in members if not (seq_dir / m).exists()]
    assert not missing, (
        f"{scan_id}: {len(missing)} files missing after unzip, e.g. {missing[0]}"
    )
    return {
        "scan_id": scan_id,
        "num_files": len(members),
        "num_frames": sum(m.endswith(".color.jpg") for m in members),
        "extracted": extracted,
    }


def read_points(root: Path, scan_id: str) -> tuple[np.ndarray, np.ndarray]:
    d = scene_dir(root, scan_id)
    v_lines, vn_lines = [], []
    with open(d / "mesh.refined.v2.obj") as f:
        for line in f:
            if line.startswith("v "):
                v_lines.append(line[2:])
            elif line.startswith("vn "):
                vn_lines.append(line[3:])
    coord = np.loadtxt(v_lines, dtype=np.float32, ndmin=2)  # (N, 3)
    normal = np.loadtxt(vn_lines, dtype=np.float32, ndmin=2)  # (N, 3)
    assert coord.shape == normal.shape and coord.shape[1] == 3, (
        coord.shape,
        normal.shape,
    )

    label_ply = d / "labels.instances.annotated.v2.ply"
    if label_ply.exists():
        v = PlyData.read(str(label_ply))["vertex"]
        ply_xyz = np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)  # (N, 3)
        assert ply_xyz.shape == coord.shape, (
            f"{scan_id}: {ply_xyz.shape} vs {coord.shape}"
        )
        assert np.abs(ply_xyz - coord).max() < 1e-4, (
            f"{scan_id}: OBJ and label ply vertices differ"
        )
    return coord, normal


def mesh_path(root: Path, scan_id: str) -> Path:
    return scene_dir(root, scan_id) / "mesh.refined.v2.obj"


def frame_ids(root: Path, scan_id: str) -> list[int]:
    seq_dir = scene_dir(root, scan_id) / "sequence"
    return sorted(
        int(p.name.split(".")[0].removeprefix("frame-"))
        for p in seq_dir.glob("frame-*.color.jpg")
    )


def parse_calibration_file(path: Path) -> dict:
    info = {}
    with open(path) as f:
        for line in f:
            key, _, *values = line.strip().split()
            if key == "m_sensorName":
                continue
            if len(values) == 1:
                # scalar: try int, then float, then string
                try:
                    info[key] = int(values[0])
                except ValueError:
                    try:
                        info[key] = float(values[0])
                    except ValueError:
                        info[key] = values[0]
            else:
                info[key] = np.array(values, dtype=np.float64)
    return info


def native_camera(root: Path, scan_id: str) -> tuple[np.ndarray, tuple[int, int]]:
    info = parse_calibration_file(scene_dir(root, scan_id) / "sequence" / "_info.txt")
    K = info["m_calibrationColorIntrinsic"].reshape(4, 4)[:3, :3]
    return K, (info["m_colorWidth"], info["m_colorHeight"])


def pose(root: Path, scan_id: str, frame_id: int) -> np.ndarray:
    return np.loadtxt(
        scene_dir(root, scan_id) / "sequence" / f"frame-{frame_id:06d}.pose.txt"
    )


def rotate_intrinsics(K: np.ndarray, curr_W: int, scale_x: float, scale_y: float):
    # 3RScan images are rotated 90, this is to maintain correction projection
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    new_K = np.array([[fy, 0, curr_W - cy], [0, fx, cx], [0, 0, 1]])
    R_z = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    new_K = new_K * np.array([[scale_x], [scale_y], [1]])
    return new_K, R_z


def image_intrinsics(K: np.ndarray, wh: tuple[int, int], img_size: int) -> np.ndarray:
    rotated_W, rotated_H = wh[1], wh[0]  # rotation swaps width and height
    K_new, R_z = rotate_intrinsics(
        K, rotated_W, img_size / rotated_W, img_size / rotated_H
    )
    return K_new @ R_z


def image_depth(depth_mm: np.ndarray, img_size: int) -> np.ndarray:
    depth = cv2.rotate(depth_mm, cv2.ROTATE_90_CLOCKWISE)
    depth = cv2.resize(depth, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return depth.astype(np.float32) / 1000.0


# --- Instance labels
def load_objects(root: Path) -> dict[str, dict[int, tuple[int, str]]]:
    scans = json.loads((Path(root) / "files" / "objects.json").read_text())["scans"]
    return {
        s["scan"]: {int(o["id"]): (int(o["nyu40"]), o["label"]) for o in s["objects"]}
        for s in scans
    }


def vertex_labels(
    root: Path, scan_id: str, objects: dict[str, dict[int, tuple[int, str]]]
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    v = PlyData.read(str(scene_dir(root, scan_id) / "labels.instances.annotated.v2.ply"))[
        "vertex"
    ]
    object_id = np.asarray(v["objectId"], dtype=np.int32)
    nyu40 = np.asarray(v["NYU40"], dtype=np.uint8)
    scan_objects = objects[scan_id]
    for oid in np.unique(object_id[object_id != 0]):
        vertex_nyu40 = np.unique(nyu40[object_id == oid])
        assert len(vertex_nyu40) == 1, (
            f"{scan_id}: object {oid} has mixed NYU40 {vertex_nyu40}"
        )
        assert int(vertex_nyu40[0]) == scan_objects[int(oid)][0], (
            f"{scan_id}: object {oid} NYU40 ply={vertex_nyu40[0]} objects.json={scan_objects[int(oid)][0]}"
        )
    return object_id, nyu40, {oid: label for oid, (_, label) in scan_objects.items()}


def label_mesh_path(root: Path, scan_id: str) -> Path:
    return scene_dir(root, scan_id) / "labels.instances.annotated.v2.ply"


def read_rgb(root: Path, scan_id: str, frame_id: int, img_size: int) -> np.ndarray:
    path = scene_dir(root, scan_id) / "sequence" / f"frame-{frame_id:06d}.color.jpg"
    bgr = cv2.imread(str(path))
    assert bgr is not None, path
    rgb = cv2.rotate(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.ROTATE_90_CLOCKWISE)
    return cv2.resize(rgb, (img_size, img_size))


def read_instance_mask(
    root: Path,
    cache_dir: Path,
    scan_id: str,
    frame_id: int,
    object_id: int,
    img_size: int,
) -> np.ndarray:
    path = Path(cache_dir) / "masks" / scan_id / f"{frame_id}.png"
    ids = cv2.imread(str(path), cv2.IMREAD_ANYDEPTH)
    assert ids is not None, path
    mask = cv2.rotate((ids == object_id).astype(np.uint8), cv2.ROTATE_90_CLOCKWISE)
    mask = cv2.resize(mask, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
    return mask.astype(np.float32)
