"""Preprocessing, in order: unpack -> encode -> frames -> eval.

    python -m crossweave.data.preprocess unpack --dataset scannet --root /path/ScanNet
    python -m crossweave.data.preprocess encode --dataset scannet --root /path/ScanNet --cache_dir /path/cache/scannet
    python -m crossweave.data.preprocess frames --dataset scannet --root /path/ScanNet --cache_dir /path/cache/scannet
    python -m crossweave.data.preprocess eval   --dataset scannet --root /path/ScanNet --cache_dir /path/cache/scannet

Same for --dataset threerscan. Re-running skips scenes that are already done.
"""

import argparse
from functools import partial
import json
import multiprocessing
from multiprocessing import Pool
from pathlib import Path
import zlib

import cv2
from loguru import logger
import numpy as np
import torch
from tqdm import tqdm

from crossweave.data import scannet, threerscan
from crossweave.data.geometry import (
    patch_point_pairs,
    pose_matrix_to_txyz_quat,
    sample_camera_pos_on_grid,
)

DATASETS = {"scannet": scannet, "threerscan": threerscan}
FRAME_STRIDE = {"scannet": 30, "threerscan": 5}  # training frames
NYU40_IGNORE = {0, 1, 2, 22, 38}  # unannotated, wall, floor, ceiling, otherstructure
MIN_INSTANCE_VERTICES = 50


def unpack(
    dataset: str, root: Path, split: str, num_workers: int, rgb_stride: int
) -> None:
    ids = DATASETS[dataset].scene_ids(root, split)
    if dataset == "scannet":
        worker = partial(scannet.unpack_scene, root, split=split, rgb_stride=rgb_stride)
    else:
        worker = partial(threerscan.unpack_scene, root)

    with Pool(num_workers) as pool:
        infos = list(
            tqdm(
                pool.imap_unordered(worker, ids),
                total=len(ids),
                desc=f"unpack {dataset}/{split}",
            )
        )

    total_frames = sum(info["num_frames"] for info in infos)
    logger.info(f"{dataset}/{split}: unpacked {len(infos)} scenes, {total_frames} frames")
    if dataset == "scannet":
        num_color = sum(info["num_color"] for info in infos)
        num_bad = sum(info["num_nonfinite_poses"] for info in infos)
        sizes = sorted({tuple(info["color_size"]) for info in infos})
        logger.info(f"  colour images written: {num_color}; colour sizes: {sizes}")
        logger.info(f"  frames with non-finite poses (skipped later): {num_bad}")
    else:
        num_extracted = sum(info["extracted"] for info in infos)
        logger.info(
            f"  extracted {num_extracted} zips; {len(infos) - num_extracted} already complete"
        )


def encode(
    dataset: str,
    root: Path,
    cache_dir: Path,
    split: str,
    encoder: "UtoniaEncoder",  # noqa: F821  (imported lazily: Utonia needs CUDA + flash-attn)
    seed: int,
    overwrite: bool,
) -> None:
    from crossweave.data.encode import encode_scene, scene_seed

    module = DATASETS[dataset]
    ids = module.scene_ids(root, split)
    num_encoded, num_skipped, num_vertices = 0, 0, 0
    for scene_id in tqdm(ids, desc=f"encode {dataset}/{split}"):
        out_path = Path(cache_dir) / "scenes" / scene_id / "pointcloud.pt"
        if out_path.exists() and not overwrite:
            num_skipped += 1
            continue
        coord, normal = module.read_points(root, scene_id)
        num_vertices += encode_scene(
            encoder, coord, normal, out_path, scene_seed(seed, scene_id)
        )
        num_encoded += 1
    logger.info(
        f"{dataset}/{split}: encoded {num_encoded} scenes ({num_vertices} vertices), "
        f"skipped {num_skipped} already done"
    )


def frame_seed(seed: int, scene_id: str, frame_id: int) -> int:
    return zlib.crc32(f"{seed}/{scene_id}/{frame_id}".encode())


def grid_name(img_size: int, patch_size: int) -> str:
    # e.g. frames_518_p14: caches depend on the backbone's patch grid
    return f"frames_{img_size}_p{patch_size}"


def frames_scene(
    scene_id: str,
    dataset: str,
    root: Path,
    cache_dir: Path,
    stride: int,
    img_size: int,
    patch_size: int,
    seed: int,
    min_pairs: int,
    depth_tol: float,
    save_rendered_depth: bool,
) -> dict:
    # frames are skipped only for a non-finite pose or < min_pairs pairs
    from crossweave.data.render import DepthRenderer  # EGL context per worker process

    torch.set_num_threads(1)  # one scene per process
    module = DATASETS[dataset]
    out_dir = Path(cache_dir) / grid_name(img_size, patch_size) / scene_id
    done_file = out_dir / "_done.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

    pc_path = Path(cache_dir) / "scenes" / scene_id / "pointcloud.pt"
    assert pc_path.exists(), f"missing {pc_path}: run the encode step first"
    pc = torch.load(pc_path, weights_only=True)
    coord, feat = pc["orig_coord"], pc["feat"]  # (N, 3) f32, (N, 1224) f16

    K, wh = module.native_camera(root, scene_id)
    K_img = module.image_intrinsics(K, wh, img_size)  # (3, 3) for the square model input
    renderer = DepthRenderer(module.mesh_path(root, scene_id), K, wh)
    depth_dir = Path(cache_dir) / "rendered_depth" / scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    samples, skipped = [], {"nonfinite_pose": 0, "few_pairs": 0}
    for frame_id in module.frame_ids(root, scene_id):
        if frame_id % stride != 0:
            continue
        cam_to_world = module.pose(root, scene_id, frame_id)
        if not np.isfinite(cam_to_world).all():
            skipped["nonfinite_pose"] += 1
            continue
        depth = renderer.render(cam_to_world)  # (H, W) metres, native resolution
        # same mm truncation as the old depth images
        depth_mm = (depth * 1000).astype(np.uint16)
        if save_rendered_depth:
            depth_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(depth_dir / f"{frame_id}.png"), depth_mm)
        depth_img = module.image_depth(depth_mm, img_size)  # (img_size, img_size) metres

        generator = torch.Generator().manual_seed(frame_seed(seed, scene_id, frame_id))
        point_idx, patch_idx = patch_point_pairs(
            coord,
            K_img,
            cam_to_world,
            depth_img,
            img_size,
            patch_size,
            generator,
            depth_tol,
        )
        if len(point_idx) < min_pairs:
            skipped["few_pairs"] += 1
            continue
        out_path = out_dir / f"{frame_id}.pt"
        tmp_path = out_path.with_suffix(".tmp")
        torch.save(
            {
                "features": feat[point_idx],  # (P, 1224) f16
                "patch_idx": patch_idx.to(torch.int16),  # (P,)
                "coords": coord[point_idx],  # (P, 3) f32, for fps_space=geometric
            },
            tmp_path,
        )
        tmp_path.rename(out_path)
        samples.append(
            {"scene_id": scene_id, "frame_id": frame_id, "num_pairs": len(point_idx)}
        )
    renderer.close()

    info = {"scene_id": scene_id, "samples": samples, "skipped": skipped}
    done_file.write_text(json.dumps(info))
    return info


def frames(
    dataset: str, root: Path, cache_dir: Path, split: str, num_workers: int, **kw
) -> None:
    ids = DATASETS[dataset].scene_ids(root, split)
    worker = partial(frames_scene, dataset=dataset, root=root, cache_dir=cache_dir, **kw)
    # spawn: each worker gets a fresh process for its own EGL context
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        infos = list(
            tqdm(
                pool.imap_unordered(worker, ids),
                total=len(ids),
                desc=f"frames {dataset}/{split}",
            )
        )

    samples = sorted(
        (s for info in infos for s in info["samples"]),
        key=lambda s: (s["scene_id"], s["frame_id"]),
    )
    skipped = (
        {k: sum(info["skipped"][k] for info in infos) for k in infos[0]["skipped"]}
        if infos
        else {}
    )
    grid = grid_name(kw["img_size"], kw["patch_size"])
    config = {
        "dataset": dataset,
        "split": split,
        "grid": grid,
        "num_scenes": len(ids),
        **kw,
    }
    out_path = Path(cache_dir) / f"{split}_{grid}.json"
    out_path.write_text(
        json.dumps({"config": config, "samples": samples, "skipped": skipped}, indent=1)
    )
    pairs = np.array([s["num_pairs"] for s in samples])
    logger.info(
        f"{dataset}/{split}: kept {len(samples)} frames from {len(ids)} scenes -> {out_path}"
    )
    logger.info(
        f"  skipped: {skipped}; pairs/frame mean {pairs.mean():.0f}, min {pairs.min()}, max {pairs.max()}"
    )


def eval_scene(
    scene_id: str, dataset: str, root: Path, cache_dir: Path, save_masks: bool
) -> dict:
    torch.set_num_threads(1)
    module = DATASETS[dataset]
    scene_cache = Path(cache_dir) / "scenes" / scene_id
    done_file = scene_cache / "eval.json"
    if done_file.exists():
        return json.loads(done_file.read_text())

    # per-vertex labels, aligned with pointcloud.pt
    if dataset == "scannet":
        object_id, nyu40, names = scannet.vertex_labels(
            root, scene_id, scannet.load_label_map(root)
        )
    else:
        object_id, nyu40, names = threerscan.vertex_labels(
            root, scene_id, threerscan.load_objects(root)
        )
    num_points = len(
        torch.load(scene_cache / "pointcloud.pt", weights_only=True, mmap=True)[
            "orig_coord"
        ]
    )
    assert len(object_id) == num_points, (
        f"{scene_id}: {len(object_id)} labels vs {num_points} points"
    )
    np.savez(scene_cache / "labels.npz", object_id=object_id, nyu40=nyu40)

    # scene retrieval: full camera-grid ordering (eval takes [:2k:2])
    frame_ids = module.frame_ids(root, scene_id)
    poses = {f: module.pose(root, scene_id, f) for f in frame_ids}
    valid = [f for f in frame_ids if np.isfinite(poses[f]).all()]
    pose_vecs = np.array([pose_matrix_to_txyz_quat(poses[f]) for f in valid])  # (K, 7)
    order = [
        valid[i] for i in sample_camera_pos_on_grid(pose_vecs, num_to_sample=len(valid))
    ]

    # instance retrieval: id != 0, > 50 vertices, NYU40 not ignored
    ids, counts = np.unique(object_id, return_counts=True)
    candidates = {}
    for oid, count in zip(ids.tolist(), counts.tolist()):
        object_nyu40 = int(nyu40[object_id == oid][0])
        if (
            oid != 0
            and count > MIN_INSTANCE_VERTICES
            and object_nyu40 not in NYU40_IGNORE
        ):
            candidates[oid] = {
                "nyu40": object_nyu40,
                "label": names[oid],
                "num_vertices": count,
            }
    visible = {oid: [] for oid in candidates}  # oid -> [(frame_id, mask_area_px), ...]
    max_id = int(object_id.max())

    def count_mask(frame_id: int, mask: np.ndarray) -> None:
        areas = np.bincount(mask.ravel(), minlength=max_id + 1)  # pixels per id
        assert len(areas) == max_id + 1, (
            f"{scene_id}/{frame_id}: mask id {len(areas) - 1} not in vertex ids"
        )
        for oid in candidates:
            if areas[oid] > 0:
                visible[oid].append((frame_id, int(areas[oid])))

    if dataset == "scannet":
        for frame_id in scannet.mask_frame_ids(root, scene_id):
            count_mask(frame_id, scannet.instance_mask(root, scene_id, frame_id))
    else:
        from crossweave.data.render import (
            ObjectIdRenderer,  # EGL context per worker process
        )

        K, wh = threerscan.native_camera(root, scene_id)
        renderer = ObjectIdRenderer(
            threerscan.label_mesh_path(root, scene_id), object_id, K, wh
        )
        mask_dir = Path(cache_dir) / "masks" / scene_id
        mask_dir.mkdir(parents=True, exist_ok=True)
        for frame_id in valid:
            mask = renderer.render(poses[frame_id])  # (H, W) uint16 objectId
            if save_masks:
                cv2.imwrite(str(mask_dir / f"{frame_id}.png"), mask)
            count_mask(frame_id, mask)
        renderer.close()

    instances, num_invisible = [], 0
    for oid, info in candidates.items():
        frames_by_area = sorted(visible[oid], key=lambda fa: fa[1], reverse=True)
        if not frames_by_area:
            num_invisible += 1  # never visible: dropped
            continue
        instances.append(
            {"scene_id": scene_id, "object_id": oid, **info, "frames": frames_by_area}
        )

    result = {
        "scene_id": scene_id,
        "frame_order": order,
        "instances": instances,
        "num_invisible": num_invisible,
    }
    done_file.write_text(json.dumps(result))
    return result


def eval_metadata(
    dataset: str,
    root: Path,
    cache_dir: Path,
    split: str,
    num_workers: int,
    save_masks: bool,
) -> None:
    ids = DATASETS[dataset].scene_ids(root, split)
    worker = partial(
        eval_scene, dataset=dataset, root=root, cache_dir=cache_dir, save_masks=save_masks
    )
    with multiprocessing.get_context("spawn").Pool(num_workers) as pool:
        results = list(
            tqdm(
                pool.imap_unordered(worker, ids),
                total=len(ids),
                desc=f"eval {dataset}/{split}",
            )
        )
    results.sort(key=lambda r: r["scene_id"])

    config = {
        "dataset": dataset,
        "split": split,
        "num_scenes": len(ids),
        "nyu40_ignore": sorted(NYU40_IGNORE),
        "min_instance_vertices": MIN_INSTANCE_VERTICES,
    }
    scene_frames = {r["scene_id"]: r["frame_order"] for r in results}
    instances = [inst for r in results for inst in r["instances"]]
    (Path(cache_dir) / f"{split}_scene_frames.json").write_text(
        json.dumps({"config": config, "scenes": scene_frames})
    )
    (Path(cache_dir) / f"{split}_instances.json").write_text(
        json.dumps({"config": config, "instances": instances})
    )
    num_invisible = sum(r["num_invisible"] for r in results)
    num_views = np.array([len(inst["frames"]) for inst in instances])
    logger.info(
        f"{dataset}/{split}: {len(scene_frames)} scenes, {len(instances)} instances "
        f"({num_invisible} never visible, dropped); views per instance median {np.median(num_views):.0f}, "
        f"min {num_views.min()}, <10 views: {(num_views < 10).sum()}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="step", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
        p.add_argument("--root", type=Path, required=True, help="raw dataset root")
        p.add_argument(
            "--split", default="train,val", help="comma-separated, from {train,val}"
        )

    p = sub.add_parser(
        "unpack", help="ScanNet .sens -> color/pose/intrinsics; 3RScan sequence.zip"
    )
    add_common(p)
    p.add_argument(
        "--num_workers", type=int, default=2, help="scenes in parallel (HDD: keep small)"
    )
    # NOTE: must equal the frames step's --frame_stride for ScanNet, or read_rgb fails in training
    p.add_argument(
        "--rgb_stride",
        type=int,
        default=30,
        help="ScanNet train: keep every n-th colour frame",
    )

    p = sub.add_parser("encode", help="Utonia point features per scene -> pointcloud.pt")
    add_common(p)
    p.add_argument(
        "--cache_dir", type=Path, required=True, help="output root for this dataset"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="seeds Utonia's per-voxel point choice"
    )
    p.add_argument(
        "--overwrite", action="store_true", help="re-encode scenes that are done"
    )

    p = sub.add_parser(
        "frames", help="point-patch pairs per training frame -> frames_<grid>/"
    )
    add_common(p)
    p.add_argument(
        "--cache_dir", type=Path, required=True, help="output root for this dataset"
    )
    p.add_argument(
        "--frame_stride", type=int, default=None, help="default: scannet 30, threerscan 5"
    )
    p.add_argument(
        "--img_size", type=int, default=518, help="square image backbone input, pixels"
    )
    p.add_argument(
        "--patch_size", type=int, default=14, help="image backbone patch size, pixels"
    )
    p.add_argument(
        "--seed", type=int, default=0, help="seeds the per-frame point-per-patch choice"
    )
    p.add_argument(
        "--min_pairs", type=int, default=128, help="drop frames with fewer pairs"
    )
    p.add_argument(
        "--depth_tol", type=float, default=0.05, help="occlusion tolerance, metres"
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="scenes in parallel (one renderer each)",
    )
    p.add_argument(
        "--save_rendered_depth",
        action="store_true",
        help="also save native depth (mm png)",
    )

    p = sub.add_parser(
        "eval", help="val: vertex labels, scene frame order, instance visibility"
    )
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--root", type=Path, required=True, help="raw dataset root")
    p.add_argument("--split", default="val", help="val only (instance eval)")
    p.add_argument(
        "--cache_dir", type=Path, required=True, help="output root for this dataset"
    )
    p.add_argument("--num_workers", type=int, default=4, help="scenes in parallel")
    # NOTE: 3RScan instance eval reads these masks, so --no_save_masks breaks it
    p.add_argument(
        "--no_save_masks", action="store_true", help="3RScan: do not save rendered masks"
    )

    args = parser.parse_args()
    splits = args.split.split(",")
    assert all(s in ("train", "val") for s in splits), f"unknown split in {args.split!r}"

    if args.step == "unpack":
        for split in splits:
            unpack(args.dataset, args.root, split, args.num_workers, args.rgb_stride)
    elif args.step == "encode":
        from crossweave.data.encode import UtoniaEncoder

        encoder = UtoniaEncoder()
        for split in splits:
            encode(
                args.dataset,
                args.root,
                args.cache_dir,
                split,
                encoder,
                args.seed,
                args.overwrite,
            )
    elif args.step == "frames":
        stride = args.frame_stride or FRAME_STRIDE[args.dataset]
        for split in splits:
            frames(
                args.dataset,
                args.root,
                args.cache_dir,
                split,
                args.num_workers,
                stride=stride,
                img_size=args.img_size,
                patch_size=args.patch_size,
                seed=args.seed,
                min_pairs=args.min_pairs,
                depth_tol=args.depth_tol,
                save_rendered_depth=args.save_rendered_depth,
            )
    elif args.step == "eval":
        assert splits == ["val"], (
            "instance / scene eval metadata is built for the val split only"
        )
        eval_metadata(
            args.dataset,
            args.root,
            args.cache_dir,
            "val",
            args.num_workers,
            not args.no_save_masks,
        )


if __name__ == "__main__":
    main()
