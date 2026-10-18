import argparse
import json
from pathlib import Path

from lightning import seed_everything
from loguru import logger
from omegaconf import OmegaConf

from crossweave.data.datasets import InstanceDataset, SceneDataset
from crossweave.eval.instance_feature_extractor import InstanceFeatureExtractor
from crossweave.eval.instance_retrieval import eval_instance_retrieval
from crossweave.eval.scene_feature_extractor import SceneFeatureExtractor
from crossweave.eval.scene_retrieval import eval_scene_retrieval
from crossweave.models.crossweave import CrossWeave

CONFIG = Path(__file__).parent / "config" / "dinov2_utonia.yaml"
DATASETS = [("scannet", "ScanNet"), ("3rscan", "3RScan")]


def report(results: dict, ckpt: Path) -> None:
    lines = [
        f"CrossWeave evaluation: {ckpt}",
        "",
        f"{'Scene retrieval (rgb -> point)':<36}{'R@1':>8}{'R@5':>8}{'R@10':>8}",
    ]
    for key, label in DATASETS:
        m = results["scene"][key]["rgb->point"]
        lines.append(f"  {label:<34}{m['R@1']:>8.2f}{m['R@5']:>8.2f}{m['R@10']:>8.2f}")
    if results["instance"]:
        cols = ["Top-1", "Top-3", "R@25%", "R@50%", "R@75%"]
        lines += [
            "",
            f"{'Instance retrieval (rgb -> point)':<36}"
            + "".join(f"{c:>8}" for c in cols),
        ]
        for key, label in DATASETS:
            m = results["instance"][key]["rgb->point"]
            lines.append(f"  {label:<34}" + "".join(f"{m[c]:>8.2f}" for c in cols))
    out = Path(ckpt).with_suffix(".eval.json")
    out.write_text(json.dumps(results, indent=2, default=float))
    lines += ["", f"all metrics (incl. rgb->rgb, point->point): {out}"]
    logger.info("\n" + "\n".join(lines))


if __name__ == "__main__":
    seed_everything(0)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt", type=Path, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=Path, default=CONFIG, help="config with the data paths"
    )
    parser.add_argument(
        "--no-aggregation",
        action="store_true",
        help="Disable SALAD aggregation, use mean pooling",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size for feature extraction"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="DataLoader workers for image loading"
    )
    parser.add_argument(
        "--scannet-scene-images",
        type=int,
        default=10,
        help="images per ScanNet scene",
    )
    parser.add_argument(
        "--threerscan-scene-images",
        type=int,
        default=10,
        help="images per 3RScan scene",
    )
    parser.add_argument(
        "--instance-images",
        type=int,
        default=10,
        help="most-visible images per object",
    )
    parser.add_argument(
        "--scene-only",
        action="store_true",
        help="stop after scene retrieval (skip instances)",
    )
    parser.add_argument(
        "overrides", nargs="*", help="config overrides, e.g. data.scannet_root=/path"
    )
    args = parser.parse_args()
    data = OmegaConf.merge(
        OmegaConf.load(args.config), OmegaConf.from_dotlist(args.overrides)
    ).data

    # backbone comes from the checkpoint's hparams
    model = CrossWeave.load_from_checkpoint(args.ckpt, strict=False)
    if args.no_aggregation:
        model.salad = None
        logger.info("Aggregation disabled — using mean pooling")
    model.eval()
    model.to("cuda")
    logger.info(f"Loaded Checkpoint from {args.ckpt}")
    results = {"checkpoint": str(args.ckpt), "scene": {}, "instance": {}}

    scannet_dataset = SceneDataset(
        "scannet", data.scannet_root, data.scannet_cache, args.scannet_scene_images
    )
    scene_feature_extractor = SceneFeatureExtractor(
        model, scannet_dataset, batch_size=args.batch_size, num_workers=args.num_workers
    )
    outputs = scene_feature_extractor()
    results["scene"]["scannet"] = eval_scene_retrieval(outputs)

    threerscan_dataset = SceneDataset(
        "threerscan",
        data.threerscan_root,
        data.threerscan_cache,
        args.threerscan_scene_images,
    )
    scene_feature_extractor = SceneFeatureExtractor(
        model,
        threerscan_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    outputs = scene_feature_extractor()
    results["scene"]["3rscan"] = eval_scene_retrieval(outputs)

    if args.scene_only:
        logger.info("--scene-only: skipping instance retrieval")
        report(results, args.ckpt)
        raise SystemExit(0)

    model.salad = None
    logger.info("Aggregation disabled — using mean pooling")

    threerscan_instance_dataset = InstanceDataset(
        "threerscan", data.threerscan_root, data.threerscan_cache, args.instance_images
    )
    instance_feature_extractor = InstanceFeatureExtractor(
        model,
        threerscan_instance_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    outputs = instance_feature_extractor()
    results["instance"]["3rscan"] = eval_instance_retrieval(outputs)

    scannet_instance_dataset = InstanceDataset(
        "scannet", data.scannet_root, data.scannet_cache, args.instance_images
    )
    instance_feature_extractor = InstanceFeatureExtractor(
        model,
        scannet_instance_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    outputs = instance_feature_extractor()
    results["instance"]["scannet"] = eval_instance_retrieval(outputs)
    report(results, args.ckpt)
