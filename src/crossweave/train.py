from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers.wandb import WandbLogger
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

from crossweave.data.datasets import FrameDataset, SceneDataset
from crossweave.eval.scene_callback import SceneRetrievalCallback
from crossweave.models.crossweave import CrossWeave


@hydra.main(config_path="config", config_name="dinov2_utonia", version_base=None)
def main(cfg):
    exp_name = (
        f"{cfg.logging.base_exp_name}"
        f"__{cfg.model.image_encoder}"
        f"__agg_{cfg.model.aggregator}"
        f"__fps{int(cfg.data.enable_fps)}"
        f"__bs{cfg.data.batch_size}"
        f"__pairs{cfg.data.num_patch_point_pairs}"
        f"__fpsr{cfg.data.fps_ratio:.6f}"
        f"__fpss{cfg.data.fps_space}"
        f"__lr{cfg.training.lr}"
    )
    # everything for this run lives in <root_dir>/<run_id>/
    run_id = cfg.logging.run_id or f"{cfg.logging.base_exp_name}_{datetime.now():%Y%m%d-%H%M%S}"
    run_dir = Path(cfg.logging.root_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=False)  # never write into an earlier run
    OmegaConf.save(cfg, run_dir / "config.yaml", resolve=True)
    logger.info(f"Run directory: {run_dir}")

    if cfg.logging.use_wandb:
        run_logger = WandbLogger(
            name=exp_name,
            id=run_id,
            save_dir=run_dir,
            entity=cfg.logging.entity,
            project=cfg.logging.project,
            group=cfg.logging.group,
        )
        run_logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True))
    else:
        run_logger = CSVLogger(run_dir, name="csv", version="")
        run_logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))

    L.seed_everything(cfg.training.seed, workers=True, verbose=False)
    logger.info(f"Seed set to {cfg.training.seed}")

    assert cfg.data.img_size[0] == cfg.data.img_size[1], "square images only"
    img_size = cfg.data.img_size[0]

    scannet_train = FrameDataset(
        dataset="scannet",
        root=Path(cfg.data.scannet_root),
        cache_dir=Path(cfg.data.scannet_cache),
        split="train",
        img_size=img_size,
        patch_size=cfg.data.patch_size,
        num_patch_point_pairs=cfg.data.num_patch_point_pairs,
        enable_fps=cfg.data.enable_fps,
        fps_ratio=cfg.data.fps_ratio,
        fps_space=cfg.data.fps_space,
    )
    logger.info(f"Loaded ScanNet Dataset (train) with {len(scannet_train)} samples.")
    threerscan_train = FrameDataset(
        dataset="threerscan",
        root=Path(cfg.data.threerscan_root),
        cache_dir=Path(cfg.data.threerscan_cache),
        split="train",
        img_size=img_size,
        patch_size=cfg.data.patch_size,
        num_patch_point_pairs=cfg.data.num_patch_point_pairs,
        enable_fps=cfg.data.enable_fps,
        fps_ratio=cfg.data.fps_ratio,
        fps_space=cfg.data.fps_space,
    )
    logger.info(f"Loaded 3RScan Dataset (train) with {len(threerscan_train)} samples.")
    train_dataset = ConcatDataset([scannet_train, threerscan_train])
    logger.info(f"Combined Datasets for a total of {len(train_dataset)} train samples.")
    if cfg.data.subset != 1:
        train_dataset = Subset(
            train_dataset, list(range(0, len(train_dataset), int(1 / cfg.data.subset)))
        )
        logger.info(
            f"Sub-sampled train subset to {len(train_dataset)} samples ({cfg.data.subset * 100:.0f}%)"
        )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        shuffle=True,
    )
    logger.info(f"Inited Train DataLoader with batch size {cfg.data.batch_size}")

    # checkpoints are selected by average val scene R@1
    scene_datasets = {}
    for name, key in [("scannet", "scannet"), ("3rscan", "threerscan")]:
        cache = Path(cfg.data[f"{key}_cache"])
        assert (cache / "val_scene_frames.json").exists(), (
            f"missing {cache / 'val_scene_frames.json'}: run `python -m crossweave.data.preprocess eval` first"
        )
        scene_datasets[name] = SceneDataset(
            key, cfg.data[f"{key}_root"], cache, num_images=10, img_size=img_size
        )

    model = CrossWeave(
        image_encoder=cfg.model.image_encoder,
        image_dim=cfg.model.image_dim,
        point_dim=cfg.model.point_dim,
        common_dim=cfg.model.common_dim,
        aggregator=cfg.model.aggregator,
        num_clusters=cfg.model.num_clusters,
        cluster_dim=cfg.model.cluster_dim,
        local_loss_weight=cfg.training.local_loss_weight,
        global_loss_weight=cfg.training.global_loss_weight,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    logger.info("Inited Model")

    trainer = L.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu",
        devices=cfg.training.devices,
        precision="bf16-mixed",
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        log_every_n_steps=1,
        callbacks=[
            SceneRetrievalCallback(scene_datasets),
            ModelCheckpoint(
                dirpath=run_dir / "checkpoints",
                monitor="val/scene_R1_avg",
                mode="max",
                save_top_k=1,
                save_last=True,
                filename="best_r1_{val/scene_R1_avg:.2f}_epoch_{epoch:02d}",
                auto_insert_metric_name=False,
                verbose=True,
            ),
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(),
        ],
        logger=run_logger,
    )

    trainer.fit(model, train_dataloader)


if __name__ == "__main__":
    main()
