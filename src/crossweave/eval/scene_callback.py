import lightning as L
import torch

from crossweave.data.datasets import SceneDataset
from crossweave.eval.scene_feature_extractor import SceneFeatureExtractor


def rgb_to_point_recall_at_1(outputs: list[dict]) -> float:
    # same computation as eval_scene_retrieval
    rgb = torch.stack([o["outputs"]["rgb"] for o in outputs])  # (S, D)
    point = torch.stack([o["outputs"]["point"] for o in outputs])  # (S, D)
    sim = torch.softmax(rgb @ point.t(), dim=-1)  # (S, S)
    rank_list = torch.argsort(1.0 - sim, dim=1)
    correct = torch.arange(rgb.shape[0], device=rank_list.device)
    return (rank_list[:, 0] == correct).float().mean().item() * 100.0


class SceneRetrievalCallback(L.Callback):
    # logs val/scene_R1_<name> and val/scene_R1_avg (rgb -> point) each epoch.
    def __init__(
        self, datasets: dict[str, SceneDataset], batch_size: int = 2, num_workers: int = 2
    ):
        self.datasets = datasets
        self.batch_size = batch_size
        self.num_workers = num_workers

    def on_train_epoch_end(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ) -> None:
        was_training = pl_module.training
        pl_module.eval()
        recalls = {}
        for name, dataset in self.datasets.items():
            outputs = SceneFeatureExtractor(
                pl_module,
                dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
            )()
            recalls[name] = rgb_to_point_recall_at_1(outputs)
            pl_module.log(f"val/scene_R1_{name}", recalls[name])
        pl_module.log("val/scene_R1_avg", sum(recalls.values()) / len(recalls))
        pl_module.train(was_training)
