import lightning as L
from pytorch_metric_learning.losses import NTXentLoss, SelfSupervisedLoss
import torch
import torch.nn as nn

from crossweave.models.aggregators.salad import SALAD
from crossweave.models.backbones import load_image_backbone
from crossweave.utils.metrics import recall_at_k, uniformity
from crossweave.utils.visualization import (
    visualise_ot_matrix,
    visualize_global_sim_matrix,
    visualize_similarity_matrix,
)
import wandb


class CrossWeave(L.LightningModule):
    def __init__(
        self,
        image_encoder: str,
        point_dim: int,
        image_dim: int,
        common_dim: int,
        num_clusters: int = 24,
        cluster_dim: int = 128,
        local_loss_weight: float = 1.0,
        global_loss_weight: float = 1.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-2,
        aggregator: str = "none",
    ):
        super().__init__()
        self.strict_loading = False
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.local_loss_weight = local_loss_weight
        self.global_loss_weight = global_loss_weight
        self.aggregator = aggregator

        # frozen DINOv2 or TIPSv2
        self.image_encoder = load_image_backbone(image_encoder)
        self.image_adapter = nn.Linear(image_dim, common_dim, bias=False)
        self.point_adapter = nn.Linear(point_dim, common_dim, bias=False)
        if self.aggregator == "salad":
            self.salad = SALAD(
                input_dim=common_dim,
                hidden_dim=common_dim // 2,
                num_clusters=num_clusters,
                cluster_dim=cluster_dim,
            )
        elif self.aggregator == "none":  # mean pooling
            self.salad = None
        else:
            raise ValueError(f"unknown aggregator {self.aggregator!r}")
        self.info_nce_loss = SelfSupervisedLoss(NTXentLoss())

    def configure_optimizers(self):
        params = list(self.image_adapter.parameters()) + list(
            self.point_adapter.parameters()
        )
        if self.salad is not None:
            params += list(self.salad.parameters())
        return torch.optim.AdamW(
            params,
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if not k.startswith("image_encoder.")
        }

    def on_load_checkpoint(self, checkpoint):
        # non-strict only because the backbone isn't saved; everything else must match exactly
        expected = {k for k in self.state_dict() if not k.startswith("image_encoder.")}
        found = set(checkpoint["state_dict"])
        assert found == expected, (
            f"checkpoint does not match the model: missing {sorted(expected - found)}, "
            f"unexpected {sorted(found - expected)}"
        )

    def _shared_step(self, batch, step):
        # optional 4th element: point coords (fps_space=geometric)
        rgb, point_feat, patch_indices = batch[
            :3
        ]  # (B, 3, H, W), (B, N, D_POINT), (B, N)

        B, N = patch_indices.shape
        # patch centres (B, N, 2), only for the debug figure
        grid_w = rgb.shape[-1] // 14
        point_pixel = torch.stack(
            [(patch_indices % grid_w + 0.5) * 14, (patch_indices // grid_w + 0.5) * 14],
            dim=-1,
        )
        batch_idx_expand = (
            torch.arange(B, device=patch_indices.device).unsqueeze(1).expand(B, N)
        )

        with torch.inference_mode():
            image_out = self.image_encoder.forward_features(rgb)
            patch_tokens = image_out["x_norm_patchtokens"]  # [B, NUM_PATCHES, D_IMG]

        # index out sampled patches
        image_feat = patch_tokens[
            batch_idx_expand, patch_indices
        ]  # [B, NUM_PAIRS, D_IMG]

        image_feat = self.image_adapter(image_feat)
        image_feat = nn.functional.normalize(
            image_feat, dim=-1
        )  # [B, NUM_PAIRS, D_COM]

        point_feat = self.point_adapter(point_feat.to(self.device))
        point_feat = nn.functional.normalize(point_feat, dim=-1)  # [B, NUM_PAIRS, D_COM]

        loss_local = self.info_nce_loss(
            image_feat.flatten(0, 1), point_feat.flatten(0, 1)
        )

        # --- Global Aggregation ---
        if self.salad is not None:
            point_desc, patch_desc = self.salad(point_feat, image_feat)
            loss_global = self.info_nce_loss(patch_desc, point_desc)

            loss = (
                self.local_loss_weight * loss_local
                + self.global_loss_weight * loss_global
            )

            # Global Metrics
            sim_matrix_global = patch_desc @ point_desc.T
            B_global = sim_matrix_global.size(0)

            diag_mask_global = torch.eye(B_global, dtype=torch.bool, device=self.device)
            diag_global = sim_matrix_global[diag_mask_global]
            off_diag_global = sim_matrix_global[~diag_mask_global]

            patch_assign = self.salad.last_patch_assignment
            point_assign = self.salad.last_point_assignment
        else:
            point_desc = nn.functional.normalize(point_feat.mean(dim=1), dim=-1)
            patch_desc = nn.functional.normalize(image_feat.mean(dim=1), dim=-1)

            loss_global = self.info_nce_loss(patch_desc, point_desc)
            loss = (
                self.local_loss_weight * loss_local
                + self.global_loss_weight * loss_global
            )

            # Global Metrics
            sim_matrix_global = patch_desc @ point_desc.T
            B_global = sim_matrix_global.size(0)

            diag_mask_global = torch.eye(B_global, dtype=torch.bool, device=self.device)
            diag_global = sim_matrix_global[diag_mask_global]
            off_diag_global = sim_matrix_global[~diag_mask_global]

            patch_assign, point_assign = None, None

        # Local Metrics
        sim_matrix_local = image_feat.flatten(0, 1) @ point_feat.flatten(0, 1).T
        B_local = sim_matrix_local.size(0)

        diag_mask_local = torch.eye(B_local, dtype=torch.bool, device=self.device)
        diag_local = sim_matrix_local[diag_mask_local]
        off_diag_local = sim_matrix_local[~diag_mask_local]

        metrics = {
            f"{step}/InfoNCELoss": loss,
            f"{step}/Loss_Local": loss_local,
            f"{step}/Loss_Global": loss_global,
            f"{step}/local/uniformity_image": uniformity(image_feat.flatten(0, 1)),
            f"{step}/local/uniformity_pc": uniformity(point_feat.flatten(0, 1)),
            f"{step}/local/sim_matrix/diag_mean": diag_local.mean(),
            f"{step}/local/sim_matrix/diag_max": diag_local.max(),
            f"{step}/local/sim_matrix/diag_min": diag_local.min(),
            f"{step}/local/sim_matrix/off_diag_mean": off_diag_local.mean(),
            f"{step}/local/sim_matrix/off_diag_max": off_diag_local.max(),
            f"{step}/local/sim_matrix/off_diag_min": off_diag_local.min(),
        }

        metrics.update(
            {
                f"{step}/global/uniformity_image": uniformity(patch_desc),
                f"{step}/global/uniformity_pc": uniformity(point_desc),
                f"{step}/global/sim_matrix/diag_mean": diag_global.mean(),
                f"{step}/global/sim_matrix/off_diag_mean": off_diag_global.mean(),
            }
        )

        prefix_sim_mats = [("local", sim_matrix_local), ("global", sim_matrix_global)]

        for prefix, sim_mat in prefix_sim_mats:
            for k in [1, 5, 10]:
                val_img2pc = recall_at_k(sim_matrix=sim_mat, k=k)
                if val_img2pc is not None:
                    metrics[f"{step}/{prefix}/R@{k}_img2pc"] = val_img2pc
                val_pc2img = recall_at_k(sim_matrix=sim_mat.T, k=k)
                if val_pc2img is not None:
                    metrics[f"{step}/{prefix}/R@{k}_pc2img"] = val_pc2img

        return (
            loss,
            image_feat,
            point_feat,
            patch_desc,
            point_desc,
            patch_assign,
            point_assign,
            rgb,
            point_pixel,
            metrics,
        )

    def training_step(self, batch, batch_idx):
        (
            loss,
            image_feat,
            point_feat,
            patch_desc,
            point_desc,
            patch_assign,
            point_assign,
            rgb,
            point_pixel,
            metrics,
        ) = self._shared_step(batch, "train")
        self.log_dict(
            metrics,
            prog_bar=True,
            sync_dist=True,
        )
        is_first = batch_idx == 0
        is_last = batch_idx == self.trainer.num_training_batches - 1
        if is_first or is_last:
            patch_desc_detach = patch_desc.detach() if patch_desc is not None else None
            point_desc_detach = point_desc.detach() if point_desc is not None else None
            patch_assign_detach = (
                patch_assign.detach() if patch_assign is not None else None
            )
            point_assign_detach = (
                point_assign.detach() if point_assign is not None else None
            )

            self._log_sim_matrix(
                image_feat.detach(),
                point_feat.detach(),
                patch_desc_detach,
                point_desc_detach,
                patch_assign_detach,
                point_assign_detach,
                rgb.detach(),
                point_pixel.detach(),
                step="train",
            )
        return loss

    def _log_sim_matrix(
        self,
        image_feat,
        point_feat,
        patch_desc,
        point_desc,
        patch_assign,
        point_assign,
        rgb,
        point_pixel,
        step,
    ):
        if not hasattr(self.logger.experiment, "log"):
            return

        if image_feat.ndim == 3:
            sim_matrix_local = image_feat.flatten(0, 1) @ point_feat.flatten(0, 1).T
        else:
            sim_matrix_local = image_feat @ point_feat.T

        fig_local = visualize_similarity_matrix(sim_matrix_local.half(), rgb, point_pixel)

        log_dict_to_add = {
            f"{step}/Local Similarity Matrix": wandb.Image(fig_local),
        }

        if patch_desc is not None and point_desc is not None:
            sim_matrix_global = patch_desc @ point_desc.T
            fig_global = visualize_global_sim_matrix(sim_matrix_global.half())

            log_dict_to_add.update(
                {
                    f"{step}/Global Similarity Matrix": wandb.Image(fig_global),
                }
            )

            if patch_assign is not None and point_assign is not None:
                fig_ot_patch = visualise_ot_matrix(
                    patch_assign[0], "Patch-to-Cluster OT Matrix"
                )
                fig_ot_point = visualise_ot_matrix(
                    point_assign[0], "Point-to-Cluster OT Matrix"
                )

                log_dict_to_add.update(
                    {
                        f"{step}/OT Matrix (Patch)": wandb.Image(fig_ot_patch),
                        f"{step}/OT Matrix (Point)": wandb.Image(fig_ot_point),
                    }
                )

        self.logger.experiment.log(log_dict_to_add, commit=False)

        import matplotlib.pyplot as plt

        plt.close(fig_local)
        if patch_desc is not None and point_desc is not None:
            plt.close(fig_global)
            if patch_assign is not None and point_assign is not None:
                plt.close(fig_ot_patch)
                plt.close(fig_ot_point)
