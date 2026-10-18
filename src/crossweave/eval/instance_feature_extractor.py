from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate_instances(batch):
    rgbs_list, masks_list, points_list, scene_ids, instance_ids, nyu40_ids, labels = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for rgbs, masks, point_feat, scene_id, instance_id, nyu40_id, label in batch:
        rgbs_list.append(rgbs)
        masks_list.append(masks)
        points_list.append(point_feat)
        scene_ids.append(scene_id)
        instance_ids.append(instance_id)
        nyu40_ids.append(nyu40_id)
        labels.append(label)
    return rgbs_list, masks_list, points_list, scene_ids, instance_ids, nyu40_ids, labels


def _finalize_modality(embeddings: list[torch.Tensor | None], mask: torch.Tensor) -> dict:
    valid = [e for e in embeddings if e is not None]
    if not valid:
        raise RuntimeError("No valid embeddings found — cannot determine embedding dim.")

    dim = valid[0].shape[-1]
    embedding_tensor = torch.zeros(len(embeddings), dim)

    for slot_idx, emb in enumerate(embeddings):
        if emb is not None:
            embedding_tensor[slot_idx] = emb

    scene_embedding = embedding_tensor[mask].mean(dim=0)

    return {
        "mask": mask,
        "embedding": embedding_tensor,
        "scene_embedding": scene_embedding,
    }


class InstanceFeatureExtractor:
    MAX_SLOTS = 150

    def __init__(
        self,
        model,
        dataset,
        batch_size=8,
        num_workers=4,
    ):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.inference_mode()
    @torch.autocast(device_type="cuda")
    def __call__(self) -> list[dict]:
        scenes: dict[str, dict] = defaultdict(
            lambda: {
                "rgb_embeddings": [None] * self.MAX_SLOTS,
                "point_embeddings": [None] * self.MAX_SLOTS,
                "rgb_mask": torch.zeros(self.MAX_SLOTS, dtype=torch.bool),
                "point_mask": torch.zeros(self.MAX_SLOTS, dtype=torch.bool),
                "label_ids": torch.full((self.MAX_SLOTS,), -100, dtype=torch.long),
                "object_ids": torch.full((self.MAX_SLOTS,), -100, dtype=torch.long),
                "slot_counter": 0,
            }
        )

        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_instances,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

        first_rgb = self.dataset[0][0]
        H, W = first_rgb.shape[2], first_rgb.shape[3]
        patch_h, patch_w = H // 14, W // 14

        pbar = tqdm(loader, desc="Extracting instance features", total=len(loader))
        for (
            batch_rgbs,
            batch_masks,
            batch_points,
            batch_scene_ids,
            batch_instance_ids,
            batch_nyu40_ids,
            batch_labels,
        ) in pbar:
            rgb_counts = [r.shape[0] for r in batch_rgbs]
            point_counts = [p.shape[0] for p in batch_points]

            cat_rgbs = torch.cat(batch_rgbs, dim=0).to(self.model.device)
            image_out = self.model.image_encoder.forward_features(cat_rgbs)
            batch_patch_feat = image_out["x_norm_patchtokens"]

            cat_points = torch.cat(batch_points, dim=0).to(self.model.device)
            point_aligned = self.model.point_adapter(cat_points)
            point_aligned = nn.functional.normalize(point_aligned, dim=-1)

            rgb_offset = 0
            point_offset = 0

            for i in range(len(batch_scene_ids)):
                scene_id = batch_scene_ids[i]
                instance_id = batch_instance_ids[i]
                n_rgbs = rgb_counts[i]
                n_points = point_counts[i]

                scene = scenes[scene_id]
                slot_idx = scene["slot_counter"]

                if slot_idx >= self.MAX_SLOTS:
                    rgb_offset += n_rgbs
                    point_offset += n_points
                    continue

                nyu40_id = batch_nyu40_ids[i]

                scene["object_ids"][slot_idx] = int(instance_id)
                if nyu40_id is not None:
                    scene["label_ids"][slot_idx] = int(nyu40_id)

                inst_patches = batch_patch_feat[rgb_offset : rgb_offset + n_rgbs]
                inst_masks = batch_masks[i].to(self.model.device)

                masks_down = nn.functional.interpolate(
                    inst_masks.unsqueeze(1).float(),
                    size=(patch_h, patch_w),
                    mode="nearest",
                )
                masks_flat = masks_down.view(n_rgbs, -1)
                patch_feat_instance = inst_patches[masks_flat > 0.5]

                if (
                    self.model.salad is not None
                    and patch_feat_instance.shape[0] < self.model.salad.num_clusters + 1
                ):
                    req_num = self.model.salad.num_clusters + 1
                    curr_num = patch_feat_instance.shape[0]
                    if curr_num > 0:
                        repeats = (req_num // curr_num) + 1
                        patch_feat_instance = patch_feat_instance.repeat(repeats, 1)[
                            :req_num
                        ]
                    else:
                        patch_feat_instance = torch.zeros(
                            (req_num, batch_patch_feat.shape[-1]),
                            device=batch_patch_feat.device,
                        )

                patch_aligned = self.model.image_adapter(patch_feat_instance)
                patch_aligned = nn.functional.normalize(patch_aligned, dim=-1)

                if self.model.salad is not None:
                    rgb_desc = self.model.salad.patch_forward(patch_aligned.unsqueeze(0))
                else:
                    rgb_desc = nn.functional.normalize(
                        patch_aligned.mean(dim=0, keepdim=True), dim=-1
                    )
                rgb_desc = rgb_desc.squeeze()

                scene["rgb_embeddings"][slot_idx] = rgb_desc
                scene["rgb_mask"][slot_idx] = True

                inst_point_aligned = point_aligned[point_offset : point_offset + n_points]

                if self.model.salad is not None:
                    point_desc = self.model.salad.point_forward(
                        inst_point_aligned.unsqueeze(0)
                    )
                else:
                    point_desc = nn.functional.normalize(
                        inst_point_aligned.mean(dim=0, keepdim=True), dim=-1
                    )
                point_desc = point_desc.squeeze()

                scene["point_embeddings"][slot_idx] = point_desc
                scene["point_mask"][slot_idx] = True

                scene["slot_counter"] += 1
                rgb_offset += n_rgbs
                point_offset += n_points

        outputs = []
        for scene_id, scene in scenes.items():
            outputs.append(
                {
                    "scan_id": scene_id,
                    "label": "NA",
                    "scene_masks": {
                        "rgb": torch.tensor(True),
                        "point": torch.tensor(True),
                    },
                    "label_ids": scene["label_ids"],
                    "object_ids": scene["object_ids"],
                    "rgb": _finalize_modality(scene["rgb_embeddings"], scene["rgb_mask"]),
                    "point": _finalize_modality(
                        scene["point_embeddings"], scene["point_mask"]
                    ),
                }
            )

        return outputs
