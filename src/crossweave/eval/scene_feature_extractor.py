import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def _collate_scenes(batch):
    rgbs_list, points_list, scan_ids = [], [], []
    for rgbs, point_feat, scan_id in batch:
        rgbs_list.append(rgbs)
        points_list.append(point_feat)
        scan_ids.append(scan_id)
    return rgbs_list, points_list, scan_ids


class SceneFeatureExtractor:
    def __init__(self, model, dataset, batch_size=8, num_workers=4):
        self.model = model
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_workers = num_workers

    @torch.inference_mode()
    @torch.autocast(device_type="cuda")
    def __call__(self):
        loader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=_collate_scenes,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

        outputs = []
        P = None

        pbar = tqdm(loader, desc="Extracting scene features", total=len(loader))
        for batch_rgbs, batch_points, batch_ids in pbar:
            if P is None:
                P = (batch_rgbs[0].shape[2] // 14) * (batch_rgbs[0].shape[3] // 14)

            rgb_counts = [r.shape[0] for r in batch_rgbs]
            point_counts = [p.shape[0] for p in batch_points]

            cat_rgbs = torch.cat(batch_rgbs, dim=0).to(self.model.device)
            total_images = cat_rgbs.shape[0]

            image_out = self.model.image_encoder.forward_features(cat_rgbs)
            patch_feat = image_out["x_norm_patchtokens"]

            patch_aligned = self.model.image_adapter(patch_feat.flatten(0, 1))
            patch_aligned = nn.functional.normalize(patch_aligned, dim=-1)
            patch_aligned = patch_aligned.view(total_images, P, -1)

            cat_points = torch.cat(batch_points, dim=0).to(self.model.device)
            point_aligned = self.model.point_adapter(cat_points)
            point_aligned = nn.functional.normalize(point_aligned, dim=-1)

            rgb_offset = 0
            point_offset = 0

            for i, scan_id in enumerate(batch_ids):
                n_rgbs = rgb_counts[i]
                n_points = point_counts[i]

                scene_patches = patch_aligned[rgb_offset : rgb_offset + n_rgbs]
                rgb_offset += n_rgbs
                scene_patches = scene_patches.flatten(0, 1).unsqueeze(0)

                if self.model.salad is not None:
                    rgb_feat = self.model.salad.patch_forward(scene_patches)
                else:
                    rgb_feat = nn.functional.normalize(scene_patches.mean(dim=1), dim=-1)
                rgb_feat = rgb_feat.squeeze()

                scene_points = point_aligned[
                    point_offset : point_offset + n_points
                ].unsqueeze(0)
                point_offset += n_points

                if self.model.salad is not None:
                    point_feat = self.model.salad.point_forward(scene_points)
                else:
                    point_feat = nn.functional.normalize(scene_points.mean(dim=1), dim=-1)
                point_feat = point_feat.squeeze()

                outputs.append(
                    {
                        "scan_id": scan_id,
                        "scene_label": "NA",
                        "masks": {
                            "rgb": torch.tensor([1.0]),
                            "point": torch.tensor([1.0]),
                        },
                        "outputs": {"rgb": rgb_feat, "point": point_feat},
                    }
                )

        return outputs
