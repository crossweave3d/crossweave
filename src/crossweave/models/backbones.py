from pathlib import Path

import numpy as np
import torch
from torch import nn

from crossweave.models import tips_image_encoder

DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"  # facebookresearch/dinov2 pinned to a working commit
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
TIPSV2_CKPT = {
    "tipsv2_l14": Path("~/.cache/tips/tips_v2_oss_l14_vision.npz").expanduser()
}


class TIPSv2(nn.Module):
    # TIPSv2 expects RGB in [0, 1]; undo the ImageNet normalisation so datasets keep one
    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.register_buffer(
            "mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False
        )
        self.register_buffer(
            "std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False
        )

    def forward_features(self, x: torch.Tensor) -> dict:
        return self.encoder.forward_features(x * self.std + self.mean)  # back to [0, 1]


def load_image_backbone(name: str) -> nn.Module:
    if name.startswith("dinov2_"):
        return torch.hub.load(
            f"facebookresearch/dinov2:{DINOV2_COMMIT}", name, skip_validation=True
        )
    if name in TIPSV2_CKPT:
        ckpt = TIPSV2_CKPT[name]
        assert ckpt.exists(), (
            f"missing {ckpt}; download https://storage.googleapis.com/tips_data/v2_0/checkpoints/pytorch/{ckpt.name}"
        )
        # as in TIPS' run_image_encoder_inference.py (448 px, 32x32 pos-embed)
        encoder = tips_image_encoder.vit_large(
            img_size=448,
            patch_size=14,
            ffn_layer="mlp",
            block_chunks=0,
            init_values=1.0,
            interpolate_antialias=True,
            interpolate_offset=0.0,
        )
        state = {k: torch.tensor(v) for k, v in np.load(ckpt, allow_pickle=False).items()}
        encoder.load_state_dict(state)
        return TIPSv2(encoder)
    raise ValueError(f"unknown image backbone {name!r}")
