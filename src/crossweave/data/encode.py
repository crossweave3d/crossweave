from pathlib import Path
import zlib

from loguru import logger
import numpy as np
import torch
import utonia

UTONIA_CKPT = Path("~/.cache/utonia/ckpt/utonia.pth").expanduser()
UTONIA_DIM = 1224
FP16_MAX = 65504.0


def scene_seed(seed: int, scene_id: str) -> int:
    return zlib.crc32(f"{seed}/{scene_id}".encode())


class UtoniaEncoder:
    def __init__(self, ckpt: Path = UTONIA_CKPT, device: str = "cuda"):
        self.device = torch.device(device)
        if ckpt.exists():
            self.model = utonia.model.load(str(ckpt)).to(self.device)
        else:
            logger.info(
                f"{ckpt} not found; downloading Utonia from Hugging Face (Pointcept/Utonia)"
            )
            self.model = utonia.model.load("utonia", repo_id="Pointcept/Utonia").to(
                self.device
            )
        self.model.eval()
        self.transform = utonia.transform.default(
            scale=1.0, apply_z_positive=True, normalize_coord=False
        )

    @torch.inference_mode()
    def encode(self, coord: np.ndarray, normal: np.ndarray, seed: int) -> torch.Tensor:
        # coord, normal (N, 3) -> (N, 1224) float16 on CPU
        n = coord.shape[0]
        assert coord.shape == normal.shape == (n, 3), (coord.shape, normal.shape)
        # geometry only: zero colour, so retrieval can't shortcut through appearance
        point = {
            "coord": coord.astype(np.float32, copy=True),
            "normal": normal.astype(np.float32, copy=True),
            "color": np.zeros((n, 3), dtype=np.float32),
        }
        # GridSample picks a random point per voxel (np.random) and shuffle_orders uses
        # torch.randperm even in eval mode, so seed both per scene
        np.random.seed(seed)
        torch.manual_seed(seed)
        point = self.transform(point)
        for key, value in point.items():
            if isinstance(value, torch.Tensor):
                point[key] = value.to(self.device, non_blocking=True)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            point = self.model(point)

        for _ in range(2):
            assert "pooling_parent" in point.keys()
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
            point = parent
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent.feat = point.feat[inverse]
            point = parent

        feat = point.feat[point.inverse].float()  # (N, 1224)
        assert feat.shape == (n, UTONIA_DIM), feat.shape
        assert torch.isfinite(feat).all(), "non-finite Utonia features"
        assert feat.abs().max() < FP16_MAX, "Utonia features exceed fp16 range"
        return feat.half().cpu()


def encode_scene(
    encoder: UtoniaEncoder,
    coord: np.ndarray,
    normal: np.ndarray,
    out_path: Path,
    seed: int,
) -> int:
    feat = encoder.encode(coord, normal, seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".tmp")
    torch.save({"orig_coord": torch.from_numpy(coord), "feat": feat}, tmp_path)
    tmp_path.rename(out_path)
    return coord.shape[0]
