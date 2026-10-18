import os

os.environ.setdefault(
    "PYOPENGL_PLATFORM", "egl"
)  # headless GPU rendering; must precede pyrender

import numpy as np
import pyrender
import trimesh

ZNEAR = 0.5  # metres
ZFAR = 50.0  # metres
T_CV2GL = np.diag(
    [1.0, -1.0, -1.0, 1.0]
)  # OpenCV camera axes (y down, z forward) -> OpenGL


class DepthRenderer:
    def __init__(
        self,
        mesh_path,
        K: np.ndarray,
        wh: tuple[int, int],
        znear: float = ZNEAR,
        zfar: float = ZFAR,
    ):
        assert K.shape == (3, 3), K.shape
        mesh = trimesh.load(mesh_path, force="mesh")
        self.scene = pyrender.Scene()
        self.scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=znear, zfar=zfar
        )
        self.camera_node = self.scene.add(camera)
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=wh[0], viewport_height=wh[1]
        )
        self.wh = wh

    def render(self, cam_to_world: np.ndarray) -> np.ndarray:
        assert cam_to_world.shape == (4, 4), cam_to_world.shape
        self.scene.set_pose(self.camera_node, cam_to_world @ T_CV2GL)
        depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.DEPTH_ONLY)
        assert depth.shape == (self.wh[1], self.wh[0]), depth.shape
        return depth

    def close(self) -> None:
        self.renderer.delete()


class ObjectIdRenderer:
    def __init__(
        self,
        mesh_path,
        object_id: np.ndarray,
        K: np.ndarray,
        wh: tuple[int, int],
        znear: float = 0.01,
        zfar: float = 100.0,
    ):
        mesh = trimesh.load(mesh_path, process=False)
        assert len(mesh.vertices) == len(object_id), (len(mesh.vertices), len(object_id))
        assert object_id.max() < 2**16, object_id.max()
        self.scene = pyrender.Scene(bg_color=[0, 0, 0, 255])  # background -> id 0
        camera = pyrender.IntrinsicsCamera(
            fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], znear=znear, zfar=zfar
        )
        self.camera_node = self.scene.add(camera)
        self.seg_node_map = {}
        face_ids = object_id[mesh.faces[:, 0]]  # (F,) label of each face's first vertex
        for oid in np.unique(face_ids):
            part = mesh.submesh([face_ids == oid], append=True)
            node = self.scene.add(pyrender.Mesh.from_trimesh(part, smooth=False))
            self.seg_node_map[node] = (int(oid) % 256, int(oid) // 256, 0)
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=wh[0], viewport_height=wh[1]
        )
        self.wh = wh

    def render(self, cam_to_world: np.ndarray) -> np.ndarray:
        assert cam_to_world.shape == (4, 4), cam_to_world.shape
        self.scene.set_pose(self.camera_node, cam_to_world @ T_CV2GL)
        seg, _ = self.renderer.render(
            self.scene, pyrender.RenderFlags.SEG, self.seg_node_map
        )
        assert seg.shape[:2] == (self.wh[1], self.wh[0]), seg.shape
        return seg[..., 0].astype(np.uint16) + 256 * seg[..., 1].astype(np.uint16)

    def close(self) -> None:
        self.renderer.delete()
