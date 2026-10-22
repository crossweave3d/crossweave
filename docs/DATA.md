# Data preparation

CrossWeave trains and evaluates on [ScanNet v2](http://www.scan-net.org/) and [3RScan](https://waldjohannau.github.io/RIO/).

Access to both datasets must be requested from their authors:
- [ScanNet v2](http://www.scan-net.org/)
- [3RScan](https://waldjohannau.github.io/RIO/)

The rest of this document refers to three root directories:

```bash
SCANNET=/path/to/ScanNet          # raw ScanNet root
THREERSCAN=/path/to/3RScan        # raw 3RScan root
CACHE=/path/to/crossweave_cache   # preprocessing outputs, one subfolder per dataset
```

## 1. Raw data

### ScanNet v2

| File | Source |
|---|---|
| `<scene_id>.sens` | ScanNet download script, `--type .sens` |
| `<scene_id>_vh_clean_2.ply` | ScanNet download script, `--type _vh_clean_2.ply` |
| `<scene_id>_vh_clean_2.0.010000.segs.json` | ScanNet download script, `--type _vh_clean_2.0.010000.segs.json` |
| `<scene_id>_vh_clean.aggregation.json` | ScanNet download script, `--type _vh_clean.aggregation.json` |
| `<scene_id>_2d-instance-filt.zip` (val only) | ScanNet download script, `--type _2d-instance-filt.zip` |
| `scannetv2-labels.combined.tsv` | ScanNet download script, `--label_map` |
| `scannetv2_train.txt`, `scannetv2_val.txt` | [ScanNet repository](https://github.com/ScanNet/ScanNet/tree/master/Tasks/Benchmark) ([train](https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_train.txt), [val](https://raw.githubusercontent.com/ScanNet/ScanNet/master/Tasks/Benchmark/scannetv2_val.txt)) |

The download script writes to the directory given by `-o`; pass `-o $SCANNET/v2` to match the layout below.

```
$SCANNET/
    splits/scannetv2_train.txt, scannetv2_val.txt
    v2/scannetv2-labels.combined.tsv
    v2/scans/<scene_id>/
        <scene_id>.sens
        <scene_id>_vh_clean_2.ply
        <scene_id>_vh_clean_2.0.010000.segs.json
        <scene_id>_vh_clean.aggregation.json
        <scene_id>_2d-instance-filt.zip      # val scenes only
```

### 3RScan

| File | Source |
|---|---|
| `sequence.zip` | 3RScan download script |
| `mesh.refined.v2.obj` (with `.mtl` and texture) | 3RScan download script |
| `labels.instances.annotated.v2.ply` | 3RScan download script |
| `train_scans.txt` | [campar.in.tum.de](https://campar.in.tum.de/public_datasets/3RScan/train_scans.txt) (reference scans and rescans) |
| `val_scans.txt` | [campar.in.tum.de](https://campar.in.tum.de/public_datasets/3RScan/val_scans.txt) (reference scans and rescans) |
| `objects.json` | [3DSSG](https://campar.in.tum.de/public_datasets/3DSSG/3DSSG/objects.json) (per-object NYU40 labels) |

```
$THREERSCAN/
    files/train_scans.txt, val_scans.txt, objects.json
    data/<scan_id>/
        sequence.zip
        mesh.refined.v2.obj
        labels.instances.annotated.v2.ply
```

## 2. Preprocessing

The four steps run in order for each dataset. Steps 2 and 3 require a GPU; step 3 and 3RScan's step 4 also require
headless EGL rendering. Each step processes scenes independently: a re-run skips finished scenes, and any error stops
the run.

| Step | Command | Splits | Writes |
|---|---|---|---|
| 1. unpack | `unpack --dataset D --root R` | train, val | ScanNet: `R/v2/scans/<scene>/{color/,pose.npy,intrinsic_color.txt,instance-filt/,unpacked.json}`; 3RScan: `R/data/<scan>/sequence/` |
| 2. encode | `encode --dataset D --root R --cache_dir C` | train, val | `C/scenes/<scene>/pointcloud.pt` |
| 3. frames | `frames --dataset D --root R --cache_dir C --split train` | train | `C/frames_518_p14/<scene>/<frame>.pt`, `C/train_frames_518_p14.json` |
| 4. eval | `eval --dataset D --root R --cache_dir C` | val | `C/scenes/<scene>/{labels.npz,eval.json}`, `C/val_{scene_frames,instances}.json`, 3RScan: `C/masks/` |

Every command is `pixi run python -m crossweave.data.preprocess <step> ...`, with `D` = `scannet` or `threerscan`.
`--num_workers` processes scenes in parallel.

```bash
pixi run python -m crossweave.data.preprocess unpack --dataset scannet --root $SCANNET
pixi run python -m crossweave.data.preprocess encode --dataset scannet --root $SCANNET --cache_dir $CACHE/scannet
pixi run python -m crossweave.data.preprocess frames --dataset scannet --root $SCANNET --cache_dir $CACHE/scannet --split train
pixi run python -m crossweave.data.preprocess eval   --dataset scannet --root $SCANNET --cache_dir $CACHE/scannet

pixi run python -m crossweave.data.preprocess unpack --dataset threerscan --root $THREERSCAN
pixi run python -m crossweave.data.preprocess encode --dataset threerscan --root $THREERSCAN --cache_dir $CACHE/3rscan
pixi run python -m crossweave.data.preprocess frames --dataset threerscan --root $THREERSCAN --cache_dir $CACHE/3rscan --split train
pixi run python -m crossweave.data.preprocess eval   --dataset threerscan --root $THREERSCAN --cache_dir $CACHE/3rscan
```

### Steps

1. **unpack**
   - **ScanNet:** exports the `.sens` files to `color/<frame>.jpg` (every frame for val, every 30th for train), `pose.npy`, `intrinsic_color.txt`, `instance-filt/` (val only) and `unpacked.json` in each scene folder.
   - **3RScan:** extracts `sequence.zip` into `data/<scan>/sequence/`.
2. **encode:** runs the frozen Utonia encoder on each mesh's vertices, using geometry and normals only (colour set to zero). Writes `pointcloud.pt` with `orig_coord (N, 3)` and `feat (N, 1224) float16`, one row per mesh vertex. Encoding is seeded per scene (`--seed`, default 0).
3. **frames:** for every training frame (ScanNet every 30th, 3RScan every 5th), renders occlusion depth from the mesh and projects the vertices into the 518 × 518 image. One visible point is kept per 14 × 14 patch, chosen with a per-frame seed. Frames with fewer than 128 pairs are dropped.
4. **eval:** writes per-vertex instance labels and orders the frames of each val scene by farthest-point sampling over camera poses. It then collects each object's views, sorted by visible mask area. For 3RScan, object masks are rendered from the label mesh into `C/masks/`.

The frame cache is keyed by the image backbone's grid (`frames_<img_size>_p<patch_size>`). DINOv2-L/14 and
TIPSv2-L/14 share `frames_518_p14`.

### Notes

- **ScanNet `unpack --rgb_stride` must equal `frames --frame_stride`** (both default to 30). Otherwise training fails on missing colour frames.
- **`eval --no_save_masks` must not be used for 3RScan.** Instance evaluation reads those masks.
- **`frames` runs on the train split only.** Nothing reads val frames.

## 3. Expected outputs

Reference values from a complete run:

| | ScanNet | 3RScan |
|---|---|---|
| Train scenes / kept frames | 1,201 / 63,705 | 1,178 / 56,889 |
| Val scenes (scene retrieval) | 312 | 157 |
| Val objects (instance retrieval) | 7,723 | 3,288 |
| Cache: `scenes/` | 514 GB | 159 GB |
| Cache: `frames_518_p14/` | 241 GB | 97 GB |
| Cache: `masks/` | – | 0.4 GB |

Final cache layout:

```
$CACHE/<dataset>/
    scenes/<scene>/pointcloud.pt, labels.npz, eval.json
    frames_518_p14/<scene>/<frame>.pt, _done.json
    masks/<scan>/<frame>.png                 # 3RScan only
    train_frames_518_p14.json                # training frame list
    val_scene_frames.json, val_instances.json
```
