from pcdet.utils import common_utils
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

sys.path.insert(
    0,
    str(OPENPCDET_ROOT),
)

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu


CFG_FILE = (
    OPENPCDET_ROOT
    / "tools"
    / "cfgs"
    / "nuscenes_models"
    / "cbgs_pp_multihead_mini.yaml"
)

CKPT = (
    OPENPCDET_ROOT
    / "checkpoints"
    / "pp_multihead_nds5823_updated.pth"
)


# =========================================================
# Config
# =========================================================

cfg_from_yaml_file(
    str(CFG_FILE),
    cfg,
)


# =========================================================
# Dataset
# =========================================================

logger = common_utils.create_logger()

dataset, dataloader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False,
    workers=0,
    logger=logger,
    training=False,
)


# =========================================================
# Model
# =========================================================

model = build_network(
    model_cfg=cfg.MODEL,
    num_class=len(cfg.CLASS_NAMES),
    dataset=dataset,
)

model.load_params_from_file(
    filename=str(CKPT),
    logger=logger,
    to_cpu=False,
)

model.cuda()
model.eval()


# =========================================================
# One batch
# =========================================================

batch_dict = next(
    iter(dataloader)
)

load_data_to_gpu(
    batch_dict
)


print()
print("=" * 70)
print("POINTPILLARS ONNX INSPECTION")
print("=" * 70)

print()
print("Model modules")
print("-" * 70)

for i, module in enumerate(
    model.module_list
):
    print(
        f"{i:02d} | "
        f"{module.__class__.__name__}"
    )


print()
print("Batch tensors")
print("-" * 70)

for key, value in batch_dict.items():

    if torch.is_tensor(value):

        print(
            f"{key:25s} "
            f"shape={tuple(value.shape)} "
            f"dtype={value.dtype} "
            f"device={value.device}"
        )


# =========================================================
# Forward each module
# =========================================================

print()
print("Forward module outputs")
print("-" * 70)

with torch.no_grad():

    working_dict = batch_dict

    for i, module in enumerate(
        model.module_list
    ):

        working_dict = module(
            working_dict
        )

        print()
        print(
            f"[{i:02d}] "
            f"{module.__class__.__name__}"
        )

        for key, value in working_dict.items():

            if torch.is_tensor(value):

                if key in {
                    "voxels",
                    "voxel_coords",
                    "voxel_num_points",
                    "pillar_features",
                    "spatial_features",
                    "spatial_features_2d",
                    "batch_cls_preds",
                    "batch_box_preds",
                }:

                    print(
                        f"  {key:22s} "
                        f"{tuple(value.shape)}"
                    )


print()
print("=" * 70)
print("Inspection finished.")
print("=" * 70)