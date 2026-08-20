import sys
from pathlib import Path

import numpy as np
import torch
import onnx
import onnxruntime as ort


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

sys.path.insert(0, str(OPENPCDET_ROOT))

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "edge"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_PATH = (
    OUTPUT_DIR
    / "pointpillars_backbone.onnx"
)


# =========================================================
# Wrapper
# =========================================================

class BackboneWrapper(torch.nn.Module):

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone

    def forward(self, spatial_features):

        batch_dict = {
            "spatial_features": spatial_features
        }

        batch_dict = self.backbone(
            batch_dict
        )

        return batch_dict[
            "spatial_features_2d"
        ]


# =========================================================
# Config / model
# =========================================================

logger = common_utils.create_logger()

cfg_from_yaml_file(
    str(CFG_FILE),
    cfg,
)

dataset, dataloader, _ = build_dataloader(
    dataset_cfg=cfg.DATA_CONFIG,
    class_names=cfg.CLASS_NAMES,
    batch_size=1,
    dist=False,
    workers=0,
    logger=logger,
    training=False,
)

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
# Prepare one real sample
# =========================================================

batch_dict = next(
    iter(dataloader)
)

load_data_to_gpu(
    batch_dict
)

with torch.no_grad():

    # PillarVFE
    batch_dict = model.module_list[0](
        batch_dict
    )

    # PointPillarScatter
    batch_dict = model.module_list[1](
        batch_dict
    )

    spatial_features = batch_dict[
        "spatial_features"
    ]


print()
print("=" * 70)
print("POINTPILLARS BACKBONE ONNX EXPORT")
print("=" * 70)

print(
    "Input spatial_features :",
    tuple(spatial_features.shape),
)


# =========================================================
# PyTorch output
# =========================================================

wrapper = BackboneWrapper(
    model.module_list[2]
)

wrapper.cuda()
wrapper.eval()

with torch.no_grad():

    torch_output = wrapper(
        spatial_features
    )


print(
    "PyTorch output         :",
    tuple(torch_output.shape),
)


# =========================================================
# Export
# =========================================================

torch.onnx.export(
    wrapper,
    spatial_features,

    str(ONNX_PATH),

    input_names=[
        "spatial_features"
    ],

    output_names=[
        "spatial_features_2d"
    ],

    opset_version=17,

    do_constant_folding=True,
)


print()
print(
    f"ONNX saved             : "
    f"{ONNX_PATH}"
)


# =========================================================
# Validate ONNX
# =========================================================

onnx_model = onnx.load(
    str(ONNX_PATH)
)

onnx.checker.check_model(
    onnx_model
)

print(
    "ONNX checker           : PASS"
)


# =========================================================
# ONNX Runtime CPU inference
# =========================================================

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

input_numpy = (
    spatial_features
    .detach()
    .cpu()
    .numpy()
)

onnx_output = session.run(
    None,
    {
        "spatial_features":
        input_numpy
    },
)[0]


# =========================================================
# Numerical comparison
# =========================================================

torch_numpy = (
    torch_output
    .detach()
    .cpu()
    .numpy()
)

abs_diff = np.abs(
    torch_numpy - onnx_output
)

max_abs_diff = np.max(
    abs_diff
)

mean_abs_diff = np.mean(
    abs_diff
)

p99 = np.percentile(
    abs_diff,
    99
)

p999 = np.percentile(
    abs_diff,
    99.9
)

relative_l2 = (
    np.linalg.norm(
        torch_numpy - onnx_output
    )
    /
    (
        np.linalg.norm(torch_numpy)
        + 1e-12
    )
)

cosine_similarity = (
    np.sum(
        torch_numpy * onnx_output
    )
    /
    (
        np.linalg.norm(torch_numpy)
        * np.linalg.norm(onnx_output)
        + 1e-12
    )
)

allclose = np.allclose(
    torch_numpy,
    onnx_output,
    rtol=1e-3,
    atol=1e-3,
)


print()

print(
    "ONNX output            :",
    onnx_output.shape,
)

print(
    f"Max absolute diff       : "
    f"{max_abs_diff:.8f}"
)

print(
    f"Mean absolute diff      : "
    f"{mean_abs_diff:.8f}"
)

print(
    f"99th percentile diff    : "
    f"{p99:.8f}"
)

print(
    f"99.9th percentile diff  : "
    f"{p999:.8f}"
)

print(
    f"Relative L2 error       : "
    f"{relative_l2:.8f}"
)

print(
    f"Cosine similarity       : "
    f"{cosine_similarity:.8f}"
)

print(
    f"np.allclose(1e-3)       : "
    f"{allclose}"
)


print()
print("=" * 70)

if (
    relative_l2 < 1e-3
    and cosine_similarity > 0.999
):
    print(
        "RESULT: ONNX numerical validation PASS"
    )
else:
    print(
        "RESULT: numerical difference needs review"
    )

print("=" * 70)