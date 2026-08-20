import sys
import time
from pathlib import Path
import copy

import numpy as np
import torch
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

ONNX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "edge"
    / "pointpillars_backbone.onnx"
)


WARMUP = 20
RUNS = 100


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
# Load model
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
# Real BEV input
# =========================================================

batch_dict = next(iter(dataloader))

load_data_to_gpu(batch_dict)

with torch.no_grad():

    batch_dict = model.module_list[0](
        batch_dict
    )

    batch_dict = model.module_list[1](
        batch_dict
    )

    spatial_features_gpu = batch_dict[
        "spatial_features"
    ].contiguous()


spatial_features_cpu = (
    spatial_features_gpu
    .detach()
    .cpu()
    .contiguous()
)

# =========================================================
# Backbone wrappers
# GPU / CPU 모델을 서로 독립적으로 복사
# =========================================================

backbone_gpu_module = copy.deepcopy(
    model.module_list[2]
).cuda().eval()

backbone_cpu_module = copy.deepcopy(
    model.module_list[2]
).cpu().eval()

backbone_gpu = BackboneWrapper(
    backbone_gpu_module
).cuda().eval()

backbone_cpu = BackboneWrapper(
    backbone_cpu_module
).cpu().eval()

# =========================================================
# PyTorch GPU benchmark
# =========================================================

with torch.no_grad():

    for _ in range(WARMUP):
        _ = backbone_gpu(
            spatial_features_gpu
        )

    torch.cuda.synchronize()

    times = []

    for _ in range(RUNS):

        start = torch.cuda.Event(
            enable_timing=True
        )

        end = torch.cuda.Event(
            enable_timing=True
        )

        start.record()

        _ = backbone_gpu(
            spatial_features_gpu
        )

        end.record()

        torch.cuda.synchronize()

        times.append(
            start.elapsed_time(end)
        )


gpu_ms = float(
    np.mean(times)
)

gpu_p50 = float(
    np.percentile(times, 50)
)

gpu_p95 = float(
    np.percentile(times, 95)
)


# =========================================================
# PyTorch CPU benchmark
# =========================================================

with torch.no_grad():

    for _ in range(WARMUP):
        _ = backbone_cpu(
            spatial_features_cpu
        )

    times = []

    for _ in range(RUNS):

        start = time.perf_counter()

        _ = backbone_cpu(
            spatial_features_cpu
        )

        end = time.perf_counter()

        times.append(
            (end - start) * 1000.0
        )


torch_cpu_ms = float(
    np.mean(times)
)

torch_cpu_p50 = float(
    np.percentile(times, 50)
)

torch_cpu_p95 = float(
    np.percentile(times, 95)
)


# =========================================================
# ONNX Runtime CPU benchmark
# =========================================================

session = ort.InferenceSession(
    str(ONNX_PATH),
    providers=[
        "CPUExecutionProvider"
    ],
)

onnx_input = (
    spatial_features_cpu
    .numpy()
)


for _ in range(WARMUP):

    _ = session.run(
        None,
        {
            "spatial_features":
            onnx_input
        },
    )


times = []

for _ in range(RUNS):

    start = time.perf_counter()

    _ = session.run(
        None,
        {
            "spatial_features":
            onnx_input
        },
    )

    end = time.perf_counter()

    times.append(
        (end - start) * 1000.0
    )


onnx_cpu_ms = float(
    np.mean(times)
)

onnx_cpu_p50 = float(
    np.percentile(times, 50)
)

onnx_cpu_p95 = float(
    np.percentile(times, 95)
)


# =========================================================
# Summary
# =========================================================

print()
print("=" * 72)
print("POINTPILLARS BACKBONE RUNTIME BENCHMARK")
print("=" * 72)

print(
    f"Input shape              : "
    f"{tuple(spatial_features_gpu.shape)}"
)

print(
    f"Warmup / Runs            : "
    f"{WARMUP} / {RUNS}"
)

print()

print(
    f"PyTorch GPU mean         : "
    f"{gpu_ms:.3f} ms"
)

print(
    f"PyTorch GPU p50 / p95    : "
    f"{gpu_p50:.3f} / "
    f"{gpu_p95:.3f} ms"
)

print(
    f"PyTorch GPU FPS          : "
    f"{1000.0 / gpu_ms:.2f}"
)

print()

print(
    f"PyTorch CPU mean         : "
    f"{torch_cpu_ms:.3f} ms"
)

print(
    f"PyTorch CPU p50 / p95    : "
    f"{torch_cpu_p50:.3f} / "
    f"{torch_cpu_p95:.3f} ms"
)

print(
    f"PyTorch CPU FPS          : "
    f"{1000.0 / torch_cpu_ms:.2f}"
)

print()

print(
    f"ONNX Runtime CPU mean    : "
    f"{onnx_cpu_ms:.3f} ms"
)

print(
    f"ONNX Runtime CPU p50/p95 : "
    f"{onnx_cpu_p50:.3f} / "
    f"{onnx_cpu_p95:.3f} ms"
)

print(
    f"ONNX Runtime CPU FPS     : "
    f"{1000.0 / onnx_cpu_ms:.2f}"
)

print()

speedup = (
    torch_cpu_ms
    / onnx_cpu_ms
)

print(
    f"ONNX CPU speedup         : "
    f"{speedup:.2f}x"
)

print("=" * 72)