import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes


# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

DATA_ROOT = (
    OPENPCDET_ROOT
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)

RESULT_PATH = (
    OPENPCDET_ROOT
    / "output"
    / "nuscenes_models"
    / "cbgs_pp_multihead_mini"
    / "default"
    / "eval"
    / "epoch_5823"
    / "val"
    / "default"
    / "result.pkl"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pointpillars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "pointpillars_bev_sample0.png"


# --------------------------------------------------
# Settings
# --------------------------------------------------
SAMPLE_INDEX = 0
SCORE_THRESHOLD = 0.30

X_LIMIT = (-50, 50)
Y_LIMIT = (-50, 50)


# --------------------------------------------------
# Helpers
# --------------------------------------------------
def box_bev_corners(x, y, length, width, yaw):
    """Return 4 BEV corners of an oriented 3D box."""
    corners = np.array(
        [
            [ length / 2,  width / 2],
            [ length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2,  width / 2],
        ]
    )

    rotation = np.array(
        [
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)],
        ]
    )

    corners = corners @ rotation.T
    corners[:, 0] += x
    corners[:, 1] += y

    return corners


def draw_box(ax, corners, linewidth=1.5, linestyle="-"):
    closed = np.vstack([corners, corners[0]])

    ax.plot(
        closed[:, 0],
        closed[:, 1],
        linewidth=linewidth,
        linestyle=linestyle,
    )


# --------------------------------------------------
# Load PointPillars predictions
# --------------------------------------------------
with open(RESULT_PATH, "rb") as f:
    results = pickle.load(f)

pred = results[SAMPLE_INDEX]

sample_token = pred["metadata"]["token"]
pred_names = pred["name"]
pred_scores = pred["score"]
pred_boxes = pred["boxes_lidar"]


# --------------------------------------------------
# Load nuScenes
# --------------------------------------------------
nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

sample = nusc.get("sample", sample_token)
lidar_token = sample["data"]["LIDAR_TOP"]

lidar_data = nusc.get("sample_data", lidar_token)
lidar_path = DATA_ROOT / lidar_data["filename"]


# nuScenes LiDAR:
# x, y, z, intensity, ring/index
points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)


# --------------------------------------------------
# Ground-truth boxes in LiDAR sensor coordinates
# --------------------------------------------------
_, gt_boxes, _ = nusc.get_sample_data(lidar_token)


# --------------------------------------------------
# Plot
# --------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# LiDAR points
mask = (
    (points[:, 0] >= X_LIMIT[0])
    & (points[:, 0] <= X_LIMIT[1])
    & (points[:, 1] >= Y_LIMIT[0])
    & (points[:, 1] <= Y_LIMIT[1])
)

bev_points = points[mask]

ax.scatter(
    bev_points[:, 0],
    bev_points[:, 1],
    s=0.4,
    alpha=0.35,
)


# --------------------------------------------------
# Ground Truth
# --------------------------------------------------
for box in gt_boxes:
    x, y, _ = box.center
    width, length, _ = box.wlh
    yaw = box.orientation.yaw_pitch_roll[0]

    corners = box_bev_corners(
        x=x,
        y=y,
        length=length,
        width=width,
        yaw=yaw,
    )

    draw_box(
        ax,
        corners,
        linewidth=1.0,
        linestyle="--",
    )


# --------------------------------------------------
# Predictions
# --------------------------------------------------
num_kept = 0

for name, score, box in zip(
    pred_names,
    pred_scores,
    pred_boxes,
):
    if score < SCORE_THRESHOLD:
        continue

    x, y, z, length, width, height, yaw, vx, vy = box

    corners = box_bev_corners(
        x=x,
        y=y,
        length=length,
        width=width,
        yaw=yaw,
    )

    draw_box(
        ax,
        corners,
        linewidth=2.0,
        linestyle="-",
    )

    ax.text(
        x,
        y,
        f"{name} {score:.2f}",
        fontsize=6,
    )

    num_kept += 1


# Ego vehicle
ax.scatter(
    [0],
    [0],
    marker="x",
    s=80,
    linewidths=2,
)

ax.text(
    0,
    1.5,
    "EGO",
    ha="center",
    fontsize=8,
)


# --------------------------------------------------
# Figure style
# --------------------------------------------------
ax.set_xlim(X_LIMIT)
ax.set_ylim(Y_LIMIT)

ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")

ax.set_title(
    f"PointPillars 3D Detection | "
    f"score ≥ {SCORE_THRESHOLD:.2f} | "
    f"predictions: {num_kept}"
)

ax.grid(alpha=0.2)

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()

print(f"sample index: {SAMPLE_INDEX}")
print(f"sample token: {sample_token}")
print(f"raw predictions: {len(pred_scores)}")
print(f"kept predictions: {num_kept}")
print(f"saved: {OUTPUT_PATH}")