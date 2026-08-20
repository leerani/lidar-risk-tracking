import ast
import csv
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from nuscenes.nuscenes import NuScenes


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

DATA_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "OpenPCDet"
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)

FUSION_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "camera_lidar"
    / "camera_lidar_fusion_scene0103_frame38.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "camera_lidar"
)

OUTPUT_PATH = (
    OUTPUT_DIR
    / "camera_lidar_fusion_portfolio.png"
)

TARGET_SCENE = "scene-0103"
TARGET_FRAME = 38
CAMERA_NAME = "CAM_FRONT"


# =========================================================
# nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Find sample
# =========================================================

scene = next(
    s
    for s in nusc.scene
    if s["name"] == TARGET_SCENE
)

sample_token = scene["first_sample_token"]

for _ in range(TARGET_FRAME):
    sample = nusc.get(
        "sample",
        sample_token,
    )
    sample_token = sample["next"]

sample = nusc.get(
    "sample",
    sample_token,
)

camera_token = sample["data"][CAMERA_NAME]

camera_data = nusc.get(
    "sample_data",
    camera_token,
)

image_path = (
    DATA_ROOT
    / camera_data["filename"]
)

image = cv2.imread(
    str(image_path)
)

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB,
)


# =========================================================
# Load fusion results
# =========================================================

rows = []

with open(FUSION_CSV, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        if row["status"] != "MATCHED":
            continue

        rows.append(
            {
                "class": row["class"],

                "camera_score": float(
                    row["camera_score"]
                ),

                "lidar_score": float(
                    row["lidar_score"]
                ),

                "fusion_score": float(
                    row["fusion_score"]
                ),

                "iou": float(
                    row["iou"]
                ),

                "distance_m": float(
                    row["distance_m"]
                ),

                "camera_bbox": ast.literal_eval(
                    row["camera_bbox"]
                ),

                "lidar_bbox": ast.literal_eval(
                    row["lidar_bbox"]
                ),
            }
        )


# highest fusion first
rows.sort(
    key=lambda x: x["fusion_score"],
    reverse=True,
)


# =========================================================
# Plot
# =========================================================

fig, ax = plt.subplots(
    figsize=(16, 9)
)

ax.imshow(image)

image_height, image_width = image.shape[:2]


for idx, row in enumerate(rows):

    cx1, cy1, cx2, cy2 = (
        row["camera_bbox"]
    )

    lx1, ly1, lx2, ly2 = (
        row["lidar_bbox"]
    )

    # ---------------------------------------------
    # Camera bbox — dashed
    # ---------------------------------------------

    camera_rect = Rectangle(
        (cx1, cy1),
        cx2 - cx1,
        cy2 - cy1,
        fill=False,
        linestyle="--",
        linewidth=1.5,
        edgecolor="deepskyblue",
    )

    ax.add_patch(camera_rect)

    # ---------------------------------------------
    # Projected LiDAR bbox — solid
    # ---------------------------------------------

    lidar_rect = Rectangle(
        (lx1, ly1),
        lx2 - lx1,
        ly2 - ly1,
        fill=False,
        linestyle="-",
        linewidth=2.0,
        edgecolor="darkorange",
    )

    ax.add_patch(lidar_rect)

    # ---------------------------------------------
    # Label
    # ---------------------------------------------

    label = (
        f"FUSED | {row['class'].capitalize()}\n"
        f"Cam {row['camera_score']:.2f} | "
        f"LiDAR {row['lidar_score']:.2f}\n"
        f"Fusion {row['fusion_score']:.2f} | "
        f"{row['distance_m']:.1f} m"
    )

    label_x = max(
        5,
        min(cx1, lx1),
    )

    label_y = max(
        20,
        min(cy1, ly1) - 8,
    )

    ax.text(
        label_x,
        label_y,
        label,
        fontsize=8.5,
        fontweight="bold",
        bbox={
            "boxstyle": "round,pad=0.3",
            "facecolor": "white",
            "edgecolor": "darkorange",
            "alpha": 0.88,
        },
    )


# =========================================================
# Legend
# =========================================================

camera_legend = Rectangle(
    (0, 0),
    1,
    1,
    fill=False,
    linestyle="--",
    linewidth=1.5,
    edgecolor="deepskyblue",
)

lidar_legend = Rectangle(
    (0, 0),
    1,
    1,
    fill=False,
    linestyle="-",
    linewidth=2,
    edgecolor="darkorange",
)

ax.legend(
    [
        camera_legend,
        lidar_legend,
    ],
    [
        "Camera 2D Detection",
        "Projected LiDAR 3D Detection",
    ],
    loc="upper left",
    fontsize=10,
)

ax.set_xlim(
    0,
    image_width,
)

ax.set_ylim(
    image_height,
    0,
)

ax.set_title(
    "Camera–LiDAR Late Fusion\n"
    "2D Appearance + 3D Position & Distance",
    fontsize=16,
    fontweight="bold",
    pad=12,
)

ax.axis("off")

plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


print()
print("=" * 60)
print("CAMERA–LIDAR FUSION VISUALIZATION")
print("=" * 60)

print(
    f"Matched objects : {len(rows)}"
)

print(
    f"Saved           : {OUTPUT_PATH}"
)

print("=" * 60)