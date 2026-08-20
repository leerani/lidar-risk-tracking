import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

OPENPCDET_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "OpenPCDet"
)

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "camera_lidar"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Settings
# =========================================================

CAMERA_NAME = "CAM_FRONT"

SCORE_THRESHOLD = 0.30

TARGET_SCENE = "scene-0103"
TARGET_FRAME = 38


# =========================================================
# Load
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

with open(RESULT_PATH, "rb") as f:
    predictions = pickle.load(f)


# =========================================================
# Find target sample
# =========================================================

scene_record = None

for scene in nusc.scene:
    if scene["name"] == TARGET_SCENE:
        scene_record = scene
        break

if scene_record is None:
    raise RuntimeError(
        f"Scene not found: {TARGET_SCENE}"
    )


sample_token = scene_record[
    "first_sample_token"
]

for _ in range(TARGET_FRAME):
    sample = nusc.get(
        "sample",
        sample_token,
    )

    if not sample["next"]:
        raise RuntimeError(
            "Frame index exceeds scene length."
        )

    sample_token = sample["next"]


sample = nusc.get(
    "sample",
    sample_token,
)


# =========================================================
# Find matching PointPillars result
# =========================================================

pred = None

for result in predictions:

    if (
        result["metadata"]["token"]
        == sample_token
    ):
        pred = result
        break


if pred is None:
    raise RuntimeError(
        "Prediction not found for target sample."
    )


# =========================================================
# Camera
# =========================================================

camera_token = sample["data"][
    CAMERA_NAME
]

camera_data = nusc.get(
    "sample_data",
    camera_token,
)

camera_calib = nusc.get(
    "calibrated_sensor",
    camera_data[
        "calibrated_sensor_token"
    ],
)

camera_ego = nusc.get(
    "ego_pose",
    camera_data[
        "ego_pose_token"
    ],
)

camera_intrinsic = np.asarray(
    camera_calib[
        "camera_intrinsic"
    ],
)

camera_path = (
    DATA_ROOT
    / camera_data["filename"]
)

image = cv2.imread(
    str(camera_path)
)

if image is None:
    raise RuntimeError(
        f"Could not load image: {camera_path}"
    )

image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB,
)


# =========================================================
# LiDAR calibration
# =========================================================

lidar_token = sample["data"][
    "LIDAR_TOP"
]

lidar_data = nusc.get(
    "sample_data",
    lidar_token,
)

lidar_calib = nusc.get(
    "calibrated_sensor",
    lidar_data[
        "calibrated_sensor_token"
    ],
)

lidar_ego = nusc.get(
    "ego_pose",
    lidar_data[
        "ego_pose_token"
    ],
)


# =========================================================
# LiDAR box → Camera coordinate
# =========================================================

def lidar_box_to_camera(box_lidar):

    x, y, z = (
        float(box_lidar[0]),
        float(box_lidar[1]),
        float(box_lidar[2]),
    )

    length = float(box_lidar[3])
    width = float(box_lidar[4])
    height = float(box_lidar[5])

    yaw = float(box_lidar[6])

    # nuScenes Box uses w, l, h
    box = Box(
        center=[x, y, z],
        size=[
            width,
            length,
            height,
        ],
        orientation=Quaternion(
            axis=[0, 0, 1],
            radians=yaw,
        ),
    )

    # ---------------------------------------------
    # LiDAR → Ego
    # ---------------------------------------------

    box.rotate(
        Quaternion(
            lidar_calib["rotation"]
        )
    )

    box.translate(
        np.asarray(
            lidar_calib["translation"]
        )
    )

    # ---------------------------------------------
    # Ego → Global
    # ---------------------------------------------

    box.rotate(
        Quaternion(
            lidar_ego["rotation"]
        )
    )

    box.translate(
        np.asarray(
            lidar_ego["translation"]
        )
    )

    # ---------------------------------------------
    # Global → Camera Ego
    # ---------------------------------------------

    box.translate(
        -np.asarray(
            camera_ego["translation"]
        )
    )

    box.rotate(
        Quaternion(
            camera_ego["rotation"]
        ).inverse
    )

    # ---------------------------------------------
    # Camera Ego → Camera Sensor
    # ---------------------------------------------

    box.translate(
        -np.asarray(
            camera_calib["translation"]
        )
    )

    box.rotate(
        Quaternion(
            camera_calib["rotation"]
        ).inverse
    )

    return box


# =========================================================
# Drawing
# =========================================================

fig, ax = plt.subplots(
    figsize=(14, 8)
)

ax.imshow(image)

image_height, image_width = image.shape[:2]

ax.set_xlim(0, image_width)
ax.set_ylim(image_height, 0)

boxes = pred["boxes_lidar"]
names = pred["name"]
scores = pred["score"]


num_drawn = 0

for box_lidar, name, score in zip(
    boxes,
    names,
    scores,
):

    score = float(score)

    if score < SCORE_THRESHOLD:
        continue

    box_cam = lidar_box_to_camera(
        box_lidar
    )

    # 객체가 카메라 뒤에 있으면 제외
    corners_3d = box_cam.corners()

    if np.all(
        corners_3d[2, :] <= 0.1
    ):
        continue

    # 일부 corner가 카메라 뒤로 넘어가면
    # projection이 과도하게 깨질 수 있으므로 제외
    if np.any(
        corners_3d[2, :] <= 0.1
    ):
        continue

    projected = view_points(
        corners_3d,
        camera_intrinsic,
        normalize=True,
    )

    xs = projected[0]
    ys = projected[1]

    image_height, image_width = (
        image.shape[:2]
    )

    # 화면과 완전히 겹치지 않으면 skip
    if (
        xs.max() < 0
        or xs.min() >= image_width
        or ys.max() < 0
        or ys.min() >= image_height
    ):
        continue

    # -------------------------------------------------
    # Projection sanity filtering
    # -------------------------------------------------

    # projected 2D bounding rectangle
    x_min = xs.min()
    x_max = xs.max()
    y_min = ys.min()
    y_max = ys.max()

    box_width = x_max - x_min
    box_height = y_max - y_min

    # 너무 거대한 projection은 제외
    if box_width > image_width * 0.8:
        continue

    if box_height > image_height * 0.8:
        continue

    # 최소한 일부 corner가 실제 이미지 안에 있어야 함
    inside = (
        (xs >= 0)
        & (xs < image_width)
        & (ys >= 0)
        & (ys < image_height)
    )

    if inside.sum() < 2:
        continue

    # box 중심 자체가 카메라에 너무 가까우면 제외
    if box_cam.center[2] < 2.0:
        continue

    # ---------------------------------------------
    # 3D box edges
    # ---------------------------------------------

    edges = [
        (0, 1), (1, 2),
        (2, 3), (3, 0),

        (4, 5), (5, 6),
        (6, 7), (7, 4),

        (0, 4), (1, 5),
        (2, 6), (3, 7),
    ]

    for start, end in edges:

        ax.plot(
            [
                xs[start],
                xs[end],
            ],
            [
                ys[start],
                ys[end],
            ],
            linewidth=1.8,
        )

    # label
    label_x = np.clip(
        xs.min(),
        0,
        image_width - 1,
    )

    label_y = np.clip(
        ys.min() - 5,
        10,
        image_height - 1,
    )

    ax.text(
        label_x,
        label_y,
        f"{name} {score:.2f}",
        fontsize=9,
        bbox={
            "facecolor": "white",
            "alpha": 0.8,
            "pad": 2,
        },
    )

    num_drawn += 1


ax.set_title(
    "Camera–LiDAR Geometric Fusion\n"
    f"{TARGET_SCENE} | frame {TARGET_FRAME} | "
    f"{CAMERA_NAME} | projected detections: {num_drawn}",
    fontsize=14,
    fontweight="bold",
)

ax.axis("off")

plt.tight_layout()


output_path = (
    OUTPUT_DIR
    / f"{TARGET_SCENE}_frame_{TARGET_FRAME:02d}_{CAMERA_NAME}.png"
)

plt.savefig(
    output_path,
    dpi=180,
    bbox_inches="tight",
)

plt.close()


print()
print("=" * 60)
print("CAMERA–LIDAR PROJECTION")
print("=" * 60)

print(f"Scene             : {TARGET_SCENE}")
print(f"Frame             : {TARGET_FRAME}")
print(f"Camera            : {CAMERA_NAME}")
print(f"Projected boxes   : {num_drawn}")

print()
print(f"Saved             : {output_path}")
print("=" * 60)