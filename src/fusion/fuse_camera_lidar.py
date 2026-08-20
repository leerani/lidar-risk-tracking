import csv
import pickle
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from scipy.optimize import linear_sum_assignment


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

CAMERA_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "camera_lidar"
    / "camera_detections_scene0103_frame38.csv"
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

OUTPUT_CSV = (
    OUTPUT_DIR
    / "camera_lidar_fusion_scene0103_frame38.csv"
)


TARGET_SCENE = "scene-0103"
TARGET_FRAME = 38
CAMERA_NAME = "CAM_FRONT"

LIDAR_SCORE_THRESHOLD = 0.30
IOU_THRESHOLD = 0.10

CAMERA_WEIGHT = 0.60
LIDAR_WEIGHT = 0.40


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# IoU
# =========================================================

def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)

    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(
        0.0,
        inter_x2 - inter_x1,
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1,
    )

    inter_area = (
        inter_w * inter_h
    )

    area_a = max(
        0.0,
        ax2 - ax1,
    ) * max(
        0.0,
        ay2 - ay1,
    )

    area_b = max(
        0.0,
        bx2 - bx1,
    ) * max(
        0.0,
        by2 - by1,
    )

    union = (
        area_a
        + area_b
        - inter_area
    )

    if union <= 0:
        return 0.0

    return (
        inter_area / union
    )


# =========================================================
# Find target sample
# =========================================================

scene_record = next(
    scene
    for scene in nusc.scene
    if scene["name"] == TARGET_SCENE
)

sample_token = scene_record[
    "first_sample_token"
]

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


# =========================================================
# Camera calibration
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

image_width = camera_data["width"]
image_height = camera_data["height"]


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
# Projection
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

    # LiDAR → ego
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

    # ego → global
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

    # global → camera ego
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

    # camera ego → camera
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


def project_box_to_2d(box_lidar):

    box_cam = lidar_box_to_camera(
        box_lidar
    )

    corners = box_cam.corners()

    # 일부라도 카메라 뒤면 제외
    if np.any(
        corners[2, :] <= 0.1
    ):
        return None

    projected = view_points(
        corners,
        camera_intrinsic,
        normalize=True,
    )

    xs = projected[0]
    ys = projected[1]

    x1 = float(xs.min())
    y1 = float(ys.min())

    x2 = float(xs.max())
    y2 = float(ys.max())

    # 화면과 전혀 안 겹침
    if (
        x2 < 0
        or x1 >= image_width
        or y2 < 0
        or y1 >= image_height
    ):
        return None

    box_width = x2 - x1
    box_height = y2 - y1

    # 비정상적으로 큰 projection 제거
    if (
        box_width > image_width * 0.8
        or box_height > image_height * 0.8
    ):
        return None

    if box_cam.center[2] < 2.0:
        return None

    # 이미지 영역으로 clipping
    x1 = max(0.0, x1)
    y1 = max(0.0, y1)

    x2 = min(
        float(image_width - 1),
        x2,
    )

    y2 = min(
        float(image_height - 1),
        y2,
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):
        return None

    return (
        x1,
        y1,
        x2,
        y2,
    )


# =========================================================
# Load Camera detections
# =========================================================

camera_detections = []

with open(
    CAMERA_CSV,
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        camera_detections.append(
            {
                "class": row["class"],
                "score": float(
                    row["score"]
                ),

                "bbox": (
                    float(row["x1"]),
                    float(row["y1"]),
                    float(row["x2"]),
                    float(row["y2"]),
                ),
            }
        )


# =========================================================
# Load LiDAR predictions
# =========================================================

with open(
    RESULT_PATH,
    "rb",
) as f:

    results = pickle.load(f)


prediction = None

for pred in results:

    if (
        pred["metadata"]["token"]
        == sample_token
    ):
        prediction = pred
        break


if prediction is None:
    raise RuntimeError(
        "PointPillars prediction not found."
    )


lidar_detections = []

for box, name, score in zip(
    prediction["boxes_lidar"],
    prediction["name"],
    prediction["score"],
):

    score = float(score)

    if score < LIDAR_SCORE_THRESHOLD:
        continue

    bbox_2d = project_box_to_2d(
        box
    )

    if bbox_2d is None:
        continue

    distance = float(
        np.hypot(
            box[0],
            box[1],
        )
    )

    lidar_detections.append(
        {
            "class": name,
            "score": score,
            "bbox": bbox_2d,
            "distance_m": distance,
            "box_lidar": box,
        }
    )


# =========================================================
# Association cost
# =========================================================

cost_matrix = np.full(
    (
        len(camera_detections),
        len(lidar_detections),
    ),
    1e6,
    dtype=float,
)

iou_matrix = np.zeros_like(
    cost_matrix
)


for i, cam in enumerate(
    camera_detections
):

    for j, lidar in enumerate(
        lidar_detections
    ):

        if (
            cam["class"]
            != lidar["class"]
        ):
            continue

        iou = bbox_iou(
            cam["bbox"],
            lidar["bbox"],
        )

        iou_matrix[i, j] = iou

        # Hungarian은 작은 cost를 선호
        cost_matrix[i, j] = (
            1.0 - iou
        )


# =========================================================
# Hungarian matching
# =========================================================

matched_camera = set()
matched_lidar = set()

fusion_rows = []


if (
    len(camera_detections) > 0
    and len(lidar_detections) > 0
):

    cam_indices, lidar_indices = (
        linear_sum_assignment(
            cost_matrix
        )
    )

    for cam_idx, lidar_idx in zip(
        cam_indices,
        lidar_indices,
    ):

        iou = iou_matrix[
            cam_idx,
            lidar_idx,
        ]

        if iou < IOU_THRESHOLD:
            continue

        cam = camera_detections[
            cam_idx
        ]

        lidar = lidar_detections[
            lidar_idx
        ]

        fusion_score = (
            CAMERA_WEIGHT
            * cam["score"]
            +
            LIDAR_WEIGHT
            * lidar["score"]
        )

        matched_camera.add(
            cam_idx
        )

        matched_lidar.add(
            lidar_idx
        )

        fusion_rows.append(
            {
                "status": "MATCHED",

                "class": cam["class"],

                "camera_score": (
                    cam["score"]
                ),

                "lidar_score": (
                    lidar["score"]
                ),

                "fusion_score": (
                    fusion_score
                ),

                "iou": iou,

                "distance_m": (
                    lidar["distance_m"]
                ),

                "camera_bbox": (
                    cam["bbox"]
                ),

                "lidar_bbox": (
                    lidar["bbox"]
                ),
            }
        )


# =========================================================
# Camera-only
# =========================================================

for i, cam in enumerate(
    camera_detections
):

    if i in matched_camera:
        continue

    fusion_rows.append(
        {
            "status": "CAMERA_ONLY",

            "class": cam["class"],

            "camera_score": (
                cam["score"]
            ),

            "lidar_score": "",
            "fusion_score": "",
            "iou": "",

            "distance_m": "",

            "camera_bbox": (
                cam["bbox"]
            ),

            "lidar_bbox": "",
        }
    )


# =========================================================
# LiDAR-only
# =========================================================

for i, lidar in enumerate(
    lidar_detections
):

    if i in matched_lidar:
        continue

    fusion_rows.append(
        {
            "status": "LIDAR_ONLY",

            "class": lidar["class"],

            "camera_score": "",

            "lidar_score": (
                lidar["score"]
            ),

            "fusion_score": "",
            "iou": "",

            "distance_m": (
                lidar["distance_m"]
            ),

            "camera_bbox": "",

            "lidar_bbox": (
                lidar["bbox"]
            ),
        }
    )


# =========================================================
# Save
# =========================================================

fieldnames = [
    "status",
    "class",

    "camera_score",
    "lidar_score",
    "fusion_score",

    "iou",
    "distance_m",

    "camera_bbox",
    "lidar_bbox",
]

with open(
    OUTPUT_CSV,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(
        fusion_rows
    )


# =========================================================
# Summary
# =========================================================

matched = [
    row
    for row in fusion_rows
    if row["status"] == "MATCHED"
]

camera_only = [
    row
    for row in fusion_rows
    if row["status"] == "CAMERA_ONLY"
]

lidar_only = [
    row
    for row in fusion_rows
    if row["status"] == "LIDAR_ONLY"
]


print()
print("=" * 72)
print("CAMERA–LIDAR LATE FUSION")
print("=" * 72)

print(
    f"Camera detections : "
    f"{len(camera_detections)}"
)

print(
    f"LiDAR detections  : "
    f"{len(lidar_detections)}"
)

print()

print(
    f"MATCHED           : "
    f"{len(matched)}"
)

print(
    f"CAMERA_ONLY       : "
    f"{len(camera_only)}"
)

print(
    f"LIDAR_ONLY        : "
    f"{len(lidar_only)}"
)

print()

print("Matched objects")
print("-" * 72)

print(
    f"{'class':12s}"
    f"{'cam':>8s}"
    f"{'lidar':>8s}"
    f"{'fusion':>9s}"
    f"{'IoU':>8s}"
    f"{'dist':>9s}"
)

for row in sorted(
    matched,
    key=lambda x: x["fusion_score"],
    reverse=True,
):

    print(
        f"{row['class']:12s}"
        f"{row['camera_score']:8.3f}"
        f"{row['lidar_score']:8.3f}"
        f"{row['fusion_score']:9.3f}"
        f"{row['iou']:8.3f}"
        f"{row['distance_m']:9.2f}"
    )


print()
print(f"Saved: {OUTPUT_CSV}")
print("=" * 72)