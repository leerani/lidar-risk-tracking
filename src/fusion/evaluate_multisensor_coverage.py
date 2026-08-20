import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
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
    / "camera_detections_scene0103_all.csv"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "camera_lidar"
    / "multisensor_coverage_scene0103_corrected.csv"
)


TARGET_SCENE = "scene-0103"
CAMERA_NAME = "CAM_FRONT"

CAMERA_IOU_THRESHOLD = 0.50
LIDAR_CENTER_THRESHOLD = 2.0
LIDAR_SCORE_THRESHOLD = 0.30


EVAL_CLASSES = {
    "car",
    "truck",
    "bus",
    "pedestrian",
    "motorcycle",
    "bicycle",
}


def bbox_iou(a, b):

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = (
        max(0.0, ax2 - ax1)
        * max(0.0, ay2 - ay1)
    )

    area_b = (
        max(0.0, bx2 - bx1)
        * max(0.0, by2 - by1)
    )

    union = (
        area_a
        + area_b
        - inter
    )

    if union <= 0:
        return 0.0

    return inter / union


def normalize_gt_name(name):

    if "vehicle.car" in name:
        return "car"

    if "vehicle.truck" in name:
        return "truck"

    if "vehicle.bus" in name:
        return "bus"

    if "vehicle.motorcycle" in name:
        return "motorcycle"

    if "vehicle.bicycle" in name:
        return "bicycle"

    if "human.pedestrian" in name:
        return "pedestrian"

    return None


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Camera detections
# =========================================================

camera_by_sample = defaultdict(list)

with open(CAMERA_CSV, newline="") as f:

    reader = csv.DictReader(f)

    for row in reader:

        camera_by_sample[
            row["sample_token"]
        ].append(
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
# LiDAR detections
# =========================================================

with open(RESULT_PATH, "rb") as f:
    results = pickle.load(f)


lidar_by_sample = {
    pred["metadata"]["token"]: pred
    for pred in results
}


# =========================================================
# Scene samples
# =========================================================

scene = next(
    s
    for s in nusc.scene
    if s["name"] == TARGET_SCENE
)

scene_samples = []

token = scene["first_sample_token"]

while token:

    scene_samples.append(token)

    sample = nusc.get(
        "sample",
        token,
    )

    token = sample["next"]


# =========================================================
# Exact GT annotation → camera bbox
# =========================================================

def get_gt_camera_bbox(
    sample_token,
    ann_token,
):

    sample = nusc.get(
        "sample",
        sample_token,
    )

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

    intrinsic = np.asarray(
        camera_calib[
            "camera_intrinsic"
        ]
    )

    image_width = camera_data["width"]
    image_height = camera_data["height"]

    # THIS exact annotation only
    box = nusc.get_box(
        ann_token
    )

    # global → ego
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

    # ego → camera
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

    corners = box.corners()

    # Camera 뒤에 걸친 객체 제외
    if np.any(
        corners[2, :] <= 0.1
    ):
        return None

    projected = view_points(
        corners,
        intrinsic,
        normalize=True,
    )

    xs = projected[0]
    ys = projected[1]

    x1 = float(xs.min())
    y1 = float(ys.min())
    x2 = float(xs.max())
    y2 = float(ys.max())

    # CAM_FRONT에 전혀 안 보임
    if (
        x2 < 0
        or x1 >= image_width
        or y2 < 0
        or y1 >= image_height
    ):
        return None

    # image clipping
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
# Evaluation
# =========================================================

output_rows = []

total_gt = 0

camera_matched = 0
lidar_matched = 0
union_matched = 0
both_matched = 0


for frame_idx, sample_token in enumerate(
    scene_samples
):

    sample = nusc.get(
        "sample",
        sample_token,
    )

    # ---------------------------------------------
    # GT boxes in LiDAR coordinates
    # ---------------------------------------------

    lidar_token = sample["data"][
        "LIDAR_TOP"
    ]

    _, gt_boxes, _ = nusc.get_sample_data(
        lidar_token
    )

    gt_objects = []

    for gt_box in gt_boxes:

        gt_class = normalize_gt_name(
            gt_box.name
        )

        if gt_class not in EVAL_CLASSES:
            continue

        ann_token = gt_box.token

        if ann_token is None:
            continue

        camera_bbox = get_gt_camera_bbox(
            sample_token,
            ann_token,
        )

        # Important:
        # compare only CAM_FRONT-visible GT
        if camera_bbox is None:
            continue

        gt_objects.append(
            {
                "class": gt_class,

                "ann_token": ann_token,

                "camera_bbox": camera_bbox,

                "lidar_xy": np.array(
                    [
                        float(
                            gt_box.center[0]
                        ),
                        float(
                            gt_box.center[1]
                        ),
                    ]
                ),
            }
        )


    if not gt_objects:
        continue


    total_gt += len(
        gt_objects
    )


    # =====================================================
    # Camera 1:1 matching
    # =====================================================

    camera_dets = camera_by_sample[
        sample_token
    ]

    camera_hits = np.zeros(
        len(gt_objects),
        dtype=bool,
    )


    if camera_dets:

        camera_cost = np.full(
            (
                len(gt_objects),
                len(camera_dets),
            ),
            1e6,
            dtype=float,
        )

        camera_iou = np.zeros(
            (
                len(gt_objects),
                len(camera_dets),
            ),
            dtype=float,
        )


        for i, gt in enumerate(
            gt_objects
        ):

            for j, det in enumerate(
                camera_dets
            ):

                if (
                    gt["class"]
                    != det["class"]
                ):
                    continue

                iou = bbox_iou(
                    gt["camera_bbox"],
                    det["bbox"],
                )

                camera_iou[i, j] = iou
                camera_cost[i, j] = (
                    1.0 - iou
                )


        gt_idx, det_idx = (
            linear_sum_assignment(
                camera_cost
            )
        )


        for i, j in zip(
            gt_idx,
            det_idx,
        ):

            if (
                camera_iou[i, j]
                >= CAMERA_IOU_THRESHOLD
            ):
                camera_hits[i] = True


    # =====================================================
    # LiDAR 1:1 matching
    # =====================================================

    lidar_hits = np.zeros(
        len(gt_objects),
        dtype=bool,
    )

    pred = lidar_by_sample.get(
        sample_token
    )


    if pred is not None:

        valid_lidar = []

        for box, name, score in zip(
            pred["boxes_lidar"],
            pred["name"],
            pred["score"],
        ):

            score = float(score)

            if (
                score
                < LIDAR_SCORE_THRESHOLD
            ):
                continue

            if name not in EVAL_CLASSES:
                continue

            valid_lidar.append(
                {
                    "class": name,

                    "xy": np.array(
                        [
                            float(box[0]),
                            float(box[1]),
                        ]
                    ),
                }
            )


        if valid_lidar:

            lidar_cost = np.full(
                (
                    len(gt_objects),
                    len(valid_lidar),
                ),
                1e6,
                dtype=float,
            )


            for i, gt in enumerate(
                gt_objects
            ):

                for j, det in enumerate(
                    valid_lidar
                ):

                    if (
                        gt["class"]
                        != det["class"]
                    ):
                        continue

                    distance = np.linalg.norm(
                        gt["lidar_xy"]
                        - det["xy"]
                    )

                    lidar_cost[i, j] = (
                        distance
                    )


            gt_idx, det_idx = (
                linear_sum_assignment(
                    lidar_cost
                )
            )


            for i, j in zip(
                gt_idx,
                det_idx,
            ):

                if (
                    lidar_cost[i, j]
                    <= LIDAR_CENTER_THRESHOLD
                ):
                    lidar_hits[i] = True


    # =====================================================
    # Sensor coverage
    # =====================================================

    for i, gt in enumerate(
        gt_objects
    ):

        camera_hit = bool(
            camera_hits[i]
        )

        lidar_hit = bool(
            lidar_hits[i]
        )

        union_hit = (
            camera_hit
            or lidar_hit
        )

        both_hit = (
            camera_hit
            and lidar_hit
        )


        camera_matched += int(
            camera_hit
        )

        lidar_matched += int(
            lidar_hit
        )

        union_matched += int(
            union_hit
        )

        both_matched += int(
            both_hit
        )


        output_rows.append(
            {
                "frame": frame_idx,

                "sample_token": (
                    sample_token
                ),

                "ann_token": (
                    gt["ann_token"]
                ),

                "class": (
                    gt["class"]
                ),

                "camera_hit": int(
                    camera_hit
                ),

                "lidar_hit": int(
                    lidar_hit
                ),

                "fusion_union_hit": int(
                    union_hit
                ),

                "both_hit": int(
                    both_hit
                ),
            }
        )


# =========================================================
# Metrics
# =========================================================

camera_recall = (
    camera_matched / total_gt
    if total_gt
    else 0.0
)

lidar_recall = (
    lidar_matched / total_gt
    if total_gt
    else 0.0
)

fusion_recall = (
    union_matched / total_gt
    if total_gt
    else 0.0
)

both_recall = (
    both_matched / total_gt
    if total_gt
    else 0.0
)


# =========================================================
# Save
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "frame",
            "sample_token",
            "ann_token",
            "class",

            "camera_hit",
            "lidar_hit",

            "fusion_union_hit",
            "both_hit",
        ],
    )

    writer.writeheader()
    writer.writerows(
        output_rows
    )


# =========================================================
# Print
# =========================================================

print()
print("=" * 72)
print("MULTISENSOR GT COVERAGE — CORRECTED")
print("CAM_FRONT-visible GT only / one-to-one matching")
print("=" * 72)

print(
    f"GT objects        : "
    f"{total_gt}"
)

print()

print(
    f"Camera matched    : "
    f"{camera_matched:4d} | "
    f"Coverage Recall "
    f"{camera_recall:.3f}"
)

print(
    f"LiDAR matched     : "
    f"{lidar_matched:4d} | "
    f"Coverage Recall "
    f"{lidar_recall:.3f}"
)

print(
    f"Fusion union      : "
    f"{union_matched:4d} | "
    f"Coverage Recall "
    f"{fusion_recall:.3f}"
)

print(
    f"Both sensors      : "
    f"{both_matched:4d} | "
    f"Coverage Recall "
    f"{both_recall:.3f}"
)

print()

print(
    f"Fusion gain over Camera : "
    f"{fusion_recall - camera_recall:+.3f}"
)

print(
    f"Fusion gain over LiDAR  : "
    f"{fusion_recall - lidar_recall:+.3f}"
)

print()

print(
    f"Saved: {OUTPUT_CSV}"
)

print("=" * 72)