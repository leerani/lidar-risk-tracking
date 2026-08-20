import csv
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import points_in_box


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

DETAIL_CSV = OUTPUT_DIR / "point_density_objects.csv"
SUMMARY_CSV = OUTPUT_DIR / "point_density_summary.csv"


SCORE_THRESHOLD = 0.30
MATCH_DISTANCE_THRESHOLD = 2.0

DISTANCE_BINS = [
    (0, 20, "0-20m"),
    (20, 40, "20-40m"),
    (40, 50, "40-50m"),
]


def get_distance_bin(distance):
    for low, high, label in DISTANCE_BINS:
        if low <= distance < high:
            return label
    return None


def normalize_gt_name(name):
    if "vehicle.car" in name:
        return "car"
    if "vehicle.truck" in name:
        return "truck"
    if "vehicle.bus" in name:
        return "bus"
    if "vehicle.trailer" in name:
        return "trailer"
    if "vehicle.construction" in name:
        return "construction_vehicle"
    if "human.pedestrian" in name:
        return "pedestrian"
    if "vehicle.motorcycle" in name:
        return "motorcycle"
    if "vehicle.bicycle" in name:
        return "bicycle"
    if "movable_object.trafficcone" in name:
        return "traffic_cone"
    if "movable_object.barrier" in name:
        return "barrier"

    return None


with open(RESULT_PATH, "rb") as f:
    results = pickle.load(f)

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

object_rows = []

for pred in results:
    sample_token = pred["metadata"]["token"]
    sample = nusc.get("sample", sample_token)

    lidar_token = sample["data"]["LIDAR_TOP"]

    # LiDAR point cloud + GT boxes:
    # 둘 다 LIDAR_TOP sensor coordinate 기준.
    lidar_path, gt_boxes, _ = nusc.get_sample_data(lidar_token)

    point_cloud = LidarPointCloud.from_file(lidar_path)

    pred_boxes = pred["boxes_lidar"]
    pred_names = pred["name"]
    pred_scores = pred["score"]

    keep = pred_scores >= SCORE_THRESHOLD

    pred_boxes = pred_boxes[keep]
    pred_names = pred_names[keep]
    pred_scores = pred_scores[keep]

    used_pred = set()

    for gt_box in gt_boxes:
        gt_class = normalize_gt_name(gt_box.name)

        if gt_class is None:
            continue

        gt_x, gt_y, _ = gt_box.center
        gt_distance = float(np.sqrt(gt_x ** 2 + gt_y ** 2))

        distance_bin = get_distance_bin(gt_distance)

        if distance_bin is None:
            continue

        # -----------------------------------------
        # Count LiDAR points inside GT 3D box
        # -----------------------------------------
        inside_mask = points_in_box(
            gt_box,
            point_cloud.points[:3, :],
        )

        num_points = int(inside_mask.sum())

        # -----------------------------------------
        # Match PointPillars prediction
        # Same rule as EXP02:
        # same class + center distance <= 2m
        # -----------------------------------------
        best_idx = None
        best_distance = float("inf")

        for i, (pred_box, pred_name) in enumerate(
            zip(pred_boxes, pred_names)
        ):
            if i in used_pred:
                continue

            if pred_name != gt_class:
                continue

            pred_x = float(pred_box[0])
            pred_y = float(pred_box[1])

            center_distance = np.sqrt(
                (gt_x - pred_x) ** 2
                + (gt_y - pred_y) ** 2
            )

            if center_distance < best_distance:
                best_distance = center_distance
                best_idx = i

        detected = False
        matched_score = np.nan

        if (
            best_idx is not None
            and best_distance <= MATCH_DISTANCE_THRESHOLD
        ):
            detected = True
            used_pred.add(best_idx)
            matched_score = float(pred_scores[best_idx])

        object_rows.append(
            {
                "sample_token": sample_token,
                "class": gt_class,
                "distance_bin": distance_bin,
                "distance_m": gt_distance,
                "num_points": num_points,
                "detected": int(detected),
                "score": matched_score,
            }
        )


# -----------------------------------------
# Save object-level results
# -----------------------------------------
with open(DETAIL_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "sample_token",
            "class",
            "distance_bin",
            "distance_m",
            "num_points",
            "detected",
            "score",
        ],
    )

    writer.writeheader()
    writer.writerows(object_rows)


# -----------------------------------------
# Aggregate statistics
# -----------------------------------------
grouped = defaultdict(list)

for row in object_rows:
    grouped[("all", row["distance_bin"])].append(row)
    grouped[(row["class"], row["distance_bin"])].append(row)


classes_to_report = [
    "all",
    "car",
    "pedestrian",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

summary_rows = []

for cls in classes_to_report:
    for _, _, distance_bin in DISTANCE_BINS:
        rows = grouped[(cls, distance_bin)]

        if not rows:
            summary_rows.append(
                {
                    "class": cls,
                    "distance_bin": distance_bin,
                    "objects": 0,
                    "mean_points": np.nan,
                    "median_points": np.nan,
                    "detected_mean_points": np.nan,
                    "missed_mean_points": np.nan,
                    "recall": np.nan,
                }
            )
            continue

        points = np.array(
            [r["num_points"] for r in rows],
            dtype=float,
        )

        detected_mask = np.array(
            [r["detected"] == 1 for r in rows]
        )

        detected_points = points[detected_mask]
        missed_points = points[~detected_mask]

        recall = float(detected_mask.mean())

        summary_rows.append(
            {
                "class": cls,
                "distance_bin": distance_bin,
                "objects": len(rows),
                "mean_points": float(points.mean()),
                "median_points": float(np.median(points)),
                "detected_mean_points": (
                    float(detected_points.mean())
                    if len(detected_points)
                    else np.nan
                ),
                "missed_mean_points": (
                    float(missed_points.mean())
                    if len(missed_points)
                    else np.nan
                ),
                "recall": recall,
            }
        )


with open(SUMMARY_CSV, "w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "class",
            "distance_bin",
            "objects",
            "mean_points",
            "median_points",
            "detected_mean_points",
            "missed_mean_points",
            "recall",
        ],
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# -----------------------------------------
# Print
# -----------------------------------------
print()
print(
    f"{'class':15s}"
    f"{'distance':10s}"
    f"{'N':>7s}"
    f"{'mean pts':>12s}"
    f"{'median':>10s}"
    f"{'det pts':>12s}"
    f"{'miss pts':>12s}"
    f"{'recall':>10s}"
)

print("-" * 88)

for row in summary_rows:

    def fmt(value):
        if isinstance(value, float) and np.isnan(value):
            return "-"
        return f"{value:.1f}"

    print(
        f"{row['class']:15s}"
        f"{row['distance_bin']:10s}"
        f"{row['objects']:7d}"
        f"{fmt(row['mean_points']):>12s}"
        f"{fmt(row['median_points']):>10s}"
        f"{fmt(row['detected_mean_points']):>12s}"
        f"{fmt(row['missed_mean_points']):>12s}"
        f"{row['recall']:10.3f}"
    )

print()
print(f"detail saved : {DETAIL_CSV}")
print(f"summary saved: {SUMMARY_CSV}")