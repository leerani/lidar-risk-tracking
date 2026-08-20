import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from nuscenes.nuscenes import NuScenes


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "pointpillars_tracks.csv"
)

DATA_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "OpenPCDet"
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)

OUTPUT_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "tracking_stability_summary.csv"
)

MATCH_DISTANCE_THRESHOLD = 2.0


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

    return None


# =========================================================
# Load predicted tracks
# =========================================================

tracks_by_sample = defaultdict(list)

with open(TRACK_CSV, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        tracks_by_sample[row["sample_token"]].append(
            {
                "track_id": int(row["track_id"]),
                "class": row["class"],
                "x": float(row["global_x"]),
                "y": float(row["global_y"]),
                "scene": row["scene"],
                "frame": int(row["frame"]),
            }
        )


# =========================================================
# Load nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# GT instance → frame별 predicted track ID
instance_history = defaultdict(list)

total_gt = 0
matched_gt = 0


# =========================================================
# Frame-by-frame GT ↔ predicted track matching
# =========================================================

for sample_token, pred_tracks in tracks_by_sample.items():

    sample = nusc.get("sample", sample_token)

    gt_objects = []

    for ann_token in sample["anns"]:

        ann = nusc.get(
            "sample_annotation",
            ann_token,
        )

        gt_class = normalize_gt_name(
            ann["category_name"]
        )

        if gt_class is None:
            continue

        gt_objects.append(
            {
                "instance_token": ann["instance_token"],
                "class": gt_class,
                "x": float(ann["translation"][0]),
                "y": float(ann["translation"][1]),
            }
        )

    total_gt += len(gt_objects)

    if not gt_objects or not pred_tracks:
        continue

    cost = np.full(
        (
            len(gt_objects),
            len(pred_tracks),
        ),
        1e6,
        dtype=float,
    )

    for i, gt in enumerate(gt_objects):

        for j, pred in enumerate(pred_tracks):

            if gt["class"] != pred["class"]:
                continue

            distance = np.hypot(
                gt["x"] - pred["x"],
                gt["y"] - pred["y"],
            )

            cost[i, j] = distance

    gt_indices, pred_indices = (
        linear_sum_assignment(cost)
    )

    for gt_idx, pred_idx in zip(
        gt_indices,
        pred_indices,
    ):

        distance = cost[
            gt_idx,
            pred_idx,
        ]

        if distance > MATCH_DISTANCE_THRESHOLD:
            continue

        gt = gt_objects[gt_idx]
        pred = pred_tracks[pred_idx]

        matched_gt += 1

        instance_history[
            gt["instance_token"]
        ].append(
            {
                "frame": pred["frame"],
                "scene": pred["scene"],
                "track_id": pred["track_id"],
                "class": gt["class"],
            }
        )


# =========================================================
# ID Switch calculation
# =========================================================

summary_rows = []

total_switches = 0
total_transitions = 0

stable_instances = 0

for instance_token, history in instance_history.items():

    history.sort(
        key=lambda x: x["frame"]
    )

    ids = [
        item["track_id"]
        for item in history
    ]

    switches = 0

    for previous_id, current_id in zip(
        ids[:-1],
        ids[1:],
    ):
        total_transitions += 1

        if previous_id != current_id:
            switches += 1
            total_switches += 1

    unique_ids = len(set(ids))

    if unique_ids == 1:
        stable_instances += 1

    summary_rows.append(
        {
            "instance_token": instance_token,
            "class": history[0]["class"],
            "scene": history[0]["scene"],
            "matched_frames": len(history),
            "unique_track_ids": unique_ids,
            "id_switches": switches,
        }
    )


# =========================================================
# Metrics
# =========================================================

num_instances = len(instance_history)

match_rate = (
    matched_gt / total_gt
    if total_gt > 0
    else 0
)

id_switch_rate = (
    total_switches / total_transitions
    if total_transitions > 0
    else 0
)

stable_instance_rate = (
    stable_instances / num_instances
    if num_instances > 0
    else 0
)


# =========================================================
# Save
# =========================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
) as f:

    fieldnames = [
        "instance_token",
        "class",
        "scene",
        "matched_frames",
        "unique_track_ids",
        "id_switches",
    ]

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# =========================================================
# Print
# =========================================================

print()
print("=" * 60)

print(f"GT observations       : {total_gt}")
print(f"Matched observations  : {matched_gt}")
print(f"GT match rate         : {match_rate:.3f}")

print()

print(f"Matched GT instances  : {num_instances}")
print(f"Stable instances      : {stable_instances}")
print(f"Stable instance rate  : {stable_instance_rate:.3f}")

print()

print(f"Track transitions     : {total_transitions}")
print(f"ID switches           : {total_switches}")
print(f"ID switch rate        : {id_switch_rate:.3f}")

print()

print(f"Saved                 : {OUTPUT_CSV}")

print("=" * 60)