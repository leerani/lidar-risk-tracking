import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from nuscenes.nuscenes import NuScenes


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"
OPENPCDET_ROOT = PROJECT_ROOT / "third_party" / "OpenPCDet"

DATA_ROOT = (
    OPENPCDET_ROOT
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)

RESULT_PATHS = {
    1: (
        OPENPCDET_ROOT
        / "output"
        / "nuscenes_models"
        / "cbgs_pp_multihead_mini_1sweep"
        / "default"
        / "eval"
    ),
    5: (
        OPENPCDET_ROOT
        / "output"
        / "nuscenes_models"
        / "cbgs_pp_multihead_mini_5sweeps"
        / "default"
        / "eval"
    ),
    10: (
        OPENPCDET_ROOT
        / "output"
        / "nuscenes_models"
        / "cbgs_pp_multihead_mini"
        / "default"
        / "eval"
    ),
}

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pointpillars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "sweep_distance_recall_comparison.csv"

SCORE_THRESHOLD = 0.30
MATCH_DISTANCE_THRESHOLD = 2.0

DISTANCE_BINS = [
    (0, 20, "0-20m"),
    (20, 40, "20-40m"),
    (40, 50, "40-50m"),
]


def find_result_pkl(base_dir):
    candidates = list(base_dir.rglob("result.pkl"))

    if not candidates:
        raise FileNotFoundError(f"result.pkl not found under: {base_dir}")

    # 가장 최근 생성된 result.pkl 사용
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)

    return candidates[0]


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


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)

all_results = []

for sweep_num, base_dir in RESULT_PATHS.items():

    result_path = find_result_pkl(base_dir)

    print(f"{sweep_num} sweep result:")
    print(result_path)

    with open(result_path, "rb") as f:
        results = pickle.load(f)

    stats = defaultdict(
        lambda: {
            "gt": 0,
            "matched": 0,
            "scores": [],
        }
    )

    for pred in results:
        sample_token = pred["metadata"]["token"]

        sample = nusc.get("sample", sample_token)
        lidar_token = sample["data"]["LIDAR_TOP"]

        _, gt_boxes, _ = nusc.get_sample_data(lidar_token)

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
            gt_distance = np.sqrt(gt_x**2 + gt_y**2)

            distance_bin = get_distance_bin(gt_distance)

            if distance_bin is None:
                continue

            stats[distance_bin]["gt"] += 1

            best_idx = None
            best_distance = float("inf")

            for i, (pred_box, pred_name) in enumerate(
                zip(pred_boxes, pred_names)
            ):
                if i in used_pred:
                    continue

                if pred_name != gt_class:
                    continue

                pred_x = pred_box[0]
                pred_y = pred_box[1]

                center_distance = np.sqrt(
                    (gt_x - pred_x) ** 2
                    + (gt_y - pred_y) ** 2
                )

                if center_distance < best_distance:
                    best_distance = center_distance
                    best_idx = i

            if (
                best_idx is not None
                and best_distance <= MATCH_DISTANCE_THRESHOLD
            ):
                used_pred.add(best_idx)

                stats[distance_bin]["matched"] += 1
                stats[distance_bin]["scores"].append(
                    float(pred_scores[best_idx])
                )

    for _, _, distance_bin in DISTANCE_BINS:
        s = stats[distance_bin]

        gt = s["gt"]
        matched = s["matched"]

        recall = matched / gt if gt > 0 else np.nan

        mean_score = (
            np.mean(s["scores"])
            if len(s["scores"]) > 0
            else np.nan
        )

        all_results.append(
            {
                "sweeps": sweep_num,
                "distance_bin": distance_bin,
                "gt_count": gt,
                "matched_count": matched,
                "recall": recall,
                "mean_score": mean_score,
            }
        )


print()
print(
    f"{'sweeps':>8s}"
    f"{'distance':>12s}"
    f"{'GT':>8s}"
    f"{'matched':>10s}"
    f"{'recall':>10s}"
    f"{'score':>10s}"
)

print("-" * 58)

for row in all_results:
    print(
        f"{row['sweeps']:8d}"
        f"{row['distance_bin']:>12s}"
        f"{row['gt_count']:8d}"
        f"{row['matched_count']:10d}"
        f"{row['recall']:10.3f}"
        f"{row['mean_score']:10.3f}"
    )


with open(OUTPUT_CSV, "w") as f:
    f.write(
        "sweeps,distance_bin,gt_count,"
        "matched_count,recall,mean_score\n"
    )

    for row in all_results:
        f.write(
            f"{row['sweeps']},"
            f"{row['distance_bin']},"
            f"{row['gt_count']},"
            f"{row['matched_count']},"
            f"{row['recall']},"
            f"{row['mean_score']}\n"
        )

print()
print(f"saved: {OUTPUT_CSV}")