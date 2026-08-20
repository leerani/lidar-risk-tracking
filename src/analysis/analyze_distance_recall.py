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

OUTPUT_CSV = OUTPUT_DIR / "distance_recall_baseline.csv"

SCORE_THRESHOLD = 0.30

DISTANCE_BINS = [
    (0, 20, "0-20m"),
    (20, 40, "20-40m"),
    (40, 50, "40-50m"),
]

# 우선 센터 거리 기준으로 간단하게 매칭
MATCH_DISTANCE_THRESHOLD = 2.0


def get_distance_bin(distance):
    for low, high, label in DISTANCE_BINS:
        if low <= distance < high:
            return label
    return None


def normalize_gt_name(name):
    """
    nuScenes category name을 OpenPCDet 10-class 이름으로 변환.
    """
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

    # GT box를 LiDAR 좌표계로 가져옴
    _, gt_boxes, _ = nusc.get_sample_data(lidar_token)

    pred_boxes = pred["boxes_lidar"]
    pred_names = pred["name"]
    pred_scores = pred["score"]

    # score threshold 적용
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
        gt_distance = np.sqrt(gt_x ** 2 + gt_y ** 2)

        distance_bin = get_distance_bin(gt_distance)

        if distance_bin is None:
            continue

        # 전체 클래스 통계
        stats[("all", distance_bin)]["gt"] += 1

        # 클래스별 통계
        stats[(gt_class, distance_bin)]["gt"] += 1

        best_idx = None
        best_distance = float("inf")

        for i, (pred_box, pred_name) in enumerate(
            zip(pred_boxes, pred_names)
        ):
            if i in used_pred:
                continue

            if pred_name != gt_class:
                continue

            pred_x, pred_y = pred_box[0], pred_box[1]

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

            score = float(pred_scores[best_idx])

            stats[("all", distance_bin)]["matched"] += 1
            stats[("all", distance_bin)]["scores"].append(score)

            stats[(gt_class, distance_bin)]["matched"] += 1
            stats[(gt_class, distance_bin)]["scores"].append(score)


rows = []

classes_to_report = [
    "all",
    "car",
    "pedestrian",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
]

for cls in classes_to_report:
    for _, _, distance_bin in DISTANCE_BINS:
        s = stats[(cls, distance_bin)]

        gt = s["gt"]
        matched = s["matched"]

        recall = matched / gt if gt > 0 else np.nan

        mean_score = (
            np.mean(s["scores"])
            if len(s["scores"]) > 0
            else np.nan
        )

        rows.append(
            [
                cls,
                distance_bin,
                gt,
                matched,
                recall,
                mean_score,
            ]
        )


print()
print(
    f"{'class':15s}"
    f"{'distance':10s}"
    f"{'GT':>8s}"
    f"{'matched':>10s}"
    f"{'recall':>10s}"
    f"{'score':>10s}"
)

print("-" * 63)

for row in rows:
    cls, distance_bin, gt, matched, recall, score = row

    recall_str = (
        f"{recall:.3f}"
        if not np.isnan(recall)
        else "-"
    )

    score_str = (
        f"{score:.3f}"
        if not np.isnan(score)
        else "-"
    )

    print(
        f"{cls:15s}"
        f"{distance_bin:10s}"
        f"{gt:8d}"
        f"{matched:10d}"
        f"{recall_str:>10s}"
        f"{score_str:>10s}"
    )


with open(OUTPUT_CSV, "w") as f:
    f.write(
        "class,distance_bin,gt_count,"
        "matched_count,recall,mean_score\n"
    )

    for row in rows:
        cls, distance_bin, gt, matched, recall, score = row

        f.write(
            f"{cls},{distance_bin},{gt},{matched},"
            f"{recall},{score}\n"
        )

print()
print(f"saved: {OUTPUT_CSV}")