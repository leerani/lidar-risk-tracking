import csv
from collections import Counter
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes


# =========================================================
# Paths
# =========================================================

PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "pointpillars_tracks.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pointpillars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "pointpillars_ttc_risk.csv"

DATA_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "OpenPCDet"
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)


# =========================================================
# TTC / Risk parameters
# =========================================================

MIN_CLOSING_SPEED = 0.1  # m/s

DANGER_TTC = 2.0
WARNING_TTC = 4.0
CAUTION_TTC = 8.0


# =========================================================
# Load nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Ego pose
# =========================================================

def get_ego_position(sample_token):
    """
    LIDAR_TOP 촬영 시점의 Ego global position 반환.
    """

    sample = nusc.get(
        "sample",
        sample_token,
    )

    lidar_token = sample["data"]["LIDAR_TOP"]

    sample_data = nusc.get(
        "sample_data",
        lidar_token,
    )

    ego_pose = nusc.get(
        "ego_pose",
        sample_data["ego_pose_token"],
    )

    position = np.asarray(
        ego_pose["translation"],
        dtype=float,
    )

    timestamp = sample_data["timestamp"]

    return position, timestamp


def get_ego_velocity(sample_token):
    """
    이전/다음 keyframe의 ego 위치를 이용해
    global 좌표계 ego velocity 추정.

    중앙 차분을 우선 사용하고,
    첫/마지막 frame에서는 forward/backward difference 사용.
    """

    sample = nusc.get(
        "sample",
        sample_token,
    )

    current_pos, current_time = (
        get_ego_position(sample_token)
    )

    prev_token = sample["prev"]
    next_token = sample["next"]

    # ---------------------------------------------
    # Central difference
    # ---------------------------------------------

    if prev_token and next_token:

        prev_pos, prev_time = (
            get_ego_position(prev_token)
        )

        next_pos, next_time = (
            get_ego_position(next_token)
        )

        dt = (
            next_time - prev_time
        ) / 1_000_000.0

        if dt > 0:
            velocity = (
                next_pos - prev_pos
            ) / dt

            return velocity[:2]

    # ---------------------------------------------
    # Forward difference
    # ---------------------------------------------

    if next_token:

        next_pos, next_time = (
            get_ego_position(next_token)
        )

        dt = (
            next_time - current_time
        ) / 1_000_000.0

        if dt > 0:
            velocity = (
                next_pos - current_pos
            ) / dt

            return velocity[:2]

    # ---------------------------------------------
    # Backward difference
    # ---------------------------------------------

    if prev_token:

        prev_pos, prev_time = (
            get_ego_position(prev_token)
        )

        dt = (
            current_time - prev_time
        ) / 1_000_000.0

        if dt > 0:
            velocity = (
                current_pos - prev_pos
            ) / dt

            return velocity[:2]

    return np.zeros(2, dtype=float)


# =========================================================
# TTC
# =========================================================

def calculate_ttc(
    object_position,
    object_velocity,
    ego_position,
    ego_velocity,
):
    """
    Radial TTC.

    rel_position:
        Ego → Object 벡터

    rel_velocity:
        Object velocity - Ego velocity

    closing_speed:
        상대속도 중 Ego 방향으로 접근하는 성분

    closing_speed <= 0 이면
    현재 서로 가까워지고 있지 않으므로 TTC = inf.
    """

    relative_position = (
        object_position - ego_position
    )

    relative_velocity = (
        object_velocity - ego_velocity
    )

    distance = np.linalg.norm(
        relative_position
    )

    if distance < 1e-6:
        return (
            0.0,
            0.0,
            0.0,
        )

    unit_direction = (
        relative_position / distance
    )

    radial_velocity = np.dot(
        relative_velocity,
        unit_direction,
    )

    # radial_velocity < 0:
    # distance 감소 중
    closing_speed = -radial_velocity

    if closing_speed <= MIN_CLOSING_SPEED:
        ttc = np.inf

    else:
        ttc = (
            distance / closing_speed
        )

    return (
        float(distance),
        float(closing_speed),
        float(ttc),
    )


# =========================================================
# Risk level
# =========================================================

def get_risk_level(ttc):

    if not np.isfinite(ttc):
        return "SAFE"

    if ttc < DANGER_TTC:
        return "DANGER"

    if ttc < WARNING_TTC:
        return "WARNING"

    if ttc < CAUTION_TTC:
        return "CAUTION"

    return "SAFE"


# =========================================================
# Load Tracking CSV
# =========================================================

rows = []

with open(
    TRACK_CSV,
    newline="",
) as f:

    reader = csv.DictReader(f)

    track_rows = list(reader)


# =========================================================
# Cache ego states
# =========================================================

sample_tokens = {
    row["sample_token"]
    for row in track_rows
}

ego_cache = {}

print(
    f"Preparing ego states for "
    f"{len(sample_tokens)} frames..."
)

for sample_token in sample_tokens:

    position, _ = (
        get_ego_position(sample_token)
    )

    velocity = (
        get_ego_velocity(sample_token)
    )

    ego_cache[sample_token] = {
        "position": position[:2],
        "velocity": velocity,
    }


# =========================================================
# Compute TTC for every confirmed track
# =========================================================

for row in track_rows:

    sample_token = row["sample_token"]

    ego = ego_cache[sample_token]

    object_position = np.array(
        [
            float(row["global_x"]),
            float(row["global_y"]),
        ]
    )

    object_velocity = np.array(
        [
            float(row["vx"]),
            float(row["vy"]),
        ]
    )

    distance, closing_speed, ttc = (
        calculate_ttc(
            object_position,
            object_velocity,
            ego["position"],
            ego["velocity"],
        )
    )

    risk_level = (
        get_risk_level(ttc)
    )

    rows.append(
        {
            "scene": row["scene"],
            "frame": row["frame"],
            "sample_token": sample_token,

            "track_id": row["track_id"],
            "class": row["class"],

            "distance_m": distance,

            "object_speed_mps": float(
                row["speed_mps"]
            ),

            "ego_speed_mps": float(
                np.linalg.norm(
                    ego["velocity"]
                )
            ),

            "closing_speed_mps": (
                closing_speed
            ),

            "ttc_s": (
                ttc
                if np.isfinite(ttc)
                else ""
            ),

            "risk_level": risk_level,

            "score": row["score"],
            "hits": row["hits"],
        }
    )


# =========================================================
# Save
# =========================================================

fieldnames = [
    "scene",
    "frame",
    "sample_token",

    "track_id",
    "class",

    "distance_m",

    "object_speed_mps",
    "ego_speed_mps",
    "closing_speed_mps",

    "ttc_s",
    "risk_level",

    "score",
    "hits",
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
    writer.writerows(rows)


# =========================================================
# Summary
# =========================================================

risk_counts = Counter(
    row["risk_level"]
    for row in rows
)

finite_ttc_rows = [
    row
    for row in rows
    if row["ttc_s"] != ""
]

finite_ttc_rows.sort(
    key=lambda x: x["ttc_s"]
)


print()
print("=" * 65)
print("TTC RISK SUMMARY")
print("=" * 65)

print(
    f"Total confirmed observations : "
    f"{len(rows)}"
)

print()

for level in [
    "SAFE",
    "CAUTION",
    "WARNING",
    "DANGER",
]:
    print(
        f"{level:8s} : "
        f"{risk_counts[level]:4d}"
    )

print()

print(
    f"Finite TTC observations      : "
    f"{len(finite_ttc_rows)}"
)

print()


# =========================================================
# Top dangerous observations
# =========================================================

print("Lowest TTC observations")
print("-" * 65)

print(
    f"{'scene':12s}"
    f"{'frame':>7s}"
    f"{'ID':>6s}"
    f"{'class':>14s}"
    f"{'dist':>9s}"
    f"{'closing':>10s}"
    f"{'TTC':>8s}"
    f"{'risk':>10s}"
)

for row in finite_ttc_rows[:15]:

    print(
        f"{row['scene']:12s}"
        f"{int(row['frame']):7d}"
        f"{int(row['track_id']):6d}"
        f"{row['class']:>14s}"
        f"{row['distance_m']:9.2f}"
        f"{row['closing_speed_mps']:10.2f}"
        f"{row['ttc_s']:8.2f}"
        f"{row['risk_level']:>10s}"
    )


print()
print(f"Saved: {OUTPUT_CSV}")
print("=" * 65)