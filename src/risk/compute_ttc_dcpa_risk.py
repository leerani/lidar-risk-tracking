import csv
from collections import Counter
from pathlib import Path

import numpy as np
from nuscenes.nuscenes import NuScenes


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "pointpillars_tracks.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pointpillars"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = OUTPUT_DIR / "pointpillars_ttc_dcpa_risk.csv"

DATA_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "OpenPCDet"
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)


# =========================================================
# Parameters
# =========================================================

MIN_REL_SPEED = 0.1

DANGER_TTC = 2.0
WARNING_TTC = 4.0
CAUTION_TTC = 8.0

DANGER_DCPA = 2.0
WARNING_DCPA = 4.0
CAUTION_DCPA = 6.0


# =========================================================
# nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


def get_ego_position(sample_token):

    sample = nusc.get("sample", sample_token)
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

    return position[:2], timestamp


def get_ego_velocity(sample_token):

    sample = nusc.get("sample", sample_token)

    current_pos, current_time = (
        get_ego_position(sample_token)
    )

    prev_token = sample["prev"]
    next_token = sample["next"]

    # 중앙 차분
    if prev_token and next_token:

        prev_pos, prev_time = get_ego_position(
            prev_token
        )

        next_pos, next_time = get_ego_position(
            next_token
        )

        dt = (
            next_time - prev_time
        ) / 1_000_000.0

        if dt > 0:
            return (
                next_pos - prev_pos
            ) / dt

    # forward
    if next_token:

        next_pos, next_time = get_ego_position(
            next_token
        )

        dt = (
            next_time - current_time
        ) / 1_000_000.0

        if dt > 0:
            return (
                next_pos - current_pos
            ) / dt

    # backward
    if prev_token:

        prev_pos, prev_time = get_ego_position(
            prev_token
        )

        dt = (
            current_time - prev_time
        ) / 1_000_000.0

        if dt > 0:
            return (
                current_pos - prev_pos
            ) / dt

    return np.zeros(2, dtype=float)


# =========================================================
# TTC + DCPA
# =========================================================

def calculate_ttc_dcpa(
    object_position,
    object_velocity,
    ego_position,
    ego_velocity,
):

    relative_position = (
        object_position - ego_position
    )

    relative_velocity = (
        object_velocity - ego_velocity
    )

    distance = np.linalg.norm(
        relative_position
    )

    rel_speed_sq = np.dot(
        relative_velocity,
        relative_velocity,
    )

    # ---------------------------------------------
    # TTC: radial closing speed 방식
    # ---------------------------------------------

    if distance < 1e-6:
        closing_speed = 0.0
        ttc = 0.0

    else:
        unit_direction = (
            relative_position / distance
        )

        radial_velocity = np.dot(
            relative_velocity,
            unit_direction,
        )

        closing_speed = -radial_velocity

        if closing_speed > MIN_REL_SPEED:
            ttc = (
                distance / closing_speed
            )
        else:
            ttc = np.inf

    # ---------------------------------------------
    # TCPA / DCPA
    #
    # t* = - (r · v) / |v|²
    # DCPA = |r + v*t*|
    # ---------------------------------------------

    if rel_speed_sq < MIN_REL_SPEED ** 2:

        tcpa = np.inf
        dcpa = distance

    else:

        tcpa = -(
            np.dot(
                relative_position,
                relative_velocity,
            )
            / rel_speed_sq
        )

        # 가장 가까운 시점이 이미 과거라면
        # 현재 거리 사용
        if tcpa < 0:
            tcpa = 0.0
            dcpa = distance

        else:
            closest_vector = (
                relative_position
                + relative_velocity * tcpa
            )

            dcpa = np.linalg.norm(
                closest_vector
            )

    return (
        float(distance),
        float(closing_speed),
        float(ttc),
        float(tcpa),
        float(dcpa),
    )


# =========================================================
# Risk
# =========================================================

def get_risk_level(
    ttc,
    dcpa,
):

    # 접근 중이 아니면 SAFE
    if not np.isfinite(ttc):
        return "SAFE"

    # 충돌 경로와 매우 가까움
    if (
        ttc < DANGER_TTC
        and dcpa < DANGER_DCPA
    ):
        return "DANGER"

    if (
        ttc < WARNING_TTC
        and dcpa < WARNING_DCPA
    ):
        return "WARNING"

    if (
        ttc < CAUTION_TTC
        and dcpa < CAUTION_DCPA
    ):
        return "CAUTION"

    return "SAFE"


# =========================================================
# Load tracks
# =========================================================

with open(
    TRACK_CSV,
    newline="",
) as f:

    reader = csv.DictReader(f)
    track_rows = list(reader)


# =========================================================
# Ego cache
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

for token in sample_tokens:

    position, _ = get_ego_position(token)
    velocity = get_ego_velocity(token)

    ego_cache[token] = {
        "position": position,
        "velocity": velocity,
    }


# =========================================================
# Calculate
# =========================================================

rows = []

for row in track_rows:

    token = row["sample_token"]

    ego = ego_cache[token]

    object_position = np.array(
        [
            float(row["global_x"]),
            float(row["global_y"]),
        ],
        dtype=float,
    )

    object_velocity = np.array(
        [
            float(row["vx"]),
            float(row["vy"]),
        ],
        dtype=float,
    )

    (
        distance,
        closing_speed,
        ttc,
        tcpa,
        dcpa,
    ) = calculate_ttc_dcpa(
        object_position,
        object_velocity,
        ego["position"],
        ego["velocity"],
    )

    risk = get_risk_level(
        ttc,
        dcpa,
    )

    rows.append(
        {
            "scene": row["scene"],
            "frame": row["frame"],
            "sample_token": token,

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

            "tcpa_s": (
                tcpa
                if np.isfinite(tcpa)
                else ""
            ),

            "dcpa_m": dcpa,

            "risk_level": risk,

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
    "tcpa_s",
    "dcpa_m",

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

finite_rows = [
    row
    for row in rows
    if row["ttc_s"] != ""
]

finite_rows.sort(
    key=lambda x: x["ttc_s"]
)


print()
print("=" * 70)
print("TTC + DCPA RISK SUMMARY")
print("=" * 70)

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
    f"{len(finite_rows)}"
)

print()

print("Lowest TTC observations")
print("-" * 90)

print(
    f"{'scene':12s}"
    f"{'frame':>7s}"
    f"{'ID':>6s}"
    f"{'class':>14s}"
    f"{'dist':>8s}"
    f"{'close':>9s}"
    f"{'TTC':>7s}"
    f"{'DCPA':>8s}"
    f"{'risk':>10s}"
)

for row in finite_rows[:20]:

    print(
        f"{row['scene']:12s}"
        f"{int(row['frame']):7d}"
        f"{int(row['track_id']):6d}"
        f"{row['class']:>14s}"
        f"{row['distance_m']:8.2f}"
        f"{row['closing_speed_mps']:9.2f}"
        f"{row['ttc_s']:7.2f}"
        f"{row['dcpa_m']:8.2f}"
        f"{row['risk_level']:>10s}"
    )


print()
print(f"Saved: {OUTPUT_CSV}")
print("=" * 70)