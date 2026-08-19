import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


PROJECT_ROOT = Path.home() / "lidar-risk-tracking"

DATA_ROOT = (
    PROJECT_ROOT
    / "third_party"
    / "OpenPCDet"
    / "data"
    / "nuscenes"
    / "v1.0-mini"
)

RISK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "pointpillars_ttc_dcpa_risk.csv"
)

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "pointpillars_tracks.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "portfolio"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_SCENE = "scene-0103"
TARGET_FRAME = 38
FOCUS_TRACK_ID = 196

BEV_RANGE = 35.0


nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Load tracking positions
# =========================================================

track_positions = {}

with open(TRACK_CSV, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        track_positions[
            (
                row["scene"],
                int(row["track_id"]),
                int(row["frame"]),
            )
        ] = np.array(
            [
                float(row["global_x"]),
                float(row["global_y"]),
            ],
            dtype=float,
        )


# =========================================================
# Load risk results
# =========================================================

rows_by_frame = defaultdict(list)
history_by_track = defaultdict(list)

with open(RISK_CSV, newline="") as f:
    reader = csv.DictReader(f)

    for row in reader:
        parsed = {
            "scene": row["scene"],
            "frame": int(row["frame"]),
            "sample_token": row["sample_token"],
            "track_id": int(row["track_id"]),
            "class": row["class"],
            "distance_m": float(row["distance_m"]),
            "ttc_s": (
                float(row["ttc_s"])
                if row["ttc_s"] != ""
                else np.inf
            ),
            "dcpa_m": float(row["dcpa_m"]),
            "risk_level": row["risk_level"],
        }

        rows_by_frame[
            (
                parsed["scene"],
                parsed["frame"],
            )
        ].append(parsed)

        history_by_track[
            (
                parsed["scene"],
                parsed["track_id"],
            )
        ].append(parsed)


for key in history_by_track:
    history_by_track[key].sort(
        key=lambda x: x["frame"]
    )


# =========================================================
# Coordinate transform
# =========================================================

def global_to_lidar(global_xy, sample_token):

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

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"],
    )

    point_global = np.array(
        [
            global_xy[0],
            global_xy[1],
            ego_pose["translation"][2],
        ]
    )

    ego_translation = np.asarray(
        ego_pose["translation"]
    )

    ego_rotation = Quaternion(
        ego_pose["rotation"]
    ).rotation_matrix

    point_ego = (
        ego_rotation.T
        @ (
            point_global
            - ego_translation
        )
    )

    sensor_translation = np.asarray(
        calibrated_sensor["translation"]
    )

    sensor_rotation = Quaternion(
        calibrated_sensor["rotation"]
    ).rotation_matrix

    point_lidar = (
        sensor_rotation.T
        @ (
            point_ego
            - sensor_translation
        )
    )

    return point_lidar[:2]


# =========================================================
# Target frame
# =========================================================

frame_rows = rows_by_frame[
    (
        TARGET_SCENE,
        TARGET_FRAME,
    )
]

if not frame_rows:
    raise RuntimeError(
        f"No data for {TARGET_SCENE}, frame {TARGET_FRAME}"
    )

sample_token = frame_rows[0]["sample_token"]

sample = nusc.get(
    "sample",
    sample_token,
)

lidar_token = sample["data"]["LIDAR_TOP"]

lidar_path = nusc.get_sample_data_path(
    lidar_token
)

pc = LidarPointCloud.from_file(
    lidar_path
)

points = pc.points


# =========================================================
# Plot
# =========================================================

fig, ax = plt.subplots(
    figsize=(10, 10)
)

# LiDAR point cloud
ax.scatter(
    points[0],
    points[1],
    s=0.35,
    alpha=0.22,
    color="gray",
)

# Ego
ax.scatter(
    0,
    0,
    marker="*",
    s=240,
    color="black",
    zorder=8,
)

ax.annotate(
    "EGO",
    (0, 0),
    xytext=(8, -16),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
)


# =========================================================
# General tracks
# =========================================================

for row in frame_rows:

    track_key = (
        row["scene"],
        row["track_id"],
        row["frame"],
    )

    if track_key not in track_positions:
        continue

    pos_lidar = global_to_lidar(
        track_positions[track_key],
        sample_token,
    )

    x, y = pos_lidar

    if (
        abs(x) > BEV_RANGE
        or abs(y) > BEV_RANGE
    ):
        continue

    if row["track_id"] == FOCUS_TRACK_ID:
        continue

    ax.scatter(
        x,
        y,
        s=28,
        color="0.55",
        alpha=0.75,
        zorder=4,
    )

    history = history_by_track[
        (
            row["scene"],
            row["track_id"],
        )
    ]

    trajectory = []

    for hist in history:

        if hist["frame"] > TARGET_FRAME:
            break

        hist_key = (
            hist["scene"],
            hist["track_id"],
            hist["frame"],
        )

        if hist_key not in track_positions:
            continue

        hist_lidar = global_to_lidar(
            track_positions[hist_key],
            sample_token,
        )

        if (
            abs(hist_lidar[0]) <= BEV_RANGE
            and abs(hist_lidar[1]) <= BEV_RANGE
        ):
            trajectory.append(
                hist_lidar
            )

    if len(trajectory) >= 2:

        trajectory = np.asarray(
            trajectory
        )

        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            linewidth=0.8,
            alpha=0.35,
            color="0.6",
            zorder=3,
        )


# =========================================================
# Focus track
# =========================================================

focus_row = None

for row in frame_rows:
    if row["track_id"] == FOCUS_TRACK_ID:
        focus_row = row
        break

if focus_row is None:
    raise RuntimeError(
        f"Track {FOCUS_TRACK_ID} not found."
    )


focus_key = (
    TARGET_SCENE,
    FOCUS_TRACK_ID,
    TARGET_FRAME,
)

focus_pos = global_to_lidar(
    track_positions[focus_key],
    sample_token,
)

focus_x, focus_y = focus_pos


# focus trajectory
focus_history = history_by_track[
    (
        TARGET_SCENE,
        FOCUS_TRACK_ID,
    )
]

focus_traj = []

for hist in focus_history:

    if hist["frame"] > TARGET_FRAME:
        break

    key = (
        hist["scene"],
        hist["track_id"],
        hist["frame"],
    )

    if key not in track_positions:
        continue

    point = global_to_lidar(
        track_positions[key],
        sample_token,
    )

    focus_traj.append(point)


if len(focus_traj) >= 2:

    focus_traj = np.asarray(
        focus_traj
    )

    ax.plot(
        focus_traj[:, 0],
        focus_traj[:, 1],
        linewidth=3.0,
        color="darkorange",
        zorder=7,
    )


ax.scatter(
    focus_x,
    focus_y,
    s=170,
    color="darkorange",
    edgecolor="black",
    linewidth=1.2,
    zorder=9,
)


# Ego → object line
ax.plot(
    [0, focus_x],
    [0, focus_y],
    linestyle="--",
    linewidth=1.3,
    color="darkorange",
    alpha=0.8,
    zorder=6,
)


# label
ttc_text = (
    f"{focus_row['ttc_s']:.2f} s"
    if np.isfinite(focus_row["ttc_s"])
    else "-"
)

label = (
    f"Track {FOCUS_TRACK_ID} | "
    f"{focus_row['class'].capitalize()}\n"
    f"Distance  {focus_row['distance_m']:.2f} m\n"
    f"TTC       {ttc_text}\n"
    f"DCPA      {focus_row['dcpa_m']:.2f} m\n"
    f"Risk      {focus_row['risk_level']}"
)

ax.annotate(
    label,
    (focus_x, focus_y),
    xytext=(20, 25),
    textcoords="offset points",
    fontsize=11,
    fontweight="bold",
    linespacing=1.4,
    bbox={
        "boxstyle": "round,pad=0.55",
        "facecolor": "white",
        "edgecolor": "darkorange",
        "linewidth": 2,
        "alpha": 0.95,
    },
    arrowprops={
        "arrowstyle": "->",
        "linewidth": 1.5,
        "color": "darkorange",
    },
)


# =========================================================
# Final design
# =========================================================

ax.set_xlim(
    -BEV_RANGE,
    BEV_RANGE,
)

ax.set_ylim(
    -BEV_RANGE,
    BEV_RANGE,
)

ax.set_aspect(
    "equal",
    adjustable="box",
)

ax.set_xlabel(
    "LiDAR X (m)",
    fontsize=11,
)

ax.set_ylabel(
    "LiDAR Y (m)",
    fontsize=11,
)

ax.set_title(
    "Tracked Object Risk Assessment in BEV\n"
    "PointPillars + Kalman/Hungarian + TTC/DCPA",
    fontsize=15,
    fontweight="bold",
    pad=12,
)

ax.grid(
    alpha=0.15,
)

plt.tight_layout()

output_path = (
    OUTPUT_DIR
    / "tracking_risk_bev_portfolio.png"
)

plt.savefig(
    output_path,
    dpi=220,
    bbox_inches="tight",
)

plt.close()

print(f"saved: {output_path}")