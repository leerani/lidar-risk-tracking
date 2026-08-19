import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion


# =========================================================
# Paths
# =========================================================

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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "tracking_risk_bev"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# Visualization target
# =========================================================

TARGET_SCENE = "scene-0103"

TARGET_FRAMES = [
    36,
    37,
    38,
]

FOCUS_TRACK_ID = 196

BEV_RANGE = 50.0


# =========================================================
# nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Load TTC/DCPA results
# =========================================================

rows_by_scene_frame = defaultdict(list)
track_history = defaultdict(list)

with open(
    RISK_CSV,
    newline="",
) as f:

    reader = csv.DictReader(f)

    for row in reader:

        parsed = {
            "scene": row["scene"],
            "frame": int(row["frame"]),
            "sample_token": row["sample_token"],

            "track_id": int(row["track_id"]),
            "class": row["class"],

            "distance_m": float(row["distance_m"]),
            "closing_speed_mps": float(
                row["closing_speed_mps"]
            ),

            "ttc_s": (
                float(row["ttc_s"])
                if row["ttc_s"] != ""
                else np.inf
            ),

            "dcpa_m": float(row["dcpa_m"]),

            "risk_level": row["risk_level"],
        }

        key = (
            parsed["scene"],
            parsed["frame"],
        )

        rows_by_scene_frame[key].append(
            parsed
        )

        track_history[
            (
                parsed["scene"],
                parsed["track_id"],
            )
        ].append(parsed)


# Sort history
for key in track_history:
    track_history[key].sort(
        key=lambda x: x["frame"]
    )


# =========================================================
# Coordinate transforms
# =========================================================

def global_to_lidar(
    global_xy,
    sample_token,
):
    """
    global XY → current LIDAR_TOP coordinate.
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

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"],
    )

    point_global = np.array(
        [
            global_xy[0],
            global_xy[1],
            ego_pose["translation"][2],
        ],
        dtype=float,
    )

    # global → ego
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

    # ego → lidar
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


def get_track_global_position(
    scene,
    track_id,
    frame,
):
    """
    같은 track의 해당 frame row에서 global position은
    risk CSV에 직접 없으므로 원 track CSV에서 읽는 대신
    여기서는 별도 로드 dictionary 사용.
    """
    return track_positions[
        (
            scene,
            track_id,
            frame,
        )
    ]


# =========================================================
# Load original tracking positions
# =========================================================

TRACK_CSV = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "pointpillars_tracks.csv"
)

track_positions = {}

with open(
    TRACK_CSV,
    newline="",
) as f:

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
            ]
        )


# =========================================================
# Risk display priority
# =========================================================

risk_priority = {
    "SAFE": 0,
    "CAUTION": 1,
    "WARNING": 2,
    "DANGER": 3,
}


# =========================================================
# Draw each frame
# =========================================================

for target_frame in TARGET_FRAMES:

    key = (
        TARGET_SCENE,
        target_frame,
    )

    frame_rows = rows_by_scene_frame.get(
        key,
        [],
    )

    if not frame_rows:
        print(
            f"No rows found: "
            f"{TARGET_SCENE} frame {target_frame}"
        )
        continue

    sample_token = (
        frame_rows[0]["sample_token"]
    )

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

    # ---------------------------------------------
    # Figure
    # ---------------------------------------------

    plt.figure(
        figsize=(10, 10)
    )

    plt.scatter(
        points[0],
        points[1],
        s=0.5,
        alpha=0.35,
    )

    # Ego
    plt.scatter(
        [0],
        [0],
        marker="*",
        s=180,
        label="Ego",
    )

    # ---------------------------------------------
    # Sort: high risk first
    # ---------------------------------------------

    frame_rows.sort(
        key=lambda x: risk_priority[
            x["risk_level"]
        ],
        reverse=True,
    )

    for row in frame_rows:

        pos_global = get_track_global_position(
            row["scene"],
            row["track_id"],
            row["frame"],
        )

        pos_lidar = global_to_lidar(
            pos_global,
            sample_token,
        )

        x, y = pos_lidar

        # ROI
        if (
            abs(x) > BEV_RANGE
            or abs(y) > BEV_RANGE
        ):
            continue

        # -----------------------------------------
        # Current track position
        # -----------------------------------------

        marker_size = (
            120
            if row["track_id"] == FOCUS_TRACK_ID
            else 55
        )

        plt.scatter(
            x,
            y,
            s=marker_size,
        )

        # -----------------------------------------
        # Track trajectory
        # -----------------------------------------

        history = track_history[
            (
                row["scene"],
                row["track_id"],
            )
        ]

        trajectory = []

        for hist in history:

            if hist["frame"] > target_frame:
                break

            history_key = (
                hist["scene"],
                hist["track_id"],
                hist["frame"],
            )

            if history_key not in track_positions:
                continue

            hist_global = track_positions[
                history_key
            ]

            hist_lidar = global_to_lidar(
                hist_global,
                sample_token,
            )

            trajectory.append(
                hist_lidar
            )

        if len(trajectory) >= 2:

            trajectory = np.asarray(
                trajectory
            )

            plt.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                linewidth=1.4,
                alpha=0.7,
            )

        # -----------------------------------------
        # Label
        # -----------------------------------------

        if np.isfinite(row["ttc_s"]):
            ttc_text = (
                f"{row['ttc_s']:.2f}s"
            )
        else:
            ttc_text = "-"

        label = (
            f"ID {row['track_id']} "
            f"{row['class']}\n"
            f"{row['distance_m']:.1f}m | "
            f"TTC {ttc_text}\n"
            f"DCPA {row['dcpa_m']:.1f}m | "
            f"{row['risk_level']}"
        )

        # 위험 객체 또는 focus track만 자세히 표시
        if (
            row["risk_level"]
            in {
                "WARNING",
                "DANGER",
            }
            or row["track_id"]
            == FOCUS_TRACK_ID
        ):

            plt.annotate(
                label,
                (x, y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=8,
                bbox={
                    "boxstyle": "round,pad=0.25",
                    "alpha": 0.75,
                },
            )

    # ---------------------------------------------
    # Plot
    # ---------------------------------------------

    plt.xlim(
        -BEV_RANGE,
        BEV_RANGE,
    )

    plt.ylim(
        -BEV_RANGE,
        BEV_RANGE,
    )

    plt.gca().set_aspect(
        "equal",
        adjustable="box",
    )

    plt.xlabel(
        "LiDAR X (m)"
    )

    plt.ylabel(
        "LiDAR Y (m)"
    )

    plt.title(
        f"PointPillars Tracking + TTC/DCPA Risk\n"
        f"{TARGET_SCENE} | frame {target_frame}"
    )

    plt.grid(
        alpha=0.2
    )

    plt.legend(
        loc="upper right"
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / f"{TARGET_SCENE}_frame_{target_frame:02d}.png"
    )

    plt.savefig(
        output_path,
        dpi=180,
    )

    plt.close()

    print(
        f"saved: {output_path}"
    )


print()
print("Done.")