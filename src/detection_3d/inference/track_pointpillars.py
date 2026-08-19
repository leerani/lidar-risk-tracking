import csv
import pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


# =========================================================
# Paths
# =========================================================

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

OUTPUT_CSV = OUTPUT_DIR / "pointpillars_tracks.csv"


# =========================================================
# Tracking parameters
# =========================================================

SCORE_THRESHOLD = 0.30

TRACK_CLASSES = {
    "car",
    "truck",
    "bus",
    "construction_vehicle",
    "trailer",
    "pedestrian",
    "motorcycle",
    "bicycle",
}

MATCH_DISTANCE_THRESHOLD = 4.0

MIN_HITS = 3
MAX_AGE = 2


# =========================================================
# Coordinate transformation
# =========================================================

def transform_lidar_to_global(point_xyz, nusc, sample_token):
    """
    PointPillars prediction center:
        LiDAR coordinate
            ↓
        ego vehicle coordinate
            ↓
        global coordinate
    """

    sample = nusc.get("sample", sample_token)
    lidar_token = sample["data"]["LIDAR_TOP"]

    sample_data = nusc.get("sample_data", lidar_token)

    calibrated_sensor = nusc.get(
        "calibrated_sensor",
        sample_data["calibrated_sensor_token"],
    )

    ego_pose = nusc.get(
        "ego_pose",
        sample_data["ego_pose_token"],
    )

    point = np.asarray(point_xyz, dtype=float)

    # LiDAR → ego
    sensor_rotation = Quaternion(
        calibrated_sensor["rotation"]
    ).rotation_matrix

    sensor_translation = np.asarray(
        calibrated_sensor["translation"]
    )

    point_ego = (
        sensor_rotation @ point
        + sensor_translation
    )

    # ego → global
    ego_rotation = Quaternion(
        ego_pose["rotation"]
    ).rotation_matrix

    ego_translation = np.asarray(
        ego_pose["translation"]
    )

    point_global = (
        ego_rotation @ point_ego
        + ego_translation
    )

    return point_global


# =========================================================
# Simple Kalman Filter
# state = [x, y, vx, vy]
# =========================================================

class KalmanTrack:

    def __init__(
        self,
        track_id,
        x,
        y,
        class_name,
        score,
        timestamp,
    ):
        self.track_id = track_id
        self.class_name = class_name

        self.state = np.array(
            [x, y, 0.0, 0.0],
            dtype=float,
        )

        self.P = np.eye(4) * 10.0

        self.score = score

        self.hits = 1
        self.age = 0

        self.last_timestamp = timestamp

    def predict(self, timestamp):
        dt = (
            timestamp - self.last_timestamp
        ) / 1_000_000.0

        if dt <= 0:
            dt = 0.5

        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        Q = np.eye(4) * 0.5

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + Q

        self.age += 1

        return self.state[:2]

    def update(
        self,
        x,
        y,
        score,
        timestamp,
    ):
        z = np.array([x, y])

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        R = np.eye(2) * 1.0

        innovation = z - H @ self.state

        S = H @ self.P @ H.T + R

        K = (
            self.P
            @ H.T
            @ np.linalg.inv(S)
        )

        self.state = (
            self.state
            + K @ innovation
        )

        I = np.eye(4)

        self.P = (
            I - K @ H
        ) @ self.P

        self.score = score
        self.hits += 1
        self.age = 0
        self.last_timestamp = timestamp


# =========================================================
# Tracker
# =========================================================

class MultiObjectTracker:

    def __init__(self):
        self.tracks = []
        self.next_id = 1

    def update(
        self,
        detections,
        timestamp,
    ):
        # ---------------------------------------------
        # Predict existing tracks
        # ---------------------------------------------

        predicted_positions = []

        for track in self.tracks:
            predicted_positions.append(
                track.predict(timestamp)
            )

        # ---------------------------------------------
        # No existing tracks
        # ---------------------------------------------

        if len(self.tracks) == 0:

            for det in detections:
                self._create_track(
                    det,
                    timestamp,
                )

            return

        # ---------------------------------------------
        # No detections
        # ---------------------------------------------

        if len(detections) == 0:
            self._remove_old_tracks()
            return

        # ---------------------------------------------
        # Hungarian cost matrix
        # ---------------------------------------------

        cost_matrix = np.full(
            (
                len(self.tracks),
                len(detections),
            ),
            1e6,
            dtype=float,
        )

        for i, track in enumerate(self.tracks):

            tx, ty = predicted_positions[i]

            for j, det in enumerate(detections):

                # Different class → cannot match
                if track.class_name != det["class"]:
                    continue

                dx = tx - det["global_x"]
                dy = ty - det["global_y"]

                distance = np.sqrt(
                    dx ** 2 + dy ** 2
                )

                cost_matrix[i, j] = distance

        track_indices, detection_indices = (
            linear_sum_assignment(cost_matrix)
        )

        matched_tracks = set()
        matched_detections = set()

        # ---------------------------------------------
        # Apply matches
        # ---------------------------------------------

        for track_idx, det_idx in zip(
            track_indices,
            detection_indices,
        ):

            cost = cost_matrix[
                track_idx,
                det_idx,
            ]

            if cost > MATCH_DISTANCE_THRESHOLD:
                continue

            track = self.tracks[track_idx]
            det = detections[det_idx]

            track.update(
                det["global_x"],
                det["global_y"],
                det["score"],
                timestamp,
            )

            matched_tracks.add(track_idx)
            matched_detections.add(det_idx)

        # ---------------------------------------------
        # Create tracks for unmatched detections
        # ---------------------------------------------

        for det_idx, det in enumerate(detections):

            if det_idx not in matched_detections:
                self._create_track(
                    det,
                    timestamp,
                )

        self._remove_old_tracks()

    def _create_track(
        self,
        det,
        timestamp,
    ):
        track = KalmanTrack(
            track_id=self.next_id,
            x=det["global_x"],
            y=det["global_y"],
            class_name=det["class"],
            score=det["score"],
            timestamp=timestamp,
        )

        self.tracks.append(track)

        self.next_id += 1

    def _remove_old_tracks(self):
        self.tracks = [
            track
            for track in self.tracks
            if track.age <= MAX_AGE
        ]


# =========================================================
# Load
# =========================================================

with open(RESULT_PATH, "rb") as f:
    predictions = pickle.load(f)

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Organize predictions by scene
# =========================================================

scene_frames = defaultdict(list)

for pred in predictions:

    sample_token = pred["metadata"]["token"]

    sample = nusc.get(
        "sample",
        sample_token,
    )

    scene_token = sample["scene_token"]

    scene_frames[scene_token].append(
        {
            "sample_token": sample_token,
            "timestamp": sample["timestamp"],
            "prediction": pred,
        }
    )


# sort chronologically
for scene_token in scene_frames:

    scene_frames[scene_token].sort(
        key=lambda x: x["timestamp"]
    )


# =========================================================
# Tracking
# =========================================================

rows = []

for scene_index, (
    scene_token,
    frames,
) in enumerate(scene_frames.items()):

    scene = nusc.get(
        "scene",
        scene_token,
    )

    scene_name = scene["name"]

    print()
    print(
        f"Tracking scene: {scene_name} "
        f"({len(frames)} frames)"
    )

    tracker = MultiObjectTracker()

    for frame_index, frame in enumerate(frames):

        pred = frame["prediction"]

        boxes = pred["boxes_lidar"]
        names = pred["name"]
        scores = pred["score"]

        detections = []

        # ---------------------------------------------
        # PointPillars → tracking detections
        # ---------------------------------------------

        for box, name, score in zip(
            boxes,
            names,
            scores,
        ):

            score = float(score)

            if score < SCORE_THRESHOLD:
                continue

            if name not in TRACK_CLASSES:
                continue

            x, y, z = (
                float(box[0]),
                float(box[1]),
                float(box[2]),
            )

            global_xyz = (
                transform_lidar_to_global(
                    [x, y, z],
                    nusc,
                    frame["sample_token"],
                )
            )

            detections.append(
                {
                    "class": name,
                    "score": score,

                    "lidar_x": x,
                    "lidar_y": y,

                    "global_x": float(global_xyz[0]),
                    "global_y": float(global_xyz[1]),
                }
            )

        tracker.update(
            detections,
            frame["timestamp"],
        )

        confirmed_tracks = 0

        # ---------------------------------------------
        # Save confirmed tracks
        # ---------------------------------------------

        for track in tracker.tracks:

            if track.hits < MIN_HITS:
                continue

            # Current frame에서 업데이트되지 않은
            # predicted-only track은 CSV에서 제외
            if track.age != 0:
                continue

            confirmed_tracks += 1

            speed = np.sqrt(
                track.state[2] ** 2
                + track.state[3] ** 2
            )

            rows.append(
                {
                    "scene": scene_name,
                    "frame": frame_index,
                    "sample_token": frame["sample_token"],
                    "timestamp": frame["timestamp"],

                    "track_id": track.track_id,
                    "class": track.class_name,

                    "global_x": track.state[0],
                    "global_y": track.state[1],

                    "vx": track.state[2],
                    "vy": track.state[3],
                    "speed_mps": speed,

                    "score": track.score,
                    "hits": track.hits,
                }
            )

        print(
            f"frame {frame_index:02d} | "
            f"detections={len(detections):3d} | "
            f"active_tracks={len(tracker.tracks):3d} | "
            f"confirmed={confirmed_tracks:3d}"
        )


# =========================================================
# Save CSV
# =========================================================

fieldnames = [
    "scene",
    "frame",
    "sample_token",
    "timestamp",
    "track_id",
    "class",
    "global_x",
    "global_y",
    "vx",
    "vy",
    "speed_mps",
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


print()
print("=" * 60)
print(f"Confirmed track rows : {len(rows)}")
print(f"Saved                : {OUTPUT_CSV}")
print("=" * 60)