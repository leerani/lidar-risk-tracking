from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.cluster import DBSCAN
from pathlib import Path
import numpy as np
import pandas as pd
import json


DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"

START_SAMPLE_IDX = 46
NUM_FRAMES = 12

OUTPUT_CSV = "outputs/logs/tracking_results.csv"
OUTPUT_JSON = "outputs/logs/tracking_results.json"


def filter_front_roi(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x > 2.0) & (x < 30) &
        (y > -10) & (y < 10) &
        (z > -3) & (z < 3)
    )
    return points[mask]


def remove_ground_by_height(points, z_threshold=-1.4):
    return points[points[:, 2] > z_threshold]


def run_dbscan(points, eps=0.6, min_samples=6):
    if len(points) == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    return clustering.labels_


def get_cluster_info(points, labels, min_cluster_points=30):
    cluster_infos = []

    for label in sorted(set(labels)):
        if label == -1:
            continue

        cluster_points = points[labels == label]

        if len(cluster_points) < min_cluster_points:
            continue

        min_xyz = cluster_points.min(axis=0)
        max_xyz = cluster_points.max(axis=0)
        center = (min_xyz + max_xyz) / 2
        size = max_xyz - min_xyz
        distance_xy = np.linalg.norm(center[:2])

        width_x, width_y, height_z = size

        # LiDAR 원점 근처 cluster 제거
        if distance_xy < 3.0:
            continue

        # 너무 얇거나 낮은 cluster 제거
        if width_x < 0.2 or width_y < 0.2 or height_z < 0.3:
            continue

        # 너무 큰 구조물 제거
        if width_x > 5.0 or width_y > 5.0 or height_z > 3.5:
            continue

        cluster_infos.append({
            "cluster_id": int(label),
            "num_points": int(len(cluster_points)),
            "center": center,
            "size": size,
            "distance_m": float(distance_xy),
        })

    return cluster_infos


class SimpleTracker:
    def __init__(self, max_match_distance=2.0, max_missing_frames=2):
        self.next_track_id = 1
        self.tracks = {}
        self.max_match_distance = max_match_distance
        self.max_missing_frames = max_missing_frames

    def update(self, detections, frame_idx):
        assigned_track_ids = set()
        results = []

        for det in detections:
            center = np.array(det["center"])
            best_track_id = None
            best_distance = float("inf")

            for track_id, track in self.tracks.items():
                if track_id in assigned_track_ids:
                    continue

                prev_center = np.array(track["center"])
                match_distance = np.linalg.norm(center[:2] - prev_center[:2])

                if match_distance < best_distance and match_distance < self.max_match_distance:
                    best_distance = match_distance
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                speed_per_frame = 0.0
                approach_delta = 0.0
            else:
                track_id = best_track_id
                prev_track = self.tracks[track_id]
                prev_center = np.array(prev_track["center"])
                prev_distance = prev_track["distance_m"]

                speed_per_frame = float(np.linalg.norm(center[:2] - prev_center[:2]))
                approach_delta = float(prev_distance - det["distance_m"])

            det["track_id"] = int(track_id)
            det["speed_per_frame"] = float(speed_per_frame)
            det["approach_delta"] = float(approach_delta)
            det["is_approaching"] = bool(approach_delta > 0.1)
            det["frame_idx"] = int(frame_idx)

            assigned_track_ids.add(track_id)
            results.append(det)

        # 현재 프레임에서 매칭된 track 업데이트
        updated_tracks = {}

        for det in results:
            updated_tracks[det["track_id"]] = {
                "center": det["center"],
                "distance_m": det["distance_m"],
                "last_seen": frame_idx,
                "missing": 0,
            }

        # 매칭되지 않은 기존 track은 missing count 증가
        for track_id, track in self.tracks.items():
            if track_id not in updated_tracks:
                missing = track.get("missing", 0) + 1
                if missing <= self.max_missing_frames:
                    track["missing"] = missing
                    updated_tracks[track_id] = track

        self.tracks = updated_tracks
        return results


def assess_risk(obj):
    distance = obj["distance_m"]
    is_approaching = obj["is_approaching"]
    speed = obj["speed_per_frame"]

    if distance < 5.0:
        return "DANGER"

    if distance < 10.0 and is_approaching:
        return "WARNING"

    if distance < 10.0:
        return "CAUTION"

    if distance < 15.0 and is_approaching and speed > 0.3:
        return "CAUTION"

    return "SAFE"


def process_sample(nusc, sample, frame_idx):
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_path = Path(DATAROOT) / lidar_data["filename"]

    pc = LidarPointCloud.from_file(str(lidar_path))
    points = pc.points[:3, :].T

    roi_points = filter_front_roi(points)
    non_ground_points = remove_ground_by_height(roi_points)
    labels = run_dbscan(non_ground_points)
    cluster_infos = get_cluster_info(non_ground_points, labels)

    return {
        "frame_idx": frame_idx,
        "sample_token": sample["token"],
        "lidar_path": str(lidar_path),
        "original_points": len(points),
        "roi_points": len(roi_points),
        "non_ground_points": len(non_ground_points),
        "cluster_count": len(cluster_infos),
        "detections": cluster_infos,
    }


def get_consecutive_samples(nusc, start_sample_idx, num_frames):
    samples = []

    sample = nusc.sample[start_sample_idx]

    for _ in range(num_frames):
        samples.append(sample)

        if sample["next"] == "":
            break

        sample = nusc.get("sample", sample["next"])

    return samples


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    samples = get_consecutive_samples(
        nusc,
        start_sample_idx=START_SAMPLE_IDX,
        num_frames=NUM_FRAMES
    )

    tracker = SimpleTracker(
        max_match_distance=2.0,
        max_missing_frames=2
    )

    all_rows = []
    all_json = []

    for frame_idx, sample in enumerate(samples):
        result = process_sample(nusc, sample, frame_idx)
        tracked_objects = tracker.update(result["detections"], frame_idx)

        frame_decision = "SAFE"

        frame_objects = []

        for obj in tracked_objects:
            risk_level = assess_risk(obj)
            obj["risk_level"] = risk_level

            if risk_level == "DANGER":
                frame_decision = "DANGER"
            elif risk_level == "WARNING" and frame_decision != "DANGER":
                frame_decision = "WARNING"
            elif risk_level == "CAUTION" and frame_decision == "SAFE":
                frame_decision = "CAUTION"

            row = {
                "frame_idx": frame_idx,
                "track_id": obj["track_id"],
                "cluster_id": obj["cluster_id"],
                "num_points": obj["num_points"],
                "center_x": obj["center"][0],
                "center_y": obj["center"][1],
                "center_z": obj["center"][2],
                "size_x": obj["size"][0],
                "size_y": obj["size"][1],
                "size_z": obj["size"][2],
                "distance_m": obj["distance_m"],
                "speed_per_frame": obj["speed_per_frame"],
                "approach_delta": obj["approach_delta"],
                "is_approaching": obj["is_approaching"],
                "risk_level": risk_level,
            }

            all_rows.append(row)

            frame_objects.append({
                "track_id": int(obj["track_id"]),
                "cluster_id": int(obj["cluster_id"]),
                "num_points": int(obj["num_points"]),
                "center": np.array(obj["center"]).round(3).tolist(),
                "size": np.array(obj["size"]).round(3).tolist(),
                "distance_m": round(float(obj["distance_m"]), 3),
                "speed_per_frame": round(float(obj["speed_per_frame"]), 3),
                "approach_delta": round(float(obj["approach_delta"]), 3),
                "is_approaching": bool(obj["is_approaching"]),
                "risk_level": risk_level,
            })

        all_json.append({
            "frame_idx": frame_idx,
            "sample_token": result["sample_token"],
            "cluster_count": result["cluster_count"],
            "frame_decision": frame_decision,
            "objects": frame_objects,
        })

        print(
            f"frame={frame_idx:02d} | "
            f"clusters={result['cluster_count']:02d} | "
            f"tracked={len(tracked_objects):02d} | "
            f"decision={frame_decision}"
        )

        for obj in frame_objects[:5]:
            print(
                f"  track={obj['track_id']:>2} | "
                f"dist={obj['distance_m']:>5.2f}m | "
                f"speed={obj['speed_per_frame']:.2f} | "
                f"approach={obj['approach_delta']:.2f} | "
                f"risk={obj['risk_level']}"
            )

    Path("outputs/logs").mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(all_json, f, indent=2, ensure_ascii=False)

    print("\nSaved:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()