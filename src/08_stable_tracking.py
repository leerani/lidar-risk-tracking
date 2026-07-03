from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.cluster import DBSCAN
from scipy.optimize import linear_sum_assignment
from pathlib import Path
import numpy as np
import pandas as pd
import json


DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"

START_SAMPLE_IDX = 46
NUM_FRAMES = 12

OUTPUT_CSV = "outputs/logs/tracking_results_stable.csv"
OUTPUT_JSON = "outputs/logs/tracking_results_stable.json"


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

        if distance_xy < 3.0:
            continue

        if width_x < 0.2 or width_y < 0.2 or height_z < 0.3:
            continue

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


class StableTrack:
    def __init__(self, track_id, detection, frame_idx):
        self.track_id = track_id
        self.center = np.array(detection["center"])
        self.size = np.array(detection["size"])
        self.distance_m = float(detection["distance_m"])
        self.smoothed_distance_m = float(detection["distance_m"])

        self.age = 1
        self.hits = 1
        self.missing = 0
        self.last_seen = frame_idx

        self.history = [float(detection["distance_m"])]


class StableTracker:
    def __init__(
        self,
        max_match_distance=2.5,
        max_missing_frames=2,
        confirm_min_hits=3,
        smoothing_alpha=0.6
    ):
        self.next_track_id = 1
        self.tracks = {}
        self.max_match_distance = max_match_distance
        self.max_missing_frames = max_missing_frames
        self.confirm_min_hits = confirm_min_hits
        self.smoothing_alpha = smoothing_alpha

    def update(self, detections, frame_idx):
        detection_centers = [
            np.array(det["center"]) for det in detections
        ]

        track_ids = list(self.tracks.keys())
        track_centers = [
            self.tracks[track_id].center for track_id in track_ids
        ]

        matched_track_ids = set()
        matched_detection_indices = set()
        results = []

        if len(track_centers) > 0 and len(detection_centers) > 0:
            cost_matrix = np.zeros((len(track_centers), len(detection_centers)))

            for i, track_center in enumerate(track_centers):
                for j, det_center in enumerate(detection_centers):
                    cost_matrix[i, j] = np.linalg.norm(track_center[:2] - det_center[:2])

            row_indices, col_indices = linear_sum_assignment(cost_matrix)

            for row_idx, col_idx in zip(row_indices, col_indices):
                match_distance = cost_matrix[row_idx, col_idx]

                if match_distance > self.max_match_distance:
                    continue

                track_id = track_ids[row_idx]
                track = self.tracks[track_id]
                det = detections[col_idx]

                prev_distance = track.distance_m
                raw_distance = float(det["distance_m"])

                smoothed_distance = (
                    self.smoothing_alpha * raw_distance
                    + (1 - self.smoothing_alpha) * track.smoothed_distance_m
                )

                speed_per_frame = float(
                    np.linalg.norm(np.array(det["center"])[:2] - track.center[:2])
                )

                raw_approach_delta = prev_distance - raw_distance
                smoothed_approach_delta = track.smoothed_distance_m - smoothed_distance

                track.center = np.array(det["center"])
                track.size = np.array(det["size"])
                track.distance_m = raw_distance
                track.smoothed_distance_m = smoothed_distance
                track.age += 1
                track.hits += 1
                track.missing = 0
                track.last_seen = frame_idx
                track.history.append(raw_distance)

                is_confirmed = track.hits >= self.confirm_min_hits

                det["track_id"] = int(track_id)
                det["speed_per_frame"] = float(speed_per_frame)
                det["raw_approach_delta"] = float(raw_approach_delta)
                det["smoothed_approach_delta"] = float(smoothed_approach_delta)
                det["smoothed_distance_m"] = float(smoothed_distance)
                det["is_approaching"] = bool(smoothed_approach_delta > 0.1)
                det["is_confirmed"] = bool(is_confirmed)
                det["track_hits"] = int(track.hits)
                det["track_age"] = int(track.age)
                det["frame_idx"] = int(frame_idx)

                matched_track_ids.add(track_id)
                matched_detection_indices.add(col_idx)
                results.append(det)

        # 새 detection → tentative track 생성
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detection_indices:
                continue

            track_id = self.next_track_id
            self.next_track_id += 1

            track = StableTrack(track_id, det, frame_idx)
            self.tracks[track_id] = track

            det["track_id"] = int(track_id)
            det["speed_per_frame"] = 0.0
            det["raw_approach_delta"] = 0.0
            det["smoothed_approach_delta"] = 0.0
            det["smoothed_distance_m"] = float(det["distance_m"])
            det["is_approaching"] = False
            det["is_confirmed"] = False
            det["track_hits"] = 1
            det["track_age"] = 1
            det["frame_idx"] = int(frame_idx)

            results.append(det)

        # 매칭되지 않은 기존 track missing 처리
        delete_track_ids = []

        for track_id, track in self.tracks.items():
            if track_id in matched_track_ids:
                continue

            # 방금 새로 만든 track은 missing 처리하지 않음
            if track.last_seen == frame_idx:
                continue

            track.missing += 1
            track.age += 1

            if track.missing > self.max_missing_frames:
                delete_track_ids.append(track_id)

        for track_id in delete_track_ids:
            del self.tracks[track_id]

        return results


def assess_risk(obj):
    distance = obj["smoothed_distance_m"]
    is_approaching = obj["is_approaching"]
    is_confirmed = obj["is_confirmed"]

    # confirmed가 아닌 track은 최종 위험 판단에서 한 단계 보수적으로 처리
    if not is_confirmed:
        if distance < 5.0:
            return "CAUTION"
        return "SAFE"

    if distance < 5.0:
        return "DANGER"

    if distance < 10.0 and is_approaching:
        return "WARNING"

    if distance < 10.0:
        return "CAUTION"

    if distance < 15.0 and is_approaching:
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

    tracker = StableTracker(
        max_match_distance=2.5,
        max_missing_frames=2,
        confirm_min_hits=3,
        smoothing_alpha=0.6
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
                "frame_idx": obj["frame_idx"],
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
                "smoothed_distance_m": obj["smoothed_distance_m"],
                "speed_per_frame": obj["speed_per_frame"],
                "raw_approach_delta": obj["raw_approach_delta"],
                "smoothed_approach_delta": obj["smoothed_approach_delta"],
                "is_approaching": obj["is_approaching"],
                "is_confirmed": obj["is_confirmed"],
                "track_hits": obj["track_hits"],
                "track_age": obj["track_age"],
                "risk_level": risk_level,
            }

            all_rows.append(row)

            frame_objects.append({
                "track_id": int(obj["track_id"]),
                "cluster_id": int(obj["cluster_id"]),
                "distance_m": round(float(obj["distance_m"]), 3),
                "smoothed_distance_m": round(float(obj["smoothed_distance_m"]), 3),
                "is_confirmed": bool(obj["is_confirmed"]),
                "track_hits": int(obj["track_hits"]),
                "is_approaching": bool(obj["is_approaching"]),
                "risk_level": risk_level,
            })

        confirmed_count = sum(1 for obj in frame_objects if obj["is_confirmed"])

        all_json.append({
            "frame_idx": frame_idx,
            "sample_token": result["sample_token"],
            "cluster_count": result["cluster_count"],
            "tracked_count": len(tracked_objects),
            "confirmed_count": confirmed_count,
            "frame_decision": frame_decision,
            "objects": frame_objects,
        })

        print(
            f"frame={frame_idx:02d} | "
            f"clusters={result['cluster_count']:02d} | "
            f"tracked={len(tracked_objects):02d} | "
            f"confirmed={confirmed_count:02d} | "
            f"decision={frame_decision}"
        )

        for obj in frame_objects[:5]:
            print(
                f"  track={obj['track_id']:>2} | "
                f"raw={obj['distance_m']:>5.2f}m | "
                f"smooth={obj['smoothed_distance_m']:>5.2f}m | "
                f"hits={obj['track_hits']} | "
                f"confirmed={obj['is_confirmed']} | "
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