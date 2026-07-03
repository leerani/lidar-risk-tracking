from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from sklearn.cluster import DBSCAN
from pathlib import Path
import numpy as np


DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"


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
    non_ground = points[points[:, 2] > z_threshold]
    return non_ground


def run_dbscan(points, eps=0.6, min_samples=6):
    if len(points) == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    return clustering.labels_


def get_cluster_info(points, labels, min_cluster_points=10):
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

        if distance_xy < 2.0:
            continue

        if width_x > 5.0 or width_y > 5.0 or height_z > 4.0:
            continue

        cluster_infos.append({
            "cluster_id": int(label),
            "num_points": int(len(cluster_points)),
            "center": center,
            "size": size,
            "distance_m": float(distance_xy),
        })

    return cluster_infos


def process_sample(nusc, sample, sample_idx):
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
        "sample_idx": sample_idx,
        "sample_token": sample["token"],
        "lidar_path": str(lidar_path),
        "original_points": len(points),
        "roi_points": len(roi_points),
        "non_ground_points": len(non_ground_points),
        "cluster_count": len(cluster_infos),
        "clusters": cluster_infos,
    }


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    results = []

    for idx, sample in enumerate(nusc.sample):
        result = process_sample(nusc, sample, idx)
        results.append(result)

    results = sorted(
        results,
        key=lambda x: x["cluster_count"],
        reverse=True
    )

    print("\nTop frames by cluster count")
    print("=" * 80)

    for r in results[:20]:
        print(
            f"sample_idx={r['sample_idx']:>3} | "
            f"clusters={r['cluster_count']:>2} | "
            f"roi={r['roi_points']:>5} | "
            f"non_ground={r['non_ground_points']:>4} | "
            f"path={r['lidar_path']}"
        )

        for c in r["clusters"][:5]:
            print(
                f"   - cluster={c['cluster_id']:>2} | "
                f"points={c['num_points']:>3} | "
                f"dist={c['distance_m']:.2f}m | "
                f"center={c['center'].round(2)} | "
                f"size={c['size'].round(2)}"
            )

        print("-" * 80)


if __name__ == "__main__":
    main()