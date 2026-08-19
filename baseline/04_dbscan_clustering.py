from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import numpy as np
from pathlib import Path
from sklearn.cluster import DBSCAN


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


def remove_ground_by_height(points, z_threshold=-1.2):
    non_ground = points[points[:, 2] > z_threshold]
    ground = points[points[:, 2] <= z_threshold]
    return non_ground, ground


def run_dbscan(points, eps=0.8, min_samples=12):
    """
    eps: 가까운 점을 같은 cluster로 볼 거리 기준
    min_samples: cluster로 인정하기 위한 최소 주변 점 개수
    """
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    return clustering.labels_


def get_cluster_info(points, labels, min_cluster_points=20):
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

        width_x = size[0]
        width_y = size[1]
        height_z = size[2]

        # 1. LiDAR 바로 근처 cluster 제거
        if distance_xy < 3.0:
            continue

        # 2. 점 수가 너무 적은 cluster 제거
        if len(cluster_points) < 30:
            continue

        # 3. 너무 얇거나 낮은 cluster 제거
        if width_x < 0.2 or width_y < 0.2 or height_z < 0.3:
            continue

        # 4. 비정상적으로 큰 구조물 제거
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

def create_colored_cluster_pcd(points, labels):
    valid_mask = labels != -1
    valid_points = points[valid_mask]
    valid_labels = labels[valid_mask]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_points)

    if len(valid_points) == 0:
        return pcd

    max_label = valid_labels.max()
    colors = np.zeros((len(valid_points), 3))

    for i, label in enumerate(valid_labels):
        # label별로 색이 다르게 보이도록 간단히 지정
        colors[i] = [
            ((label * 37) % 255) / 255.0,
            ((label * 67) % 255) / 255.0,
            ((label * 97) % 255) / 255.0,
        ]

    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def create_bbox_lines(cluster_infos):
    geometries = []

    for info in cluster_infos:
        center = info["center"]
        size = info["size"]

        bbox = o3d.geometry.OrientedBoundingBox()
        bbox.center = center
        bbox.extent = size
        bbox.R = np.eye(3)
        bbox.color = [1, 0, 0]

        geometries.append(bbox)

    return geometries


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    SAMPLE_IDX = 48
    sample = nusc.sample[SAMPLE_IDX]
    lidar_token = sample["data"]["LIDAR_TOP"]
    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_path = Path(DATAROOT) / lidar_data["filename"]

    pc = LidarPointCloud.from_file(str(lidar_path))
    points = pc.points[:3, :].T

    roi_points = filter_front_roi(points)
    non_ground_points, ground_points = remove_ground_by_height(
        roi_points,
        z_threshold=-1.4
    )

    labels = run_dbscan(
        non_ground_points,
        eps=0.6,
        min_samples=6
    )

    cluster_infos = get_cluster_info(
        non_ground_points,
        labels,
        min_cluster_points=20
    )

    num_clusters = len(cluster_infos)
    num_noise = int(np.sum(labels == -1))

    print("Original point count:", len(points))
    print("ROI point count:", len(roi_points))
    print("Non-ground point count:", len(non_ground_points))
    print("DBSCAN cluster count:", num_clusters)
    print("Noise point count:", num_noise)
    print("Sample index:", SAMPLE_IDX)

    print("\nCluster info:")
    for info in cluster_infos:
        print(
            f"cluster={info['cluster_id']:>2} | "
            f"points={info['num_points']:>4} | "
            f"distance={info['distance_m']:.2f}m | "
            f"center={info['center'].round(2)} | "
            f"size={info['size'].round(2)}"
        )

    cluster_pcd = create_colored_cluster_pcd(non_ground_points, labels)
    bboxes = create_bbox_lines(cluster_infos)

    o3d.visualization.draw_geometries(
        [cluster_pcd, *bboxes],
        window_name="DBSCAN Clustering Result"
    )


if __name__ == "__main__":
    main()