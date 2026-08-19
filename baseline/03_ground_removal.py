from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import numpy as np
from pathlib import Path


DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"


def filter_front_roi(points):
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x > 0) & (x < 30) &
        (y > -10) & (y < 10) &
        (z > -3) & (z < 3)
    )
    return points[mask]


def remove_ground_by_height(points, z_threshold=-1.2):
    """
    1차 ground removal:
    LiDAR 좌표계에서 z 값이 낮은 점들을 바닥으로 보고 제거.
    threshold는 데이터에 따라 조정 가능.
    """
    non_ground = points[points[:, 2] > z_threshold]
    ground = points[points[:, 2] <= z_threshold]
    return non_ground, ground


def make_pcd(points, color):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    return pcd


def main():
    nusc = NuScenes(version=VERSION, dataroot=DATAROOT, verbose=True)

    sample = nusc.sample[0]
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

    print("Original point count:", len(points))
    print("ROI point count:", len(roi_points))
    print("Ground point count:", len(ground_points))
    print("Non-ground point count:", len(non_ground_points))

    # Ground는 회색, Non-ground는 빨강으로 표시
    ground_pcd = make_pcd(ground_points, [0.6, 0.6, 0.6])
    non_ground_pcd = make_pcd(non_ground_points, [1.0, 0.0, 0.0])

    o3d.visualization.draw_geometries(
        [ground_pcd, non_ground_pcd],
        window_name="Ground Removal Result"
    )


if __name__ == "__main__":
    main()