from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import numpy as np
from pathlib import Path


DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"


def visualize_points(points_xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="nuScenes LiDAR Point Cloud"
    )


def main():
    nusc = NuScenes(
        version=VERSION,
        dataroot=DATAROOT,
        verbose=True
    )

    sample = nusc.sample[0]
    lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_data = nusc.get("sample_data", lidar_token)
    lidar_path = Path(DATAROOT) / lidar_data["filename"]

    print("LiDAR file:", lidar_path)

    pc = LidarPointCloud.from_file(str(lidar_path))

    # nuScenes point cloud shape: 4 x N
    points = pc.points[:3, :].T  # N x 3

    print("Point cloud shape:", points.shape)
    print("First 5 points:")
    print(points[:5])

    visualize_points(points)


if __name__ == "__main__":
    main()