from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
import open3d as o3d
import numpy as np
from pathlib import Path


DATAROOT = "data/nuscenes"
VERSION = "v1.0-mini"


def filter_front_roi(points):
    """
    nuScenes LiDAR coordinate 기준:
    x: 전방/후방
    y: 좌우
    z: 높이

    전방 위험 판단에 필요한 영역만 남김.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    mask = (
        (x > 0) & (x < 30) &
        (y > -10) & (y < 10) &
        (z > -3) & (z < 3)
    )

    return points[mask]


def make_pcd(points, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    if color is not None:
        pcd.paint_uniform_color(color)

    return pcd


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

    pc = LidarPointCloud.from_file(str(lidar_path))
    points = pc.points[:3, :].T

    roi_points = filter_front_roi(points)

    print("Original point count:", len(points))
    print("ROI point count:", len(roi_points))
    print("Removed point count:", len(points) - len(roi_points))

    pcd = make_pcd(roi_points)

    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Front ROI Filtered LiDAR Point Cloud"
    )


if __name__ == "__main__":
    main()