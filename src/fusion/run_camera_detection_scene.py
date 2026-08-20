import csv
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "pointpillars"
    / "camera_lidar"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = (
    OUTPUT_DIR
    / "camera_detections_scene0103_all.csv"
)


# =========================================================
# Settings
# =========================================================

TARGET_SCENE = "scene-0103"
CAMERA_NAME = "CAM_FRONT"

CONF_THRESHOLD = 0.25


# =========================================================
# Class mapping
# =========================================================

CLASS_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
}


# =========================================================
# nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# YOLO
# =========================================================

model = YOLO("yolo11n.pt")


# =========================================================
# Find scene
# =========================================================

scene_record = next(
    scene
    for scene in nusc.scene
    if scene["name"] == TARGET_SCENE
)


# =========================================================
# Iterate scene
# =========================================================

rows = []

sample_token = scene_record["first_sample_token"]

frame_index = 0

while sample_token:

    sample = nusc.get(
        "sample",
        sample_token,
    )

    camera_token = sample["data"][
        CAMERA_NAME
    ]

    camera_data = nusc.get(
        "sample_data",
        camera_token,
    )

    image_path = (
        DATA_ROOT
        / camera_data["filename"]
    )

    # ---------------------------------------------
    # Camera detection
    # ---------------------------------------------

    results = model.predict(
        source=str(image_path),
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    result = results[0]

    frame_count = 0

    for box in result.boxes:

        class_id = int(
            box.cls.item()
        )

        camera_class = result.names[
            class_id
        ]

        if camera_class not in CLASS_MAP:
            continue

        mapped_class = CLASS_MAP[
            camera_class
        ]

        score = float(
            box.conf.item()
        )

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .tolist()
        )

        rows.append(
            {
                "scene": TARGET_SCENE,
                "frame": frame_index,

                "sample_token": sample_token,
                "camera_token": camera_token,

                "class": mapped_class,
                "camera_class": camera_class,

                "score": score,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }
        )

        frame_count += 1

    print(
        f"frame {frame_index:02d} | "
        f"camera detections = {frame_count:3d}"
    )

    sample_token = sample["next"]

    frame_index += 1


# =========================================================
# Save CSV
# =========================================================

fieldnames = [
    "scene",
    "frame",

    "sample_token",
    "camera_token",

    "class",
    "camera_class",

    "score",

    "x1",
    "y1",
    "x2",
    "y2",
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


# =========================================================
# Summary
# =========================================================

print()
print("=" * 65)
print("SCENE CAMERA DETECTION SUMMARY")
print("=" * 65)

print(
    f"Scene              : "
    f"{TARGET_SCENE}"
)

print(
    f"Frames processed   : "
    f"{frame_index}"
)

print(
    f"Total detections   : "
    f"{len(rows)}"
)

print()

print(
    f"Saved              : "
    f"{OUTPUT_CSV}"
)

print("=" * 65)