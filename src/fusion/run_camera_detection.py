import csv
from pathlib import Path

from nuscenes.nuscenes import NuScenes
from ultralytics import YOLO


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
    / "camera_detections_scene0103_frame38.csv"
)

TARGET_SCENE = "scene-0103"
TARGET_FRAME = 38
CAMERA_NAME = "CAM_FRONT"

CONF_THRESHOLD = 0.25


# =========================================================
# nuScenes
# =========================================================

nusc = NuScenes(
    version="v1.0-mini",
    dataroot=str(DATA_ROOT),
    verbose=False,
)


# =========================================================
# Find sample
# =========================================================

scene_record = next(
    scene
    for scene in nusc.scene
    if scene["name"] == TARGET_SCENE
)

sample_token = scene_record[
    "first_sample_token"
]

for _ in range(TARGET_FRAME):

    sample = nusc.get(
        "sample",
        sample_token,
    )

    sample_token = sample["next"]


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


# =========================================================
# YOLO
# =========================================================

model = YOLO("yolo11n.pt")

results = model.predict(
    source=str(image_path),
    conf=CONF_THRESHOLD,
    verbose=False,
)

result = results[0]


# =========================================================
# Keep classes relevant to nuScenes
# =========================================================

CLASS_MAP = {
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
    "person": "pedestrian",
}


rows = []

for box in result.boxes:

    class_id = int(
        box.cls.item()
    )

    class_name = result.names[
        class_id
    ]

    if class_name not in CLASS_MAP:
        continue

    mapped_class = CLASS_MAP[
        class_name
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
            "sample_token": sample_token,
            "camera": CAMERA_NAME,

            "class": mapped_class,
            "camera_class": class_name,

            "score": score,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
        }
    )


# =========================================================
# Save
# =========================================================

fieldnames = [
    "sample_token",
    "camera",

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
# Print
# =========================================================

print()
print("=" * 60)
print("CAMERA 2D DETECTION")
print("=" * 60)

print(f"Scene       : {TARGET_SCENE}")
print(f"Frame       : {TARGET_FRAME}")
print(f"Camera      : {CAMERA_NAME}")
print(f"Detections  : {len(rows)}")

print()

for i, row in enumerate(rows):

    print(
        f"{i:02d} | "
        f"{row['class']:12s} | "
        f"score={row['score']:.3f} | "
        f"bbox=("
        f"{row['x1']:.0f},"
        f"{row['y1']:.0f},"
        f"{row['x2']:.0f},"
        f"{row['y2']:.0f})"
    )

print()
print(f"Saved: {OUTPUT_CSV}")
print("=" * 60)