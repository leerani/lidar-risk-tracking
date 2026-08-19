# LiDAR 3D Perception, Tracking, Risk Assessment, and Camera–LiDAR Fusion

## 1. Project Overview

This project started as a classical LiDAR point-cloud pipeline based on ROI filtering, ground removal, DBSCAN clustering, and frame-to-frame tracking. The original version focused on converting raw LiDAR data into structured proximity-risk decisions. It has since been extended into a broader **3D perception and risk-assessment pipeline** with semantic 3D detection, temporal tracking, dynamic risk estimation, camera–LiDAR fusion, and deployment-oriented runtime optimization.

The final project covers:

- PointPillars-based 3D object detection
- LiDAR distance / point-density analysis
- Multi-sweep and pillar-resolution ablation
- Hungarian Matching + Kalman Filter tracking
- TTC / DCPA-based dynamic risk assessment
- Camera 2D detection and Camera–LiDAR late fusion
- GT-based multisensor coverage evaluation
- ONNX Runtime deployment optimization

The goal is not simply to apply a 3D detector, but to analyze how sensor characteristics affect perception, connect detections over time, convert motion into interpretable risk, combine complementary sensor information, and evaluate runtime efficiency for deployment.

---

## 2. Project Evolution

### Stage 1 — Classical LiDAR Baseline

```text
LiDAR Point Cloud
→ Front ROI Filtering
→ Ground Removal
→ DBSCAN Clustering
→ 3D Bounding Box Generation
→ Hungarian Matching
→ Confirmed Track Filtering
→ Smoothed Distance
→ Proximity Risk Assessment
```

This stage established an interpretable baseline without semantic 3D detection.

### Stage 2 — Deep Learning-based 3D Perception

```text
LiDAR Point Cloud
→ PointPillars 3D Detection
→ Sensor Characteristic Analysis
→ Hungarian + Kalman Tracking
→ Relative Motion Estimation
→ TTC + DCPA Risk Assessment
→ Camera–LiDAR Association
→ Multisensor Coverage Evaluation
→ ONNX Runtime Optimization
```

The second stage replaces spatial DBSCAN candidates with semantic 3D detections and expands the system toward multimodal perception and deployment-oriented evaluation.

---

## 3. Dataset

### nuScenes Mini

The project uses the **nuScenes v1.0-mini** dataset.

Used modalities and metadata:

- `LIDAR_TOP`
- `CAM_FRONT`
- ego pose
- calibrated sensor information
- sample annotations
- LiDAR sweeps

Dataset layout:

```text
nuscenes/
├── maps/
├── samples/
│   ├── LIDAR_TOP/
│   └── CAM_FRONT/
├── sweeps/
│   └── LIDAR_TOP/
└── v1.0-mini/
```

The dataset is stored outside the Git repository and linked symbolically:

```text
data/nuscenes -> /media/rani/새 볼륨/nuscenes
```

PointPillars evaluation split:

| Split | Samples |
|---|---:|
| Train | 323 |
| Validation | 81 |
| Total | 404 |

---

## 4. End-to-End Architecture

```text
nuScenes LiDAR + Camera
        │
        ├── LiDAR → PointPillars 3D Detection
        │              │
        │              ├── Distance / Density / Sweep Analysis
        │              │
        │              └── Hungarian + Kalman Tracking
        │                              │
        │                              └── TTC + DCPA Risk
        │
        └── CAM_FRONT → YOLO 2D Detection
                               │
                     Camera–LiDAR Association
                               │
                     Multisensor Coverage Eval
                               │
                     Structured Risk / Fusion Output
                               │
                     ONNX Runtime Benchmark
```

---

## 5. Classical LiDAR Baseline

### 5.1 Front ROI Filtering

For forward proximity-risk analysis, the following ROI was used:

```text
x:  2.0 m ~ 30.0 m
y: -10.0 m ~ 10.0 m
z: -3.0 m ~  3.0 m
```

Example:

| Stage | Point Count |
|---|---:|
| Original Point Cloud | 34,688 |
| Front ROI | 9,924 |
| Removed | 24,764 |

### 5.2 Ground Removal

A height-based ground removal baseline was applied before clustering.

```text
z_threshold = -1.4
```

### 5.3 DBSCAN Object Candidate Extraction

Baseline DBSCAN configuration:

```text
eps = 0.6
min_samples = 6
```

Example frame:

| Metric | Result |
|---|---:|
| ROI Points | 8,662 |
| Non-ground Points | 2,421 |
| DBSCAN Clusters | 13 |
| Noise Points | 225 |

DBSCAN generated **spatial object candidates**, not semantic classes such as car or pedestrian.

### 5.4 Stable Tracking Baseline

The original nearest-neighbor tracker was improved using:

- Hungarian Matching
- confirmed-track logic
- exponential moving average distance smoothing

Confirmed-track rule:

```text
tentative: hits < 3
confirmed: hits >= 3
```

Result over the selected 11-frame sequence:

| Metric | Result |
|---|---:|
| Total Detections | 61 |
| Total Tracks | 35 |
| Confirmed Tracks | 9 |
| Stable Approaching Tracks | 4 |
| Confirmed Detections | 13 |
| Minimum Distance | 5.623 m |

This baseline showed why temporal stabilization is necessary before making risk decisions.

---

## 6. PointPillars 3D Detection

### 6.1 Model Configuration

The extended pipeline uses the OpenPCDet implementation of **PointPillars**.

```text
Point Cloud Range
[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

Pillar Size
[0.2, 0.2, 8.0]

Max Points per Pillar
20

Input Sweeps
10
```

Architecture:

```text
Voxel / Pillar Generation
→ PillarVFE
→ PointPillarScatter
→ BaseBEVBackbone
→ AnchorHeadMulti
→ 3D Detection
```

Classes:

```text
car, truck, construction_vehicle, bus, trailer,
barrier, motorcycle, bicycle, pedestrian, traffic_cone
```

### 6.2 Pretrained Baseline Evaluation

A pretrained PointPillars checkpoint trained on the full nuScenes dataset was evaluated on the nuScenes mini validation split.

> **Important:** this is a pretrained baseline evaluation, not a model trained from scratch on nuScenes mini.

| Metric | Result |
|---|---:|
| Validation Samples | 81 |
| mAP | 0.4134 |
| NDS | 0.4920 |

---

## 7. LiDAR Sensor Characteristic Analysis

### 7.1 Distance-based Detection Recall

Custom diagnostic condition:

```text
same class
prediction score >= 0.30
center distance <= 2 m
```

This is an auxiliary diagnostic, not the official nuScenes mAP metric.

| Distance | GT | Matched | Recall |
|---|---:|---:|---:|
| 0–20 m | 1,513 | 1,151 | 0.761 |
| 20–40 m | 1,967 | 1,180 | 0.600 |
| 40–50 m | 461 | 123 | 0.267 |

Detection recall decreased sharply with distance.

### 7.2 Current-sweep Point Density

Current-frame LiDAR points inside GT boxes were counted to inspect sensor sparsity.

| Distance | Objects | Mean Points | Median Points | Recall |
|---|---:|---:|---:|---:|
| 0–20 m | 1,513 | 115.8 | 29 | 0.761 |
| 20–40 m | 1,967 | 9.6 | 4 | 0.600 |
| 40–50 m | 461 | 2.7 | 1 | 0.267 |

Observed trend:

```text
increasing distance
→ lower point density
→ lower detection recall
```

Point count is not the only cause; class, occlusion, orientation, and scene structure also affect detection.

### 7.3 Multi-sweep Ablation

The same pretrained 10-sweep checkpoint was evaluated with reduced input sweeps.

> This is an **input ablation / robustness experiment**, not sweep-specific retraining.

| Input Sweeps | mAP | NDS |
|---:|---:|---:|
| 1 | 0.3135 | 0.3535 |
| 5 | 0.3717 | 0.4485 |
| 10 | **0.4134** | **0.4920** |

Distance recall:

| Sweeps | 0–20 m | 20–40 m | 40–50 m |
|---:|---:|---:|---:|
| 1 | 0.619 | 0.378 | 0.072 |
| 5 | 0.705 | 0.541 | 0.206 |
| 10 | **0.761** | **0.600** | **0.267** |

Temporal sweep accumulation helped compensate for sparse LiDAR information, especially at longer ranges.

### 7.4 Pillar Resolution Ablation

The pretrained `0.20 m` checkpoint was evaluated with different pillar sizes.

| Pillar Size | mAP | NDS | sec / example |
|---:|---:|---:|---:|
| 0.16 m | 0.3480 | 0.4363 | 0.1209 |
| 0.20 m | **0.4134** | **0.4920** | 0.0692 |
| 0.32 m | 0.1401 | 0.2538 | 0.0883 |

> The checkpoint was trained with `0.20 m` pillars. This is therefore a **resolution-mismatch ablation**, not a fair comparison of independently trained pillar resolutions.

---

## 8. 3D Object Tracking

PointPillars detections were connected across time using:

```text
PointPillars Detection
→ Class-aware Hungarian Matching
→ Kalman Filter
→ Confirmed Track
```

Tracking is performed in the global coordinate frame to account for ego motion.

Kalman state:

```text
[x, y, vx, vy]
```

Main parameters:

```text
score threshold = 0.30
matching distance = 4 m
minimum hits = 3
maximum age = 2
```

Evaluation result:

| Metric | Result |
|---|---:|
| Confirmed Track Rows | 2,010 |
| GT Observations | 4,267 |
| Matched Observations | 1,947 |
| GT Match Rate | 0.456 |
| Matched GT Instances | 157 |
| Stable Instances | 96 |
| Stable Instance Rate | 0.611 |
| ID Switches | 115 |
| ID Switch Rate | 0.064 |

The GT match rate is influenced by detector recall, score threshold, and confirmed-track filtering, so it is not a pure tracker metric.

---

## 9. Dynamic Risk Assessment

### 9.1 TTC

For each confirmed track, ego motion and object motion were used to estimate radial closing speed and Time-To-Collision.

Initial heuristic thresholds:

```text
DANGER  : TTC < 2 s
WARNING : TTC < 4 s
CAUTION : TTC < 8 s
SAFE    : otherwise
```

Result:

| Risk | Count |
|---|---:|
| SAFE | 1,368 |
| CAUTION | 351 |
| WARNING | 264 |
| DANGER | 27 |

TTC alone can overestimate risk when paths cross without an actual near-collision.

### 9.2 TTC + DCPA

DCPA was added to estimate minimum spatial separation at the closest point of approach.

```text
DANGER  : TTC < 2 s AND DCPA < 2 m
WARNING : TTC < 4 s AND DCPA < 4 m
CAUTION : TTC < 8 s AND DCPA < 6 m
SAFE    : otherwise
```

Result:

| Risk | Count |
|---|---:|
| SAFE | 1,781 |
| CAUTION | 207 |
| WARNING | 22 |
| DANGER | 0 |

Representative case:

```text
Track 196
Class = car
Distance = 7.09 m
Closing Speed = 12.88 m/s
TTC = 0.55 s
DCPA = 3.44 m
Risk = WARNING
```

DCPA suppresses TTC-only over-warning when an object is rapidly approaching but is not predicted to pass inside the closest-danger separation threshold.

> TTC / DCPA thresholds are experimental heuristics and are not certified automotive safety thresholds.

---

## 10. Camera–LiDAR Fusion

### 10.1 Geometric Alignment

nuScenes calibration information was used to transform LiDAR 3D detections into the CAM_FRONT image plane.

```text
LiDAR
→ ego vehicle
→ global
→ camera ego
→ camera sensor
→ image projection
```

This verified spatial alignment between PointPillars detections and camera objects.

### 10.2 Camera Detection

CAM_FRONT images were processed using YOLO11n.

Mapped classes:

```text
car, truck, bus, motorcycle, bicycle, pedestrian
```

For `scene-0103`:

| Metric | Result |
|---|---:|
| Frames | 40 |
| Camera Detections | 672 |

### 10.3 Late Fusion

Camera and LiDAR detections were associated in the image plane using:

```text
same class
+ 2D IoU
+ Hungarian Matching
```

For matched objects, camera and LiDAR confidence scores were combined while preserving LiDAR 3D position and distance.

```text
fusion_score
= 0.60 × camera_score
+ 0.40 × lidar_score
```

The fusion score is an object-level combined score and is not guaranteed to be higher than either individual sensor score.

### 10.4 Multisensor Coverage Evaluation

A custom GT-coverage diagnostic was performed over `scene-0103`.

Evaluation conditions:

- CAM_FRONT-visible GT only
- class-consistent one-to-one matching
- Camera: 2D IoU >= 0.50
- LiDAR: center distance <= 2 m
- LiDAR score >= 0.30

| Method | Matched GT | Coverage Recall |
|---|---:|---:|
| Camera-only | 316 | 0.432 |
| LiDAR-only | 195 | 0.266 |
| **Fusion Union** | **377** | **0.515** |
| Both Sensors | 134 | 0.183 |

Total CAM_FRONT-visible GT objects:

```text
732
```

Fusion gain:

```text
vs Camera-only : +0.083
vs LiDAR-only  : +0.249
```

This indicates that the two sensors have complementary detection behavior and can compensate for different failure cases.

> Coverage Recall is a custom diagnostic metric, not an official nuScenes multimodal benchmark metric.

---

## 11. Edge / Deployment Optimization

### 11.1 ONNX Export

The PointPillars `BaseBEVBackbone` was exported to ONNX.

```text
Input : spatial_features    (1, 64, 512, 512)
Output: spatial_features_2d (1, 384, 128, 128)
```

Validation:

| Metric | Result |
|---|---:|
| ONNX Checker | PASS |
| 99th Percentile Abs. Diff | 0.00143294 |
| 99.9th Percentile Abs. Diff | 0.00395483 |
| Relative L2 Error | 0.00173204 |

This verifies feature-level numerical consistency within a small error range.

> Only the BEV backbone was exported. This is not a full end-to-end PointPillars ONNX deployment.

### 11.2 Runtime Benchmark

Benchmark condition:

```text
Input shape: (1, 64, 512, 512)
Warmup: 20
Runs: 100
```

| Runtime | Device | Mean Latency | p50 | p95 | FPS |
|---|---|---:|---:|---:|---:|
| PyTorch | GPU | **12.333 ms** | 12.410 ms | 12.727 ms | **81.08** |
| PyTorch | CPU | 352.772 ms | 264.290 ms | 1230.809 ms | 2.83 |
| ONNX Runtime | CPU | **162.872 ms** | 162.843 ms | 163.803 ms | **6.14** |

CPU optimization result:

```text
PyTorch CPU → ONNX Runtime CPU
352.772 ms → 162.872 ms
2.17× speedup
```

> GPU FPS refers only to the `BaseBEVBackbone` benchmark and must not be interpreted as end-to-end PointPillars FPS.

---

## 12. Key Results

| Area | Result |
|---|---|
| PointPillars Baseline | mAP 0.4134 / NDS 0.4920 |
| Distance Analysis | Recall 0.761 → 0.600 → 0.267 with increasing distance |
| Sweep Ablation | mAP 0.3135 → 0.3717 → 0.4134 for 1 / 5 / 10 sweeps |
| Tracking | ID Switch Rate 0.064 |
| Dynamic Risk | TTC + DCPA reduced TTC-only over-warning |
| Camera Coverage | 0.432 |
| LiDAR Coverage | 0.266 |
| Fusion Coverage | **0.515** |
| Fusion Gain | +0.083 vs Camera / +0.249 vs LiDAR |
| ONNX CPU Optimization | **2.17× speedup** |
| Backbone GPU Runtime | 12.333 ms / 81.08 FPS |

---

## 13. Project Structure

```text
lidar-risk-tracking/
├── data/
│   └── nuscenes -> external dataset
│
├── experiments/
│
├── outputs/
│   ├── images/
│   ├── logs/
│   └── pointpillars/
│
├── src/
│   ├── baseline/
│   │   ├── 01_load_lidar.py
│   │   ├── 02_roi_filter.py
│   │   ├── 03_ground_removal.py
│   │   ├── 04_dbscan_clustering.py
│   │   ├── 05_find_good_frames.py
│   │   ├── 06_baseline_tracking.py
│   │   ├── 07_baseline_tracking_summary.py
│   │   ├── 08_stable_tracking.py
│   │   ├── 09_stable_tracking_summary.py
│   │   ├── 10_plot_stable_distance_clean.py
│   │   └── 11_plot_stable_bev_tracking.py
│   │
│   └── detection_3d/
│       ├── analysis/
│       │   ├── analyze_distance_recall.py
│       │   ├── analyze_point_density.py
│       │   ├── compare_sweep_distance_recall.py
│       │   └── evaluate_tracking_stability.py
│       ├── inference/
│       │   ├── track_pointpillars.py
│       │   └── visualize_pointpillars_bev.py
│       ├── risk/
│       │   ├── compute_ttc_risk.py
│       │   └── compute_ttc_dcpa_risk.py
│       ├── fusion/
│       │   ├── run_camera_detection.py
│       │   ├── run_camera_detection_scene.py
│       │   ├── visualize_camera_lidar_fusion.py
│       │   ├── fuse_camera_lidar.py
│       │   ├── evaluate_multisensor_coverage.py
│       │   └── visualize_camera_lidar_fusion_result.py
│       ├── edge/
│       │   ├── inspect_pointpillars_for_onnx.py
│       │   ├── export_pointpillars_backbone_onnx.py
│       │   └── benchmark_backbone_runtime.py
│       └── visualization/
│           ├── visualize_tracking_risk_bev.py
│           └── visualize_tracking_risk_bev_portfolio.py
│
├── third_party/
│   └── OpenPCDet/
├── README.md
└── requirements.txt
```

---

## 14. Main Outputs

```text
outputs/pointpillars/
├── pointpillars_bev_sample0.png
├── distance_recall_baseline.csv
├── point_density_objects.csv
├── point_density_summary.csv
├── pointpillars_tracks.csv
├── pointpillars_ttc_risk.csv
├── pointpillars_ttc_dcpa_risk.csv
├── portfolio/
│   └── tracking_risk_bev_portfolio.png
├── camera_lidar/
│   ├── camera_detections_scene0103_all.csv
│   ├── camera_lidar_fusion_portfolio.png
│   └── multisensor_coverage_scene0103_corrected.csv
└── edge/
    └── pointpillars_backbone.onnx
```

---

## 15. Tech Stack

### Programming

- Python
- NumPy
- Pandas

### Deep Learning / 3D Detection

- PyTorch
- OpenPCDet
- PointPillars
- spconv

### Point Cloud / Sensor Processing

- nuScenes-devkit
- Open3D
- scikit-learn
- SciPy

### Tracking / Risk

- Hungarian Matching
- Kalman Filter
- TTC
- DCPA

### Camera / Multisensor

- Ultralytics YOLO11
- nuScenes calibration
- Camera–LiDAR geometric projection
- IoU-based late fusion

### Deployment

- ONNX
- ONNX Runtime

### Visualization

- Matplotlib
- Open3D

---

## 16. Limitations

1. The PointPillars checkpoint was pretrained on the full nuScenes dataset. Reported mAP / NDS values are evaluation results, not self-trained model performance.
2. Sweep and pillar-size experiments reuse the same pretrained checkpoint and are robustness / mismatch ablations rather than independently trained fair comparisons.
3. Tracking metrics are affected by detector recall and confirmed-track filtering.
4. TTC / DCPA thresholds are heuristic experimental settings and are not automotive safety-certified thresholds.
5. Camera–LiDAR Coverage Recall is a custom diagnostic metric, not an official nuScenes multimodal metric.
6. The current late-fusion method uses geometric association and confidence combination rather than an end-to-end learned multimodal model.
7. ONNX optimization was evaluated for the PointPillars BEV backbone, not the full end-to-end detector.
8. The project uses recorded nuScenes data rather than live physical LiDAR input.

---

## 17. Key Insights

1. LiDAR detection performance decreases strongly with range as point density becomes sparse.
2. Temporal sweep accumulation can mitigate sparse long-range observations.
3. Detection quality cannot be explained by point count alone; class, occlusion, and geometry also matter.
4. Replacing DBSCAN candidates with semantic PointPillars detections enables class-aware tracking and risk analysis.
5. Hungarian Matching and Kalman filtering provide a practical temporal layer over frame-level 3D detections.
6. TTC alone can over-warn; DCPA adds predicted path-separation information.
7. Camera and LiDAR show complementary detection behavior, raising union coverage to 0.515.
8. ONNX Runtime reduced BEV-backbone CPU latency from 352.772 ms to 162.872 ms, a 2.17× speedup.

---

## 18. Conclusion

This project evolved from a classical LiDAR clustering baseline into a complete perception and risk-analysis pipeline:

```text
LiDAR Sensor Analysis
→ PointPillars 3D Detection
→ Hungarian + Kalman Tracking
→ TTC + DCPA Risk Assessment
→ Camera–LiDAR Fusion
→ GT-based Coverage Evaluation
→ ONNX Runtime Optimization
```

The most important outcome is not simply the use of PointPillars.

> **LiDAR sensor characteristics were analyzed quantitatively, 3D detections were connected over time, relative motion was converted into interpretable risk, camera information was used to compensate for LiDAR detection failures, and deployment efficiency was evaluated through ONNX Runtime optimization.**

This provides a practical foundation for real-time multimodal perception, edge deployment, and sensor-based safety monitoring.
