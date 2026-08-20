# LiDAR 3D 인지·추적·위험도 평가 및 Camera–LiDAR Fusion

## 1. 프로젝트 개요

본 프로젝트는 **nuScenes mini 데이터셋을 활용해 LiDAR 기반 3D 객체 인지와 위험 판단 파이프라인을 구축한 프로젝트**입니다.

PointPillars 기반 3D 객체 검출에서 시작해 LiDAR 센서 특성 분석, 객체 추적, TTC·DCPA 기반 동적 위험 판단, Camera–LiDAR Late Fusion, ONNX Runtime 기반 추론 최적화까지 확장했습니다.

주요 구성:

- PointPillars 기반 3D 객체 검출
- 거리별 검출 성능 및 Point Density 분석
- Multi-sweep 및 Pillar Resolution 분석
- Hungarian Matching + Kalman Filter 기반 객체 추적
- TTC / DCPA 기반 동적 위험 판단
- Camera 2D Detection 및 Camera–LiDAR Late Fusion
- GT 기반 Multisensor Coverage 평가
- ONNX Runtime 기반 배포 최적화

단순히 3D Detection 모델을 적용하는 것에 그치지 않고, **LiDAR 센서 특성이 검출 성능에 어떤 영향을 주는지 분석하고, 검출 결과를 시간축으로 연결해 객체를 추적한 뒤 위험도를 계산하며, Camera 정보를 결합해 단일 센서의 한계를 보완하고, 마지막으로 배포 환경에서의 실행 성능까지 검증하는 것**을 목표로 했습니다.

---

# 2. 데이터셋

## nuScenes Mini

본 프로젝트는 **nuScenes v1.0-mini** 데이터셋을 사용했습니다.

프로젝트에서 활용한 주요 데이터는 다음과 같습니다.

- `LIDAR_TOP`
- `CAM_FRONT`
- Ego Pose
- Sensor Calibration 정보
- Sample Annotation
- LiDAR Sweep

데이터셋 구조:

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

대용량 데이터는 Git 저장소 외부에 저장하고 Symbolic Link로 연결했습니다.

```text
data/nuscenes -> /media/rani/새 볼륨/nuscenes
```

PointPillars 평가에 사용한 nuScenes mini split:

| Split | Samples |
|---|---:|
| Train | 323 |
| Validation | 81 |
| Total | 404 |

---

# 3. 전체 시스템 구조

```text
                         ┌──────────────────────────┐
                         │      nuScenes Sensor     │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │      LiDAR Point Cloud   │
                         └────────────┬─────────────┘
                                      │
                         ┌────────────▼─────────────┐
                         │   PointPillars 3D Det.   │
                         └────────────┬─────────────┘
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
                    ▼                 ▼                 ▼
             Sensor Analysis      Tracking          CAM_FRONT
             - Distance           Hungarian          YOLO 2D
             - Point Density      + Kalman              │
             - Sweeps                │                  │
                    │                ▼                  │
                    │          Relative Motion          │
                    │                │                  │
                    │          TTC + DCPA               │
                    │                │                  │
                    └────────────────┼──────────────────┘
                                     ▼
                           Camera–LiDAR Fusion
                                     │
                                     ▼
                           Structured Risk Output
                                     │
                                     ▼
                           ONNX Runtime Benchmark
```

---

# 4. PointPillars 3D Detection

## 6.1 모델 구성

확장된 파이프라인에서는 OpenPCDet의 **PointPillars** 구현을 사용했습니다.

주요 설정:

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

구조:

```text
Voxel / Pillar Generation
→ PillarVFE
→ PointPillarScatter
→ BaseBEVBackbone
→ AnchorHeadMulti
→ 3D Detection
```

검출 클래스:

```text
car
truck
construction_vehicle
bus
trailer
barrier
motorcycle
bicycle
pedestrian
traffic_cone
```

---

## 6.2 Pretrained Baseline 평가

Full nuScenes 데이터셋으로 학습된 PointPillars Pretrained Checkpoint를 nuScenes mini Validation Split에서 평가했습니다.

> 이 결과는 nuScenes mini에서 직접 학습한 모델의 성능이 아니라 **Pretrained Baseline 평가 결과**입니다.

| Metric | Result |
|---|---:|
| Validation Samples | 81 |
| mAP | 0.4134 |
| NDS | 0.4920 |

이 Baseline을 기준으로 Sensor Characteristic Analysis, Tracking, Risk Assessment, Fusion 실험을 진행했습니다.

---

# 5. LiDAR 센서 특성 분석

## 5.1 거리별 Detection Recall

다음 기준으로 Custom Diagnostic Recall을 계산했습니다.

```text
동일 Class
Prediction Score >= 0.30
Center Distance <= 2 m
```

이는 공식 nuScenes mAP가 아닌 **보조 분석 지표**입니다.

### 전체 클래스

| Distance | GT | Matched | Recall |
|---|---:|---:|---:|
| 0–20 m | 1,513 | 1,151 | 0.761 |
| 20–40 m | 1,967 | 1,180 | 0.600 |
| 40–50 m | 461 | 123 | 0.267 |

거리가 증가할수록 검출 Recall이 크게 감소했습니다.

---

## 5.2 Point Density 분석

현재 Keyframe의 LiDAR Point가 각 GT Box 내부에 몇 개 존재하는지 계산해 Sensor Sparsity를 분석했습니다.

| Distance | Objects | Mean Points | Median Points | Recall |
|---|---:|---:|---:|---:|
| 0–20 m | 1,513 | 115.8 | 29 | 0.761 |
| 20–40 m | 1,967 | 9.6 | 4 | 0.600 |
| 40–50 m | 461 | 2.7 | 1 | 0.267 |

전체적으로 다음 관계를 확인했습니다.

```text
거리 증가
→ 객체에 포함되는 LiDAR Point 감소
→ Detection Recall 감소
```

다만 Point 수만으로 검출 성능을 모두 설명할 수는 없으며 Class, Occlusion, Object Orientation, Scene Structure 등의 영향도 존재합니다.

---

## 5.3 Multi-sweep Ablation

동일한 10-sweep Pretrained Checkpoint에 입력 Sweep 수만 줄여 성능 변화를 비교했습니다.

> 이는 Sweep 수별 모델을 각각 재학습한 공정 비교가 아니라 **Input Ablation / Robustness Experiment**입니다.

| Input Sweeps | mAP | NDS |
|---:|---:|---:|
| 1 | 0.3135 | 0.3535 |
| 5 | 0.3717 | 0.4485 |
| 10 | **0.4134** | **0.4920** |

거리별 Recall:

| Sweeps | 0–20 m | 20–40 m | 40–50 m |
|---:|---:|---:|---:|
| 1 | 0.619 | 0.378 | 0.072 |
| 5 | 0.705 | 0.541 | 0.206 |
| 10 | **0.761** | **0.600** | **0.267** |

Temporal Sweep를 누적할수록 Sparse한 LiDAR 정보를 보완할 수 있었으며, 특히 원거리에서 차이가 크게 나타났습니다.

---

## 5.4 Pillar Resolution Ablation

`0.20 m` Pillar 설정으로 학습된 동일 Pretrained Checkpoint에 서로 다른 Pillar Size를 적용했습니다.

| Pillar Size | mAP | NDS | sec / example |
|---:|---:|---:|---:|
| 0.16 m | 0.3480 | 0.4363 | 0.1209 |
| 0.20 m | **0.4134** | **0.4920** | 0.0692 |
| 0.32 m | 0.1401 | 0.2538 | 0.0883 |

> Checkpoint 자체가 `0.20 m` 기준으로 학습되었으므로 이는 독립적으로 학습된 Resolution 간 공정 비교가 아니라 **Resolution Mismatch Ablation**입니다.

학습 조건과 다른 Pillar Resolution을 적용하면 Detection Accuracy가 저하되었으며, 특히 `0.32 m`에서 큰 성능 하락이 나타났습니다.

---

# 6. 3D Object Tracking

PointPillars Detection 결과를 시간축으로 연결하기 위해 다음 Tracking Pipeline을 구성했습니다.

```text
PointPillars Detection
→ Class-aware Hungarian Matching
→ Kalman Filter
→ Confirmed Track
```

Ego Vehicle의 이동 영향을 줄이기 위해 Global Coordinate에서 Tracking을 수행했습니다.

Kalman State:

```text
[x, y, vx, vy]
```

주요 Tracking Parameter:

```text
score threshold = 0.30
matching distance = 4 m
minimum hits = 3
maximum age = 2
```

결과:

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

GT Match Rate는 Detector Recall, Score Threshold, Confirmed Track 조건의 영향을 함께 받기 때문에 순수 Tracking 성능 지표로 해석하지 않았습니다.

---

# 7. 동적 위험도 평가

## 7.1 TTC

Confirmed Track의 상대 위치와 상대 속도를 이용해 Radial Closing Speed와 TTC(Time-To-Collision)를 계산했습니다.

```text
TTC = distance / closing speed
```

초기 실험 Threshold:

```text
DANGER  : TTC < 2 s
WARNING : TTC < 4 s
CAUTION : TTC < 8 s
SAFE    : otherwise
```

결과:

| Risk | Count |
|---|---:|
| SAFE | 1,368 |
| CAUTION | 351 |
| WARNING | 264 |
| DANGER | 27 |

하지만 TTC만 사용할 경우 실제 충돌 경로가 아닌 옆으로 지나가는 객체도 과도하게 위험하게 판단할 수 있었습니다.

---

## 7.2 TTC + DCPA

TTC-only Risk의 과잉 경고를 줄이기 위해 DCPA(Distance at Closest Point of Approach)를 추가했습니다.

DCPA는 현재 상대 위치와 상대 속도를 이용해 두 객체가 가장 가까워지는 시점의 최소 거리를 계산합니다.

Risk Logic:

```text
DANGER  : TTC < 2 s AND DCPA < 2 m
WARNING : TTC < 4 s AND DCPA < 4 m
CAUTION : TTC < 8 s AND DCPA < 6 m
SAFE    : otherwise
```

결과:

| Risk | Count |
|---|---:|
| SAFE | 1,781 |
| CAUTION | 207 |
| WARNING | 22 |
| DANGER | 0 |

대표 사례:

```text
Track 196
Distance = 7.09 m
Closing Speed = 12.88 m/s
TTC = 0.55 s
DCPA = 3.44 m
Risk = WARNING
```

TTC만 보면 매우 짧은 충돌 예상 시간이지만, DCPA가 3.44 m이므로 가장 가까운 시점에서도 DANGER 기준보다 충분한 횡방향 간격이 남는 것으로 판단해 WARNING으로 분류했습니다.

> TTC / DCPA Threshold는 프로젝트 실험을 위한 Heuristic 기준이며, 실제 자동차 안전 규격에 따른 인증 기준이 아닙니다.

---

# 8. Camera–LiDAR Fusion

## 8.1 Geometric Alignment

nuScenes Calibration 정보를 이용해 PointPillars 3D Detection 결과를 CAM_FRONT Image Plane으로 투영했습니다.

```text
LiDAR
→ Ego Vehicle
→ Global
→ Camera Ego
→ Camera Sensor
→ Image Projection
```

이를 통해 Camera와 LiDAR 좌표계에서 동일 객체가 공간적으로 정렬되는지 확인했습니다.

---

## 8.2 Camera Detection

CAM_FRONT 영상에는 YOLO11n을 적용했습니다.

Mapping Class:

```text
car
truck
bus
motorcycle
bicycle
pedestrian
```

`scene-0103` 전체 결과:

| Metric | Result |
|---|---:|
| Frames | 40 |
| Camera Detections | 672 |

---

## 8.3 Late Fusion

Camera와 LiDAR Detection을 Image Plane에서 다음 조건으로 Association했습니다.

```text
동일 Class
+ 2D IoU
+ Hungarian Matching
```

동일 객체로 Matching된 경우 Camera Confidence와 LiDAR Confidence를 결합하고, LiDAR에서 얻은 3D Position과 Distance 정보는 유지했습니다.

Confidence Fusion 예시:

```text
fusion_score
= 0.60 × camera_score
+ 0.40 × lidar_score
```

Fusion Score는 두 센서의 Confidence를 통합하기 위한 객체 단위 점수이며, 개별 Camera 또는 LiDAR Confidence보다 반드시 높아지는 값은 아닙니다.

---

## 8.4 Multisensor Coverage 평가

Camera와 LiDAR가 서로의 검출 실패를 얼마나 보완하는지 확인하기 위해 `scene-0103` 전체에서 GT 기반 Custom Coverage Diagnostic을 수행했습니다.

평가 조건:

- CAM_FRONT에서 보이는 GT만 평가
- Class가 같은 객체끼리 1:1 Matching
- Camera: 2D IoU >= 0.50
- LiDAR: Center Distance <= 2 m
- LiDAR Score >= 0.30

| Method | Matched GT | Coverage Recall |
|---|---:|---:|
| Camera-only | 316 | 0.432 |
| LiDAR-only | 195 | 0.266 |
| **Fusion Union** | **377** | **0.515** |
| Both Sensors | 134 | 0.183 |

CAM_FRONT-visible GT:

```text
732
```

Fusion 개선폭:

```text
vs Camera-only : +0.083
vs LiDAR-only  : +0.249
```

Camera와 LiDAR가 서로 다른 객체를 놓치는 특성을 보였고, 두 센서의 Union을 사용했을 때 전체 객체 Coverage가 증가했습니다.

> Coverage Recall은 센서 간 보완 효과를 확인하기 위해 정의한 Custom Diagnostic이며 공식 nuScenes Multimodal Benchmark Metric은 아닙니다.

---

# 9. Edge / 배포 최적화

## 9.1 ONNX Export

PointPillars의 `BaseBEVBackbone`을 ONNX로 변환했습니다.

실제 Backbone 입력:

```text
spatial_features
shape = (1, 64, 512, 512)
```

출력:

```text
spatial_features_2d
shape = (1, 384, 128, 128)
```

ONNX Validation:

| Metric | Result |
|---|---:|
| ONNX Checker | PASS |
| 99th Percentile Abs. Diff | 0.00143294 |
| 99.9th Percentile Abs. Diff | 0.00395483 |
| Relative L2 Error | 0.00173204 |

작은 수치 오차 범위에서 PyTorch와 ONNX Backbone Feature가 유사하게 출력되는 것을 확인했습니다.

> 현재 ONNX 변환 범위는 BEV Backbone이며, 전체 PointPillars End-to-End ONNX Deployment 결과는 아닙니다.

---

## 9.2 Runtime Benchmark

Benchmark 조건:

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

CPU 기준:

```text
PyTorch CPU → ONNX Runtime CPU
352.772 ms → 162.872 ms
2.17× speedup
```

동일 CPU 환경에서 PyTorch 대비 ONNX Runtime의 평균 추론 시간이 약 2.17배 개선되었습니다.

> GPU FPS 81.08은 `BaseBEVBackbone`만 측정한 값으로, 전체 PointPillars End-to-End FPS를 의미하지 않습니다.

---

# 10. 주요 결과 요약

| Area | Result |
|---|---|
| PointPillars Baseline | mAP 0.4134 / NDS 0.4920 |
| Distance Analysis | 거리 증가에 따라 Recall 0.761 → 0.600 → 0.267 |
| Sweep Ablation | 1 / 5 / 10 Sweep에서 mAP 0.3135 → 0.3717 → 0.4134 |
| Tracking | ID Switch Rate 0.064 |
| Dynamic Risk | TTC + DCPA로 TTC-only 과잉 경고 감소 |
| Camera Coverage | 0.432 |
| LiDAR Coverage | 0.266 |
| Fusion Coverage | **0.515** |
| Fusion Gain | Camera 대비 +0.083 / LiDAR 대비 +0.249 |
| ONNX CPU 최적화 | **2.17× Speedup** |
| Backbone GPU Runtime | 12.333 ms / 81.08 FPS |

---

# 11. 주요 시각화 및 결과 파일

```text
outputs/pointpillars/
├── camera_lidar/
│   ├── camera_lidar_fusion_portfolio.png
│   └── multisensor_coverage_scene0103_corrected.csv
├── edge/
│   └── pointpillars_backbone.onnx
├── portfolio/
│   └── tracking_risk_bev_portfolio.png
├── tracking_risk_bev/
│   ├── scene-0103_frame_36.png
│   ├── scene-0103_frame_37.png
│   └── scene-0103_frame_38.png
├── distance_recall_baseline.csv
├── point_density_summary.csv
├── pointpillars_bev_sample0.png
├── pointpillars_ttc_dcpa_risk.csv
├── sweep_distance_recall_comparison.csv
└── tracking_stability_summary.csv
```

---

# 12. 프로젝트 구조

```text
lidar-risk-tracking/
├── assets/
├── configs/
├── data/
│   └── nuscenes -> external dataset
├── outputs/
│   └── pointpillars/
├── src/
│   ├── analysis/
│   ├── edge/
│   ├── fusion/
│   ├── inference/
│   └── risk/
├── .gitignore
├── README.md
├── README_ko.md
└── requirements.txt
```

`third_party/OpenPCDet/`는 로컬 실행을 위해 유지하되 Git 추적에서는 제외합니다. 재현 시에는 OpenPCDet를 별도로 설치하거나 clone해야 합니다.

---

# 13. 기술 스택

## Programming

- Python
- NumPy
- Pandas

## Deep Learning / 3D Detection

- PyTorch
- OpenPCDet
- PointPillars
- spconv

## Point Cloud / Sensor Processing

- nuScenes-devkit
- scikit-learn
- SciPy

## Tracking / Risk

- Hungarian Matching
- Kalman Filter
- TTC
- DCPA

## Camera / Multisensor

- Ultralytics YOLO11
- nuScenes Calibration
- Camera–LiDAR Geometric Projection
- IoU 기반 Late Fusion

## Deployment

- ONNX
- ONNX Runtime

## Visualization

- Matplotlib

---

# 14. 한계점

현재 프로젝트에는 다음과 같은 한계가 있습니다.

1. PointPillars Checkpoint는 Full nuScenes 데이터셋으로 사전 학습된 모델입니다. 보고된 mAP / NDS는 직접 학습 성능이 아니라 Pretrained Model 평가 결과입니다.
2. Sweep 및 Pillar Size 실험은 동일한 Pretrained Checkpoint를 사용했으므로 각각 독립적으로 학습된 모델 간의 공정 비교가 아니라 Robustness / Mismatch Ablation입니다.
3. Tracking 평가는 Detector Recall과 Confirmed Track Filtering의 영향을 함께 받습니다.
4. TTC / DCPA Threshold는 프로젝트 실험을 위해 설정한 Heuristic 기준이며 실제 자동차 안전 인증 기준이 아닙니다.
5. Camera–LiDAR Coverage Recall은 공식 nuScenes Multimodal Metric이 아니라 센서 보완 효과를 보기 위한 Custom Diagnostic입니다.
6. 현재 Fusion은 End-to-End Multimodal Network가 아니라 Calibration, Geometric Association, Confidence Combination 기반 Late Fusion입니다.
7. ONNX 최적화는 PointPillars 전체가 아닌 BEV Backbone에 대해 수행했습니다.
8. 실제 물리 LiDAR 센서의 실시간 입력이 아니라 기록된 nuScenes Sensor Data를 사용했습니다.

---

# 15. 핵심 인사이트

1. LiDAR는 거리가 증가할수록 객체에 포함되는 Point가 Sparse해지고 Detection 성능도 크게 감소했습니다.
2. 여러 Sweep의 Point Cloud를 누적하면 Sparse한 원거리 정보를 보완할 수 있었습니다.
3. Point Count만으로 Detection 성능을 모두 설명할 수 없으며 Class, Occlusion, Object Geometry 역시 영향을 줍니다.
5. Hungarian Matching과 Kalman Filter를 결합해 프레임 단위 Detection 결과를 시간축 객체 Track으로 연결했습니다.
6. TTC만 사용하면 과잉 경고가 발생할 수 있으며, DCPA를 함께 사용하면 실제 이동 경로의 최소 이격 거리를 반영할 수 있습니다.
7. Camera와 LiDAR는 서로 다른 Detection 실패 특성을 보였고, Fusion Union Coverage는 Camera 0.432 / LiDAR 0.266에서 0.515까지 증가했습니다.
8. ONNX Runtime을 적용해 BEV Backbone의 CPU Latency를 352.772 ms에서 162.872 ms로 줄여 배포 환경을 고려한 Runtime Optimization 효과를 확인했습니다.

---

# 16. 결론

```text
PointPillars 3D Detection
→ LiDAR Sensor Analysis
→ Hungarian + Kalman Tracking
→ TTC + DCPA Risk Assessment
→ Camera–LiDAR Fusion
→ GT-based Coverage Evaluation
→ ONNX Runtime Optimization
```

이 프로젝트의 핵심은 단순히 PointPillars 모델을 적용한 것이 아닙니다.

> **LiDAR 센서 특성을 정량적으로 분석하고, 3D Detection 결과를 시간축으로 연결해 객체를 추적하고, 상대 운동을 위험도로 변환하며, Camera 정보를 결합해 LiDAR의 검출 한계를 보완하고, 마지막으로 ONNX Runtime을 통해 배포 환경에서의 실행 성능까지 검증했습니다.**

이를 통해 LiDAR 기반 3D Perception을 모델 단위가 아닌 **Sensor Analysis → Detection → Tracking → Risk → Multisensor Fusion → Deployment**의 전체 시스템 관점에서 구성하고 검증했습니다.
