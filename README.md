# Binpicking Vision Pipeline

RTMDet-Ins 기반 2D 인스턴스 세그멘테이션과 ICP 3D 정합을 활용한 브라켓 빈피킹 시스템.

---

## 목차

1. [환경 요구사항](#1-환경-요구사항)
2. [설치](#2-설치)
3. [프로젝트 구조](#3-프로젝트-구조)
4. [사용자 조정 변수](#4-사용자-조정-변수)
5. [실행 순서](#5-실행-순서)
6. [출력 파일 설명](#6-출력-파일-설명)

---

## 1. 환경 요구사항

| 항목 | 버전 |
|------|------|
| OS | Ubuntu 22.04 |
| Python | 3.10 |
| CUDA | 11.8 이상 (GPU 추론용) |
| GPU | NVIDIA (RTMDet CUDA 추론) |
| 카메라 | LUCID Helios ToF 카메라 |

---

## 2. 설치

### 2-1. Conda 환경 생성

```bash
conda create -n vision_env python=3.9 -y
conda activate vision_env
```

### 2-2. PyTorch 설치 (CUDA 버전에 맞게)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### 2-3. MMDetection 계열 설치

```bash
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install "mmdet>=3.0.0"
```

### 2-4. 기타 의존성 설치

```bash
pip install open3d opencv-python numpy pyyaml
```

### 2-5. LUCID 카메라 SDK 설치

```bash
# Arena SDK 설치 후
pip install arena_api
```

### 2-6. 프로젝트 설치

```bash
git clone <repo_url>
cd binpicking_vision/RTM_test
pip install -e .
```

---

## 3. 프로젝트 구조

```
RTM_test/
├── config/
│   └── config.yaml                  ← 카메라 설정
├── data/
│   ├── cad/
│   │   └── bracket_v2.stl           ← 브라켓 CAD 모델 (mm 단위)
│   ├── dataset_input/               ← 배치 처리용 입력 데이터
│   └── captures/live/               ← 실전 캡처 저장 경로
├── scripts/
│   ├── 0_1_Camera_capture_test.py   ← 카메라 연결 테스트
│   ├── 1_Collect_dataset.py         ← 학습 데이터 수집
│   ├── 3_Detect_and_PickPoint.py    ← 배치 처리 (저장 파일 → 픽포인트)
│   └── run_binpicking.py            ← 실전 파이프라인 (카메라 → 픽포인트)
├── src/
│   ├── camera/                      ← 카메라 드라이버
│   └── detection/                   ← RTMDet inferencer
└── work_dirs/
    └── rtmdet-ins_bracket_v1/
        ├── rtmdet-ins_bracket.py    ← 모델 config
        └── best_coco_bbox_mAP_epoch_50.pth  ← 학습된 가중치
```

---

## 4. 사용자 조정 변수

### 4-1. `config/config.yaml` — 카메라 설정

카메라 하드웨어 환경에 맞게 조정.

```yaml
camera:
  type: helios
  exposure_time_selector: ...   # 노출 시간
  operating_mode: ...           # 동작 모드
  pixel_format: ...
```

### 4-2. `run_binpicking.py` / `3_Detect_and_PickPoint.py` — 공통 설정

**Detection**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `SCORE_THRESHOLD` | `0.3` | 검출 신뢰도 임계값. 낮추면 더 많이 검출, 노이즈 증가 |
| `MIN_POINTS_PER_INSTANCE` | `100` | 이 이하 포인트 인스턴스는 노이즈로 제거 |

**CAD 축 보정**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `CAD_AXIS_CORRECTION_DEG` | `(-90, 90, 90)` | STL 좌표계 → 센서 좌표계 보정 회전각 (Rx, Ry, Rz). CAD 교체 시 재확인 필요 |
| `CAD_PICK_LOCAL` | `[0.000, -0.100, 0.031, 1.0]` | 축 보정 후 CAD 로컬 좌표계에서 픽포인트 위치 (m 단위). CAD 교체 시 재측정 필요 |

**ICP**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `VOXEL_SIZE_CAD` | `0.002` (2mm) | CAD 다운샘플 크기. 작을수록 정밀하지만 느림 |
| `VOXEL_SIZE_SCENE` | `0.003` (3mm) | Scene 다운샘플 크기 |
| `ICP_STAGES` | `20→10→5mm` | 다단계 ICP 수렴 거리. 정합 실패 시 첫 단계 거리를 키움 |
| `ICP_FITNESS_THRESHOLD` | `0.5` | 이 값 미만이면 정합 실패 처리. 너무 높으면 과도하게 실패 처리됨 |

**픽포인트 오프셋**

| 변수 | 기본값 | 설명 |
|------|--------|------|
| `PICK_OFFSET_X_MM` | `-5.0` | 브라켓 폭 방향 오프셋 (mm, +: 오른쪽) |
| `PICK_OFFSET_Y_MM` | `0.0` | 브라켓 길이 방향 오프셋 (mm, +: 앞) |
| `PICK_OFFSET_Z_MM` | `0.0` | 브라켓 높이 방향 오프셋 (mm, +: 위) |

### 4-3. `1_Collect_dataset.py` — 데이터 수집 설정

실행 인자로 조정.

| 인자 | 기본값 | 설명 |
|------|--------|------|
| `--out` | `data/dataset/brackets_for_train` | 저장 경로 |
| `--num` | `10` | 캡처할 프레임 수 |
| `--warmup` | `3` | 버리는 워밍업 프레임 수 |
| `--start-index` | `1` | 시작 프레임 번호 (이어서 수집 시 변경) |

---

## 5. 실행 순서

모든 명령은 프로젝트 루트에서 실행.

```bash
cd ~/binpicking_vision/RTM_test
conda activate vision_env
```

### Step 0. 카메라 연결 확인

처음 설치 후 또는 카메라 교체 시 실행.

```bash
python scripts/0_1_Camera_capture_test.py
```

확인 항목: 카메라 연결, 프레임 캡처, intensity PNG + PLY 저장, 통계 출력.

---

### Step 1. 학습 데이터 수집

RTMDet 모델 학습용 데이터가 없거나 추가 수집이 필요할 때.

```bash
python scripts/1_Collect_dataset.py \
    --out data/dataset/brackets_for_train \
    --num 50 \
    --start-index 1
```

프레임마다 Enter를 눌러 부품 배치를 바꿔가며 촬영.

---

### Step 2. RTMDet 모델 학습

수집한 데이터에 라벨링 후 학습. 학습된 가중치가 이미 있으면 스킵.

```bash
# 라벨링: CVAT 등 사용
# 학습:
python tools/train.py work_dirs/rtmdet-ins_bracket_v1/rtmdet-ins_bracket.py
```

학습 완료 후 `work_dirs/rtmdet-ins_bracket_v1/` 에 `best_coco_bbox_mAP_epoch_*.pth` 생성 확인.

---

### Step 3-A. 실전 실행 (카메라 → 픽포인트, 권장)

카메라가 연결된 상태에서 실시간으로 픽포인트를 산출.

```bash
python scripts/run_binpicking.py
```

실행 후 조작:

```
Enter  → 캡처 → Detection → ICP → 픽포인트 출력
q      → 종료
```

출력 예시:

```
┌──────────────────────────────────────────────────
│ 픽포인트 목록 (2개)
├──────────────────────────────────────────────────
│  frame_0001_obj0
│    위치:     (50.5, 172.9, 566.4) mm
│    접근벡터: (0.199, -0.042, 0.979)
│  frame_0001_obj1
│    위치:     (97.1, 180.1, 532.7) mm
│    접근벡터: (0.008, 0.001, 0.999)
└──────────────────────────────────────────────────
```

---

### Step 3-B. 배치 처리 (저장된 파일 → 픽포인트)

카메라 없이 이미 저장된 데이터로 실행하거나 파라미터 튜닝 시 사용.

```bash
# 스크립트 내 input_data 변수를 처리할 폴더명으로 수정 후 실행
python scripts/3_Detect_and_PickPoint.py
```

---

## 6. 출력 파일 설명

### 실전 실행 (`run_binpicking.py`) 기준

```
data/captures/live/
├── intensity/frame_NNNN.png               ← 캡처 강도 이미지
├── pointcloud_organized/frame_NNNN.npy   ← organized PCD (H,W,3) mm
├── valid_mask/frame_NNNN.npy             ← 유효 마스크 (H,W) bool
├── metadata/frame_NNNN.json              ← 캡처 통계
└── results/
    ├── frame_NNNN_overlay.png            ← 2D detection 시각화
    ├── frame_NNNN_colored.ply            ← 전체 PCD (인스턴스 컬러)
    ├── frame_NNNN_obj{i}.ply             ← 인스턴스 단독 PCD
    ├── frame_NNNN_summary.json           ← detection 통계
    ├── frame_NNNN_obj{i}_icp_vis.ply     ← ICP 시각화 (Open3D로 확인)
    └── frame_NNNN_obj{i}_pose.json       ← 6DoF 자세 + 픽포인트 ★
```

### `pose.json` 구조 (로봇 제어에 사용)

```json
{
  "icp_fitness": 0.91,
  "pose": {
    "xyz_mm": [50.5, 170.8, 533.4],
    "euler_deg": { "roll_deg": 2.1, "pitch_deg": 11.5, "yaw_deg": -1.3 },
    "transform_matrix": [[ ... ]]
  },
  "pick_point": {
    "position_mm":  [50.5, 172.9, 566.4],
    "approach_vec": [0.199, -0.042, 0.979],
    "approach_deg": { "roll_deg": 2.1, "pitch_deg": 11.5, "yaw_deg": -1.3 }
  }
}
```

### PLY 파일 색상 규칙 (Open3D 시각화)

| 색상 | 의미 |
|------|------|
| 회색 | 실측 포인트 클라우드 (scene) |
| 초록 | ICP 정합된 CAD 모델 |
| 빨강 | 픽포인트 (그리퍼 목표 위치) |
| 파랑 | 접근 방향 벡터 |

### PLY 확인 명령

```bash
python -c "
import open3d as o3d
o3d.visualization.draw_geometries([
    o3d.io.read_point_cloud('data/captures/live/results/frame_0001_obj0_icp_vis.ply')
])
"
```
