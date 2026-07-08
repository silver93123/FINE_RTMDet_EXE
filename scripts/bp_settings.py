"""설정 상수 모듈 — 모든 튜닝 파라미터는 여기서 관리합니다."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

try:
    import open3d as o3d
except ImportError:
    print("ERROR: open3d 필요. pip install open3d", flush=True)
    sys.exit(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# =============================================================================
# Detection
# =============================================================================
WORK_DIR      = ROOT / "work_dirs" / "rtmdet-ins_bracket_v1"
CONFIG_PATH   = WORK_DIR / "rtmdet-ins_bracket.py"

_candidates = sorted(WORK_DIR.glob("best_*.pth"))
if not _candidates:
    print(f"ERROR: best 모델이 없습니다: {WORK_DIR}", flush=True)
    sys.exit(1)
CHECKPOINT_PATH = _candidates[-1]

SCORE_THRESHOLD         = 0.3
MIN_POINTS_PER_INSTANCE = 100
MASK_IOU_THRESHOLD      = 0.65

# =============================================================================
# ICP
# =============================================================================
CAD_PATH             = ROOT / "data" / "cad" / "bracket_v2.stl"
CAD_SAMPLE_POINTS    = 20000
VOXEL_SIZE_CAD       = 0.002
VOXEL_SIZE_SCENE     = 0.003
OUTLIER_NB_NEIGHBORS = 20
OUTLIER_STD_RATIO    = 1.5

ICP_STAGES = [
    {"max_dist": 0.020, "max_iter": 100},
    {"max_dist": 0.010, "max_iter": 100},
    {"max_dist": 0.005, "max_iter": 100},
]

ICP_FITNESS_THRESHOLD   = 0.7
XYZ_MAX_M               = 2.0
CAD_AXIS_CORRECTION_DEG = (0, 90, 90)

# ICP 고정 초기 자세 (관찰 평균값으로 교체)
ICP_INIT_ROLL_DEG  = 0.0
ICP_INIT_PITCH_DEG = 0.0
ICP_INIT_YAW_DEG   = 0.0

# ICP 회전 구속 조건
ICP_ROLL_RANGE  = (-45.0, 45.0)
ICP_PITCH_RANGE = (-45.0, 45.0)
ICP_YAW_RANGE   = (-45.0, 45.0)

# =============================================================================
# 픽포인트
# =============================================================================
CAD_PICK_LOCAL          = np.array([0.000, -0.100, 0.031, 1.0])
PICK_OFFSET_X_MM        = 5.0
PICK_OFFSET_Y_MM        = 7.0
PICK_OFFSET_Z_MM        = 0.0
PICK_2D_MASK_WEIGHT     = 0.5
PICK_Z_NEIGHBOR_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
HEIGHT_POINT_OFFSET_Y_MM = 15.0

# =============================================================================
# TCP 서버
# =============================================================================
TCP_HOST = "192.168.0.22"
TCP_PORT = 29999

# =============================================================================
# 저장 옵션 (True = 저장, False = 저장 안 함)
# =============================================================================
SAVE_INTENSITY       = True   # intensity/*.png         캡처 강도 이미지
SAVE_POINTCLOUD      = False  # pointcloud_organized/*.npy  원시 포인트클라우드
SAVE_VALID_MASK      = False  # valid_mask/*.npy         유효 픽셀 마스크
SAVE_METADATA        = True   # metadata/*.json          캡처 메타데이터
SAVE_OVERLAY_PNG     = True   # results/*_overlay.png   Detection+픽포인트 오버레이
SAVE_COLORED_PLY     = False  # results/*_colored.ply   컬러 포인트클라우드
SAVE_INSTANCE_PLY    = False  # results/*_obj*.ply      인스턴스별 포인트클라우드
SAVE_RESULT_JSON     = False  # results/*_result.json   ICP 결과 JSON
SAVE_PICK_LOG_CSV    = True   # pick_log.csv            픽포인트 CSV 로그

# =============================================================================
# 색상 팔레트
# =============================================================================
_PALETTE_BGR = np.array([
    [ 50,  50, 255], [ 50, 200,  50], [255, 100,  50],
    [ 30, 180, 255], [230,  50, 180], [200, 200,  30],
], dtype=np.uint8)
_PALETTE_RGB_FLOAT = _PALETTE_BGR[:, ::-1].astype(np.float64) / 255.0
_BG_COLOR = np.array([0.55, 0.55, 0.55], dtype=np.float64)
