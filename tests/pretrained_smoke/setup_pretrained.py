"""Step 1: RTMDet-Ins 사전학습 모델 + 샘플 이미지 다운로드.

실행:
    python scripts/1_setup_pretrained.py

다운로드 항목:
    1. RTMDet-Ins tiny 모델 가중치 (~10MB) + config
       출처: https://github.com/open-mmlab/mmdetection/blob/main/configs/rtmdet/README.md
    2. COCO 샘플 이미지 5장 (~1MB)
       다양한 일상 객체 포함 - 파이프라인 검증용

저장 위치:
    models/rtmdet-ins_tiny_8xb32-300e_coco_*.pth
    configs/rtmdet-ins_tiny_8xb32-300e_coco.py
    data/samples/coco_*.jpg
"""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


# 이 스크립트가 있는 폴더 기준 (tests/pretrained_smoke/)
SMOKE_ROOT = Path(__file__).resolve().parent
MODELS_DIR = SMOKE_ROOT / "models"
CONFIGS_DIR = SMOKE_ROOT / "configs"
SAMPLES_DIR = SMOKE_ROOT / "samples"


# RTMDet-Ins tiny - mmdetection 공식 model zoo
# 출처: https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet
MODEL_URL = (
    "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/"
    "rtmdet-ins_tiny_8xb32-300e_coco/"
    "rtmdet-ins_tiny_8xb32-300e_coco_20221130_151727-ec670f7e.pth"
)
MODEL_FILENAME = "rtmdet-ins_tiny_8xb32-300e_coco.pth"


# COCO val2017 샘플 이미지 5장
# 일상 물체가 풍부하게 들어있어 다양한 클래스 검출 확인 가능
SAMPLE_URLS = [
    # (filename, url)
    ("coco_000000397133.jpg",  # 식탁: 음식, 컵, 그릇 등
     "http://images.cocodataset.org/val2017/000000397133.jpg"),
    ("coco_000000037777.jpg",  # 길거리: 자동차, 사람 등
     "http://images.cocodataset.org/val2017/000000037777.jpg"),
    ("coco_000000252219.jpg",  # 주방: 다양한 물건
     "http://images.cocodataset.org/val2017/000000252219.jpg"),
    ("coco_000000087038.jpg",  # 거실: 가구, 가전
     "http://images.cocodataset.org/val2017/000000087038.jpg"),
    ("coco_000000174482.jpg",  # 동물 또는 사람
     "http://images.cocodataset.org/val2017/000000174482.jpg"),
]


def download(url: str, dst: Path, label: str = "") -> bool:
    """파일 다운로드 (이미 존재하면 스킵)."""
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  [skip] {label or dst.name} 이미 존재 ({dst.stat().st_size // 1024} KB)",
              flush=True)
        return True

    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [down] {label or dst.name}", flush=True)
    print(f"         from {url}", flush=True)

    try:
        # 진행 표시용 reporthook
        def _progress(block_num, block_size, total_size):
            if total_size > 0:
                pct = min(100, 100 * block_num * block_size / total_size)
                mb = block_num * block_size / 1024 / 1024
                total_mb = total_size / 1024 / 1024
                print(f"\r         {pct:5.1f}% ({mb:.1f}/{total_mb:.1f} MB)",
                      end="", flush=True)

        urllib.request.urlretrieve(url, dst, reporthook=_progress)
        print("", flush=True)  # 줄바꿈
        size_kb = dst.stat().st_size // 1024
        print(f"         ✓ saved ({size_kb} KB)", flush=True)
        return True
    except Exception as e:
        print(f"\n         ✗ FAILED: {e}", flush=True)
        if dst.exists():
            dst.unlink()
        return False


def get_config_via_mim() -> bool:
    """mim 명령으로 config 다운로드 (가장 안전한 방법).

    mim은 OpenMMLab 모델/config 통합 다운로더입니다.
    """
    cfg_dst = CONFIGS_DIR / "rtmdet-ins_tiny_8xb32-300e_coco.py"

    if cfg_dst.exists():
        print(f"  [skip] config 이미 존재: {cfg_dst.name}", flush=True)
        return True

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    # mmdetection 패키지 내부의 config를 찾아서 복사
    try:
        import mmdet
        mmdet_root = Path(mmdet.__file__).parent.parent
        src_cfg = (mmdet_root / "configs" / "rtmdet"
                   / "rtmdet-ins_tiny_8xb32-300e_coco.py")

        if src_cfg.exists():
            import shutil
            shutil.copy(src_cfg, cfg_dst)
            print(f"  [copy] config: {src_cfg.name} → {cfg_dst}", flush=True)
            return True
        else:
            print(f"  [warn] mmdet 내부에서 config를 찾지 못함: {src_cfg}", flush=True)
    except Exception as e:
        print(f"  [warn] mmdet config 복사 실패: {e}", flush=True)

    # 대안: GitHub raw URL에서 다운로드
    config_url = (
        "https://raw.githubusercontent.com/open-mmlab/mmdetection/main/"
        "configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py"
    )
    return download(config_url, cfg_dst, "config (RTMDet-Ins tiny)")


def main() -> int:
    print("=" * 70, flush=True)
    print("  RTMDet-Ins 사전학습 모델 + 샘플 이미지 다운로드", flush=True)
    print("=" * 70, flush=True)
    print(f"  Project root: {SMOKE_ROOT}", flush=True)
    print("", flush=True)

    # 1. 모델 가중치
    print("[1/3] 모델 가중치 다운로드", flush=True)
    model_dst = MODELS_DIR / MODEL_FILENAME
    ok_model = download(MODEL_URL, model_dst, "RTMDet-Ins tiny weights")
    print("", flush=True)

    # 2. Config 파일
    print("[2/3] Config 파일 준비", flush=True)
    ok_cfg = get_config_via_mim()
    print("", flush=True)

    # 3. 샘플 이미지 5장
    print("[3/3] COCO 샘플 이미지 5장 다운로드", flush=True)
    n_ok = 0
    for fname, url in SAMPLE_URLS:
        dst = SAMPLES_DIR / fname
        if download(url, dst, fname):
            n_ok += 1
    print("", flush=True)

    # 요약
    print("=" * 70, flush=True)
    print("  요약", flush=True)
    print("=" * 70, flush=True)
    print(f"  모델 가중치: {'OK' if ok_model else 'FAIL'}", flush=True)
    print(f"  Config:     {'OK' if ok_cfg else 'FAIL'}", flush=True)
    print(f"  샘플 이미지: {n_ok}/{len(SAMPLE_URLS)} 성공", flush=True)
    print("", flush=True)
    print(f"  모델 경로:   {model_dst}", flush=True)
    print(f"  Config 경로: {CONFIGS_DIR / 'rtmdet-ins_tiny_8xb32-300e_coco.py'}", flush=True)
    print(f"  샘플 경로:   {SAMPLES_DIR}", flush=True)

    if ok_model and ok_cfg and n_ok >= 3:
        print("\n  ✓ 다음 단계로 진행하세요: python scripts/2_inference_demo.py",
              flush=True)
        return 0
    else:
        print("\n  ✗ 일부 다운로드 실패. 네트워크 확인 후 다시 시도하세요.",
              flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
