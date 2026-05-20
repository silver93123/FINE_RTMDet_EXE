"""Stage 3: RTMDet-Ins fine-tuning 실행 스크립트.

실행:
    cd ~/binpicking_vision/RTM_test
    python scripts/4_train_rtmdet.py

내부 동작:
    mmdet의 'tools/train.py' 로직을 그대로 호출.
    config 파일 경로만 지정.

참고:
    실제로는 mmdet 패키지가 제공하는 명령:
        python -m mmdet.tools.train configs/rtmdet-ins_bracket.py
    를 호출하는 것과 동일. 이 스크립트는 그것을 우리 프로젝트 구조에
    맞춰 한 줄로 감싸준 것.

학습 중 출력:
    - epoch별 loss
    - val_interval마다 mAP 평가
    - 체크포인트 저장 알림

학습 결과:
    work_dirs/rtmdet-ins_bracket_v1/
    ├── epoch_10.pth, epoch_20.pth, ...  ← 체크포인트
    ├── best_*.pth                        ← 최고 성능 모델
    ├── last_checkpoint                   ← 마지막 학습 위치
    ├── *.log                             ← 학습 로그
    └── vis_data/                         ← TensorBoard 로그
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    config_path = ROOT / "configs" / "rtmdet-ins_bracket.py"

    if not config_path.exists():
        print(f"ERROR: config 파일이 없습니다: {config_path}", flush=True)
        return 1

    print("=" * 70, flush=True)
    print("  STEP 2: RTMDet-Ins fine-tuning", flush=True)
    print("=" * 70, flush=True)
    print(f"  Config: {config_path}", flush=True)
    print(f"  Project root: {ROOT}", flush=True)
    print("", flush=True)
    print("  실시간 진행 상황은 아래 로그 + work_dirs/.../*.log 파일에서 확인.", flush=True)
    print("  중단하려면 Ctrl+C", flush=True)
    print("=" * 70, flush=True)
    print("", flush=True)

    # mmdet의 train 모듈을 직접 호출
    # (sys.argv를 mmdet/tools/train.py 형식으로 세팅)
    import os
    os.chdir(ROOT)  # 상대경로 안전성을 위해 프로젝트 루트로 이동

    # mmdet 3.x에선 tools/train.py를 직접 import해서 호출하는 방식 권장
    from mmengine.config import Config
    from mmengine.runner import Runner

    cfg = Config.fromfile(str(config_path))

    # work_dir 보장
    work_dir = Path(cfg.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    # Runner 생성 + 학습 시작
    runner = Runner.from_cfg(cfg)
    runner.train()

    print("\n" + "=" * 70, flush=True)
    print("  학습 완료", flush=True)
    print("=" * 70, flush=True)
    print(f"  체크포인트: {work_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
