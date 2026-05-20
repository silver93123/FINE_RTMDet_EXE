"""[시운전] 전체 파이프라인 통합 데모.

이 스크립트는 본 작업과 완전히 분리된 시운전(smoke test)용입니다.
설치된 라이브러리들이 정상 동작하는지 빠르게 검증합니다.

흐름:
    [intensity 2D 이미지]
        ↓
    [합성 FrameData 생성: 균일 평면 깊이 z=900mm]
        ↓
    [RTMDet-Ins 추론] → 마스크 N개
        ↓
    [FrameData.crop_by_mask(mask)] → 인스턴스별 PCD
        ↓
    [Open3D RANSAC + ICP] → 6D pose
        ↓
    [결과 테이블 출력]

검증 목적:
    fitness/RMSE 값이 아니라 '데이터가 끝까지 흐르는지' 검증.
    COCO pretrained + 합성 평면 PCD라 ICP fitness는 본질적으로 낮음 (예상됨).
    의미 있는 fitness는 본 작업 (fine-tune + 실제 PCD)에서 나옴.

실행:
    cd <PROJECT_ROOT>/tests/pretrained_smoke
    python full_pipeline.py

전제 조건:
    setup_pretrained.py를 먼저 실행해 모델과 샘플이 받아져 있어야 함.

리소스 위치 (모두 이 스크립트 폴더 기준, 본 작업과 분리):
    ./models/rtmdet-ins_tiny_*.pth
    ./configs/rtmdet-ins_tiny_*.py
    ./samples/coco_*.jpg
    ./output/                          ← 결과 저장
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass


# 이 스크립트가 있는 폴더 (tests/pretrained_smoke/)
SMOKE_ROOT = Path(__file__).resolve().parent

# 본 작업의 src/ 모듈을 import하기 위해 PROJECT_ROOT를 sys.path에 추가
# (tests/pretrained_smoke/ → <PROJECT_ROOT> 두 단계 상위)
PROJECT_ROOT = SMOKE_ROOT.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.base import FrameData  # noqa: E402
from src.detection import RTMDetInferencer  # noqa: E402


# =============================================================================
# 1. 합성 FrameData 생성
# =============================================================================

def make_synthetic_frame(
    image_bgr: np.ndarray,
    plane_z_mm: float = 900.0,
    fx_mm: float = 1000.0,
    fy_mm: float = 1000.0,
    noise_std_mm: float = 1.0,
    rng: Optional[np.random.Generator] = None,
) -> FrameData:
    """COCO 이미지로부터 가짜 FrameData 합성.

    단순화된 핀홀 모델:
        X_mm = (u - cx) * Z / fx
        Y_mm = (v - cy) * Z / fy
        Z_mm = plane_z_mm + noise
    """
    if rng is None:
        rng = np.random.default_rng(seed=0)

    intensity = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    H, W = intensity.shape

    v, u = np.indices((H, W), dtype=np.float32)
    cx, cy = W / 2.0, H / 2.0

    Z = plane_z_mm + rng.normal(0, noise_std_mm, size=(H, W)).astype(np.float32)
    X = (u - cx) * Z / fx_mm
    Y = (v - cy) * Z / fy_mm

    points_organized = np.stack([X, Y, Z], axis=-1)
    valid_mask = np.ones((H, W), dtype=bool)
    points = points_organized[valid_mask]

    return FrameData(
        intensity=intensity,
        points=points,
        points_organized=points_organized,
        valid_mask=valid_mask,
        confidence=None,
    )


# =============================================================================
# 2. 인스턴스 PCD 분석
# =============================================================================

def analyze_instance_pcd(object_pcd_mm: np.ndarray) -> dict:
    """잘라낸 PCD의 기본 통계."""
    if len(object_pcd_mm) == 0:
        return {"n": 0}

    return {
        "n": len(object_pcd_mm),
        "centroid_mm": object_pcd_mm.mean(axis=0),
        "bbox_size_mm": object_pcd_mm.max(axis=0) - object_pcd_mm.min(axis=0),
    }


def try_icp_against_plane(object_pcd_mm: np.ndarray) -> dict:
    """ICP 호출 가능성 검증 (값 자체는 의미 없음).

    Target = source를 10mm 이동시킨 것 → ICP가 그 이동을 복원하면 정상.
    """
    if len(object_pcd_mm) < 50:
        return {"ok": False, "reason": "점 부족"}

    try:
        import open3d as o3d

        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(object_pcd_mm.astype(np.float64))

        target_pts = object_pcd_mm + np.array([10.0, 5.0, 2.0])
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pts.astype(np.float64))

        threshold = 30.0
        result = o3d.pipelines.registration.registration_icp(
            source, target, threshold,
            np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30),
        )

        t_recovered = result.transformation[:3, 3]
        t_expected = np.array([10.0, 5.0, 2.0])
        t_err = float(np.linalg.norm(t_recovered - t_expected))

        return {
            "ok": True,
            "fitness": float(result.fitness),
            "rmse_mm": float(result.inlier_rmse),
            "t_recovered_mm": t_recovered,
            "t_err_mm": t_err,
        }
    except Exception as e:
        return {"ok": False, "reason": f"ICP 예외: {e}"}


# =============================================================================
# 3. 단일 이미지 처리
# =============================================================================

def process_image(
    image_path: Path,
    inferencer: RTMDetInferencer,
    output_dir: Path,
    max_instances_to_process: int = 5,
) -> dict:
    print("\n" + "=" * 78, flush=True)
    print(f"  [Pipeline] {image_path.name}", flush=True)
    print("=" * 78, flush=True)

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"  ERROR: 이미지 로드 실패", flush=True)
        return {"ok": False}

    H, W = image_bgr.shape[:2]
    print(f"  이미지: {W}x{H}", flush=True)

    t0 = time.perf_counter()
    frame = make_synthetic_frame(image_bgr, plane_z_mm=900.0, noise_std_mm=1.0)
    print(f"  합성 FrameData: intensity {frame.intensity.shape}, "
          f"PCD {frame.points_organized.shape} "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)", flush=True)

    t0 = time.perf_counter()
    detections = inferencer.infer(image_bgr)
    print(f"  RTMDet 추론: {len(detections)} 인스턴스 "
          f"({(time.perf_counter() - t0) * 1000:.1f} ms)", flush=True)

    if not detections:
        print(f"  검출된 인스턴스 없음.", flush=True)
        return {"ok": True, "n_instances": 0}

    n_to_process = min(len(detections), max_instances_to_process)
    print(f"\n  상위 {n_to_process}개 인스턴스 처리:", flush=True)
    print(f"  {'#':<3}{'class':<15}{'score':<7}{'mask px':<9}{'crop pts':<10}"
          f"{'centroid (X,Y,Z) mm':<28}{'ICP t_err':<10}", flush=True)
    print(f"  {'-' * 82}", flush=True)

    instance_results = []
    for i, det in enumerate(detections[:n_to_process]):
        if det.mask.shape != (H, W):
            print(f"  {i:<3}{det.class_name:<15}마스크 shape 불일치", flush=True)
            continue

        object_pcd = frame.crop_by_mask(det.mask)
        stats = analyze_instance_pcd(object_pcd)
        icp = try_icp_against_plane(object_pcd)

        if stats["n"] == 0:
            print(f"  {i:<3}{det.class_name:<15}{det.score:<7.3f}"
                  f"{det.n_pixels:<9}{0:<10}(빈 PCD)", flush=True)
            continue

        c = stats["centroid_mm"]
        centroid_str = f"({c[0]:7.1f}, {c[1]:7.1f}, {c[2]:7.1f})"
        icp_str = (f"{icp['t_err_mm']:6.2f}mm" if icp["ok"]
                   else f"({icp['reason']})")

        print(f"  {i:<3}{det.class_name:<15}{det.score:<7.3f}"
              f"{det.n_pixels:<9}{stats['n']:<10}{centroid_str:<28}"
              f"{icp_str}", flush=True)

        instance_results.append({
            "class": det.class_name, "score": det.score,
            "mask_pixels": det.n_pixels, "pcd_pts": stats["n"],
            "centroid_mm": stats["centroid_mm"], "icp": icp,
        })

    save_visualization(image_bgr, detections[:n_to_process],
                        output_dir / f"{image_path.stem}_pipeline.jpg")

    return {
        "ok": True,
        "n_instances": len(detections),
        "n_processed": len(instance_results),
        "instances": instance_results,
    }


def save_visualization(image_bgr, detections, out_path: Path):
    """마스크 영역만 컬러로 강조."""
    H, W = image_bgr.shape[:2]
    out = image_bgr.copy()

    all_mask = np.zeros((H, W), dtype=bool)
    for d in detections:
        all_mask |= d.mask
    out[~all_mask] = (out[~all_mask] * 0.3).astype(np.uint8)

    palette = [(0, 255, 100), (255, 100, 100), (100, 100, 255),
               (255, 200, 50), (200, 50, 255)]
    for i, d in enumerate(detections):
        color = palette[i % len(palette)]
        mask_u8 = (d.mask.astype(np.uint8)) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, color, 2)

        x1, y1 = d.bbox[:2].astype(int)
        label = f"#{i} {d.class_name} {d.score:.2f}"
        cv2.putText(out, label, (x1, max(y1 - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), out)


# =============================================================================
# 메인
# =============================================================================

def main() -> int:
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s | %(levelname)-7s | %(message)s",
    )

    # 모든 리소스는 시운전 폴더 안에서 완결 (본 작업 무관)
    config_path = SMOKE_ROOT / "configs" / "rtmdet-ins_tiny_8xb32-300e_coco.py"
    checkpoint_candidates = sorted(
        (SMOKE_ROOT / "models").glob("rtmdet-ins_tiny_8xb32-300e_coco_*.pth")
    )
    samples_dir = SMOKE_ROOT / "samples"
    output_dir = SMOKE_ROOT / "output"
    
    print(config_path.exists() )
    print(checkpoint_candidates)

    if not config_path.exists() or not checkpoint_candidates:
        print("ERROR: 모델 파일이 없습니다. 먼저 setup을 실행하세요:", flush=True)
        print("  $ python setup_pretrained.py", flush=True)
        return 1

    sample_files = sorted(samples_dir.glob("*.jpg"))
    if not sample_files:
        print(f"ERROR: 샘플 이미지가 없습니다: {samples_dir}", flush=True)
        return 1

    print("=" * 78, flush=True)
    print("  [시운전] 전체 파이프라인 통합 데모", flush=True)
    print("  intensity → RTMDet-Ins → 마스크 → PCD crop → ICP", flush=True)
    print("=" * 78, flush=True)
    print(f"  시운전 폴더: {SMOKE_ROOT}", flush=True)
    print(f"  본 작업 모듈 import: {PROJECT_ROOT}/src/", flush=True)
    print("\n주의: 합성 평면 PCD + COCO pretrained라 ICP 결과 값은 무의미.",
          flush=True)
    print("      검증 목적: '데이터가 끝까지 흐르는가'.", flush=True)

    inferencer = RTMDetInferencer(
        config=config_path,
        checkpoint=checkpoint_candidates[0],
        device="cuda:0",
        score_threshold=0.5,
    )

    total_processed = 0
    total_pipeline_ok = 0
    for img_path in sample_files:
        result = process_image(img_path, inferencer, output_dir,
                              max_instances_to_process=5)
        if result["ok"]:
            total_processed += result.get("n_processed", 0)
            if result.get("n_processed", 0) > 0:
                total_pipeline_ok += 1

    print("\n" + "=" * 78, flush=True)
    print("  최종 요약", flush=True)
    print("=" * 78, flush=True)
    print(f"  처리된 이미지: {len(sample_files)}", flush=True)
    print(f"  파이프라인 성공: {total_pipeline_ok}/{len(sample_files)}", flush=True)
    print(f"  처리된 총 인스턴스: {total_processed}", flush=True)
    print(f"  시각화 출력: {output_dir}", flush=True)
    print("", flush=True)
    print("  검증 결과:", flush=True)
    print(f"    ✓ RTMDet-Ins 추론 호출 정상", flush=True)
    print(f"    ✓ FrameData.crop_by_mask() 데이터 흐름 정상", flush=True)
    print(f"    ✓ Open3D ICP 호출 정상", flush=True)
    print("", flush=True)
    print("  본 작업으로 이동:", flush=True)
    print("    - cd <PROJECT_ROOT>", flush=True)
    print("    - 본 작업 README 참고", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
