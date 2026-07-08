"""로거 / 디렉토리 설정 / 캡처 저장 모듈"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from scripts.bp_settings import (
    SAVE_INTENSITY, SAVE_POINTCLOUD, SAVE_VALID_MASK, SAVE_METADATA,
)


# =============================================================================
# 로거
# =============================================================================
_logger: logging.Logger | None = None


def log(msg: str) -> None:
    print(msg, flush=True)
    if _logger is not None:
        _logger.info(msg)


def setup_file_logger(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "binpicking.log"
    logger = logging.getLogger("binpicking")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    fh = logging.FileHandler(str(log_path), mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    sep        = "=" * 70
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"\n{sep}")
    logger.info(f" 서버 기동: {start_time}")
    logger.info(sep)
    return logger


# =============================================================================
# 디렉터리 / 캡처 저장
# =============================================================================
def setup_dirs(out_dir: Path) -> dict:
    subdirs = {
        "intensity":            out_dir / "intensity",
        "pointcloud_organized": out_dir / "pointcloud_organized",
        "valid_mask":           out_dir / "valid_mask",
        "metadata":             out_dir / "metadata",
        "results":              out_dir / "results",
    }
    for p in subdirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return subdirs


def save_capture(frame, dirs: dict, idx: int, cfg_camera: dict) -> dict:
    dt_name         = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname_intensity = f"intensity_{dt_name}.png"
    fname_pcd       = f"pointcloud_{dt_name}.npy"
    fname_mask      = f"mask_{dt_name}.npy"
    fname_meta      = f"metadata_{dt_name}.json"

    if SAVE_INTENSITY:
        cv2.imwrite(str(dirs["intensity"] / fname_intensity), frame.intensity)
    if SAVE_POINTCLOUD:
        np.save(dirs["pointcloud_organized"] / fname_pcd,
                frame.points_organized.astype(np.float32))
    if SAVE_VALID_MASK:
        np.save(dirs["valid_mask"] / fname_mask,
                frame.valid_mask.astype(bool))

    pts       = frame.points
    valid_cnt = int(frame.valid_mask.sum())
    total     = frame.height * frame.width

    if pts.size > 0:
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        z_med = float(np.median(pts[:, 2]))
    else:
        z_min = z_max = z_med = float("nan")

    metadata = {
        "frame_index": idx,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
        "image":  {"width": int(frame.width), "height": int(frame.height)},
        "stats": {
            "valid_pixels": valid_cnt,
            "total_pixels": total,
            "valid_ratio":  round(100.0 * valid_cnt / total, 2),
            "z_min_mm":     round(z_min, 1) if not np.isnan(z_min) else None,
            "z_max_mm":     round(z_max, 1) if not np.isnan(z_max) else None,
            "z_median_mm":  round(z_med, 1) if not np.isnan(z_med) else None,
            "num_points":   int(len(pts)),
        },
        "files": {
            "intensity":            f"intensity/{fname_intensity}",
            "pointcloud_organized": f"pointcloud_organized/{fname_pcd}",
            "valid_mask":           f"valid_mask/{fname_mask}",
        },
    }
    if SAVE_METADATA:
        with (dirs["metadata"] / fname_meta).open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata
