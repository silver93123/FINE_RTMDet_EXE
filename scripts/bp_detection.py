"""Detection / NMS / PCD 저장 / CSV 로깅 모듈"""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from scripts.bp_settings import (
    MIN_POINTS_PER_INSTANCE, MASK_IOU_THRESHOLD,
    PICK_Z_NEIGHBOR_OFFSETS,
    _PALETTE_BGR, _PALETTE_RGB_FLOAT, _BG_COLOR,
    SAVE_OVERLAY_PNG, SAVE_COLORED_PLY, SAVE_INSTANCE_PLY, SAVE_PICK_LOG_CSV,
)
from scripts.bp_logger import log

# =============================================================================
# Detection 오버레이
# =============================================================================
def overlay_results(image_bgr, results, valid_mask=None):
    overlay = image_bgr.copy()
    if valid_mask is not None:
        overlay[~valid_mask] = (overlay[~valid_mask] * 0.4).astype(np.uint8)
    for i, r in enumerate(results):
        color = _PALETTE_BGR[i % len(_PALETTE_BGR)]
        layer = np.zeros_like(overlay)
        layer[r.mask] = color
        overlay[r.mask] = (0.5 * overlay[r.mask] + 0.5 * layer[r.mask]).astype(np.uint8)
    for i, r in enumerate(results):
        color = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
        x1, y1, x2, y2 = r.bbox.astype(int)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        label = f"#{i} {r.class_name} {r.score:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay

# =============================================================================
# 픽셀 ↔ 3D 포인트 변환
# =============================================================================
def pick_to_pixel(pick_mm: list, pcd_organized: np.ndarray,
                  valid_mask: np.ndarray, fallback_xy: tuple):
    target = np.array(pick_mm, dtype=np.float32)
    H, W   = pcd_organized.shape[:2]
    vr, vc = np.where(valid_mask)
    if len(vr) == 0:
        return fallback_xy
    pts  = pcd_organized[vr, vc]
    dist = ((pts - target) ** 2).sum(axis=1)
    idx  = int(np.argmin(dist))
    if dist[idx] > 50.0 ** 2:
        return fallback_xy
    return int(vc[idx]), int(vr[idx])

def pixel_to_point(px: float, py: float, pcd_organized: np.ndarray,
                   valid_mask: np.ndarray):
    H, W = pcd_organized.shape[:2]
    ix = min(max(int(round(px)), 0), W - 1)
    iy = min(max(int(round(py)), 0), H - 1)
    if valid_mask[iy, ix]:
        return pcd_organized[iy, ix].astype(np.float64).copy()
    vr, vc = np.where(valid_mask)
    if len(vr) == 0:
        return None
    d2  = (vr - iy) ** 2 + (vc - ix) ** 2
    idx = int(np.argmin(d2))
    return pcd_organized[vr[idx], vc[idx]].astype(np.float64).copy()

def robust_z_from_neighbors(px: float, py: float, pcd_organized: np.ndarray,
                             valid_mask: np.ndarray):
    H, W = pcd_organized.shape[:2]
    ix = min(max(int(round(px)), 0), W - 1)
    iy = min(max(int(round(py)), 0), H - 1)
    zs = []
    for dx, dy in PICK_Z_NEIGHBOR_OFFSETS:
        nx, ny = ix + dx, iy + dy
        if 0 <= nx < W and 0 <= ny < H and valid_mask[ny, nx]:
            zs.append(float(pcd_organized[ny, nx, 2]))
    if not zs:
        return None, 0
    return float(np.median(zs)), len(zs)

# =============================================================================
# CSV 로깅
# =============================================================================
PICK_LOG_CSV_NAME = "pick_log.csv"

PICK_LOG_FIELDS = [
    "timestamp", "frame_name", "instance_id", "status", "error_msg",
    "det_score",
    "icp_fitness", "icp_rmse_m",
    "num_points_scene", "num_points_after_outlier_removal", "was_flipped",
    "pos_icp_x_mm", "pos_icp_y_mm", "pos_icp_z_mm",
    "pos_2d_x_mm",  "pos_2d_y_mm",  "pos_2d_z_mm",
    "pos_blend_x_mm", "pos_blend_y_mm", "pos_blend_z_mm",
    "z_robust_mm", "z_neighbor_count",
    "final_x_mm", "final_y_mm", "final_z_mm",
    "roll_deg", "pitch_deg", "yaw_deg",
    "height_x_mm", "height_y_mm", "height_z_mm",
    "height_z_neighbor_count", "height_z_is_fallback",
]

def append_pick_log_csv(csv_path: Path, row: dict) -> None:
    if not SAVE_PICK_LOG_CSV:
        return
    is_new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PICK_LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in PICK_LOG_FIELDS})

def build_pick_log_row(frame_name, inst_idx, det_score,
                       status, error_msg="",
                       icp_fitness=None, icp_rmse_m=None,
                       n_pts=None, n_after=None, flipped=None,
                       pick=None, height_point=None) -> dict:
    row = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "frame_name":  frame_name,
        "instance_id": inst_idx,
        "status":      status,
        "error_msg":   error_msg,
        "det_score":   None if det_score is None else round(det_score, 4),
        "icp_fitness": None if icp_fitness is None else round(icp_fitness, 4),
        "icp_rmse_m":  None if icp_rmse_m is None else round(icp_rmse_m, 6),
        "num_points_scene":                 n_pts,
        "num_points_after_outlier_removal": n_after,
        "was_flipped": flipped,
    }
    if pick is not None:
        icp_mm   = pick["_pos_icp_mm"]
        pos2d_mm = pick["_pos_2d_mm"]
        blend_mm = pick["_pos_blend_mm"]
        final_mm = pick["position_mm"]
        deg      = pick["approach_deg"]
        row.update({
            "pos_icp_x_mm":   round(icp_mm[0], 2),
            "pos_icp_y_mm":   round(icp_mm[1], 2),
            "pos_icp_z_mm":   round(icp_mm[2], 2),
            "pos_2d_x_mm":    "" if pos2d_mm is None else round(pos2d_mm[0], 2),
            "pos_2d_y_mm":    "" if pos2d_mm is None else round(pos2d_mm[1], 2),
            "pos_2d_z_mm":    "" if pos2d_mm is None else round(pos2d_mm[2], 2),
            "pos_blend_x_mm": round(blend_mm[0], 2),
            "pos_blend_y_mm": round(blend_mm[1], 2),
            "pos_blend_z_mm": round(blend_mm[2], 2),
            "z_robust_mm":      "" if pick["_z_robust_mm"] is None else round(pick["_z_robust_mm"], 2),
            "z_neighbor_count": pick["_z_neighbor_count"],
            "final_x_mm": round(final_mm[0], 2),
            "final_y_mm": round(final_mm[1], 2),
            "final_z_mm": round(final_mm[2], 2),
            "roll_deg":   deg["roll_deg"],
            "pitch_deg":  deg["pitch_deg"],
            "yaw_deg":    deg["yaw_deg"],
        })
    if height_point is not None:
        hp = height_point["position_mm"]
        row.update({
            "height_x_mm":             hp[0],
            "height_y_mm":             hp[1],
            "height_z_mm":             hp[2],
            "height_z_neighbor_count": height_point["z_neighbor_count"],
            "height_z_is_fallback":    height_point["z_is_fallback"],
        })
    return row

# =============================================================================
# 오버레이 픽포인트 표시
# =============================================================================
def draw_picks_on_overlay(image_bgr: np.ndarray, picks_2d: list) -> np.ndarray:
    out  = image_bgr.copy()
    H, W = out.shape[:2]
    for i, (px, py, pick, icp_fitness, bbox, px_h, py_h) in enumerate(picks_2d):
        color        = tuple(int(c) for c in _PALETTE_BGR[i % len(_PALETTE_BGR)])
        pp           = pick["position_mm"]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(px), int(py)), color,
                       cv2.MARKER_CROSS, 24, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (int(px_h), int(py_h)), color,
                       cv2.MARKER_CROSS, 12, 1, cv2.LINE_AA)
        line1 = f"#{i} ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm"
        line2 = f"ICP fit: {icp_fitness:.3f}"
        font, font_scale, thickness, line_gap = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1, 4
        (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
        (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
        box_w = max(w1, w2) + 8
        box_h = h1 + h2 + line_gap + 8
        tx = max(x1, 0)
        ty = y1 - box_h - 4
        if ty < 0:
            ty = y2 + 4
        ty = min(ty, H - box_h - 2)
        tx = min(tx, W - box_w - 2)
        cv2.rectangle(out, (tx - 2, ty), (tx + box_w, ty + box_h), (0, 0, 0), -1)
        cv2.putText(out, line1, (tx + 2, ty + h1 + 2),
                    font, font_scale, color, thickness, cv2.LINE_AA)
        cv2.putText(out, line2, (tx + 2, ty + h1 + h2 + line_gap + 4),
                    font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)
    return out

# =============================================================================
# 마스크 NMS
# =============================================================================
def mask_nms(results, iou_threshold: float = MASK_IOU_THRESHOLD):
    keep, removed = [], []
    suppressed    = [False] * len(results)
    for i, ri in enumerate(results):
        if suppressed[i]:
            continue
        keep.append(ri)
        area_i = ri.mask.sum()
        if area_i == 0:
            continue
        for j in range(i + 1, len(results)):
            if suppressed[j]:
                continue
            rj    = results[j]
            inter = (ri.mask & rj.mask).sum()
            if inter == 0:
                continue
            area_j = rj.mask.sum()
            union  = area_i + area_j - inter
            iou    = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                suppressed[j] = True
                removed.append((rj, ri, float(iou)))
    return keep, removed

# =============================================================================
# PCD 저장
# =============================================================================
def save_instance_pcd(points, out_path, color):
    if points.size == 0:
        return False
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(
        np.tile(np.array(color, dtype=np.float64), (len(points), 1)))
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))

def save_colored_full_pcd(pcd_organized, valid_mask, results, out_path):
    all_pts = pcd_organized[valid_mask]
    if len(all_pts) == 0:
        return False
    colors = np.tile(_BG_COLOR, (len(all_pts), 1))
    H, W   = valid_mask.shape
    lookup = np.full((H, W), -1, dtype=np.int32)
    vr, vc = np.where(valid_mask)
    lookup[vr, vc] = np.arange(len(vr))
    for i, r in enumerate(results):
        ir, ic = np.where(r.mask & valid_mask)
        if len(ir):
            colors[lookup[ir, ic]] = _PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(all_pts / 1000.0)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return bool(o3d.io.write_point_cloud(str(out_path), pcd, write_ascii=False))

# =============================================================================
# Detection 실행
# =============================================================================
def run_detection(frame_name, gray, pcd_organized, valid_mask, inferencer, result_dir, tmp_dir):
    bgr = np.stack([gray, gray, gray], axis=-1)

    results, nms_removed = mask_nms(inferencer.infer(bgr))
    for rem, winner, iou in nms_removed:
        print(f" [NMS] score={rem.score:.2f} 제거 "
              f"(IoU={iou:.2f}, winner score={winner.score:.2f})", flush=True)

    if SAVE_OVERLAY_PNG:
        cv2.imwrite(str(result_dir / f"{frame_name}_overlay.png"),
                    overlay_results(bgr, results, valid_mask))
    if SAVE_COLORED_PLY:
        save_colored_full_pcd(pcd_organized, valid_mask, results,
                              result_dir / f"{frame_name}_colored.ply")

    instances_info, instance_plys = [], []
    for i, r in enumerate(results):
        combined = r.mask & valid_mask
        obj_pts  = pcd_organized[combined]
        if len(obj_pts) < MIN_POINTS_PER_INSTANCE:
            instances_info.append({
                "instance_id": i, "class": r.class_name,
                "score": float(r.score),
                "skipped": "점이 너무 적음", "num_points": int(len(obj_pts)),
            })
            continue
        color_rgb = tuple(_PALETTE_RGB_FLOAT[i % len(_PALETTE_RGB_FLOAT)].tolist())
        # tmp_dir 에 저장 → 매 프레임 덮어씀, ICP 처리 후 자동 삭제
        ply_path  = tmp_dir / f"obj{i}.ply"
        ok        = save_instance_pcd(obj_pts, ply_path, color=color_rgb)

        center    = obj_pts.mean(axis=0)
        size      = obj_pts.max(axis=0) - obj_pts.min(axis=0)
        cx_2d     = float((r.bbox[0] + r.bbox[2]) / 2)
        cy_2d     = float((r.bbox[1] + r.bbox[3]) / 2)
        instances_info.append({
            "instance_id": i, "class": r.class_name,
            "score": float(r.score),
            "num_points_3d": int(len(obj_pts)),
            "center_mm": center.tolist(), "size_mm": size.tolist(),
            "bbox_center_2d": [cx_2d, cy_2d],
        })
        if ok:
            instance_plys.append((ply_path, cx_2d, cy_2d, r.bbox, float(r.score)))

    instance_mask_union = np.zeros(valid_mask.shape, dtype=bool)
    for r in results:
        instance_mask_union |= (r.mask & valid_mask)
    bg_pts = pcd_organized[valid_mask & ~instance_mask_union]
    bg_pcd = o3d.geometry.PointCloud()
    if len(bg_pts) > 0:
        bg_pcd.points = o3d.utility.Vector3dVector(bg_pts / 1000.0)
        bg_pcd.colors = o3d.utility.Vector3dVector(
            np.tile([0.55, 0.55, 0.55], (len(bg_pts), 1)))

    return ({"frame": frame_name,
             "num_detected": len(results),
             "num_with_pcd": len(instance_plys),
             "instances":    instances_info},
            instance_plys, bgr, bg_pcd)
