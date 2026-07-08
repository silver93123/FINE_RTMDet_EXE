"""ICP 포즈 추정 모듈"""
from __future__ import annotations

import copy
import json

import cv2
import numpy as np
import open3d as o3d

from scripts.bp_settings import (
    CAD_AXIS_CORRECTION_DEG, CAD_SAMPLE_POINTS,
    CAD_PICK_LOCAL,
    VOXEL_SIZE_CAD, VOXEL_SIZE_SCENE,
    OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO,
    ICP_STAGES, ICP_FITNESS_THRESHOLD, XYZ_MAX_M,
    ICP_INIT_ROLL_DEG, ICP_INIT_PITCH_DEG, ICP_INIT_YAW_DEG,
    ICP_ROLL_RANGE, ICP_PITCH_RANGE, ICP_YAW_RANGE,
    PICK_OFFSET_X_MM, PICK_OFFSET_Y_MM, PICK_OFFSET_Z_MM,
    PICK_2D_MASK_WEIGHT, HEIGHT_POINT_OFFSET_Y_MM,
    _PALETTE_RGB_FLOAT,
    SAVE_COLORED_PLY, SAVE_RESULT_JSON, SAVE_OVERLAY_PNG,
)
from scripts.bp_logger import log
from scripts.bp_detection import (
    pick_to_pixel, pixel_to_point, robust_z_from_neighbors,
    draw_picks_on_overlay,
    append_pick_log_csv, build_pick_log_row, PICK_LOG_CSV_NAME,
)


# =============================================================================
# 회전 행렬 헬퍼
# =============================================================================
def _Rx(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[1, 1] = c; R[1, 2] = -s; R[2, 1] = s; R[2, 2] = c
    return R

def _Ry(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[0, 0] = c; R[0, 2] = s; R[2, 0] = -s; R[2, 2] = c
    return R

def _Rz(d):
    c, s = np.cos(np.radians(d)), np.sin(np.radians(d))
    R = np.eye(3); R[0, 0] = c; R[0, 1] = -s; R[1, 0] = s; R[1, 1] = c
    return R


# =============================================================================
# CAD 로드
# =============================================================================
def load_cad_as_pcd(cad_path):
    mesh = o3d.io.read_triangle_mesh(str(cad_path))
    ext  = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent())
    if ext.max() > 10.0:
        mesh.scale(1.0 / 1000.0, center=np.zeros(3))
    rx, ry, rz = CAD_AXIS_CORRECTION_DEG
    R      = _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    center = np.asarray(mesh.get_center())
    T_fix  = np.eye(4); T_fix[:3, :3] = R; T_fix[:3, 3] = center - R @ center
    mesh.transform(T_fix)
    return mesh.sample_points_poisson_disk(CAD_SAMPLE_POINTS)


# =============================================================================
# ICP 초기값 + 구속 검사
# =============================================================================
def build_icp_init(scene_down, cad_down) -> np.ndarray:
    R_init    = _Rz(ICP_INIT_YAW_DEG) @ _Ry(ICP_INIT_PITCH_DEG) @ _Rx(ICP_INIT_ROLL_DEG)
    sc_center = np.asarray(scene_down.get_center())
    cd_center = np.asarray(cad_down.get_center())
    T_init    = np.eye(4)
    T_init[:3, :3] = R_init
    T_init[:3, 3]  = sc_center - R_init @ cd_center
    return T_init


def check_rotation_constraint(T: np.ndarray):
    R     = T[:3, :3]
    pitch = np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2)))
    cp    = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp))
        yaw  = np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp))
    else:
        roll, yaw = 0.0, np.degrees(np.arctan2(-R[0, 1], R[1, 1]))

    violations = []
    if not (ICP_ROLL_RANGE[0]  <= roll  <= ICP_ROLL_RANGE[1]):
        violations.append(f"roll={roll:.1f}° (허용 [{ICP_ROLL_RANGE[0]}, {ICP_ROLL_RANGE[1]}])")
    if not (ICP_PITCH_RANGE[0] <= pitch <= ICP_PITCH_RANGE[1]):
        violations.append(f"pitch={pitch:.1f}° (허용 [{ICP_PITCH_RANGE[0]}, {ICP_PITCH_RANGE[1]}])")
    if not (ICP_YAW_RANGE[0]   <= yaw   <= ICP_YAW_RANGE[1]):
        violations.append(f"yaw={yaw:.1f}° (허용 [{ICP_YAW_RANGE[0]}, {ICP_YAW_RANGE[1]}])")

    if violations:
        return False, "회전 구속 위반: " + ", ".join(violations)
    return True, f"roll={roll:.1f}° pitch={pitch:.1f}° yaw={yaw:.1f}°"


# =============================================================================
# ICP 실행
# =============================================================================
def run_icp_multistage(src, tgt, T_init):
    T = T_init.copy()
    for stage in ICP_STAGES:
        res = o3d.pipelines.registration.registration_icp(
            src, tgt, stage["max_dist"], T,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=stage["max_iter"]),
        )
        T = np.asarray(res.transformation)
    final = o3d.pipelines.registration.evaluate_registration(
        src, tgt, ICP_STAGES[-1]["max_dist"], T)
    return T, float(final.fitness), float(final.inlier_rmse)


def correct_flipped_pose(T, src, tgt):
    if T[:3, :3][2, 2] >= 0:
        final = o3d.pipelines.registration.evaluate_registration(
            src, tgt, ICP_STAGES[-1]["max_dist"], T)
        return T, float(final.fitness), float(final.inlier_rmse), False
    R_flip = np.diag([-1.0, -1.0, 1.0])
    T_flip = np.eye(4); T_flip[:3, :3] = R_flip
    c = T[:3, 3]; T_flip[:3, 3] = c - R_flip @ c
    T_f, fit, rmse = run_icp_multistage(src, tgt, T_flip @ T)
    return T_f, fit, rmse, True


def transform_to_pose(T):
    xyz_mm = (T[:3, 3] * 1000.0).tolist()
    R      = T[:3, :3]
    pitch  = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    cp     = np.cos(pitch)
    if abs(cp) > 1e-6:
        roll = np.arctan2(R[2, 1] / cp, R[2, 2] / cp)
        yaw  = np.arctan2(R[1, 0] / cp, R[0, 0] / cp)
    else:
        roll, yaw = 0.0, np.arctan2(-R[0, 1], R[1, 1])
    e = np.degrees([roll, pitch, yaw]).tolist()
    return {
        "xyz_mm":    [round(v, 3) for v in xyz_mm],
        "euler_deg": {"roll_deg":  round(e[0], 4),
                      "pitch_deg": round(e[1], 4),
                      "yaw_deg":   round(e[2], 4)},
        "transform_matrix": T.tolist(),
    }


# =============================================================================
# 픽포인트 계산
# =============================================================================
def compute_pick_point(T, pcd_organized=None, valid_mask=None,
                       cx_2d=None, cy_2d=None):
    pl_icp     = CAD_PICK_LOCAL.copy()
    pos_icp_mm = (T @ pl_icp)[:3] * 1000.0

    pt2d_mm = None
    if pcd_organized is not None and valid_mask is not None \
            and cx_2d is not None and cy_2d is not None:
        pt2d_mm = pixel_to_point(cx_2d, cy_2d, pcd_organized, valid_mask)

    w = PICK_2D_MASK_WEIGHT
    pos_blend_mm = pos_icp_mm * (1.0 - w) + pt2d_mm * w \
                   if pt2d_mm is not None else pos_icp_mm

    z_robust_mm, z_neighbor_count = None, 0
    if pcd_organized is not None and valid_mask is not None:
        fallback_px = (cx_2d, cy_2d) if cx_2d is not None else (0, 0)
        px_blend, py_blend = pick_to_pixel(
            pos_blend_mm.tolist(), pcd_organized, valid_mask, fallback_px)
        z_robust_mm, z_neighbor_count = robust_z_from_neighbors(
            px_blend, py_blend, pcd_organized, valid_mask)

    if z_robust_mm is not None:
        pos_blend_mm = np.array([pos_blend_mm[0], pos_blend_mm[1], z_robust_mm])

    pos_final_mm = pos_blend_mm + np.array(
        [PICK_OFFSET_X_MM, PICK_OFFSET_Y_MM, PICK_OFFSET_Z_MM])

    R     = T[:3, :3]
    pitch = float(np.degrees(np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))))
    cp    = np.cos(np.radians(pitch))
    if abs(cp) > 1e-6:
        roll = float(np.degrees(np.arctan2(R[2, 1] / cp, R[2, 2] / cp)))
        yaw  = float(np.degrees(np.arctan2(R[1, 0] / cp, R[0, 0] / cp)))
    else:
        roll, yaw = 0.0, float(np.degrees(np.arctan2(-R[0, 1], R[1, 1])))

    return {
        "position_mm":  [round(v, 3) for v in pos_final_mm.tolist()],
        "approach_deg": {"roll_deg":  round(roll,  4),
                         "pitch_deg": round(pitch, 4),
                         "yaw_deg":   round(yaw,   4)},
        "_pos_icp_mm":       pos_icp_mm.tolist(),
        "_pos_2d_mm":        None if pt2d_mm is None else pt2d_mm.tolist(),
        "_pos_blend_mm":     pos_blend_mm.tolist(),
        "_z_robust_mm":      z_robust_mm,
        "_z_neighbor_count": z_neighbor_count,
    }


def compute_height_point(pick_position_mm, pcd_organized=None,
                         valid_mask=None, fallback_xy=(0, 0)):
    target_x = pick_position_mm[0]
    target_y = pick_position_mm[1] + HEIGHT_POINT_OFFSET_Y_MM
    approx_z = pick_position_mm[2]

    z_robust_mm, z_neighbor_count = None, 0
    if pcd_organized is not None and valid_mask is not None:
        px_h, py_h = pick_to_pixel(
            [target_x, target_y, approx_z], pcd_organized, valid_mask, fallback_xy)
        z_robust_mm, z_neighbor_count = robust_z_from_neighbors(
            px_h, py_h, pcd_organized, valid_mask)

    final_z = approx_z if z_robust_mm is None else z_robust_mm
    return {
        "position_mm":      [round(target_x, 3), round(target_y, 3), round(final_z, 3)],
        "z_neighbor_count": z_neighbor_count,
        "z_is_fallback":    z_robust_mm is None,
    }


def build_icp_elements(scene_pcd, cad_pcd, T, pick, inst_color):
    sv = copy.deepcopy(scene_pcd)
    sv.colors = o3d.utility.Vector3dVector(
        np.tile(inst_color, (len(np.asarray(sv.points)), 1)))
    cv = copy.deepcopy(cad_pcd); cv.transform(T)
    cv.colors = o3d.utility.Vector3dVector(
        np.tile([0.1, 0.9, 0.3], (len(np.asarray(cv.points)), 1)))
    pm  = np.array(pick["position_mm"]) / 1000.0
    sp  = o3d.geometry.TriangleMesh.create_sphere(radius=0.005)
    sp.translate(pm); sp.paint_uniform_color([1.0, 0.1, 0.1])
    sp_pcd = sp.sample_points_uniformly(500)
    deg    = pick["approach_deg"]
    cr, sr   = np.cos(np.radians(deg["roll_deg"])),  np.sin(np.radians(deg["roll_deg"]))
    cp_, sp_ = np.cos(np.radians(deg["pitch_deg"])), np.sin(np.radians(deg["pitch_deg"]))
    cy, sy   = np.cos(np.radians(deg["yaw_deg"])),   np.sin(np.radians(deg["yaw_deg"]))
    app = np.array([cr * sy * sp_ + sr * cy,
                    sr * sy - cr * cy * sp_,
                    cr * cp_])
    app    = app / (np.linalg.norm(app) + 1e-9)
    ap     = np.array([pm + t * app * 0.03 for t in np.linspace(0, 1, 50)])
    ap_pcd = o3d.geometry.PointCloud()
    ap_pcd.points = o3d.utility.Vector3dVector(ap)
    ap_pcd.colors = o3d.utility.Vector3dVector(np.tile([0.1, 0.3, 1.0], (50, 1)))
    return sv + cv + sp_pcd + ap_pcd


# =============================================================================
# ICP 프레임 처리
# =============================================================================
def run_icp_for_frame(instance_plys, cad_pcd, cad_down,
                      result_dir, frame_name, bgr_image,
                      pcd_organized=None, valid_mask=None, bg_pcd=None,
                      csv_path=None):
    icp_results  = []
    picks_2d     = []
    combined_pcd = bg_pcd if bg_pcd is not None else o3d.geometry.PointCloud()

    for ply_path, cx_2d, cy_2d, bbox, det_score in instance_plys:
        inst_idx  = int(ply_path.stem.split("obj")[-1])
        scene_pcd = o3d.io.read_point_cloud(str(ply_path))
        n_pts     = len(np.asarray(scene_pcd.points))

        if n_pts < 50:
            icp_results.append({"instance_id": inst_idx, "error": f"포인트 부족: {n_pts}개"})
            if csv_path is not None:
                append_pick_log_csv(csv_path, build_pick_log_row(
                    frame_name, inst_idx, det_score, status="error",
                    error_msg=f"포인트 부족: {n_pts}개", n_pts=n_pts))
            continue

        log(f"   obj{inst_idx}: {n_pts} pts")
        sc, _   = scene_pcd.remove_statistical_outlier(OUTLIER_NB_NEIGHBORS, OUTLIER_STD_RATIO)
        n_after = len(np.asarray(sc.points))
        sd      = sc.voxel_down_sample(VOXEL_SIZE_SCENE)

        T_init = build_icp_init(sd, cad_down)
        log(f"   T_init roll={ICP_INIT_ROLL_DEG}° pitch={ICP_INIT_PITCH_DEG}° "
            f"yaw={ICP_INIT_YAW_DEG}° "
            f"t=[{T_init[0,3]:.4f}, {T_init[1,3]:.4f}, {T_init[2,3]:.4f}]")

        T, fit, rmse = run_icp_multistage(cad_down, sd, T_init)
        T, fit, rmse, flipped = correct_flipped_pose(T, cad_down, sd)
        if flipped:
            log(f"   △ 뒤집힘 보정 후 fitness={fit:.4f}")

        if fit < ICP_FITNESS_THRESHOLD:
            log(f"   ✗ ICP 실패 (fitness={fit:.4f})")
            icp_results.append({"instance_id": inst_idx, "error": "ICP 정합 실패", "icp_fitness": float(fit)})
            if csv_path is not None:
                append_pick_log_csv(csv_path, build_pick_log_row(
                    frame_name, inst_idx, det_score, status="error",
                    error_msg="ICP 정합 실패", icp_fitness=fit,
                    n_pts=n_pts, n_after=n_after, flipped=flipped))
            ply_path.unlink(missing_ok=True); continue

        if max(abs(v) for v in T[:3, 3]) > XYZ_MAX_M:
            icp_results.append({"instance_id": inst_idx, "error": "xyz 범위 이상", "icp_fitness": float(fit)})
            if csv_path is not None:
                append_pick_log_csv(csv_path, build_pick_log_row(
                    frame_name, inst_idx, det_score, status="error",
                    error_msg="xyz 범위 이상", icp_fitness=fit,
                    n_pts=n_pts, n_after=n_after, flipped=flipped))
            ply_path.unlink(missing_ok=True); continue

        rot_ok, rot_msg = check_rotation_constraint(T)
        if not rot_ok:
            log(f"   ✗ {rot_msg} → 기각")
            icp_results.append({"instance_id": inst_idx, "error": rot_msg, "icp_fitness": float(fit)})
            if csv_path is not None:
                append_pick_log_csv(csv_path, build_pick_log_row(
                    frame_name, inst_idx, det_score, status="error",
                    error_msg=rot_msg, icp_fitness=fit,
                    n_pts=n_pts, n_after=n_after, flipped=flipped))
            ply_path.unlink(missing_ok=True); continue
        log(f"   ✓ 회전 OK: {rot_msg}")

        pose         = transform_to_pose(T)
        pick         = compute_pick_point(T, pcd_organized, valid_mask, cx_2d, cy_2d)
        ppos         = pick["position_mm"]
        deg          = pick["approach_deg"]
        height_point = compute_height_point(ppos, pcd_organized, valid_mask,
                                            fallback_xy=(int(cx_2d), int(cy_2d)))
        inst_color   = _PALETTE_RGB_FLOAT[inst_idx % len(_PALETTE_RGB_FLOAT)].tolist()
        combined_pcd += build_icp_elements(scene_pcd, cad_pcd, T, pick, inst_color)

        if pcd_organized is not None and valid_mask is not None:
            px_2d, py_2d = pick_to_pixel(pick["position_mm"], pcd_organized, valid_mask,
                                         fallback_xy=(int(cx_2d), int(cy_2d)))
        else:
            px_2d, py_2d = int(cx_2d), int(cy_2d)

        w     = PICK_2D_MASK_WEIGHT
        z_dbg = ("Z없음(폴백)" if pick["_z_robust_mm"] is None
                 else f"Z={pick['_z_robust_mm']:.1f}(n={pick['_z_neighbor_count']})")
        log(f"   픽포인트: ICP{tuple(round(v,1) for v in pick['_pos_icp_mm'])}"
            f" x{1-w:.0%} + 2D{None if pick['_pos_2d_mm'] is None else tuple(round(v,1) for v in pick['_pos_2d_mm'])} x{w:.0%}"
            f" = XY블렌딩 + 이웃median{z_dbg}"
            f" = 블렌딩{tuple(round(v,1) for v in pick['_pos_blend_mm'])}"
            f" + offset({PICK_OFFSET_X_MM},{PICK_OFFSET_Y_MM},{PICK_OFFSET_Z_MM})"
            f" = 최종{tuple(ppos)}  →  px,py=({px_2d},{py_2d})")
        hz_dbg = ("Z없음(픽포인트Z 폴백)" if height_point["z_is_fallback"]
                  else f"n={height_point['z_neighbor_count']}")
        log(f"   높이포인트: 픽포인트 Y+{HEIGHT_POINT_OFFSET_Y_MM}mm"
            f" = {tuple(height_point['position_mm'])}  (이웃median {hz_dbg})")

        if pcd_organized is not None and valid_mask is not None:
            px_h, py_h = pick_to_pixel(height_point["position_mm"], pcd_organized, valid_mask,
                                       fallback_xy=(px_2d, py_2d))
        else:
            px_h, py_h = px_2d, py_2d

        picks_2d.append((px_2d, py_2d, pick, float(fit), bbox, px_h, py_h))

        result = {
            "instance_id":  inst_idx,
            "icp_fitness":  float(fit),
            "icp_rmse_m":   float(rmse),
            "was_flipped":  flipped,
            "num_points_scene":                 n_pts,
            "num_points_after_outlier_removal": n_after,
            "pose":         pose,
            "pick_point":   pick,
            "height_point": height_point,
        }
        if csv_path is not None:
            append_pick_log_csv(csv_path, build_pick_log_row(
                frame_name, inst_idx, det_score, status="ok",
                icp_fitness=fit, icp_rmse_m=rmse,
                n_pts=n_pts, n_after=n_after, flipped=flipped,
                pick=pick, height_point=height_point))
        print(f"   ✓ 픽포인트: ({ppos[0]:.1f}, {ppos[1]:.1f}, {ppos[2]:.1f}) mm "
              f"fit={fit:.3f} roll={deg['roll_deg']:.2f} "
              f"pitch={deg['pitch_deg']:.2f} yaw={deg['yaw_deg']:.2f}", flush=True)
        icp_results.append(result)
        ply_path.unlink(missing_ok=True)

    if SAVE_COLORED_PLY and len(np.asarray(combined_pcd.points)) > 0:
        ply_out = result_dir / f"{frame_name}_colored.ply"
        o3d.io.write_point_cloud(str(ply_out), combined_pcd, write_ascii=False)
        log(f"   ✓ 통합 PLY: {ply_out.name}")

    if SAVE_RESULT_JSON:
        success  = [r for r in icp_results if "error" not in r]
        json_out = result_dir / f"{frame_name}_result.json"
        with json_out.open("w", encoding="utf-8") as f:
            json.dump({"frame": frame_name,
                       "num_total":   len(icp_results),
                       "num_success": len(success),
                       "instances":   icp_results},
                      f, indent=2, ensure_ascii=False)
        log(f"   ✓ 통합 JSON: {json_out.name}")

    overlay_final = draw_picks_on_overlay(bgr_image, picks_2d) if picks_2d \
                    else bgr_image.copy()
    if SAVE_OVERLAY_PNG:
        cv2.imwrite(str(result_dir / f"{frame_name}_overlay.png"), overlay_final)
        log(f"   ✓ overlay PNG: {frame_name}_overlay.png")

    return icp_results, overlay_final
