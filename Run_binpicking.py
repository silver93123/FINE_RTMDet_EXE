"""TCP 서버 + 파이프라인 + main 모듈"""
from __future__ import annotations

import argparse
import socket
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.bp_settings import (
    ROOT, CONFIG_PATH, CHECKPOINT_PATH, SCORE_THRESHOLD,
    CAD_PATH, VOXEL_SIZE_CAD,
    TCP_HOST, TCP_PORT,
    ICP_ROLL_RANGE, ICP_PITCH_RANGE, ICP_YAW_RANGE,
)
from scripts.bp_logger import log, setup_file_logger, setup_dirs, save_capture
from scripts.bp_detection import run_detection, PICK_LOG_CSV_NAME
from scripts.bp_icp import load_cad_as_pcd, run_icp_for_frame
from scripts.bp_gui import (
    gui_loop, show_monitor, init_monitor,
    _gui_stop,
)

from src.camera import create_camera
from src.detection import RTMDetInferencer


# =============================================================================
# TCP 헬퍼
# =============================================================================
def format_response(payload: dict) -> str:
    status = payload.get("status")
    if status == "ok":
        picks = payload["picks"]
        parts = ["'ok'", str(len(picks))]
        for pk in picks:
            pp  = pk["position_mm"]
            deg = pk["approach_deg"]
            fit = round(pk["icp_fitness"], 2)
            tup = (round(pp[0], 3), round(pp[1], 3), round(pp[2], 3),
                   round(deg["roll_deg"], 3), round(deg["pitch_deg"], 3),
                   round(deg["yaw_deg"], 3), fit)
            parts.append(str(tup))
        return "{" + ", ".join(parts) + "}"
    elif status in ("no_object", "No"):
        return "{'No'}"
    else:
        msg = payload.get("message", "unknown error")
        return "{" + f"'error', '{msg}'" + "}"


def send_response(conn: socket.socket, payload: dict) -> None:
    conn.sendall((format_response(payload) + "\n").encode("utf-8"))


def recv_command(conn: socket.socket) -> str:
    buf = b""
    while b"\n" not in buf:
        chunk = conn.recv(1024)
        if not chunk:
            return ""
        buf += chunk
    return buf.decode("utf-8").strip()


# =============================================================================
# 한 프레임 처리
# =============================================================================
def process_one_frame(cam, dirs, frame_idx, cfg_camera,
                      inferencer, cad_pcd, cad_down) -> dict:
    _now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"\n{'─'*70}")
    log(f" [frame_{frame_idx:04d}] 캡처 중... {_now}")

    t0    = time.perf_counter()
    frame = cam.capture()
    dt_ms = (time.perf_counter() - t0) * 1000.0
    meta  = save_capture(frame, dirs, frame_idx, cfg_camera)
    s     = meta["stats"]
    print(f" Captured: {dt_ms:.1f} ms | valid {s['valid_ratio']:.1f}% | "
          f"Z {s['z_min_mm']}~{s['z_max_mm']} mm", flush=True)

    _cap_stat = {"capture_ms": dt_ms, "valid_ratio": s["valid_ratio"],
                 "z_min": s["z_min_mm"] or 0, "z_max": s["z_max_mm"] or 0}

    dt_name       = Path(meta["files"]["intensity"]).stem.replace("intensity_", "")
    frame_name    = f"result_{dt_name}"
    gray          = frame.intensity
    pcd_organized = frame.points_organized.astype(np.float32)
    valid_mask    = frame.valid_mask.astype(bool)

    log(" [Detection]")
    t0 = time.perf_counter()
    summary, inst_plys, bgr_image, bg_pcd = run_detection(
        frame_name, gray, pcd_organized, valid_mask,
        inferencer, dirs["results"], dirs["tmp"])
    det_ms = (time.perf_counter() - t0) * 1000.0
    n_det  = summary["num_detected"]
    n_pcd  = summary["num_with_pcd"]
    log(f" [RTMDet] 검출: {n_det}개  유효 PCD: {n_pcd}개  ({det_ms:.0f} ms)")

    if not inst_plys:
        log(f" 브라켓 없음 (RTMDet 검출: {n_det}개)")
        return {"status": "No",
                "_overlay": bgr_image, "_cap_stat": _cap_stat,
                "_info": {"status": "No detection",
                          "num_detected": n_det, "num_icp_ok": 0,
                          "det_ms": det_ms, "icp_ms": 0, "picks": []}}

    log(" [ICP]")
    t0 = time.perf_counter()
    icp_results, final_overlay = run_icp_for_frame(
        inst_plys, cad_pcd, cad_down, dirs["results"],
        frame_name, bgr_image,
        pcd_organized=pcd_organized, valid_mask=valid_mask,
        bg_pcd=bg_pcd,
        csv_path=dirs["results"].parent / PICK_LOG_CSV_NAME)
    icp_ms  = (time.perf_counter() - t0) * 1000.0
    success = [r for r in icp_results if "error" not in r]
    n_fail  = len(icp_results) - len(success)
    log(f" [ICP]  성공: {len(success)}개  실패: {n_fail}개  ({icp_ms:.0f} ms)")
    log(f"        RTMDet {n_det}개 → 유효PCD {n_pcd}개 → ICP성공 {len(success)}개")

    if not success:
        return {"status": "No",
                "_overlay": final_overlay, "_cap_stat": _cap_stat,
                "_info": {"status": "No detection",
                          "num_detected": n_det, "num_icp_ok": 0,
                          "det_ms": det_ms, "icp_ms": icp_ms, "picks": []}}

    picks = [{"position_mm":  r["pick_point"]["position_mm"],
               "approach_deg": r["pick_point"]["approach_deg"],
               "icp_fitness":  r["icp_fitness"],
               "height_point": r["height_point"]}
             for r in success]

    for i, pk in enumerate(picks):
        pp  = pk["position_mm"]
        deg = pk["approach_deg"]
        fit = pk["icp_fitness"]
        log(f" #{i} 위치: ({pp[0]:.1f}, {pp[1]:.1f}, {pp[2]:.1f}) mm fit={fit:.2f}"
            f" roll={deg['roll_deg']:.2f} pitch={deg['pitch_deg']:.2f}"
            f" yaw={deg['yaw_deg']:.2f}")

    return {"status": "ok", "picks": picks,
            "_overlay": final_overlay, "_cap_stat": _cap_stat,
            "_info": {"status": "Done",
                      "num_detected": n_det, "num_icp_ok": len(success),
                      "det_ms": det_ms, "icp_ms": icp_ms, "picks": picks}}


# =============================================================================
# main
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser(description="빈피킹 TCP 서버")
    p.add_argument("--config", type=Path, default=ROOT / "config" / "config.yaml")
    p.add_argument("--out",    type=Path, default=ROOT / "data" / "captures" / "live")
    p.add_argument("--warmup", type=int,  default=3)
    p.add_argument("--host",   type=str,  default=TCP_HOST)
    p.add_argument("--port",   type=int,  default=TCP_PORT)
    return p.parse_args()


def main():
    import scripts.bp_logger as _lg
    args = parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg_camera = cfg.get("camera", {})

    dirs          = setup_dirs(args.out)
    _lg._logger   = setup_file_logger(args.out)

    log(f"모델 로드: {CHECKPOINT_PATH.name}")
    inferencer = RTMDetInferencer(
        config=str(CONFIG_PATH),
        checkpoint=str(CHECKPOINT_PATH),
        score_threshold=SCORE_THRESHOLD,
    )

    log(f"CAD 로드: {CAD_PATH.name}")
    cad_pcd  = load_cad_as_pcd(CAD_PATH)
    cad_down = cad_pcd.voxel_down_sample(VOXEL_SIZE_CAD)
    log(f"CAD 포인트 수: {len(np.asarray(cad_pcd.points))} "
        f"(다운샘플: {len(np.asarray(cad_down.points))})")

    log("카메라 초기화 중...")
    cam = create_camera(cfg_camera)
    cam.open()
    log(f"카메라 IP: {getattr(cam, 'ip', 'N/A')}")

    log(f"워밍업 {args.warmup}프레임...")
    for _ in range(args.warmup):
        cam.capture()

    dummy = cam.capture()
    init_monitor(dummy.height, dummy.width)

    gui_thread = threading.Thread(target=gui_loop, name="gui", daemon=True)
    gui_thread.start()

    frame_idx = 0
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    log(f"\nTCP 서버 대기 중: {args.host}:{args.port}")
    log(f"회전 구속: roll {ICP_ROLL_RANGE}°  pitch {ICP_PITCH_RANGE}°  yaw {ICP_YAW_RANGE}°")

    try:
        while True:
            conn, addr = srv.accept()
            log(f"연결: {addr}")
            try:
                while True:
                    cmd = recv_command(conn)
                    if not cmd:
                        log("연결 종료 (클라이언트)")
                        break
                    log(f"수신: {cmd!r}")

                    if cmd.upper() == "QUIT":
                        log("QUIT 수신 → 서버 종료")
                        send_response(conn, {"status": "ok", "picks": []})
                        conn.close(); srv.close()
                        cv2.destroyAllWindows()
                        return

                    elif cmd.upper() in ("CAPTURE", "C"):
                        frame_idx += 1
                        _ts = datetime.now().strftime("%H:%M:%S")
                        show_monitor(
                            np.zeros((dummy.height, dummy.width, 3), dtype=np.uint8),
                            {"status": "Processing", "frame_idx": frame_idx,
                             "timestamp": _ts, "picks": []})
                        try:
                            payload = process_one_frame(
                                cam, dirs, frame_idx, cfg_camera,
                                inferencer, cad_pcd, cad_down)
                            send_response(conn, payload)
                            overlay = payload.get("_overlay")
                            cs      = payload.get("_cap_stat", {})
                            info    = payload.get("_info", {})
                            info.update({"frame_idx": frame_idx, "timestamp": _ts, **cs})
                            if overlay is not None:
                                show_monitor(overlay, info)
                        except Exception as e:
                            log(f"ERROR: {e}")
                            send_response(conn, {"status": "error", "message": str(e)})
                    else:
                        log(f"알 수 없는 명령: {cmd!r}")
                        send_response(conn, {"status": "error",
                                             "message": f"unknown command: {cmd}"})
            finally:
                conn.close()
    except KeyboardInterrupt:
        log("\nKeyboardInterrupt → 종료")
    finally:
        _gui_stop.set()
        gui_thread.join(timeout=2.0)
        srv.close()
        try:
            cam.close(); log("카메라 닫힘")
        except Exception:
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
