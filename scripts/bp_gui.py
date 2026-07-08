"""실시간 GUI 모니터 모듈"""
from __future__ import annotations

import queue
import threading

import cv2
import numpy as np

from scripts.bp_settings import _PALETTE_BGR

# =============================================================================
# GUI 스레드
# =============================================================================
_WIN       = "BinPicking Monitor"
_gui_queue: queue.Queue = queue.Queue(maxsize=2)
_gui_stop  = threading.Event()


def _put(img, text, x, y, color=(220, 220, 220), scale=0.46, thickness=1):
    cv2.putText(img, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def build_info_panel(h: int, w: int, info: dict) -> np.ndarray:
    MIN_PANEL_H = 680
    ph    = max(h, MIN_PANEL_H)
    panel = np.full((ph, w, 3), 28, dtype=np.uint8)
    x0, lh = 12, 20

    def sep(y, c=(55, 55, 55)):
        cv2.line(panel, (x0, y), (w - x0, y), c, 1)

    def section(text, y, color=(100, 200, 255)):
        _put(panel, text, x0, y, color=color, scale=0.47)

    def row(text, y, color=(190, 190, 190)):
        _put(panel, text, x0 + 6, y, color=color, scale=0.41)

    y = 19
    _put(panel, "BinPicking Monitor", x0, y, color=(255, 200, 60), scale=0.53, thickness=1)
    y += lh + 2; sep(y); y += lh

    status  = info.get("status", "Waiting")
    s_color = {"Waiting": (150, 150, 150), "Processing": (60, 200, 255),
               "Done": (60, 220, 60), "No detection": (60, 140, 255),
               "Error": (60, 60, 220)}.get(status, (200, 200, 200))

    section("[ Server Status ]", y); y += lh
    row(f"Status : {status}", y, color=s_color);         y += lh
    row(f"Frame # : {info.get('frame_idx', '-')}", y);   y += lh
    row(f"Time : {info.get('timestamp', '-')}", y);      y += lh + 2
    sep(y); y += lh

    section("[ Capture Info ]", y); y += lh
    row(f"Capture : {info.get('capture_ms', 0):.0f} ms", y); y += lh
    row(f"Valid : {info.get('valid_ratio', 0):.1f} %",   y); y += lh
    row(f"Z range : {info.get('z_min', 0):.0f} ~ {info.get('z_max', 0):.0f} mm", y)
    y += lh + 2; sep(y); y += lh

    section("[ Timing ]", y); y += lh
    row(f"Detection: {info.get('det_ms', 0):.0f} ms", y); y += lh
    row(f"ICP : {info.get('icp_ms', 0):.0f} ms",     y); y += lh
    total = info.get("capture_ms", 0) + info.get("det_ms", 0) + info.get("icp_ms", 0)
    row(f"Total : {total:.0f} ms", y, color=(255, 200, 60)); y += lh + 2
    sep(y); y += lh

    picks      = info.get("picks", [])
    n_det_info = info.get("num_detected", "-")
    n_icp_info = info.get("num_icp_ok", len(picks))
    section(f"[ Results : ICP {n_icp_info} obj(s) ]", y); y += lh
    det_color = (60, 220, 60) if n_det_info != "-" and int(str(n_det_info)) > 0 \
                else (120, 120, 120)
    row(f" RTMDet : {n_det_info} detected", y, color=det_color); y += lh - 2
    row(f" ICP OK : {n_icp_info} / {n_det_info}", y,
        color=(60, 220, 60) if n_icp_info and int(str(n_icp_info)) > 0
              else (100, 100, 220)); y += lh
    sep(y, c=(45, 45, 45)); y += lh - 4
    if not picks:
        row(" No objects detected", y, color=(100, 100, 220))
    else:
        for i, pk in enumerate(picks):
            pp  = pk["position_mm"]
            deg = pk["approach_deg"]
            fit = pk["icp_fitness"]
            ci  = tuple(int(v) for v in _PALETTE_BGR[i % len(_PALETTE_BGR)])
            sep(y, c=(45, 45, 45)); y += lh - 4
            _put(panel, f" Obj #{i}", x0, y, color=ci, scale=0.43)
            y += lh - 2
            row(f" X= {pp[0]:+8.2f} Y= {pp[1]:+8.2f}", y);    y += lh - 3
            z_text = f" Z= {pp[2]:+8.2f} mm"
            row(z_text, y)
            hp_dict = pk.get("height_point")
            if hp_dict is not None:
                (tw, _), _ = cv2.getTextSize(z_text, cv2.FONT_HERSHEY_SIMPLEX, 0.41, 1)
                hz = hp_dict["position_mm"][2]
                _put(panel, f" (H:{hz:+.2f})", x0 + 6 + tw, y,
                     color=(0, 255, 255), scale=0.41)
            y += lh - 3
            row(f" R= {deg['roll_deg']:+7.2f} P= {deg['pitch_deg']:+7.2f}", y); y += lh - 3
            row(f" Yaw= {deg['yaw_deg']:+7.2f} deg", y);        y += lh - 3
            fc = (60, 220, 60) if fit >= 0.7 else (60, 140, 255) if fit >= 0.5 else (60, 60, 220)
            row(f" ICP fit : {fit:.3f}", y, color=fc);          y += lh

    sep(ph - 22)
    _put(panel, "QUIT=exit  ESC=close window",
         x0, ph - 8, color=(80, 80, 80), scale=0.37)
    return panel


def _render_frame(overlay_bgr: np.ndarray, info: dict):
    PANEL_W = 310
    SCALE   = 1.5
    oh, ow  = overlay_bgr.shape[:2]
    overlay_bgr = cv2.resize(overlay_bgr,
                              (int(ow * SCALE), int(oh * SCALE)),
                              interpolation=cv2.INTER_LINEAR)
    img_h = overlay_bgr.shape[0]
    panel = build_info_panel(img_h, PANEL_W, info)
    ph    = panel.shape[0]
    if img_h < ph:
        pad = np.zeros((ph - img_h, overlay_bgr.shape[1], 3), dtype=np.uint8)
        overlay_bgr = np.vstack([overlay_bgr, pad])
    divider = np.full((ph, 2, 3), 70, dtype=np.uint8)
    cv2.imshow(_WIN, np.hstack([overlay_bgr, divider, panel]))


def gui_loop():
    while not _gui_stop.is_set():
        try:
            overlay_bgr, info = _gui_queue.get(timeout=0.05)
            _render_frame(overlay_bgr, info)
        except queue.Empty:
            pass
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
    cv2.destroyAllWindows()


def show_monitor(overlay_bgr: np.ndarray, info: dict):
    try:
        if _gui_queue.full():
            try:
                _gui_queue.get_nowait()
            except queue.Empty:
                pass
        _gui_queue.put_nowait((overlay_bgr.copy(), info))
    except Exception:
        pass


def init_monitor(h: int, w: int):
    blank = np.full((h, w, 3), 18, dtype=np.uint8)
    _put(blank, "Waiting for capture command...",
         w // 2 - 160, h // 2, color=(110, 110, 110), scale=0.58)
    show_monitor(blank, {"status": "Waiting", "frame_idx": 0,
                         "timestamp": "-", "picks": []})
