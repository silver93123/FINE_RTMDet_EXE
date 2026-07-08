"""BinPicking Vision System — PyQt6 런처

실행:
    conda activate vision_env
    pip install PyQt6
    python bp_launcher.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QStackedWidget,
    QHBoxLayout, QVBoxLayout,
    QLabel, QLineEdit, QPushButton, QCheckBox,
    QDoubleSpinBox, QSpinBox, QComboBox,
    QTextEdit, QFileDialog, QGroupBox, QFrame, QScrollArea,
)

ROOT = Path(__file__).resolve().parent

# =============================================================================
# 설정 기본값
# =============================================================================
DEFAULTS = {
    "tcp_host":         "0.0.0.0",
    "tcp_port":         29999,
    "warmup":           3,
    "out_dir":          str(ROOT / "data" / "captures" / "live"),
    "cam_mode":         "Distance1500mm",
    "cam_exp":          "Exp250Us",
    "config_path":      str(ROOT / "work_dirs" / "rtmdet-ins_bracket_v1" / "rtmdet-ins_bracket.py"),
    "checkpoint_path":  "",
    "score_threshold":  0.30,
    "min_points":       100,
    "mask_iou":         0.65,
    "cad_path":         str(ROOT / "data" / "cad" / "bracket_v2.stl"),
    "icp_fitness":      0.70,
    "voxel_cad":        0.002,
    "voxel_scene":      0.003,
    "init_roll":        0.0,
    "init_pitch":       0.0,
    "init_yaw":         0.0,
    "roll_min":        -45.0, "roll_max":   45.0,
    "pitch_min":       -45.0, "pitch_max":  45.0,
    "yaw_min":         -45.0, "yaw_max":    45.0,
    "pick_x":    0.000, "pick_y":  -0.100, "pick_z":   0.031,
    "off_x":     5.0,   "off_y":    7.0,   "off_z":    0.0,
    "mask_w":    0.5,
    "height_y":  15.0,
    "save_intensity":    True,
    "save_pointcloud":   False,
    "save_valid_mask":   False,
    "save_metadata":     True,
    "save_overlay":      True,
    "save_colored_ply":  False,
    "save_result_json":  False,
    "save_pick_csv":     True,
}

# =============================================================================
# 스타일
# =============================================================================
STYLE = """
QWidget {
    font-family: 'Segoe UI', 'Apple SD Gothic Neo', sans-serif;
    font-size: 12px;
    color: #111827;
}
QGroupBox {
    background: transparent;
    border: none;
    border-top: 1px solid #e5e7eb;
    margin-top: 18px;
    padding-top: 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 0px;
    top: -8px;
    padding: 0 10px 0 2px;
    color: #6b7280;
    font-size: 11px;
    font-weight: 500;
    background: transparent;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background: #ffffff;
    border: 1px solid #d1d5db;
    border-radius: 4px;
    padding: 3px 8px;
    color: #111827;
    min-height: 26px;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #2563eb;
}
QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
    border-color: #9ca3af;
}
QComboBox::drop-down { border: none; width: 18px; }
QComboBox::down-arrow { width: 0; }
QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {
    width: 16px; background: #f3f4f6;
    border: none; border-left: 1px solid #e5e7eb;
}
QCheckBox { color: #111827; spacing: 8px; }
QCheckBox::indicator {
    width: 15px; height: 15px;
    border-radius: 3px;
    border: 1px solid #d1d5db;
    background: #ffffff;
}
QCheckBox::indicator:checked { background: #2563eb; border-color: #2563eb; }
QPushButton {
    background: #f3f4f6;
    border: 1px solid #d1d5db;
    border-radius: 5px;
    padding: 5px 16px;
    color: #374151;
    min-height: 30px;
    font-size: 12px;
}
QPushButton:hover { background: #e5e7eb; border-color: #9ca3af; }
QPushButton:pressed { background: #d1d5db; }
QPushButton#btn_start {
    background: #2563eb; border-color: #1d4ed8;
    color: #ffffff; font-weight: 500; padding: 6px 22px;
    font-size: 13px;
}
QPushButton#btn_start:hover { background: #1d4ed8; }
QPushButton#btn_start:pressed { background: #1e40af; }
QPushButton#btn_start:disabled { background: #bfdbfe; border-color: #bfdbfe; }
QPushButton#btn_stop {
    background: #dc2626; border-color: #b91c1c;
    color: #ffffff; font-weight: 500; padding: 6px 22px;
    font-size: 13px;
}
QPushButton#btn_stop:hover { background: #b91c1c; }
QPushButton#btn_stop:disabled { background: #fecaca; border-color: #fecaca; }
QPushButton#btn_save {
    background: #f9fafb; border: 1px solid #d1d5db;
    color: #6b7280; padding: 6px 14px;
}
QPushButton#btn_save:hover { background: #f3f4f6; color: #374151; }
QPushButton#btn_browse {
    min-width: 30px; max-width: 30px;
    padding: 0; background: #f3f4f6;
    color: #6b7280; font-size: 14px;
    border: 1px solid #d1d5db;
}
QPushButton#btn_browse:hover { background: #e5e7eb; }
QTextEdit {
    background: #0f172a;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    color: #94a3b8;
    font-family: 'Consolas', 'Menlo', 'D2Coding', monospace;
    font-size: 11px;
    padding: 8px;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical {
    background: #f3f4f6; width: 6px; border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #d1d5db; border-radius: 3px; min-height: 24px;
}
QScrollBar::handle:vertical:hover { background: #9ca3af; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }

/* objectName 기반 배경 - 자식 위젯에 상속 안 됨 */
QWidget#right_panel   { background: #f3f4f6; }
QWidget#central_panel { background: #f3f4f6; }
QStackedWidget#stack_panel { background: #ffffff; border-right: 1px solid #d1d5db; }
"""


# =============================================================================
class ServerThread(QThread):
    log_signal    = pyqtSignal(str, str)
    status_signal = pyqtSignal(str)

    def __init__(self, settings: dict):
        super().__init__()
        self.settings = settings
        self._proc = None

    def run(self):
        s = self.settings
        patch = self._build_patch(s)
        script = (
            f"import sys\n"
            f"sys.path.insert(0, r'{ROOT}')\n"
            f"import scripts.bp_settings as _s\n"
            f"{patch}\n"
            f"from scripts.Run_binpicking import main\n"
            f"main()\n"
        )
        cmd = [sys.executable, "-c", script,
               "--host", s["tcp_host"], "--port", str(s["tcp_port"]),
               "--warmup", str(s["warmup"]), "--out", s["out_dir"]]
        self.log_signal.emit("비전시스템 시작 중...", "info")
        self.status_signal.emit("running")
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            for line in self._proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                level = ("ok"   if "✓" in line else
                         "err"  if ("✗" in line or "ERROR" in line) else
                         "info" if any(x in line for x in ["[frame", "TCP", "서버"]) else
                         "normal")
                self.log_signal.emit(line, level)
            self._proc.wait()
        except Exception as e:
            self.log_signal.emit(f"오류: {e}", "err")
        finally:
            self.status_signal.emit("stopped")
            self.log_signal.emit("비전시스템 종료됨", "info")

    def stop(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()

    def _build_patch(self, s: dict) -> str:
        lines = [
            f"_s.TCP_HOST = '{s['tcp_host']}'",
            f"_s.TCP_PORT = {s['tcp_port']}",
            f"_s.SCORE_THRESHOLD = {s['score_threshold']}",
            f"_s.MIN_POINTS_PER_INSTANCE = {s['min_points']}",
            f"_s.MASK_IOU_THRESHOLD = {s['mask_iou']}",
            f"_s.ICP_FITNESS_THRESHOLD = {s['icp_fitness']}",
            f"_s.VOXEL_SIZE_CAD = {s['voxel_cad']}",
            f"_s.VOXEL_SIZE_SCENE = {s['voxel_scene']}",
            f"_s.ICP_INIT_ROLL_DEG = {s['init_roll']}",
            f"_s.ICP_INIT_PITCH_DEG = {s['init_pitch']}",
            f"_s.ICP_INIT_YAW_DEG = {s['init_yaw']}",
            f"_s.ICP_ROLL_RANGE = ({s['roll_min']}, {s['roll_max']})",
            f"_s.ICP_PITCH_RANGE = ({s['pitch_min']}, {s['pitch_max']})",
            f"_s.ICP_YAW_RANGE = ({s['yaw_min']}, {s['yaw_max']})",
            f"_s.CAD_PICK_LOCAL = __import__('numpy').array([{s['pick_x']}, {s['pick_y']}, {s['pick_z']}, 1.0])",
            f"_s.PICK_OFFSET_X_MM = {s['off_x']}",
            f"_s.PICK_OFFSET_Y_MM = {s['off_y']}",
            f"_s.PICK_OFFSET_Z_MM = {s['off_z']}",
            f"_s.PICK_2D_MASK_WEIGHT = {s['mask_w']}",
            f"_s.HEIGHT_POINT_OFFSET_Y_MM = {s['height_y']}",
            f"_s.SAVE_INTENSITY = {s['save_intensity']}",
            f"_s.SAVE_POINTCLOUD = {s['save_pointcloud']}",
            f"_s.SAVE_VALID_MASK = {s['save_valid_mask']}",
            f"_s.SAVE_METADATA = {s['save_metadata']}",
            f"_s.SAVE_OVERLAY_PNG = {s['save_overlay']}",
            f"_s.SAVE_COLORED_PLY = {s['save_colored_ply']}",
            f"_s.SAVE_RESULT_JSON = {s['save_result_json']}",
            f"_s.SAVE_PICK_LOG_CSV = {s['save_pick_csv']}",
        ]
        if s.get("config_path"):
            lines.append(f"_s.CONFIG_PATH = __import__('pathlib').Path(r'{s['config_path']}')")
        if s.get("checkpoint_path"):
            lines.append(f"_s.CHECKPOINT_PATH = __import__('pathlib').Path(r'{s['checkpoint_path']}')")
        if s.get("cad_path"):
            lines.append(f"_s.CAD_PATH = __import__('pathlib').Path(r'{s['cad_path']}')")
        return "\n".join(lines)


# =============================================================================
# 헬퍼
# =============================================================================
def lbl(text, color="#111827", size=12, bold=False) -> QLabel:
    w = QLabel(text)
    fw = "500" if bold else "400"
    w.setStyleSheet(f"color:{color}; font-size:{size}px; font-weight:{fw}; background:transparent;")
    return w

def spin_d(val, step=0.01, dec=3, lo=-9999, hi=9999) -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setDecimals(dec); w.setSingleStep(step); w.setRange(lo, hi); w.setValue(val)
    return w

def spin_i(val, lo=0, hi=99999) -> QSpinBox:
    w = QSpinBox(); w.setRange(lo, hi); w.setValue(val); return w

def path_row(placeholder, default, file_filter="", folder=False):
    edit = QLineEdit(default); edit.setPlaceholderText(placeholder)
    btn = QPushButton("…"); btn.setObjectName("btn_browse")
    def browse():
        p = (QFileDialog.getExistingDirectory(None, "폴더 선택", default) if folder
             else QFileDialog.getOpenFileName(None, "파일 선택", str(ROOT), file_filter)[0])
        if p: edit.setText(p)
    btn.clicked.connect(browse)
    row = QHBoxLayout(); row.setSpacing(4)
    row.addWidget(edit); row.addWidget(btn)
    return edit, row

def field(label_text, widget, label_w=120) -> QHBoxLayout:
    row = QHBoxLayout(); row.setSpacing(8)
    l = lbl(label_text, "#374151"); l.setFixedWidth(label_w)
    row.addWidget(l); row.addWidget(widget); return row

def range_field(label_text, wmin, wmax, label_w=120) -> QHBoxLayout:
    row = QHBoxLayout(); row.setSpacing(6)
    l = lbl(label_text, "#374151"); l.setFixedWidth(label_w)
    row.addWidget(l); row.addWidget(wmin)
    row.addWidget(lbl("~", "#94a3b8")); row.addWidget(wmax)
    return row

def grp(title, parent_lay: QVBoxLayout) -> QVBoxLayout:
    """섹션 타이틀 + 구분선을 parent_lay 에 직접 추가하고 내용 레이아웃 반환"""
    # 위 간격
    parent_lay.addSpacing(12)
    # 타이틀
    t = QLabel(title)
    t.setMinimumHeight(18)
    t.setStyleSheet("color:#6b7280; font-size:11px; font-weight:500;")
    parent_lay.addWidget(t)
    # 구분선
    line = QFrame(); line.setFrameShape(QFrame.Shape.HLine)
    line.setFixedHeight(1)
    line.setStyleSheet("background:#e5e7eb; border:none; margin:0;")
    parent_lay.addWidget(line)
    # 내용 레이아웃 반환
    return parent_lay

def scroll_page(inner: QWidget) -> QScrollArea:
    sa = QScrollArea(); sa.setWidgetResizable(True)
    sa.setWidget(inner); return sa


# =============================================================================
# 콘텐츠 패널들
# =============================================================================
class ServerPanel(QWidget):
    def __init__(self, d):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        grp("TCP 서버", lay)
        self.host = QLineEdit(d["tcp_host"])
        self.port = spin_i(d["tcp_port"], 1024, 65535)
        self.warmup = spin_i(d["warmup"], 0, 20)
        lay.addLayout(field("Host", self.host))
        lay.addLayout(field("Port", self.port))
        lay.addLayout(field("Warmup 프레임", self.warmup))

        grp("출력 경로", lay)
        self.out_edit, out_row = path_row("출력 폴더", d["out_dir"], folder=True)
        lay.addLayout(out_row)

        grp("카메라 (config.yaml)", lay)
        self.cam_mode = QComboBox()
        self.cam_mode.addItems(["Distance1500mm","Distance3000mm","Distance4000mm","Distance5000mm"])
        self.cam_mode.setCurrentText(d["cam_mode"])
        self.cam_exp = QComboBox()
        self.cam_exp.addItems(["Exp62_5Us","Exp250Us","Exp1000Us"])
        self.cam_exp.setCurrentText(d["cam_exp"])
        lay.addLayout(field("동작 거리", self.cam_mode))
        lay.addLayout(field("노출 시간", self.cam_exp))
        lay.addStretch()

    def values(self):
        return {"tcp_host": self.host.text(), "tcp_port": self.port.value(),
                "warmup": self.warmup.value(), "out_dir": self.out_edit.text(),
                "cam_mode": self.cam_mode.currentText(), "cam_exp": self.cam_exp.currentText()}


class DetectionPanel(QWidget):
    def __init__(self, d):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        grp("RTMDet 모델 파일", lay)
        self.config_edit, cfg_row = path_row("rtmdet-ins_bracket.py", d["config_path"], "Config (*.py)")
        self.ckpt_edit,  ckpt_row = path_row("best_*.pth (비워두면 자동탐색)", d["checkpoint_path"], "Checkpoint (*.pth)")
        lay.addWidget(lbl("Config 파일", "#6b7280", 11))
        lay.addLayout(cfg_row)
        lay.addWidget(lbl("Checkpoint (.pth)", "#6b7280", 11))
        lay.addLayout(ckpt_row)

        grp("Detection 파라미터", lay)
        self.score    = spin_d(d["score_threshold"], 0.05, 2, 0.0, 1.0)
        self.min_pts  = spin_i(d["min_points"], 1, 10000)
        self.mask_iou = spin_d(d["mask_iou"], 0.05, 2, 0.0, 1.0)
        lay.addLayout(field("Score Threshold", self.score))
        lay.addLayout(field("Min Points", self.min_pts))
        lay.addLayout(field("Mask IOU", self.mask_iou))
        lay.addStretch()

    def values(self):
        return {"config_path": self.config_edit.text(), "checkpoint_path": self.ckpt_edit.text(),
                "score_threshold": self.score.value(), "min_points": self.min_pts.value(),
                "mask_iou": self.mask_iou.value()}


class ICPPanel(QWidget):
    def __init__(self, d):
        super().__init__()
        inner = QWidget(); lay = QVBoxLayout(inner); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        grp("CAD 파일", lay)
        self.cad_edit, cad_row = path_row("bracket_v2.stl", d["cad_path"], "STL Files (*.stl)")
        lay.addLayout(cad_row)

        grp("ICP 파라미터", lay)
        self.fitness = spin_d(d["icp_fitness"], 0.05, 2, 0.0, 1.0)
        self.v_cad   = spin_d(d["voxel_cad"],   0.001, 3, 0.001, 0.1)
        self.v_scene = spin_d(d["voxel_scene"],  0.001, 3, 0.001, 0.1)
        lay.addLayout(field("Fitness Threshold", self.fitness))
        lay.addLayout(field("Voxel CAD (m)", self.v_cad))
        lay.addLayout(field("Voxel Scene (m)", self.v_scene))

        grp("초기 자세 (deg)", lay)
        self.ir = spin_d(d["init_roll"],  1.0, 1, -180, 180)
        self.ip = spin_d(d["init_pitch"], 1.0, 1, -180, 180)
        self.iy = spin_d(d["init_yaw"],   1.0, 1, -180, 180)
        lay.addLayout(field("Init Roll", self.ir))
        lay.addLayout(field("Init Pitch", self.ip))
        lay.addLayout(field("Init Yaw", self.iy))

        grp("회전 구속 범위 (deg)", lay)
        self.rmin = spin_d(d["roll_min"],  1.0, 1, -180, 0)
        self.rmax = spin_d(d["roll_max"],  1.0, 1, 0, 180)
        self.pmin = spin_d(d["pitch_min"], 1.0, 1, -180, 0)
        self.pmax = spin_d(d["pitch_max"], 1.0, 1, 0, 180)
        self.ymin = spin_d(d["yaw_min"],   1.0, 1, -180, 0)
        self.ymax = spin_d(d["yaw_max"],   1.0, 1, 0, 180)
        lay.addLayout(range_field("Roll",  self.rmin, self.rmax))
        lay.addLayout(range_field("Pitch", self.pmin, self.pmax))
        lay.addLayout(range_field("Yaw",   self.ymin, self.ymax))
        lay.addStretch()

        vlay = QVBoxLayout(self); vlay.setContentsMargins(0,0,0,0)
        vlay.addWidget(scroll_page(inner))

    def values(self):
        return {"cad_path": self.cad_edit.text(), "icp_fitness": self.fitness.value(),
                "voxel_cad": self.v_cad.value(), "voxel_scene": self.v_scene.value(),
                "init_roll": self.ir.value(), "init_pitch": self.ip.value(), "init_yaw": self.iy.value(),
                "roll_min": self.rmin.value(), "roll_max": self.rmax.value(),
                "pitch_min": self.pmin.value(), "pitch_max": self.pmax.value(),
                "yaw_min": self.ymin.value(), "yaw_max": self.ymax.value()}


class PickPanel(QWidget):
    def __init__(self, d):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)

        grp("CAD 픽포인트 로컬 좌표 (m)", lay)
        self.px = spin_d(d["pick_x"], 0.001, 3)
        self.py = spin_d(d["pick_y"], 0.001, 3)
        self.pz = spin_d(d["pick_z"], 0.001, 3)
        lay.addLayout(field("X", self.px)); lay.addLayout(field("Y", self.py)); lay.addLayout(field("Z", self.pz))

        grp("픽포인트 오프셋 (mm)", lay)
        self.ox = spin_d(d["off_x"], 0.5, 1)
        self.oy = spin_d(d["off_y"], 0.5, 1)
        self.oz = spin_d(d["off_z"], 0.5, 1)
        lay.addLayout(field("Offset X", self.ox)); lay.addLayout(field("Offset Y", self.oy)); lay.addLayout(field("Offset Z", self.oz))

        grp("블렌딩 / 높이", lay)
        self.mw = spin_d(d["mask_w"], 0.05, 2, 0.0, 1.0)
        self.hy = spin_d(d["height_y"], 0.5, 1, 0.0, 200.0)
        lay.addLayout(field("2D Mask Weight", self.mw))
        lay.addLayout(field("Height Offset Y (mm)", self.hy))
        lay.addStretch()

    def values(self):
        return {"pick_x": self.px.value(), "pick_y": self.py.value(), "pick_z": self.pz.value(),
                "off_x": self.ox.value(), "off_y": self.oy.value(), "off_z": self.oz.value(),
                "mask_w": self.mw.value(), "height_y": self.hy.value()}


class SavePanel(QWidget):
    ITEMS = [
        ("save_intensity",   "Intensity PNG",  "intensity/*.png — 캡처 강도 이미지"),
        ("save_pointcloud",  "Pointcloud NPY", "pointcloud_organized/*.npy"),
        ("save_valid_mask",  "Valid Mask NPY", "valid_mask/*.npy — 유효 픽셀 마스크"),
        ("save_metadata",    "Metadata JSON",  "metadata/*.json — 캡처 메타데이터"),
        ("save_overlay",     "Overlay PNG",    "results/*_overlay.png — 오버레이"),
        ("save_colored_ply", "Colored PLY",    "results/*_colored.ply — 포인트클라우드"),
        ("save_result_json", "Result JSON",    "results/*_result.json — ICP 결과"),
        ("save_pick_csv",    "Pick Log CSV",   "pick_log.csv — 픽포인트 로그"),
    ]
    def __init__(self, d):
        super().__init__()
        lay = QVBoxLayout(self); lay.setContentsMargins(16,16,16,16); lay.setSpacing(12)
        grp("저장 항목 선택", lay)
        self.checks = {}
        for key, name, desc in self.ITEMS:
            cb = QCheckBox(name); cb.setChecked(d.get(key, False))
            dl = lbl(desc, "#94a3b8", 11); dl.setStyleSheet("color:#6b7280; font-size:11px; margin-left:22px; background:transparent;")
            lay.addWidget(cb); lay.addWidget(dl)
            self.checks[key] = cb
        lay.addStretch()

    def values(self):
        return {k: cb.isChecked() for k, cb in self.checks.items()}


# =============================================================================
# 커스텀 사이드바 탭 버튼
# =============================================================================
class SidebarBtn(QPushButton):
    def __init__(self, text, idx, on_click):
        super().__init__(text)
        self.idx = idx
        self.setCheckable(True)
        self.setFixedHeight(44)
        self.setStyleSheet(self._style(False))
        self.clicked.connect(lambda: on_click(self.idx))

    def set_active(self, active: bool):
        self.setChecked(active)
        self.setStyleSheet(self._style(active))

    def _style(self, active: bool) -> str:
        if active:
            return ("QPushButton { background:#1d4ed8; color:#ffffff; border:none;"
                    " border-left:3px solid #93c5fd; font-size:13px; font-weight:500;"
                    " text-align:left; padding-left:20px; border-radius:0; }")
        return ("QPushButton { background:transparent; color:#bfdbfe; border:none;"
                " font-size:13px; text-align:left; padding-left:23px; border-radius:0; }"
                "QPushButton:hover { background:#1e56d8; color:#ffffff; }")


# =============================================================================
# 메인 윈도우
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bracket Vision System")
        self.setMinimumSize(940, 640)
        self._server = None
        self._frame_count = 0
        self._ok_count = 0
        self._sidebar_btns: list[SidebarBtn] = []
        self._build_ui()

    def _build_ui(self):
        central = QWidget()
        central.setObjectName("central_panel")
        self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0,0,0,0); root.setSpacing(0)

        # ── 사이드바 ──────────────────────────────────────────────────────
        sidebar = QWidget(); sidebar.setFixedWidth(190)
        sidebar.setStyleSheet("background:#2563eb;")
        slay = QVBoxLayout(sidebar); slay.setContentsMargins(0,0,0,0); slay.setSpacing(0)

        # 로고 영역
        logo_area = QWidget(); logo_area.setFixedHeight(52)
        logo_area.setStyleSheet("background:#1d4ed8; border-bottom:1px solid #2563eb;")
        ll = QHBoxLayout(logo_area); ll.setContentsMargins(16,0,0,0)
        ll.addWidget(lbl("화인산업", "#ffffff", 14, bold=True))
        slay.addWidget(logo_area)

        # 탭 버튼
        TABS = ["비전실행 / 연결", "Detection", "ICP", "픽포인트", "저장 옵션"]
        for i, name in enumerate(TABS):
            btn = SidebarBtn(name, i, self._switch_tab)
            self._sidebar_btns.append(btn)
            slay.addWidget(btn)

        slay.addStretch()
        root.addWidget(sidebar)

        # ── 왼쪽 콘텐츠 패널 ─────────────────────────────────────────────
        d = DEFAULTS
        self.panels = [
            ServerPanel(d), DetectionPanel(d), ICPPanel(d),
            PickPanel(d),   SavePanel(d),
        ]
        self.stack = QStackedWidget()
        self.stack.setFixedWidth(340)
        self.stack.setObjectName("stack_panel")
        for p in self.panels:
            self.stack.addWidget(p)
        root.addWidget(self.stack)
        self._switch_tab(0)

        # ── 오른쪽: 상태 + 로그 ──────────────────────────────────────────
        right = QWidget()
        right.setObjectName("right_panel")
        rlay = QVBoxLayout(right); rlay.setContentsMargins(20,20,20,0); rlay.setSpacing(14)

        # 버튼 행
        btn_row = QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_start = QPushButton("▶  비전 시작")
        self.btn_start.setStyleSheet(
            "QPushButton { background:#2563eb; color:#ffffff; border:none;"
            " border-radius:5px; padding:7px 22px; font-size:13px; font-weight:500; min-height:32px; }"
            "QPushButton:hover { background:#1d4ed8; }"
            "QPushButton:pressed { background:#1e40af; }"
            "QPushButton:disabled { background:#93c5fd; color:#ffffff; }"
        )
        self.btn_stop = QPushButton("■  비전 중지")
        self.btn_stop.setStyleSheet(
            "QPushButton { background:#dc2626; color:#ffffff; border:none;"
            " border-radius:5px; padding:7px 22px; font-size:13px; font-weight:500; min-height:32px; }"
            "QPushButton:hover { background:#b91c1c; }"
            "QPushButton:pressed { background:#991b1b; }"
            "QPushButton:disabled { background:#fca5a5; color:#ffffff; }"
        )
        self.btn_save = QPushButton("↺  설정 저장")
        self.btn_save.setStyleSheet(
            "QPushButton { background:#f3f4f6; color:#6b7280; border:1px solid #d1d5db;"
            " border-radius:5px; padding:7px 14px; font-size:12px; min-height:32px; }"
            "QPushButton:hover { background:#e5e7eb; color:#374151; }"
        )
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self._start_server)
        self.btn_stop.clicked.connect(self._stop_server)
        self.btn_save.clicked.connect(lambda: self._on_log("설정은 시스템 시작 시 자동 적용됩니다.", "info"))
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_save)
        rlay.addLayout(btn_row)

        # 상태 카드
        card_row = QHBoxLayout(); card_row.setSpacing(10)
        self.card_status  = self._card("시스템 상태",   "대기 중", "#6b7280")
        self.card_frames  = self._card("처리 프레임", "0",       "#111827")
        self.btn_open_result = QPushButton("📂  결과 폴더 열기")
        self.btn_open_result.setStyleSheet(
            "QPushButton { background:#ffffff; border:1px solid #d1d5db; border-radius:6px;"
            " color:#374151; font-size:12px; padding:0 16px; min-height:64px; text-align:left; }"
            "QPushButton:hover { background:#f3f4f6; border-color:#9ca3af; }"
        )
        self.btn_open_result.clicked.connect(self._open_result_folder)
        for c in [self.card_status, self.card_frames]:
            card_row.addWidget(c)
        card_row.addWidget(self.btn_open_result)
        rlay.addLayout(card_row)

        # 로그
        rlay.addWidget(lbl("실행 로그", "#6b7280", 10))
        self.log_edit = QTextEdit(); self.log_edit.setReadOnly(True)
        rlay.addWidget(self.log_edit)

        # 상태바
        sb = QWidget(); sb.setFixedHeight(30)
        sb.setStyleSheet("background:#ffffff; border-top:1px solid #d1d5db;")
        sbl = QHBoxLayout(sb); sbl.setContentsMargins(12,0,12,0)
        self.lbl_conn  = QLabel("●  대기 중")
        self.lbl_conn.setStyleSheet("color:#6b7280; font-size:11px;")
        self.lbl_model = QLabel("")
        self.lbl_model.setStyleSheet("color:#6b7280; font-size:11px;")
        sbl.addWidget(self.lbl_conn); sbl.addStretch(); sbl.addWidget(self.lbl_model)
        rlay.addWidget(sb)
        rlay.setContentsMargins(20,16,20,0)

        root.addWidget(right)

    def _switch_tab(self, idx: int):
        for i, btn in enumerate(self._sidebar_btns):
            btn.set_active(i == idx)
        self.stack.setCurrentIndex(idx)

    def _card(self, title, value, color) -> QWidget:
        card = QWidget()
        card.setStyleSheet("background:#ffffff; border:1px solid #d1d5db; border-radius:6px;")
        card.setFixedHeight(64)
        lay = QVBoxLayout(card); lay.setContentsMargins(14,8,14,8); lay.setSpacing(4)
        lay.addWidget(lbl(title, "#6b7280", 10))
        v = lbl(value, color, 16, bold=True)
        v.setObjectName("card_val")
        lay.addWidget(v)
        card._val_lbl = v
        return card

    def _collect(self) -> dict:
        s = dict(DEFAULTS)
        for p in self.panels:
            s.update(p.values())
        return s

    def _start_server(self):
        s = self._collect()
        self._frame_count = self._ok_count = 0
        self.log_edit.clear()
        self._server = ServerThread(s)
        self._server.log_signal.connect(self._on_log)
        self._server.status_signal.connect(self._on_status)
        self._server.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        ckpt = s.get("checkpoint_path") or "auto"
        self.lbl_model.setText(f"모델: {Path(ckpt).name if ckpt != 'auto' else '자동탐색'}")

    def _stop_server(self):
        if self._server: self._server.stop()

    def _on_log(self, msg, level):
        c = {"ok":"#22c55e","err":"#ef4444","info":"#60a5fa","normal":"#64748b"}.get(level,"#64748b")
        self.log_edit.append(f'<span style="color:{c}; font-family:Consolas,monospace; font-size:11px;">{msg}</span>')
        if "[frame_" in msg:
            self._frame_count += 1
            self.card_frames._val_lbl.setText(str(self._frame_count))

    def _open_result_folder(self):
        import subprocess as _sp
        out = self.panels[0].out_edit.text()
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        _sp.Popen(["xdg-open", str(path)])

    def _on_status(self, status):
        if status == "running":
            self.card_status._val_lbl.setText("실행 중")
            self.card_status._val_lbl.setStyleSheet("color:#16a34a; font-size:16px; font-weight:500; background:transparent;")
            self.lbl_conn.setText(f"●  {self.panels[0].host.text()}:{self.panels[0].port.value()}")
            self.lbl_conn.setStyleSheet("color:#16a34a; font-size:11px;")
        else:
            self.card_status._val_lbl.setText("중지됨")
            self.card_status._val_lbl.setStyleSheet("color:#6b7280; font-size:16px; font-weight:500; background:transparent;")
            self.lbl_conn.setText("●  대기 중")
            self.lbl_conn.setStyleSheet("color:#6b7280; font-size:11px;")
            self.btn_start.setEnabled(True)
            self.btn_stop.setEnabled(False)

    def closeEvent(self, event):
        if self._server and self._server.isRunning():
            self._server.stop(); self._server.wait(2000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLE)
    win = MainWindow(); win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()