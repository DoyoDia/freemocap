"""
Experimental launcher for the skellycam live FreeMoCap -> GMR bridge.

This stays separate from the main FreeMoCap GUI. It reuses the existing
skellycam camera selector/config widgets, adds a small calibration workflow, and
starts the bridge as a child process with the selected camera configuration.
"""

from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from PySide6.QtCore import QProcess, Qt, QThread, QTimer, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from skellycam_live_source import configure_skellycam_runtime_home
from gmr_runtime import validate_gmr_runtime
from groundplane_only_calibration import apply_groundplane_to_calibration_toml
from skellycam_preview_latency_patch import install_latest_frame_preview_patch

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_HOME = configure_skellycam_runtime_home(REPO_ROOT / ".venv" / ".skellycam_live_gui_home")
GUI_SETTINGS_PATH = RUNTIME_HOME / "live_bridge_gui_settings.json"
GUI_SETTINGS_VERSION = 1

from freemocap.core_processes.capture_volume_calibration.charuco_stuff.charuco_board_definition import CHARUCO_BOARDS
from freemocap.gui.qt.workers.anipose_calibration_thread_worker import AniposeCalibrationThreadWorker

from skellycam import CameraConfig, SkellyCamParameterTreeWidget, SkellyCamWidget


class GroundplaneCalibrationThreadWorker(QThread):
    finished = Signal(str)
    failed = Signal(str)
    in_progress = Signal(str)

    def __init__(
        self,
        *,
        calibration_toml: Path,
        calibration_videos_folder: Path,
        charuco_square_size: float,
        charuco_board_name: str,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._calibration_toml = calibration_toml
        self._calibration_videos_folder = calibration_videos_folder
        self._charuco_square_size = charuco_square_size
        self._charuco_board_name = charuco_board_name

    def run(self) -> None:
        try:
            result = apply_groundplane_to_calibration_toml(
                calibration_toml=self._calibration_toml,
                calibration_videos_folder=self._calibration_videos_folder,
                charuco_square_size=self._charuco_square_size,
                charuco_board_name=self._charuco_board_name,
                progress_callback=self.in_progress.emit,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.finished.emit(str(result.output_toml_path))


TRANSLATIONS = {
    "zh": {
        "window_title": "Skellycam 实时 FreeMoCap -> GMR 桥",
        "camera_hint": "相机预览和相机参数来自 skellycam；下方英文参数会原样保留。",
        "language_group": "语言",
        "language": "界面语言",
        "calibration_group": "标定",
        "charuco_square_size": "Charuco 方格边长 (mm)",
        "charuco_board": "Charuco 板型",
        "groundplane": "用初始 Charuco 板作为地面原点",
        "preserve_ground_height": "保留地面高度（使用地面标定 TOML 时建议开启）",
        "annotate_charuco": "预览叠加 Charuco 检测（切换后可能需要重连相机）",
        "auto_fast_connect": "启动时快速连接上次相机",
        "fast_connect": "快速连接上次相机",
        "record_calibration": "开始录制标定视频",
        "stop_calibration": "停止录制标定视频",
        "run_calibration": "运行 FreeMoCap 标定",
        "run_groundplane": "只重标定地面",
        "active_recording": "当前标定录制",
        "no_calibration_recording": "还没有录制标定视频。",
        "bridge_group": "实时桥",
        "calibration_toml": "标定 TOML",
        "browse": "选择",
        "camera_id_order": "相机顺序",
        "camera_id_placeholder": "例如：0,1",
        "human_height": "人体身高",
        "model_complexity": "模型复杂度",
        "parallel_tracking": "并行相机 2D 识别",
        "leg_width_scale": "腿宽压缩",
        "min_valid_2d_ratio": "最低 2D 有效比例",
        "mujoco_viewer": "MuJoCo 可视化",
        "mujoco_fps": "MuJoCo FPS",
        "max_camera_skew": "相机最大时间差 ms",
        "request_bind": "请求地址",
        "reply_bind": "回复地址",
        "control_bind": "控制地址",
        "start_bridge": "启动实时桥",
        "stop_bridge": "停止实时桥",
        "start_diagnostics": "诊断双相机 3D",
        "stop_diagnostics": "停止诊断",
        "status": "状态",
        "stopped": "实时桥已停止。",
        "diagnostics_starting": "双相机 3D 诊断启动中...",
        "starting": "实时桥启动中...",
        "select_toml_title": "选择相机标定 TOML",
        "toml_missing": "标定 TOML 不存在：{path}",
        "no_cameras_selected": "没有选择相机。请先检测相机，并至少启用一个相机。",
        "preview_close_warning": "警告：关闭预览相机失败：{error}",
        "bridge_command": "启动实时桥：{command}",
        "bridge_finished": "实时桥进程已结束。",
        "diagnostics_command": "启动双相机 3D 诊断：{command}",
        "diagnostics_finished": "双相机 3D 诊断进程已结束。",
        "diagnostics_running": "诊断已经在运行啦，先停止当前诊断喵。",
        "bridge_running_block_diagnostics": "实时桥正在运行。请先停止实时桥，再诊断双相机 3D。",
        "bridge_running_block_calibration": "实时桥正在运行。请先停止实时桥，再录制标定视频。",
        "cameras_not_connected": "相机还没有连接。请先点击 skellycam 的 Detect Available Cameras。",
        "settings_loaded": "已读取上次配置：{path}",
        "settings_saved": "已保存 GUI 配置：{path}",
        "settings_save_failed": "保存 GUI 配置失败：{error}",
        "fast_connect_no_config": "还没有上次相机配置。请先 Detect Available Cameras 并 Apply 一次。",
        "fast_connect_started": "正在用上次配置快速连接相机：{camera_ids}",
        "fast_connect_failed": "快速连接上次相机失败：{error}",
        "calibration_recording_started": "开始录制标定视频：{path}",
        "calibration_overlay_paused": "录制标定视频时已临时关闭 Charuco 预览叠加，避免把叠加标记写进视频。",
        "calibration_recording_stopped": "标定录制已停止，正在保存同步视频...",
        "calibration_videos_saved": "标定视频已保存：{path}",
        "no_calibration_videos": "还没有可标定的视频。请先录制并等待保存完成。",
        "calibration_started": "开始运行 FreeMoCap 标定：{path}",
        "calibration_finished": "标定完成：{path}",
        "calibration_failed": "标定失败：{message}",
        "groundplane_started": "开始只重标定地面：{path}",
        "groundplane_finished": "地面重标定完成：{path}",
        "groundplane_requires_toml": "请先选择一个已经成功的相机标定 TOML，再只重标定地面。",
        "groundplane_failed": "地面原点标定失败：{message}",
        "runtime_not_ready": "GMR / MuJoCo 运行时环境未就绪：{error}",
    },
    "en": {
        "window_title": "Skellycam Live FreeMoCap -> GMR Bridge",
        "camera_hint": "Camera preview and camera parameters come from skellycam; its embedded labels stay in English.",
        "language_group": "Language",
        "language": "Language",
        "calibration_group": "Calibration",
        "charuco_square_size": "Charuco square size (mm)",
        "charuco_board": "Charuco board",
        "groundplane": "Use initial Charuco board as groundplane origin",
        "preserve_ground_height": "Preserve ground height (recommended with groundplane TOML)",
        "annotate_charuco": "Overlay Charuco detection in preview (may need reconnect)",
        "auto_fast_connect": "Fast-connect last cameras on startup",
        "fast_connect": "Fast Connect Last Cameras",
        "record_calibration": "Start Calibration Recording",
        "stop_calibration": "Stop Calibration Recording",
        "run_calibration": "Run FreeMoCap Calibration",
        "run_groundplane": "Groundplane Only",
        "active_recording": "Active calibration recording",
        "no_calibration_recording": "No calibration recording yet.",
        "bridge_group": "Live Bridge",
        "calibration_toml": "Calibration TOML",
        "browse": "Browse",
        "camera_id_order": "Camera id order",
        "camera_id_placeholder": "Example: 0,1",
        "human_height": "Human height",
        "model_complexity": "Model complexity",
        "parallel_tracking": "Parallel camera tracking",
        "leg_width_scale": "Leg width scale",
        "min_valid_2d_ratio": "Minimum valid 2D ratio",
        "mujoco_viewer": "MuJoCo viewer",
        "mujoco_fps": "MuJoCo FPS",
        "max_camera_skew": "Max camera skew ms",
        "request_bind": "Request bind",
        "reply_bind": "Reply bind",
        "control_bind": "Control bind",
        "start_bridge": "Start Bridge",
        "stop_bridge": "Stop Bridge",
        "start_diagnostics": "Diagnose 2-Camera 3D",
        "stop_diagnostics": "Stop Diagnostics",
        "status": "Status",
        "stopped": "Bridge stopped.",
        "diagnostics_starting": "2-camera 3D diagnostics starting...",
        "starting": "Bridge starting...",
        "select_toml_title": "Select camera calibration TOML",
        "toml_missing": "Calibration TOML does not exist: {path}",
        "no_cameras_selected": "No cameras selected. Detect cameras and enable at least one camera first.",
        "preview_close_warning": "Warning: could not close preview cameras cleanly: {error}",
        "bridge_command": "Starting bridge: {command}",
        "bridge_finished": "Bridge process finished.",
        "diagnostics_command": "Starting 2-camera 3D diagnostics: {command}",
        "diagnostics_finished": "2-camera 3D diagnostics process finished.",
        "diagnostics_running": "Diagnostics is already running. Stop the current diagnostics process first.",
        "bridge_running_block_diagnostics": "Bridge is running. Stop it before diagnosing 2-camera 3D.",
        "bridge_running_block_calibration": "Bridge is running. Stop it before recording calibration videos.",
        "cameras_not_connected": "Cameras are not connected. Click skellycam's Detect Available Cameras first.",
        "settings_loaded": "Loaded previous settings: {path}",
        "settings_saved": "Saved GUI settings: {path}",
        "settings_save_failed": "Could not save GUI settings: {error}",
        "fast_connect_no_config": "No saved camera config yet. Detect cameras and apply settings once first.",
        "fast_connect_started": "Fast-connecting cameras from previous config: {camera_ids}",
        "fast_connect_failed": "Fast connect failed: {error}",
        "calibration_recording_started": "Started calibration recording: {path}",
        "calibration_overlay_paused": "Temporarily disabled Charuco preview overlay while recording calibration videos so overlays are not written into the videos.",
        "calibration_recording_stopped": "Calibration recording stopped; saving synchronized videos...",
        "calibration_videos_saved": "Calibration videos saved: {path}",
        "no_calibration_videos": "No calibration videos are ready. Record and wait for saving first.",
        "calibration_started": "Starting FreeMoCap calibration: {path}",
        "calibration_finished": "Calibration finished: {path}",
        "calibration_failed": "Calibration failed: {message}",
        "groundplane_started": "Starting groundplane-only calibration: {path}",
        "groundplane_finished": "Groundplane-only calibration finished: {path}",
        "groundplane_requires_toml": "Select an existing successful camera calibration TOML before running groundplane-only calibration.",
        "groundplane_failed": "Groundplane calibration failed: {message}",
        "runtime_not_ready": "GMR/MuJoCo runtime is not ready: {error}",
    },
}


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _set_combo_text(combo: QComboBox, text: str) -> None:
    index = combo.findText(text)
    if index >= 0:
        combo.setCurrentIndex(index)


def _find_default_calibration_toml() -> Optional[Path]:
    candidates: list[Path] = []
    local_data = REPO_ROOT / "freemocap_data"
    if local_data.exists():
        candidates.extend(local_data.rglob("*camera_calibration.toml"))
        candidates.extend(local_data.rglob("*calibration*.toml"))

    unique_existing = sorted(
        {candidate.resolve() for candidate in candidates if candidate.exists()},
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return unique_existing[0] if unique_existing else None


def _create_live_calibration_recording_folder() -> Path:
    session_folder = REPO_ROOT / "freemocap_data" / "recording_sessions" / "live_bridge_calibrations"
    recording_name = datetime.now().strftime("recording_%Y_%m_%d_%H_%M_%S_calibration")
    recording_folder = session_folder / recording_name
    suffix = 1
    while recording_folder.exists():
        recording_folder = session_folder / f"{recording_name}_{suffix}"
        suffix += 1
    return recording_folder


class SkellycamLiveBridgeLauncher(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self._language = "zh"
        self.resize(1280, 860)

        self._bridge_process: Optional[QProcess] = None
        self._diagnostics_process: Optional[QProcess] = None
        self._calibration_worker: Optional[AniposeCalibrationThreadWorker] = None
        self._groundplane_worker: Optional[GroundplaneCalibrationThreadWorker] = None
        self._kill_thread_event = threading.Event()
        self._camera_config_json_path = RUNTIME_HOME / "live_bridge_camera_configs.json"
        self._active_calibration_recording_folder: Optional[Path] = None
        self._active_calibration_videos_folder: Optional[Path] = None
        self._next_recording_folder: Optional[Path] = None
        self._restore_annotate_charuco_after_recording: Optional[bool] = None
        self._loaded_settings: dict[str, Any] = {}
        self._loaded_camera_configs: dict[str, dict[str, Any]] = {}
        self._loading_settings = False

        self._layout = QHBoxLayout()
        self.setLayout(self._layout)

        patch_installed = install_latest_frame_preview_patch()
        print(
            "skellycam latest-frame preview patch installed"
            if patch_installed
            else "skellycam latest-frame preview patch already installed"
        )

        self._camera_viewer = SkellyCamWidget(
            get_new_synchronized_videos_folder_callable=self._get_synchronized_videos_folder,
            parent=self,
        )
        self._camera_viewer.videos_saved_to_this_folder_signal.connect(self._handle_videos_saved)
        self._camera_viewer.camera_group_created_signal.connect(self._handle_camera_group_created)
        self._camera_config_tree = SkellyCamParameterTreeWidget(self._camera_viewer)
        self._camera_config_tree.emitting_camera_configs_signal.connect(self._handle_camera_configs_changed)

        self._left_column = QVBoxLayout()
        self._camera_hint_label = QLabel()
        self._camera_hint_label.setWordWrap(True)
        self._left_column.addWidget(self._camera_hint_label)
        self._left_column.addWidget(self._camera_viewer, stretch=3)
        self._left_column.addWidget(self._camera_config_tree, stretch=2)
        self._layout.addLayout(self._left_column, stretch=3)

        self._right_column = QVBoxLayout()
        self._layout.addLayout(self._right_column, stretch=2)

        self._build_language_controls()
        self._build_calibration_controls()
        self._build_bridge_controls()
        self._build_log_view()
        self._apply_language()
        self._load_gui_settings()
        QTimer.singleShot(0, self._auto_fast_connect_if_enabled)

    def _tr(self, key: str, **kwargs) -> str:
        template = TRANSLATIONS.get(self._language, TRANSLATIONS["zh"]).get(key, key)
        return template.format(**kwargs) if kwargs else template

    def _build_language_controls(self) -> None:
        self._language_group = QGroupBox()
        self._right_column.addWidget(self._language_group)
        layout = QFormLayout()
        self._language_group.setLayout(layout)
        self._language_label = QLabel()
        self._language_combo = QComboBox()
        self._language_combo.addItem("中文", "zh")
        self._language_combo.addItem("English", "en")
        self._language_combo.currentIndexChanged.connect(self._handle_language_changed)
        layout.addRow(self._language_label, self._language_combo)

    def _build_calibration_controls(self) -> None:
        self._calibration_group = QGroupBox()
        self._right_column.addWidget(self._calibration_group)
        form = QFormLayout()
        self._calibration_group.setLayout(form)

        self._charuco_square_size_label = QLabel()
        self._charuco_square_size_spin = QDoubleSpinBox()
        self._charuco_square_size_spin.setRange(1.0, 500.0)
        self._charuco_square_size_spin.setDecimals(2)
        self._charuco_square_size_spin.setSingleStep(1.0)
        self._charuco_square_size_spin.setValue(39.0)
        form.addRow(self._charuco_square_size_label, self._charuco_square_size_spin)

        self._charuco_board_label = QLabel()
        self._charuco_board_combo = QComboBox()
        self._charuco_board_combo.addItems(list(CHARUCO_BOARDS.keys()))
        self._charuco_board_combo.currentTextChanged.connect(self._set_preview_charuco_board)
        form.addRow(self._charuco_board_label, self._charuco_board_combo)

        self._groundplane_checkbox = QCheckBox()
        form.addRow("", self._groundplane_checkbox)

        self._annotate_charuco_checkbox = QCheckBox()
        self._annotate_charuco_checkbox.setChecked(True)
        self._annotate_charuco_checkbox.toggled.connect(self._set_annotate_charuco)
        self._set_annotate_charuco(self._annotate_charuco_checkbox.isChecked())
        form.addRow("", self._annotate_charuco_checkbox)

        self._auto_fast_connect_checkbox = QCheckBox()
        self._auto_fast_connect_checkbox.setChecked(False)
        form.addRow("", self._auto_fast_connect_checkbox)

        self._fast_connect_button = QPushButton()
        self._fast_connect_button.clicked.connect(self._fast_connect_last_cameras)
        form.addRow("", self._fast_connect_button)

        button_row = QHBoxLayout()
        self._start_calibration_recording_button = QPushButton()
        self._stop_calibration_recording_button = QPushButton()
        self._run_calibration_button = QPushButton()
        self._run_groundplane_button = QPushButton()
        self._stop_calibration_recording_button.setEnabled(False)
        self._run_calibration_button.setEnabled(False)
        self._run_groundplane_button.setEnabled(False)
        self._start_calibration_recording_button.clicked.connect(self._start_calibration_recording)
        self._stop_calibration_recording_button.clicked.connect(self._stop_calibration_recording)
        self._run_calibration_button.clicked.connect(self._run_calibration)
        self._run_groundplane_button.clicked.connect(self._run_groundplane_calibration)
        button_row.addWidget(self._start_calibration_recording_button)
        button_row.addWidget(self._stop_calibration_recording_button)
        button_row.addWidget(self._run_calibration_button)
        button_row.addWidget(self._run_groundplane_button)
        form.addRow("", button_row)

        self._active_recording_label = QLabel()
        self._active_recording_value_label = QLabel()
        self._active_recording_value_label.setWordWrap(True)
        form.addRow(self._active_recording_label, self._active_recording_value_label)

    def _build_bridge_controls(self) -> None:
        self._bridge_group = QGroupBox()
        self._right_column.addWidget(self._bridge_group)
        form = QFormLayout()
        self._bridge_group.setLayout(form)

        self._calibration_line_edit = QLineEdit()
        default_calibration = _find_default_calibration_toml()
        if default_calibration is not None:
            self._calibration_line_edit.setText(str(default_calibration))
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(self._calibration_line_edit)
        self._browse_button = QPushButton()
        self._browse_button.clicked.connect(self._browse_calibration_toml)
        calibration_row.addWidget(self._browse_button)
        self._calibration_toml_label = QLabel()
        form.addRow(self._calibration_toml_label, calibration_row)

        self._camera_ids_label = QLabel()
        self._camera_ids_line_edit = QLineEdit()
        form.addRow(self._camera_ids_label, self._camera_ids_line_edit)

        self._human_height_label = QLabel()
        self._human_height_spin = QDoubleSpinBox()
        self._human_height_spin.setRange(0.8, 2.4)
        self._human_height_spin.setDecimals(2)
        self._human_height_spin.setSingleStep(0.01)
        self._human_height_spin.setValue(1.6)
        form.addRow(self._human_height_label, self._human_height_spin)

        self._model_complexity_label = QLabel()
        self._model_complexity_spin = QSpinBox()
        self._model_complexity_spin.setRange(0, 2)
        self._model_complexity_spin.setValue(0)
        form.addRow(self._model_complexity_label, self._model_complexity_spin)

        self._parallel_tracking_checkbox = QCheckBox()
        self._parallel_tracking_checkbox.setChecked(True)
        form.addRow("", self._parallel_tracking_checkbox)

        self._preserve_ground_height_checkbox = QCheckBox()
        self._preserve_ground_height_checkbox.setChecked(True)
        form.addRow("", self._preserve_ground_height_checkbox)

        self._leg_width_scale_label = QLabel()
        self._leg_width_scale_spin = QDoubleSpinBox()
        self._leg_width_scale_spin.setRange(0.0, 2.0)
        self._leg_width_scale_spin.setDecimals(2)
        self._leg_width_scale_spin.setSingleStep(0.05)
        self._leg_width_scale_spin.setValue(0.55)
        form.addRow(self._leg_width_scale_label, self._leg_width_scale_spin)

        self._min_valid_2d_ratio_label = QLabel()
        self._min_valid_2d_ratio_spin = QDoubleSpinBox()
        self._min_valid_2d_ratio_spin.setRange(0.0, 1.0)
        self._min_valid_2d_ratio_spin.setDecimals(2)
        self._min_valid_2d_ratio_spin.setSingleStep(0.05)
        self._min_valid_2d_ratio_spin.setValue(0.75)
        form.addRow(self._min_valid_2d_ratio_label, self._min_valid_2d_ratio_spin)

        self._mujoco_viewer_checkbox = QCheckBox()
        self._mujoco_viewer_checkbox.setChecked(True)
        form.addRow("", self._mujoco_viewer_checkbox)

        self._mujoco_fps_label = QLabel()
        self._mujoco_fps_spin = QDoubleSpinBox()
        self._mujoco_fps_spin.setRange(1.0, 120.0)
        self._mujoco_fps_spin.setDecimals(1)
        self._mujoco_fps_spin.setSingleStep(5.0)
        self._mujoco_fps_spin.setValue(30.0)
        form.addRow(self._mujoco_fps_label, self._mujoco_fps_spin)

        self._max_camera_skew_label = QLabel()
        self._max_camera_skew_spin = QDoubleSpinBox()
        self._max_camera_skew_spin.setRange(1.0, 500.0)
        self._max_camera_skew_spin.setDecimals(1)
        self._max_camera_skew_spin.setSingleStep(5.0)
        self._max_camera_skew_spin.setValue(50.0)
        form.addRow(self._max_camera_skew_label, self._max_camera_skew_spin)

        self._request_bind_label = QLabel()
        self._reply_bind_label = QLabel()
        self._control_bind_label = QLabel()
        self._req_addr_line_edit = QLineEdit("tcp://*:28701")
        self._rep_addr_line_edit = QLineEdit("tcp://*:28702")
        self._ctrl_addr_line_edit = QLineEdit("tcp://*:28703")
        form.addRow(self._request_bind_label, self._req_addr_line_edit)
        form.addRow(self._reply_bind_label, self._rep_addr_line_edit)
        form.addRow(self._control_bind_label, self._ctrl_addr_line_edit)

        button_row = QHBoxLayout()
        self._start_button = QPushButton()
        self._stop_button = QPushButton()
        self._stop_button.setEnabled(False)
        self._start_button.clicked.connect(self._start_bridge)
        self._stop_button.clicked.connect(self._stop_bridge)
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._stop_button)
        form.addRow("", button_row)

        diagnostics_button_row = QHBoxLayout()
        self._start_diagnostics_button = QPushButton()
        self._stop_diagnostics_button = QPushButton()
        self._stop_diagnostics_button.setEnabled(False)
        self._start_diagnostics_button.clicked.connect(self._start_diagnostics)
        self._stop_diagnostics_button.clicked.connect(self._stop_diagnostics)
        diagnostics_button_row.addWidget(self._start_diagnostics_button)
        diagnostics_button_row.addWidget(self._stop_diagnostics_button)
        form.addRow("", diagnostics_button_row)

        self._status_label_text = QLabel()
        self._status_label = QLabel()
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow(self._status_label_text, self._status_label)

    def _build_log_view(self) -> None:
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._right_column.addWidget(self._log_view, stretch=1)

    def _apply_language(self) -> None:
        self.setWindowTitle(self._tr("window_title"))
        self._camera_hint_label.setText(self._tr("camera_hint"))
        self._language_group.setTitle(self._tr("language_group"))
        self._language_label.setText(self._tr("language"))

        self._calibration_group.setTitle(self._tr("calibration_group"))
        self._charuco_square_size_label.setText(self._tr("charuco_square_size"))
        self._charuco_board_label.setText(self._tr("charuco_board"))
        self._groundplane_checkbox.setText(self._tr("groundplane"))
        self._preserve_ground_height_checkbox.setText(self._tr("preserve_ground_height"))
        self._annotate_charuco_checkbox.setText(self._tr("annotate_charuco"))
        self._auto_fast_connect_checkbox.setText(self._tr("auto_fast_connect"))
        self._fast_connect_button.setText(self._tr("fast_connect"))
        self._start_calibration_recording_button.setText(self._tr("record_calibration"))
        self._stop_calibration_recording_button.setText(self._tr("stop_calibration"))
        self._run_calibration_button.setText(self._tr("run_calibration"))
        self._run_groundplane_button.setText(self._tr("run_groundplane"))
        self._active_recording_label.setText(self._tr("active_recording"))
        if self._active_calibration_recording_folder is None:
            self._active_recording_value_label.setText(self._tr("no_calibration_recording"))

        self._bridge_group.setTitle(self._tr("bridge_group"))
        self._calibration_toml_label.setText(self._tr("calibration_toml"))
        self._browse_button.setText(self._tr("browse"))
        self._camera_ids_label.setText(self._tr("camera_id_order"))
        self._camera_ids_line_edit.setPlaceholderText(self._tr("camera_id_placeholder"))
        self._human_height_label.setText(self._tr("human_height"))
        self._model_complexity_label.setText(self._tr("model_complexity"))
        self._parallel_tracking_checkbox.setText(self._tr("parallel_tracking"))
        self._leg_width_scale_label.setText(self._tr("leg_width_scale"))
        self._min_valid_2d_ratio_label.setText(self._tr("min_valid_2d_ratio"))
        self._mujoco_viewer_checkbox.setText(self._tr("mujoco_viewer"))
        self._mujoco_fps_label.setText(self._tr("mujoco_fps"))
        self._max_camera_skew_label.setText(self._tr("max_camera_skew"))
        self._request_bind_label.setText(self._tr("request_bind"))
        self._reply_bind_label.setText(self._tr("reply_bind"))
        self._control_bind_label.setText(self._tr("control_bind"))
        self._start_button.setText(self._tr("start_bridge"))
        self._stop_button.setText(self._tr("stop_bridge"))
        self._start_diagnostics_button.setText(self._tr("start_diagnostics"))
        self._stop_diagnostics_button.setText(self._tr("stop_diagnostics"))
        self._status_label_text.setText(self._tr("status"))
        if self._bridge_process is None and self._diagnostics_process is None:
            self._status_label.setText(self._tr("stopped"))

    def _handle_language_changed(self, *_args) -> None:
        self._language = self._language_combo.currentData() or "zh"
        self._apply_language()
        if not self._loading_settings:
            self._save_gui_settings()

    def _handle_camera_group_created(self, camera_configs: Dict[str, Any]) -> None:
        self._loaded_camera_configs = {
            str(camera_id): _model_to_dict(config)
            for camera_id, config in camera_configs.items()
        }

    def _handle_camera_configs_changed(self, camera_configs: Dict[str, Any]) -> None:
        self._loaded_camera_configs = {
            str(camera_id): _model_to_dict(config)
            for camera_id, config in camera_configs.items()
        }
        self._save_gui_settings()

    def _load_gui_settings(self) -> None:
        try:
            if not GUI_SETTINGS_PATH.exists():
                return
            with GUI_SETTINGS_PATH.open("r", encoding="utf-8") as file:
                settings = json.load(file)
        except Exception as exc:
            self._append_log(f"Could not load GUI settings: {exc}")
            return

        settings_loaded = False
        self._loading_settings = True
        try:
            self._loaded_settings = settings
            self._loaded_camera_configs = {
                str(camera_id): config
                for camera_id, config in settings.get("camera_configs", {}).items()
                if isinstance(config, dict)
            }

            language = settings.get("language")
            if language in {"zh", "en"}:
                index = self._language_combo.findData(language)
                if index >= 0:
                    self._language_combo.setCurrentIndex(index)

            calibration_toml = settings.get("calibration_toml")
            if calibration_toml:
                self._calibration_line_edit.setText(str(calibration_toml))
            self._camera_ids_line_edit.setText(str(settings.get("camera_ids", "")))

            charuco_square_size = settings.get("charuco_square_size")
            if charuco_square_size is not None:
                self._charuco_square_size_spin.setValue(float(charuco_square_size))
            _set_combo_text(self._charuco_board_combo, str(settings.get("charuco_board", "")))
            self._groundplane_checkbox.setChecked(bool(settings.get("groundplane", False)))
            self._annotate_charuco_checkbox.setChecked(bool(settings.get("annotate_charuco", True)))
            self._auto_fast_connect_checkbox.setChecked(bool(settings.get("auto_fast_connect", False)))

            self._human_height_spin.setValue(float(settings.get("human_height", self._human_height_spin.value())))
            self._model_complexity_spin.setValue(int(settings.get("model_complexity", self._model_complexity_spin.value())))
            self._parallel_tracking_checkbox.setChecked(bool(settings.get("parallel_tracking", True)))
            self._preserve_ground_height_checkbox.setChecked(bool(settings.get("preserve_ground_height", True)))
            self._leg_width_scale_spin.setValue(float(settings.get("leg_width_scale", self._leg_width_scale_spin.value())))
            self._min_valid_2d_ratio_spin.setValue(
                float(settings.get("min_valid_2d_ratio", self._min_valid_2d_ratio_spin.value()))
            )
            self._mujoco_viewer_checkbox.setChecked(bool(settings.get("mujoco_viewer", True)))
            self._mujoco_fps_spin.setValue(float(settings.get("mujoco_fps", self._mujoco_fps_spin.value())))
            self._max_camera_skew_spin.setValue(float(settings.get("max_camera_skew_ms", self._max_camera_skew_spin.value())))
            self._req_addr_line_edit.setText(str(settings.get("req_bind_addr", self._req_addr_line_edit.text())))
            self._rep_addr_line_edit.setText(str(settings.get("rep_bind_addr", self._rep_addr_line_edit.text())))
            self._ctrl_addr_line_edit.setText(str(settings.get("ctrl_bind_addr", self._ctrl_addr_line_edit.text())))
            settings_loaded = True
        except Exception as exc:
            self._append_log(f"Could not load GUI settings: {exc}")
        finally:
            self._loading_settings = False

        if settings_loaded:
            self._append_log(self._tr("settings_loaded", path=GUI_SETTINGS_PATH))

    def _collect_gui_settings(self) -> dict[str, Any]:
        camera_configs = self._extract_camera_configs()
        if not camera_configs:
            camera_configs = self._loaded_camera_configs
        return {
            "version": GUI_SETTINGS_VERSION,
            "language": self._language,
            "calibration_toml": self._calibration_line_edit.text().strip(),
            "camera_ids": self._camera_ids_line_edit.text().strip(),
            "charuco_square_size": self._charuco_square_size_spin.value(),
            "charuco_board": self._charuco_board_combo.currentText(),
            "groundplane": self._groundplane_checkbox.isChecked(),
            "annotate_charuco": self._annotate_charuco_checkbox.isChecked(),
            "auto_fast_connect": self._auto_fast_connect_checkbox.isChecked(),
            "human_height": self._human_height_spin.value(),
            "model_complexity": self._model_complexity_spin.value(),
            "parallel_tracking": self._parallel_tracking_checkbox.isChecked(),
            "preserve_ground_height": self._preserve_ground_height_checkbox.isChecked(),
            "leg_width_scale": self._leg_width_scale_spin.value(),
            "min_valid_2d_ratio": self._min_valid_2d_ratio_spin.value(),
            "mujoco_viewer": self._mujoco_viewer_checkbox.isChecked(),
            "mujoco_fps": self._mujoco_fps_spin.value(),
            "max_camera_skew_ms": self._max_camera_skew_spin.value(),
            "req_bind_addr": self._req_addr_line_edit.text().strip(),
            "rep_bind_addr": self._rep_addr_line_edit.text().strip(),
            "ctrl_bind_addr": self._ctrl_addr_line_edit.text().strip(),
            "camera_configs": camera_configs,
        }

    def _save_gui_settings(self, log_success: bool = False) -> None:
        try:
            settings = self._collect_gui_settings()
            GUI_SETTINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with GUI_SETTINGS_PATH.open("w", encoding="utf-8") as file:
                json.dump(settings, file, ensure_ascii=False, indent=2)
            self._loaded_settings = settings
            self._loaded_camera_configs = settings.get("camera_configs", {})
            if log_success:
                self._append_log(self._tr("settings_saved", path=GUI_SETTINGS_PATH))
        except Exception as exc:
            self._append_log(self._tr("settings_save_failed", error=exc))

    def _saved_camera_ids(self) -> list[str]:
        raw = self._camera_ids_line_edit.text().strip() or str(self._loaded_settings.get("camera_ids", "")).strip()
        if raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [
            camera_id
            for camera_id, config in self._loaded_camera_configs.items()
            if bool(config.get("use_this_camera", True))
        ]

    def _saved_camera_config_models(self, camera_ids: list[str]) -> dict[str, CameraConfig]:
        output: dict[str, CameraConfig] = {}
        for camera_id in camera_ids:
            config = dict(self._loaded_camera_configs.get(str(camera_id), {}))
            config["camera_id"] = str(camera_id)
            output[str(camera_id)] = CameraConfig(**config)
        return output

    def _auto_fast_connect_if_enabled(self) -> None:
        if self._auto_fast_connect_checkbox.isChecked():
            self._fast_connect_last_cameras()

    def _fast_connect_last_cameras(self) -> None:
        camera_ids = self._saved_camera_ids()
        if not camera_ids:
            self._append_log(self._tr("fast_connect_no_config"))
            return

        try:
            if self._cameras_connected():
                self._camera_viewer.disconnect_from_cameras()
            camera_configs = self._saved_camera_config_models(camera_ids)
            worker = getattr(self._camera_viewer, "_cam_group_frame_worker")
            worker.annotate_images = self._annotate_charuco_checkbox.isChecked()
            worker._camera_ids = camera_ids
            worker._camera_group = worker._create_camera_group(
                camera_ids=camera_ids,
                camera_config_dictionary=camera_configs,
            )
            worker._video_recorder_dictionary = worker._initialize_video_recorder_dictionary()

            self._camera_viewer._clear_camera_grid_view(
                getattr(self._camera_viewer, "_dictionary_of_single_camera_view_widgets", None)
            )
            self._camera_viewer._detect_available_cameras_push_button.hide()
            self._camera_viewer._dictionary_of_single_camera_view_widgets = (
                self._camera_viewer._create_camera_view_widgets_and_add_them_to_grid_layout(
                    camera_config_dictionary=worker.camera_config_dictionary
                )
            )
            try:
                worker.new_image_signal.disconnect(self._camera_viewer._handle_image_update)
            except (RuntimeError, TypeError):
                pass
            worker.new_image_signal.connect(self._camera_viewer._handle_image_update)
            worker.start()
            self._set_preview_charuco_board(self._charuco_board_combo.currentText())
            self._append_log(self._tr("fast_connect_started", camera_ids=",".join(camera_ids)))
        except Exception as exc:
            self._append_log(self._tr("fast_connect_failed", error=exc))

    def _set_annotate_charuco(self, checked: bool) -> None:
        self._camera_viewer.annotate_images = checked
        worker = getattr(self._camera_viewer, "_cam_group_frame_worker", None)
        if worker is not None:
            worker.annotate_images = checked

    def _set_preview_charuco_board(self, board_name: str) -> None:
        worker = getattr(self._camera_viewer, "_cam_group_frame_worker", None)
        if worker is None:
            return
        skellycam_board_name = {
            "7x5 Charuco": "Full Charuco (7x5)",
            "5x3 Charuco": "Mini Charuco (5x3)",
        }.get(board_name)
        if skellycam_board_name is not None and hasattr(worker, "charuco_board"):
            worker.charuco_board = skellycam_board_name

    def _cameras_connected(self) -> bool:
        try:
            return bool(self._camera_viewer.cameras_connected)
        except AttributeError:
            return False

    def _get_synchronized_videos_folder(self) -> str:
        if self._next_recording_folder is not None:
            synchronized_videos_folder = self._next_recording_folder / "synchronized_videos"
        else:
            synchronized_videos_folder = RUNTIME_HOME / "preview_recording" / "synchronized_videos"
        synchronized_videos_folder.mkdir(parents=True, exist_ok=True)
        return str(synchronized_videos_folder)

    def _browse_calibration_toml(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            self._tr("select_toml_title"),
            str(REPO_ROOT),
            "TOML files (*.toml);;All files (*.*)",
        )
        if selected:
            self._calibration_line_edit.setText(selected)
            self._save_gui_settings()

    def _extract_camera_configs(self) -> Dict[str, dict]:
        try:
            configs = self._camera_config_tree._extract_dictionary_of_camera_configs()
        except Exception:
            configs = self._camera_viewer.camera_config_dicationary or {}

        output: Dict[str, dict] = {}
        for camera_id, config in configs.items():
            config_dict = _model_to_dict(config)
            config_dict["camera_id"] = str(camera_id)
            output[str(camera_id)] = config_dict
        return output

    def _selected_camera_ids(self, configs: Dict[str, dict]) -> list[str]:
        raw = self._camera_ids_line_edit.text().strip()
        if raw:
            return [part.strip() for part in raw.split(",") if part.strip()]
        return [
            camera_id
            for camera_id, config in configs.items()
            if bool(config.get("use_this_camera", True))
        ]

    def _write_camera_config_json(self, configs: Dict[str, dict]) -> None:
        self._camera_config_json_path.parent.mkdir(parents=True, exist_ok=True)
        with self._camera_config_json_path.open("w", encoding="utf-8") as file:
            json.dump(configs, file, indent=2)

    def _validate_bridge_runtime(self) -> bool:
        try:
            # Importing MuJoCo inside the long-lived Qt preview process can fail on
            # some Windows setups with WinError 1114 even though the bridge child
            # process can launch the viewer successfully. Preflight only the GMR
            # package + patch here and let the bridge/mujoco worker perform the
            # actual MuJoCo import check in its own process.
            validate_gmr_runtime(
                REPO_ROOT,
                require_import=False,
                require_mujoco=False,
                require_patch=True,
            )
        except Exception as exc:
            self._append_log(self._tr("runtime_not_ready", error=str(exc)))
            return False
        return True

    def _start_calibration_recording(self) -> None:
        if self._bridge_process is not None:
            self._append_log(self._tr("bridge_running_block_calibration"))
            return
        if not self._cameras_connected():
            self._append_log(self._tr("cameras_not_connected"))
            return
        self._save_gui_settings()

        recording_folder = _create_live_calibration_recording_folder()
        self._active_calibration_recording_folder = recording_folder
        self._active_calibration_videos_folder = None
        self._next_recording_folder = recording_folder
        self._active_recording_value_label.setText(str(recording_folder))
        self._run_calibration_button.setEnabled(False)
        self._run_groundplane_button.setEnabled(False)
        self._set_preview_charuco_board(self._charuco_board_combo.currentText())
        self._restore_annotate_charuco_after_recording = self._annotate_charuco_checkbox.isChecked()
        self._annotate_charuco_checkbox.setEnabled(False)
        if self._restore_annotate_charuco_after_recording:
            self._annotate_charuco_checkbox.setChecked(False)
            self._append_log(self._tr("calibration_overlay_paused"))

        self._camera_viewer.controller_slot_dictionary["start_recording"]()
        self._start_calibration_recording_button.setEnabled(False)
        self._stop_calibration_recording_button.setEnabled(True)
        self._append_log(self._tr("calibration_recording_started", path=recording_folder))

    def _stop_calibration_recording(self) -> None:
        self._camera_viewer.controller_slot_dictionary["stop_recording"]()
        self._stop_calibration_recording_button.setEnabled(False)
        self._start_calibration_recording_button.setEnabled(True)
        self._next_recording_folder = None
        if self._restore_annotate_charuco_after_recording:
            self._annotate_charuco_checkbox.setChecked(True)
        self._annotate_charuco_checkbox.setEnabled(True)
        self._restore_annotate_charuco_after_recording = None
        self._append_log(self._tr("calibration_recording_stopped"))

    def _handle_videos_saved(self, folder_path: str) -> None:
        videos_folder = Path(folder_path)
        if self._active_calibration_recording_folder is None:
            return
        if videos_folder.resolve().parent != self._active_calibration_recording_folder.resolve():
            return

        self._active_calibration_videos_folder = videos_folder
        self._active_recording_value_label.setText(str(self._active_calibration_recording_folder))
        self._run_calibration_button.setEnabled(True)
        self._run_groundplane_button.setEnabled(True)
        self._save_gui_settings()
        self._append_log(self._tr("calibration_videos_saved", path=videos_folder))

    def _run_calibration(self) -> None:
        if self._active_calibration_videos_folder is None:
            self._append_log(self._tr("no_calibration_videos"))
            return

        board_name = self._charuco_board_combo.currentText()
        charuco_board_definition = CHARUCO_BOARDS[board_name]()
        self._calibration_worker = AniposeCalibrationThreadWorker(
            calibration_videos_folder_path=self._active_calibration_videos_folder,
            charuco_square_size=float(self._charuco_square_size_spin.value()),
            kill_thread_event=self._kill_thread_event,
            charuco_board_definition=charuco_board_definition,
            use_charuco_as_groundplane=self._groundplane_checkbox.isChecked(),
        )
        self._calibration_worker.in_progress.connect(self._append_log)
        self._calibration_worker.finished.connect(self._handle_calibration_finished)
        self._calibration_worker.failed.connect(self._handle_calibration_failed)
        self._calibration_worker.groundplane_failed.connect(self._handle_groundplane_failed)
        self._run_calibration_button.setEnabled(False)
        self._run_groundplane_button.setEnabled(False)
        self._append_log(self._tr("calibration_started", path=self._active_calibration_videos_folder))
        self._calibration_worker.start()

    def _run_groundplane_calibration(self) -> None:
        if self._active_calibration_videos_folder is None:
            self._append_log(self._tr("no_calibration_videos"))
            return
        calibration_toml = Path(self._calibration_line_edit.text()).expanduser()
        if not calibration_toml.exists():
            self._append_log(self._tr("groundplane_requires_toml"))
            return

        self._groundplane_worker = GroundplaneCalibrationThreadWorker(
            calibration_toml=calibration_toml,
            calibration_videos_folder=self._active_calibration_videos_folder,
            charuco_square_size=float(self._charuco_square_size_spin.value()),
            charuco_board_name=self._charuco_board_combo.currentText(),
            parent=self,
        )
        self._groundplane_worker.in_progress.connect(self._append_log)
        self._groundplane_worker.finished.connect(self._handle_groundplane_finished)
        self._groundplane_worker.failed.connect(self._handle_groundplane_only_failed)
        self._run_calibration_button.setEnabled(False)
        self._run_groundplane_button.setEnabled(False)
        self._append_log(self._tr("groundplane_started", path=self._active_calibration_videos_folder))
        self._groundplane_worker.start()

    def _handle_calibration_finished(self, toml_path: str) -> None:
        self._calibration_line_edit.setText(toml_path)
        self._run_calibration_button.setEnabled(True)
        self._run_groundplane_button.setEnabled(self._active_calibration_videos_folder is not None)
        self._save_gui_settings()
        self._append_log(self._tr("calibration_finished", path=toml_path))

    def _handle_calibration_failed(self, message: str) -> None:
        self._run_calibration_button.setEnabled(True)
        self._run_groundplane_button.setEnabled(self._active_calibration_videos_folder is not None)
        self._append_log(self._tr("calibration_failed", message=message))

    def _handle_groundplane_finished(self, toml_path: str) -> None:
        self._calibration_line_edit.setText(toml_path)
        self._run_calibration_button.setEnabled(True)
        self._run_groundplane_button.setEnabled(True)
        self._preserve_ground_height_checkbox.setChecked(True)
        self._save_gui_settings()
        self._append_log(self._tr("groundplane_finished", path=toml_path))

    def _handle_groundplane_only_failed(self, message: str) -> None:
        self._run_calibration_button.setEnabled(True)
        self._run_groundplane_button.setEnabled(self._active_calibration_videos_folder is not None)
        self._append_log(self._tr("groundplane_failed", message=message))

    def _handle_groundplane_failed(self, message: str) -> None:
        self._append_log(self._tr("groundplane_failed", message=message))

    def _start_bridge(self) -> None:
        if self._diagnostics_process is not None:
            self._append_log(self._tr("diagnostics_running"))
            return
        calibration_toml = Path(self._calibration_line_edit.text()).expanduser()
        if not calibration_toml.exists():
            self._append_log(self._tr("toml_missing", path=calibration_toml))
            return

        configs = self._extract_camera_configs()
        camera_ids = self._selected_camera_ids(configs)
        if not camera_ids:
            self._append_log(self._tr("no_cameras_selected"))
            return
        if not self._validate_bridge_runtime():
            return

        self._save_gui_settings(log_success=True)
        self._write_camera_config_json(configs)
        try:
            self._camera_viewer.disconnect_from_cameras()
        except Exception as exc:
            self._append_log(self._tr("preview_close_warning", error=exc))

        args = [
            "-u",
            str(REPO_ROOT / "experimental" / "freemocap_to_gmr_bridge.py"),
            "--source",
            "skellycam",
            "--calibration-toml",
            str(calibration_toml),
            "--camera-ids",
            ",".join(camera_ids),
            "--camera-config-json",
            str(self._camera_config_json_path),
            "--skellycam-home",
            str(RUNTIME_HOME),
            "--tracker",
            "pose",
            "--model-complexity",
            str(self._model_complexity_spin.value()),
            "--actual-human-height",
            str(self._human_height_spin.value()),
            "--leg-width-scale",
            str(self._leg_width_scale_spin.value()),
            "--min-valid-2d-ratio",
            str(self._min_valid_2d_ratio_spin.value()),
            "--max-camera-skew-ms",
            str(self._max_camera_skew_spin.value()),
            "--req-bind-addr",
            self._req_addr_line_edit.text().strip(),
            "--rep-bind-addr",
            self._rep_addr_line_edit.text().strip(),
            "--ctrl-bind-addr",
            self._ctrl_addr_line_edit.text().strip(),
        ]
        if self._parallel_tracking_checkbox.isChecked():
            args.append("--parallel-camera-tracking")
        if self._preserve_ground_height_checkbox.isChecked():
            args.append("--preserve-ground-height")
        if self._mujoco_viewer_checkbox.isChecked():
            args.extend(
                [
                    "--mujoco-viewer",
                    "--mujoco-fps",
                    str(self._mujoco_fps_spin.value()),
                ]
            )

        self._bridge_process = QProcess(self)
        self._bridge_process.setWorkingDirectory(str(REPO_ROOT))
        self._bridge_process.setProgram(sys.executable)
        self._bridge_process.setArguments(args)
        self._bridge_process.readyReadStandardOutput.connect(self._handle_stdout)
        self._bridge_process.readyReadStandardError.connect(self._handle_stderr)
        self._bridge_process.finished.connect(lambda *_: self._handle_bridge_finished())
        self._bridge_process.start()

        self._start_button.setEnabled(False)
        self._stop_button.setEnabled(True)
        self._start_diagnostics_button.setEnabled(False)
        self._status_label.setText(self._tr("starting"))
        self._append_log(self._tr("bridge_command", command=f"{sys.executable} {' '.join(args)}"))

    def _start_diagnostics(self) -> None:
        if self._bridge_process is not None:
            self._append_log(self._tr("bridge_running_block_diagnostics"))
            return
        if self._diagnostics_process is not None:
            self._append_log(self._tr("diagnostics_running"))
            return

        calibration_toml = Path(self._calibration_line_edit.text()).expanduser()
        if not calibration_toml.exists():
            self._append_log(self._tr("toml_missing", path=calibration_toml))
            return

        configs = self._extract_camera_configs()
        camera_ids = self._selected_camera_ids(configs)
        if not camera_ids:
            self._append_log(self._tr("no_cameras_selected"))
            return

        self._save_gui_settings(log_success=True)
        self._write_camera_config_json(configs)
        try:
            self._camera_viewer.disconnect_from_cameras()
        except Exception as exc:
            self._append_log(self._tr("preview_close_warning", error=exc))

        args = [
            "-u",
            str(REPO_ROOT / "experimental" / "skellycam_mocap_diagnostics.py"),
            "--calibration-toml",
            str(calibration_toml),
            "--camera-ids",
            ",".join(camera_ids),
            "--camera-config-json",
            str(self._camera_config_json_path),
            "--skellycam-home",
            str(RUNTIME_HOME),
            "--tracker",
            "pose",
            "--model-complexity",
            str(self._model_complexity_spin.value()),
            "--max-camera-skew-ms",
            str(self._max_camera_skew_spin.value()),
            "--max-frames",
            "900",
        ]
        if self._parallel_tracking_checkbox.isChecked():
            args.append("--parallel-camera-tracking")

        self._diagnostics_process = QProcess(self)
        self._diagnostics_process.setWorkingDirectory(str(REPO_ROOT))
        self._diagnostics_process.setProgram(sys.executable)
        self._diagnostics_process.setArguments(args)
        self._diagnostics_process.readyReadStandardOutput.connect(self._handle_diagnostics_stdout)
        self._diagnostics_process.readyReadStandardError.connect(self._handle_diagnostics_stderr)
        self._diagnostics_process.finished.connect(lambda *_: self._handle_diagnostics_finished())
        self._diagnostics_process.start()

        self._start_diagnostics_button.setEnabled(False)
        self._stop_diagnostics_button.setEnabled(True)
        self._start_button.setEnabled(False)
        self._status_label.setText(self._tr("diagnostics_starting"))
        self._append_log(self._tr("diagnostics_command", command=f"{sys.executable} {' '.join(args)}"))

    def _stop_bridge(self) -> None:
        if self._bridge_process is None:
            return
        self._bridge_process.terminate()
        if not self._bridge_process.waitForFinished(3000):
            self._bridge_process.kill()
            self._bridge_process.waitForFinished(1000)

    def _stop_diagnostics(self) -> None:
        if self._diagnostics_process is None:
            return
        self._diagnostics_process.terminate()
        if not self._diagnostics_process.waitForFinished(3000):
            self._diagnostics_process.kill()
            self._diagnostics_process.waitForFinished(1000)

    def _handle_stdout(self) -> None:
        if self._bridge_process is None:
            return
        text = bytes(self._bridge_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self._append_log(line)
            if line.startswith("[BridgeStats]"):
                self._status_label.setText(line)

    def _handle_stderr(self) -> None:
        if self._bridge_process is None:
            return
        text = bytes(self._bridge_process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self._append_log(line)

    def _handle_diagnostics_stdout(self) -> None:
        if self._diagnostics_process is None:
            return
        text = bytes(self._diagnostics_process.readAllStandardOutput()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self._append_log(line)
            if line.startswith("[MocapDiagStats]"):
                self._status_label.setText(line)

    def _handle_diagnostics_stderr(self) -> None:
        if self._diagnostics_process is None:
            return
        text = bytes(self._diagnostics_process.readAllStandardError()).decode("utf-8", errors="replace")
        for line in text.splitlines():
            self._append_log(line)

    def _handle_bridge_finished(self) -> None:
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._start_diagnostics_button.setEnabled(True)
        self._status_label.setText(self._tr("stopped"))
        self._append_log(self._tr("bridge_finished"))
        self._bridge_process = None

    def _handle_diagnostics_finished(self) -> None:
        self._start_diagnostics_button.setEnabled(True)
        self._stop_diagnostics_button.setEnabled(False)
        self._start_button.setEnabled(True)
        self._status_label.setText(self._tr("stopped"))
        self._append_log(self._tr("diagnostics_finished"))
        self._diagnostics_process = None

    def _append_log(self, text: str) -> None:
        self._log_view.appendPlainText(text)

    def closeEvent(self, event) -> None:
        self._save_gui_settings()
        self._stop_bridge()
        self._stop_diagnostics()
        self._kill_thread_event.set()
        try:
            self._camera_viewer.close()
        finally:
            super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    launcher = SkellycamLiveBridgeLauncher()
    launcher.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
