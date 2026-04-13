"""
Experimental launcher for the skellycam live FreeMoCap -> GMR bridge.

This is intentionally separate from the main FreeMoCap GUI. It reuses the
existing skellycam camera selector/config widgets, then starts the bridge as a
child process with the selected camera configuration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Optional

from PySide6.QtCore import QProcess, Qt
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
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

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_HOME = configure_skellycam_runtime_home(REPO_ROOT / ".venv" / ".skellycam_live_gui_home")

from skellycam import SkellyCamParameterTreeWidget, SkellyCamWidget


def _model_to_dict(model) -> dict:
    if hasattr(model, "model_dump"):
        return model.model_dump(mode="json")
    if hasattr(model, "dict"):
        return model.dict()
    return dict(model)


def _find_default_calibration_toml() -> Optional[Path]:
    candidates: list[Path] = []
    try:
        from freemocap.system.paths_and_filenames.path_getters import (
            get_last_successful_calibration_toml_path,
            get_most_recent_recording_path,
        )

        last_successful = Path(get_last_successful_calibration_toml_path())
        if last_successful.exists():
            candidates.append(last_successful)

        most_recent = get_most_recent_recording_path()
        if most_recent is not None:
            recent_path = Path(most_recent)
            candidates.extend(recent_path.glob("*camera_calibration.toml"))
            candidates.extend(recent_path.glob("*calibration*.toml"))
    except Exception:
        pass

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


class SkellycamLiveBridgeLauncher(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Skellycam Live FreeMoCap -> GMR Bridge")
        self.resize(1280, 820)

        self._bridge_process: Optional[QProcess] = None
        self._camera_config_json_path = RUNTIME_HOME / "live_bridge_camera_configs.json"

        self._layout = QHBoxLayout()
        self.setLayout(self._layout)

        self._camera_viewer = SkellyCamWidget(
            get_new_synchronized_videos_folder_callable=self._get_preview_recording_folder,
            parent=self,
        )
        self._camera_config_tree = SkellyCamParameterTreeWidget(self._camera_viewer)

        self._left_column = QVBoxLayout()
        self._left_column.addWidget(self._camera_viewer, stretch=3)
        self._left_column.addWidget(self._camera_config_tree, stretch=2)
        self._layout.addLayout(self._left_column, stretch=3)

        self._right_column = QVBoxLayout()
        self._layout.addLayout(self._right_column, stretch=2)

        self._build_bridge_controls()
        self._build_log_view()

    def _get_preview_recording_folder(self) -> str:
        preview_folder = RUNTIME_HOME / "preview_recording" / "synchronized_videos"
        preview_folder.mkdir(parents=True, exist_ok=True)
        return str(preview_folder)

    def _build_bridge_controls(self) -> None:
        bridge_group = QGroupBox("Live Bridge")
        self._right_column.addWidget(bridge_group)
        form = QFormLayout()
        bridge_group.setLayout(form)

        self._calibration_line_edit = QLineEdit()
        default_calibration = _find_default_calibration_toml()
        if default_calibration is not None:
            self._calibration_line_edit.setText(str(default_calibration))
        calibration_row = QHBoxLayout()
        calibration_row.addWidget(self._calibration_line_edit)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._browse_calibration_toml)
        calibration_row.addWidget(browse_button)
        form.addRow("Calibration TOML", calibration_row)

        self._camera_ids_line_edit = QLineEdit()
        self._camera_ids_line_edit.setPlaceholderText("Example: 0,1")
        form.addRow("Camera id order", self._camera_ids_line_edit)

        self._human_height_spin = QDoubleSpinBox()
        self._human_height_spin.setRange(0.8, 2.4)
        self._human_height_spin.setDecimals(2)
        self._human_height_spin.setSingleStep(0.01)
        self._human_height_spin.setValue(1.6)
        form.addRow("Human height", self._human_height_spin)

        self._model_complexity_spin = QSpinBox()
        self._model_complexity_spin.setRange(0, 2)
        self._model_complexity_spin.setValue(0)
        form.addRow("Model complexity", self._model_complexity_spin)

        self._parallel_tracking_checkbox = QCheckBox("Parallel camera tracking")
        self._parallel_tracking_checkbox.setChecked(True)
        form.addRow("", self._parallel_tracking_checkbox)

        self._max_camera_skew_spin = QDoubleSpinBox()
        self._max_camera_skew_spin.setRange(1.0, 500.0)
        self._max_camera_skew_spin.setDecimals(1)
        self._max_camera_skew_spin.setSingleStep(5.0)
        self._max_camera_skew_spin.setValue(50.0)
        form.addRow("Max camera skew ms", self._max_camera_skew_spin)

        self._req_addr_line_edit = QLineEdit("tcp://*:28701")
        self._rep_addr_line_edit = QLineEdit("tcp://*:28702")
        self._ctrl_addr_line_edit = QLineEdit("tcp://*:28703")
        form.addRow("Request bind", self._req_addr_line_edit)
        form.addRow("Reply bind", self._rep_addr_line_edit)
        form.addRow("Control bind", self._ctrl_addr_line_edit)

        button_row = QHBoxLayout()
        self._start_button = QPushButton("Start Bridge")
        self._stop_button = QPushButton("Stop Bridge")
        self._stop_button.setEnabled(False)
        self._start_button.clicked.connect(self._start_bridge)
        self._stop_button.clicked.connect(self._stop_bridge)
        button_row.addWidget(self._start_button)
        button_row.addWidget(self._stop_button)
        form.addRow("", button_row)

        self._status_label = QLabel("Bridge stopped.")
        self._status_label.setWordWrap(True)
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        form.addRow("Status", self._status_label)

    def _build_log_view(self) -> None:
        self._log_view = QPlainTextEdit()
        self._log_view.setReadOnly(True)
        self._right_column.addWidget(self._log_view, stretch=1)

    def _browse_calibration_toml(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select camera calibration TOML",
            str(REPO_ROOT),
            "TOML files (*.toml);;All files (*.*)",
        )
        if selected:
            self._calibration_line_edit.setText(selected)

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

    def _start_bridge(self) -> None:
        calibration_toml = Path(self._calibration_line_edit.text()).expanduser()
        if not calibration_toml.exists():
            self._append_log(f"Calibration TOML does not exist: {calibration_toml}")
            return

        configs = self._extract_camera_configs()
        camera_ids = self._selected_camera_ids(configs)
        if not camera_ids:
            self._append_log("No cameras selected. Detect cameras and enable at least one camera first.")
            return

        self._write_camera_config_json(configs)
        try:
            self._camera_viewer.disconnect_from_cameras()
        except Exception as exc:
            self._append_log(f"Warning: could not close preview cameras cleanly: {exc}")

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
        self._status_label.setText("Bridge starting...")
        self._append_log(f"Starting bridge: {sys.executable} {' '.join(args)}")

    def _stop_bridge(self) -> None:
        if self._bridge_process is None:
            return
        self._bridge_process.terminate()
        if not self._bridge_process.waitForFinished(3000):
            self._bridge_process.kill()
            self._bridge_process.waitForFinished(1000)

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

    def _handle_bridge_finished(self) -> None:
        self._start_button.setEnabled(True)
        self._stop_button.setEnabled(False)
        self._status_label.setText("Bridge stopped.")
        self._append_log("Bridge process finished.")
        self._bridge_process = None

    def _append_log(self, text: str) -> None:
        self._log_view.appendPlainText(text)

    def closeEvent(self, event) -> None:
        self._stop_bridge()
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
