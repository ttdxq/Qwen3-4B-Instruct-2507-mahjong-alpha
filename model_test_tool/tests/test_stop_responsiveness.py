import sys
import time
from pathlib import Path
from unittest.mock import Mock

try:
    from PyQt6.QtWidgets import QApplication  # noqa: F401
except ModuleNotFoundError:
    from PyQt5.QtWidgets import QApplication  # noqa: F401


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _create_main_window():
    from src.gui.main_window import MainWindow

    return MainWindow()


def test_stop_task_evaluation_is_non_blocking(qtbot):
    window = _create_main_window()
    qtbot.addWidget(window)

    mock_thread = Mock()
    mock_thread.isRunning.return_value = True
    window.task_evaluation_thread = mock_thread
    window._set_task_evaluation_ui_state("Running")

    started_at = time.perf_counter()
    window.stop_task_evaluation()
    elapsed_ms = (time.perf_counter() - started_at) * 1000

    mock_thread.request_stop.assert_called_once_with()
    assert elapsed_ms < 100, f"stop_task_evaluation took {elapsed_ms:.3f}ms"
    assert window._task_evaluation_state == "Stopping"
    assert window.task_progress_label.text() == "正在停止..."
