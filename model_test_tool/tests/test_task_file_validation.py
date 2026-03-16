"""
Test for task file validation - prevents crashes on missing files.
"""

import pytest
from PyQt5.QtWidgets import QApplication


@pytest.fixture
def main_window(qtbot):
    """
    Create MainWindow with mocked evaluator for testing file validation.
    """
    from src.gui.main_window import MainWindow

    # Mock evaluator to avoid actual evaluation
    class MockEvaluator:
        def __init__(self):
            self.datasets = []
            self.models = []
            self.config = {}

        def evaluate_with_progress_tracking(
            self,
            datasets,
            models,
            config,
            progress_callback,
            result_callback,
            max_requests=None,
            task_name="unknown_task",
            total_task_requests=0,
            previous_completed=0,
            stop_flag=False,
        ):
            return {"total": 0, "successful": 0}

    # Mock evaluation thread
    class MockEvalThread:
        def __init__(self):
            self.isRunning = False

        def isRunning(self):
            return self.isRunning

        def request_stop(self):
            self.isRunning = False

        def start(self):
            self.isRunning = True

    # Create mocked evaluator and thread
    mock_evaluator = MockEvaluator()
    mock_thread = MockEvalThread()

    # Create main window with mock evaluator
    main_window = MainWindow(None, mock_evaluator, datasets=[], models=[])

    # Simulate selecting non-existent dataset file
    main_window.eval_dataset_list.clear()
    dataset_item = main_window.eval_dataset_list.item(0)
    dataset_item.setText("nonexistent_dataset.json")

    # Try to create evaluation task with invalid dataset
    main_window.start_task_evaluation()

    # Verify: program should NOT crash
    assert main_window.task_evaluation_thread is not None
    assert main_window._task_evaluation_state == "Idle"
    print("✅ Test passed: Program handles missing dataset files correctly")
