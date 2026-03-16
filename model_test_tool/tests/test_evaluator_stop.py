import sys
import threading
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from core.evaluator import ModelEvaluator  # noqa: E402
from core.model_config import ModelConfig  # noqa: E402


def _make_sample(text: str = "[任务]\n问题\nA"):
    return {"text": text}


def _make_config(model_name: str = "test-model", max_concurrent_total: int = 1):
    return {
        "filter_empty_results": False,
        "model_configs": {
            model_name: {
                "model": model_name,
                "request_model": model_name,
                "max_concurrent_total": max_concurrent_total,
            }
        },
    }


def test_stop_flag_checked_before_scheduling(monkeypatch):
    evaluator = ModelEvaluator()
    evaluator.set_stop_flag(True)

    scheduled_calls = {"count": 0}

    def fake_call(*_args, **_kwargs):
        scheduled_calls["count"] += 1
        return ("A", 0.01)

    monkeypatch.setattr(evaluator, "_call_model_api_with_timing", fake_call)

    results = evaluator.process_samples_concurrent_with_limits(
        model_name="test-model",
        samples=[_make_sample(), _make_sample()],
        eval_times=2,
        config=_make_config(),
        max_concurrent_requests=2,
        progress_callback=None,
        total_tasks=4,
        completed_tasks_start=0,
        max_requests=None,
        model_max_requests=None,
        model_request_limit=None,
        current_total_requests=0,
        current_model_requests=0,
        dataset_path="dummy.jsonl",
    )

    assert scheduled_calls["count"] == 0
    assert results == []


def test_no_new_requests_scheduled_after_stop(monkeypatch):
    evaluator = ModelEvaluator()
    call_count = {"count": 0}
    lock = threading.Lock()

    def fake_call(_model_name, _sample, _config):
        with lock:
            call_count["count"] += 1
            if call_count["count"] == 1:
                evaluator.set_stop_flag(True)
        time.sleep(0.12)
        return ("A", 0.01)

    monkeypatch.setattr(evaluator, "_call_model_api_with_timing", fake_call)

    evaluator.process_samples_concurrent_with_limits(
        model_name="test-model",
        samples=[_make_sample(), _make_sample(), _make_sample()],
        eval_times=3,
        config=_make_config(max_concurrent_total=1),
        max_concurrent_requests=4,
        progress_callback=None,
        total_tasks=9,
        completed_tasks_start=0,
        max_requests=None,
        model_max_requests=None,
        model_request_limit=None,
        current_total_requests=0,
        current_model_requests=0,
        dataset_path="dummy.jsonl",
    )

    assert call_count["count"] == 1


def test_active_request_counter_decrements_in_finally(monkeypatch):
    evaluator = ModelEvaluator()

    class FakeCompletions:
        @staticmethod
        def create(**_kwargs):
            raise RuntimeError("network should not be used in test")

    class FakeChat:
        completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **_kwargs):
            self.chat = FakeChat()

    monkeypatch.setattr("core.evaluator.OpenAI", FakeOpenAI)

    model_config = ModelConfig("test-model")
    result = evaluator.call_model_api_normal(model_config, _make_sample(), config={})

    assert result == ("", 0, 0, 0)
    assert evaluator.get_active_requests_count() == 0
