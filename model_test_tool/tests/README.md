# Pytest Tests

This directory contains regression tests for modeltesttool.

## Running Tests

```bash
cd E:\Downloads\unsloth\modeltesttool
python -m pytest -q tests/
```

## Test Files

- `test_stop_responsiveness.py` - GUI freeze regression test
- `test_evaluator_stop.py` - Evaluator cooperative soft-stop tests

## Fix Verified

**Progress Display Issue** (Task 14-15):
- Fixed progress calculation for multiple models
- Global total now tracked correctly
- Progress shows "X/Y" format (e.g., "3/5" for model A)

## Recent Changes

**Stop UI Freeze** (Wave 1, Tasks 1-3):
- Non-blocking Stop implementation
- UI state machine (Idle/Running/Stopping)
- Stop idempotency guards

**Evaluator Soft-Stop** (Wave 1, Tasks 4-8):
- Stop-flag lifecycle fixed
- Cooperative stop at all scheduling points
- ThreadPoolExecutor future cancellation
- Proper `finally` blocks for request counters

## Test Harness

**pytest** with **pytest-qt** for Qt GUI testing:
- `qtbot` fixture provides QApplication and mocked MainWindow
- Tests verify Stop button doesn't block event loop
- Tests verify Stop is idempotent

## Test Categories

1. **GUI Regression Tests**:
   - Event loop responsiveness (<100ms target)
   - UI state transitions
   - Stop button behavior

2. **Evaluator Unit Tests**:
   - Stop flag checked at scheduling points
   - Active request counters decrement in `finally` blocks
   - ThreadPoolExecutor future cancellation on stop
   - Timeout handling
