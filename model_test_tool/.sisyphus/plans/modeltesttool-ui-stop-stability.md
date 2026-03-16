# modeltesttool: Stop UI Freeze + Stability Plan

## TL;DR

> **Quick Summary**: Fix the UI freeze when clicking Stop by making Stop non-blocking on the Qt GUI thread, implementing a cooperative “soft stop” workflow in the worker, and adding an automated GUI-level regression test to prove the event loop stays responsive.
>
> **Deliverables**:
> - Non-blocking Stop behavior in the PyQt GUI (immediate “Stopping…” state; no UI thread waits)
> - Cooperative soft-stop in evaluation logic (stop scheduling new requests; let in-flight finish with bounded timeouts)
> - Automated regression test (preferred: `pytest-qt`) verifying Stop does not block the event loop
> - Cleanup of duplicated/dead code that makes stop/resume behavior fragile
>
> **Estimated Effort**: Medium
> **Parallel Execution**: YES (2 waves)
> **Critical Path**: Stop workflow refactor → regression test → cleanup + throttling

---

## Context

### Original Request
User asked for optimization points; prioritized **stability/resume**, with the concrete pain point: **UI freezes / becomes unresponsive when clicking Stop**.

### Interview Summary
- Priority: “稳定性/续跑”.
- Main pain: “点停止时卡死”.
- Stop semantics: **Soft stop** (do not issue new requests; allow in-flight requests to finish).

### Code Evidence (why UI freezes)
- `src/gui/main_window.py` `stop_task_evaluation()` performs blocking waits on the GUI thread (e.g., calling evaluator wait loops and `QThread.wait()`), which blocks the Qt event loop.

### Metis Review (gaps addressed in this plan)
- Explicit guardrails: never block GUI thread; Stop is idempotent; no force-killing threads.
- Edge cases: in-flight request never returns (timeouts), Stop spam-click, Stop then Start, Stop during saving.
- Verification: must be agent-executable; for desktop Qt apps, prefer `pytest-qt` to simulate Stop and assert event loop remains alive.

---

## Work Objectives

### Core Objective
Make Stop **immediately responsive** (no “Not Responding”), while preserving correctness: after Stop, the system stops scheduling new requests, finishes in-flight requests (bounded by timeouts), saves partial progress safely, and returns UI to a consistent state.

### Concrete Deliverables
- Stop button returns control to GUI thread immediately and UI shows “正在停止...” state.
- Evaluation worker cooperatively exits once stop is requested.
- Regression test proving the event loop continues to tick shortly after Stop.
- Remove duplicate method definitions and unreachable duplicated UI code that could mask future fixes.

### Definition of Done
- Clicking Stop no longer blocks the UI thread.
- Stop triggers cooperative shutdown of evaluation.
- Automated test suite includes a regression test for Stop responsiveness.

### Must NOT Have (Guardrails)
- No long waits (`.wait()`), busy loops, heavy I/O, or network calls on the Qt GUI thread.
- No thread force-termination patterns (avoid `terminate()` / killing threads).
- No “manual verification required” acceptance criteria.

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> Every task must be verifiable via automated commands/tests run by the agent.

### Test Decision
- **Infrastructure exists**: NO (no project-level pytest config found; avoid relying on `.conda/` tests)
- **Automated tests**: YES (Tests-after; targeted regression test)
- **Framework**: `pytest` + `pytest-qt` (recommended for Qt event-loop assertions)

### Agent-Executed QA Scenarios (Desktop Qt)
Because this is a PyQt desktop app, verification should be done via:
- `pytest-qt` tests that instantiate the window, start a stubbed evaluation, trigger Stop, and assert UI remains responsive.
- For non-UI pieces (evaluator stop flags / counters), unit tests using fakes/mocks.

Evidence capture:
- Save pytest output: `.sisyphus/evidence/task-N-pytest.txt` (redirect stdout/stderr)

---

## Execution Strategy

Wave 1 (Stop responsiveness + worker semantics):
1) Refactor Stop workflow to be non-blocking on UI thread
2) Ensure evaluator stop & active request tracking is cooperative and bounded

Wave 2 (Regression tests + cleanup + throttling):
3) Add `pytest-qt` regression tests for Stop responsiveness
4) Throttle UI updates and clean duplicate/dead code

Critical Path: 1 → 3 → 4

---

## TODOs

- [ ] 1. Make Stop non-blocking on the Qt GUI thread

  **What to do**:
  - Update `stop_task_evaluation()` so it does NOT call blocking waits on the GUI thread.
  - On Stop click, immediately:
    - Request stop on the worker (`request_stop()`),
    - Update UI state to “正在停止...” (disable Start, disable Stop or turn it into “Stopping…”, disable config edits if needed),
    - Return to event loop immediately.
  - Move any “wait for active requests”, “save partial results”, and “thread join” behavior into the worker completion path (signals) instead of inline blocking in the UI slot.

  **Must NOT do**:
  - No `self.task_evaluation_thread.wait()` (or any join/wait) inside the UI slot.
  - No polling loops/sleeps on UI thread.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multi-threaded GUI correctness (race/deadlock risks).
  - **Skills**: (none)
  - **Skills Evaluated but Omitted**:
    - `playwright`: Web-only; this is a desktop Qt app.

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `src/gui/main_window.py:1261` - Current `stop_task_evaluation()` blocks UI via waits; primary change site.
  - `src/gui/main_window.py:1170` - `start_task_evaluation()` sets up the worker and signal connections.
  - `src/gui/main_window.py:1337` - Progress/result UI updates; may need to set “stopping” state here.

  **Acceptance Criteria**:
  - [ ] Calling Stop (slot) returns quickly (target: < 100ms in a test harness) and does not block the Qt event loop.
  - [ ] UI label updates immediately to include “正在停止” (or equivalent) when Stop is triggered.
  - [ ] Evaluation finishes via signals; UI returns to idle state without manual intervention.

  **Agent-Executed QA Scenarios**:
  
  ```
  Scenario: Stop click does not freeze event loop
    Tool: pytest (pytest-qt)
    Preconditions: Tests installed; a stub evaluator simulates a long in-flight request.
    Steps:
      1. python -c "import os; os.makedirs('.sisyphus/evidence', exist_ok=True)"
      2. python -m pytest -q tests/test_stop_responsiveness.py > .sisyphus/evidence/task-1-pytest.txt 2>&1
      2. Test creates MainWindow, starts evaluation thread, triggers stop_task_evaluation()
      3. Test schedules QTimer.singleShot(0, callback) and asserts callback runs within 250ms
    Expected Result: Test passes; proves GUI thread remains responsive.
    Evidence: .sisyphus/evidence/task-1-pytest.txt
  ```


- [ ] 2. Implement cooperative soft-stop in evaluation execution (bounded)

  **What to do**:
  - Ensure stop flag is checked at all scheduling points so no new requests are created after Stop.
  - Ensure any “active request” counters always decrement in `finally` blocks so Stop completion doesn’t wait forever.
  - Ensure OpenAI calls have a bounded timeout (already present in config: `timeout`) and that the SDK call path actually uses it.
  - For concurrent modes (ThreadPoolExecutor): on stop request, stop submitting new futures; attempt to cancel futures not yet started; allow running futures to finish.
  - Define a clear “soft stop boundary”:
    - Default: stop after current in-flight requests complete; then exit loops.

  **Must NOT do**:
  - Don’t force-kill threads.
  - Don’t introduce infinite waits if a request hangs; rely on timeouts.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Concurrency + cancellation semantics.
  - **Skills**: (none)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `src/core/evaluator.py:23` - stop flag + locks.
  - `src/core/evaluator.py:90` - `wait_for_active_requests()` behavior; should not be used by UI thread.
  - `src/core/model_config.py:13` - per-model `timeout` configuration.
  - `src/gui/model_config_dialog.py:75` - UI for timeout field.

  **Acceptance Criteria**:
  - [ ] When stop flag is set, no new requests are issued after a bounded point in the loops.
  - [ ] Active request counter never goes negative and reaches 0 on completion (no leaks).
  - [ ] A “hanging request” is bounded by configured timeout; soft stop eventually completes.

  **Agent-Executed QA Scenarios**:
  
  ```
  Scenario: Soft stop stops scheduling new work
    Tool: pytest
    Preconditions: Fake OpenAI client that sleeps for N seconds and records call count.
    Steps:
      1. python -c "import os; os.makedirs('.sisyphus/evidence', exist_ok=True)"
      2. python -m pytest -q tests/test_soft_stop_scheduling.py > .sisyphus/evidence/task-2-pytest.txt 2>&1
      2. Start evaluation with many planned requests
      3. Trigger stop flag early
      4. Assert: call count stops increasing shortly after stop
    Expected Result: Requests stop being scheduled; only in-flight complete.
    Evidence: .sisyphus/evidence/task-2-pytest.txt
  ```


- [ ] 3. Add automated regression test suite for Stop responsiveness (pytest-qt)

  **What to do**:
  - Add test dependencies (`pytest`, `pytest-qt`).
  - Create `tests/` with a minimal Qt test that:
    - Creates `MainWindow`,
    - Starts a controlled/stubbed evaluation thread (avoid real network),
    - Triggers Stop,
    - Asserts the event loop still processes queued callbacks within a short deadline.
  - Ensure tests do not touch `.conda/` (limit test discovery to the project `tests/` directory).

  **Must NOT do**:
  - Don’t require a human to click buttons.
  - Don’t rely on real OpenAI network calls.

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Qt testing harness and reliable timing assertions.
  - **Skills**: (none)

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 4
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `src/main.py` - App entry and Qt app instantiation reference.
  - `src/gui/main_window.py:124` - Main window class.

  **Acceptance Criteria**:
  - [ ] `python -m pytest -q tests` exits 0.
  - [ ] Regression test fails on old blocking Stop implementation and passes on new implementation.
  - [ ] Pytest output saved to `.sisyphus/evidence/task-3-pytest.txt`.


- [ ] 4. Reduce UI workload during evaluation (throttle and cleanup)

  **What to do**:
  - Throttle frequent progress/result UI updates to avoid saturating the GUI thread (e.g., buffer updates and flush on a QTimer every 100–250ms).
  - Remove/resolve duplicated methods and dead/unreachable duplicated code blocks that complicate future fixes.
    - Example: duplicated method definitions in `src/core/evaluator.py`.
    - Example: duplicated unreachable block after `return` in `create_general_settings_group()` in `src/gui/main_window.py`.
  - Make Stop idempotent: multiple Stop clicks should not create additional waits or errors.

  **Must NOT do**:
  - Avoid large refactors unrelated to stop/resume.
  - Avoid changing result formats unless necessary for stability.

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mostly mechanical cleanup + small performance mitigation.
  - **Skills**: (none)

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (can overlap with Task 3 after baseline tests exist)
  - **Blocks**: None
  - **Blocked By**: Task 1 (for Stop idempotency changes)

  **References**:
  - `src/gui/main_window.py:315` - `create_general_settings_group()` contains duplicated unreachable code.
  - `src/core/evaluator.py:34` - duplicated `set_stop_flag` definitions.
  - `src/core/evaluator.py:277` - duplicated `save_partial_results` definitions.

  **Acceptance Criteria**:
  - [ ] UI remains responsive during evaluation (measured indirectly by the Stop responsiveness regression test and absence of excessive UI update load).
  - [ ] `python -m compileall -q src` exits 0.
  - [ ] `python -m pytest -q tests` still exits 0 after cleanup.

---

## Success Criteria

### Verification Commands
```bash
python -m compileall -q src
python -m pytest -q tests
```

Evidence capture (optional but recommended):
```bash
python -c "import os; os.makedirs('.sisyphus/evidence', exist_ok=True)"
python -m compileall -q src > .sisyphus/evidence/compileall.txt 2>&1
python -m pytest -q tests > .sisyphus/evidence/pytest.txt 2>&1
```

### Final Checklist
- [ ] Stop click immediately updates UI state and does not block the event loop
- [ ] Soft stop stops scheduling new requests, finishes in-flight with timeouts
- [ ] Partial results remain consistent (no deadlocks / no stuck “active requests”)
- [ ] Regression test suite reliably catches the freeze
