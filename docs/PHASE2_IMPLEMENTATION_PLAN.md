# EVORA Phase 2 Implementation Plan

## Overview

Transform EVORA from a linear pipeline (PLAN → ASK → CODE → TEST → FIX → REPORT) into a dynamic autonomous agent loop:

`UNDERSTAND → ANALYZE → PLAN → DECIDE → ACT → OBSERVE → EVALUATE → REASON → DECIDE AGAIN`

## Implementation Order

### Step 1: Extend Logger Stages

**File**: `evora/logger.py`

Add new Stage enum values:
- `UNDERSTAND = "UNDERSTAND"` — Understanding the request
- `ANALYZE = "ANALYZE"` — Workspace inspection
- `DECIDE = "DECIDE"` — Decision-making
- `OBSERVE = "OBSERVE"` — Capturing action results
- `EVALUATE = "EVALUATE"` — Assessing progress
- `REASON = "REASON"` — Analyzing failures/success

Add corresponding colors and emojis.

**No changes to existing stages** — backward compatible.

### Step 2: Create Task State Module

**File**: `evora/task.py` (new)

Implement `@dataclass TaskState`:

```python
@dataclass
class TaskState:
    task_id: str                    # UUID
    request: str                    # User's request
    goal: str                       # Current objective
    current_step: Optional[str]     # Current executing step
    workspace: str                  # Workspace directory
    project_context: dict           # Analysis results
    plan: Optional[dict]            # Current plan
    attempts: int = 0               # Action retry count
    max_attempts: int = 10          # Configurable max
    observations: list[Observation] # All observations
    actions: list[dict]             # Actions attempted
    errors: list[str]               # Errors encountered
    decisions: list[Decision]       # All decisions made
    test_results: list[TestResult]  # Test outcomes
    status: str = "idle"            # idle, running, completed, failed, cancelled
    completion_criteria: list[str]  # How to determine success
    final_result: str = ""          # Final output
    elapsed: float = 0.0
    is_complete: bool = False
    is_failed: bool = False
    is_cancelled: bool = False
```

Plus `@dataclass Observation`:
```python
@dataclass
class Observation:
    type: str          # file_created, file_modified, command_success, 
                       # command_failed, test_passed, test_failed, 
                       # build_failed, file_missing, approval_granted, 
                       # approval_denied, error
    source: str        # Which action produced this
    data: dict         # Structured data about the observation
    timestamp: float
    success: bool
```

Plus `@dataclass TestResult`:
```python
@dataclass
class TestResult:
    command: str
    passed: bool
    output: str
    error: str
    return_code: int
```

Plus `@dataclass Decision`:
```python
@dataclass
class Decision:
    action: str         # next_action, execute_tool, run_tests, 
                        # fix_error, report, ask_approval, cancel, done
    reason: str         # Why this decision
    tool: Optional[str] # Which tool to use
    arguments: dict     # Tool arguments
    expected_outcome: str
    risk_level: str     # SAFE, ASK, DANGEROUS
    requires_approval: bool
    confidence: float   # 0.0 to 1.0
```

Plus `@dataclass ActionResult` (renamed from existing ToolResult to avoid collision):
```python
@dataclass
class ActionResult:
    success: bool
    tool: str
    arguments: dict
    output: str
    error: str
    observations: list[Observation]
```

**Serialization**: All dataclasses have `to_dict()` / `from_dict()` methods for JSON persistence.

### Step 3: Create Decision Engine

**File**: `evora/decision.py` (new)

Class `DecisionEngine`:

```python
class DecisionEngine:
    def __init__(self, tools: ToolRegistry, logger: Logger, 
                 max_retries: int = 3, auto_approve: bool = False)
    
    def decide_next(self, state: TaskState) -> Decision:
        """Main entry point — determines what to do next based on current state."""
        
    def _decide_understand(self, state: TaskState) -> Decision
    def _decide_analyze(self, state: TaskState) -> Decision
    def _decide_plan(self, state: TaskState) -> Decision
    def _decide_ask(self, state: TaskState) -> Decision
    def _decide_execute(self, state: TaskState) -> Decision
    def _decide_test(self, state: TaskState) -> Decision
    def _decide_fix(self, state: TaskState) -> Decision
    def _decide_report(self, state: TaskState) -> Decision
    def _decide_done(self, state: TaskState) -> Decision
```

The `decide_next()` method implements the state machine:

```
state = idle          → UNDERSTAND → ANALYZE → PLAN → ASK
state = planning      → wait for plan
state = awaiting_approval → ASK
state = approved      → EXECUTE (first step)
state = executing     → OBSERVE → EVALUATE → (execute next step | run tests | fix | report)
state = testing       → OBSERVE → EVALUATE → (report success | decide fix)
state = fixing        → OBSERVE → EVALUATE → (execute fix | report failure)
state = completed     → REPORT → MEMORY → DONE
state = failed        → REPORT → MEMORY → DONE
state = cancelled     → REPORT → MEMORY → DONE
```

**Key logic**: The engine considers:
- Remaining plan steps
- Test results
- Error history
- Retry counts
- Completion criteria

### Step 4: Create Observation Manager

**File**: `evora/observation.py` (new) — or integrate into task.py

Class `ObservationManager`:

```python
class ObservationManager:
    def observe(self, state: TaskState, action_result: ActionResult) -> list[Observation]
    
    def observe_file_created(self, path: str) -> Observation
    def observe_file_modified(self, path: str) -> Observation
    def observe_command_success(self, command: str) -> Observation
    def observe_command_failed(self, command: str, error: str, return_code: int) -> Observation
    def observe_test_passed(self, output: str) -> Observation
    def observe_test_failed(self, output: str, error: str) -> Observation
    def observe_error(self, error: str, context: str) -> Observation
    def observe_approval(self, granted: bool, reason: str) -> Observation
```

Each observation is added to `state.observations` and feeds the decision engine.

### Step 5: Create Evaluator

**File**: `evora/evaluation.py` (new)

Class `Evaluator`:

```python
class Evaluator:
    def evaluate(self, state: TaskState, observation: Observation) -> EvaluationResult
    
    def _check_completion(self, state: TaskState) -> bool
    def _check_stuck(self, state: TaskState) -> bool  # retry limit exceeded
    def _check_blocked(self, state: TaskState) -> bool
```

`EvaluationResult`:
```python
@dataclass
class EvaluationResult:
    outcome: str  # SUCCESS, PROGRESS, FAILURE, NEEDS_INFORMATION, 
                 # NEEDS_USER_APPROVAL, BLOCKED
    confidence: float
    reason: str
    recommendations: list[str]
```

### Step 6: Create Autonomous Agent

**File**: `evora/autonomous.py` (new) — or replace agent.py

Class `AutonomousAgent`:

```python
class AutonomousAgent:
    def __init__(self, model_manager, tools, security, memory, 
                 approval, analyzer, logger, config)
    
    async def run(self, request: str) -> str:
        """Main autonomous loop."""
        
    def _understand(self, request: str) -> str
    async def _analyze(self, workspace: str) -> dict
    async def _plan(self, request: str, context: dict) -> Plan
    async def _ask_approval(self, plan: Plan) -> Decision
    async def _act(self, state: TaskState, decision: Decision) -> ActionResult
    def _observe(self, state: TaskState, result: ActionResult) -> list[Observation]
    def _evaluate(self, state: TaskState, observation: Observation) -> EvaluationResult
    async def _reason(self, state: TaskState, eval_result: EvaluationResult) -> None
    def _report(self, state: TaskState) -> str
```

**Main loop**:
```python
while not state.is_complete and not state.is_failed and not state.is_cancelled:
    decision = self.decision_engine.decide_next(state)
    result = await self._act(state, decision)
    observations = self._observe(state, result)
    eval_result = self._evaluate(state, observations[-1])
    self._reason(state, eval_result)
```

### Step 7: Integrate with CLI

**File**: `evora/cli.py` (modify)

- Add new stages to logger initialization
- Wire up `AutonomousAgent` alongside existing `Agent`
- Add `--provider` flag (optional, for future use)
- Keep existing commands working (`run`, `plan`, `analyze`, `config`, `memory`)
- `evora run` uses the new `AutonomousAgent` instead of the old `Agent`

### Step 8: Keep Backward Compatibility

- The old `Agent` class remains available for direct use
- `AutonomousAgent` wraps/enhances the existing components
- All existing tests continue to pass

---

## Files to Create

| File | Purpose |
|------|---------|
| `evora/task.py` | TaskState, Observation, TestResult, Decision, ActionResult dataclasses |
| `evora/decision.py` | DecisionEngine — determines next action |
| `evora/observation.py` | ObservationManager — captures action results |
| `evora/evaluation.py` | Evaluator — assesses action outcomes |
| `evora/autonomous.py` | AutonomousAgent — the main loop |

## Files to Modify

| File | Changes |
|------|---------|
| `evora/logger.py` | Add UNDERSTAND, ANALYZE, DECIDE, OBSERVE, EVALUATE, REASON stages |
| `evora/cli.py` | Wire up AutonomousAgent, add new logger stages |
| `evora/agent.py` | Keep as-is (backward compatibility) |

## Files NOT Modified

- `evora/model.py` — No changes (model abstraction stays the same)
- `evora/tools.py` — No changes (tools stay the same)
- `evora/security.py` — No changes (permission system stays the same)
- `evora/approval.py` — No changes (approval system stays the same)
- `evora/memory.py` — No changes (memory system stays the same)
- `evora/analyzer.py` — No changes (analyzer stays the same)
- `evora/planner.py` — No changes (planner stays the same)
- `requirements.txt` — No new external dependencies needed
- `pyproject.toml` — No changes to dependencies

## Test Plan

### New Tests

| Test File | Tests |
|-----------|-------|
| `tests/test_task.py` | TaskState creation, serialization, observations, decisions |
| `tests/test_decision.py` | DecisionEngine state machine, all decision paths |
| `tests/test_observation.py` | ObservationManager capture, all observation types |
| `tests/test_evaluation.py` | Evaluator outcomes, completion detection, stuck detection |
| `tests/test_autonomous.py` | Full autonomous loop: success, failure, retry limit, cancellation |

### Test Principles
- Use mock model provider (existing pattern from test_agent.py)
- Test the actual orchestration logic, not just mock responses
- Verify the decision engine correctly transitions states
- Verify retry limits are enforced
- Verify cancellation works
- Verify completion detection works
- No real API access required

---

## Timeline

1. **Logger stages** — 30 min
2. **Task state module** — 30 min
3. **Decision engine** — 45 min
4. **Observation manager** — 30 min
5. **Evaluator** — 30 min
6. **Autonomous agent** — 45 min
7. **CLI integration** — 30 min
8. **Tests** — 60 min
9. **Verification** — 30 min

**Total estimated time**: ~5 hours

---

## STOP — Awaiting Approval

This plan does not modify Phase 1 functionality. The existing `Agent` class and all current tests remain intact. The Phase 2 implementation adds new modules that provide the autonomous loop foundation.
