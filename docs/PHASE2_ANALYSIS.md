# EVORA Phase 2 Analysis

## What Already Exists

### 1. Core Modules (evora/)

#### `model.py` — Model Abstraction Layer
- **Providers**: `OpenAIProvider`, `AnthropicProvider` (both fully implemented)
- **Abstract base**: `ModelProvider` (ABC with `name()`, `model()`, `chat()`, `chat_stream()`, `close()`)
- **Data types**: `Message`, `Role`, `ToolSpec`, `ToolCall`, `ToolResult`, `Usage`, `ModelResponse`, `ChatRequest`
- **Registry**: `ModelManager` — registers providers, manages active provider, delegates `chat()`/`chat_stream()`
- **Mock support**: CLI has `MockModelProvider` for testing without API keys

#### `planner.py` — Plan Generation
- **Data types**: `PlanStep` (id, name, description, action_type, action_args, depends_on, estimated_effort), `Plan` (title, description, steps, raw_output)
- **Planner class**: Uses AI model to decompose requests into structured plan steps
- **Parser**: `_parse_plan()` — robust JSON parsing with fallback plan
- **Formatter**: `format_plan()` — human-readable plan display
- **Action types**: `create_file`, `edit_file`, `run_command`, `read_file`, `run_tests`, `analyze`, `create_directory`

#### `tools.py` — Tool System
- **8 tools implemented**: `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `CreateDirTool`, `ListDirTool`, `SearchFilesTool`, `SearchContentTool`, `ExecuteCommandTool`, `RunTestsTool`
- **ToolResult**: success, output, error, data (serializable)
- **ToolRegistry**: Registers all tools, `execute()` dispatcher, `get_specs()` for LLM consumption
- **Permission integration**: Each tool checks `PermissionManager` before executing
- **Note**: `RunTestsTool` delegates to `ExecuteCommandTool` (code smell)

#### `security.py` — Permission System
- **PermissionLevel**: SAFE, ASK, DANGEROUS
- **Command classification**: Pattern-based — `DANGEROUS_PATTERNS` (rm -rf, dd, shutdown, etc.), `ASK_PATTERNS` (pip install, chmod, sudo, etc.)
- **Path checking**: `check_workspace_path()` — restricts file operations to workspace
- **File write/delete checking**: `check_file_write()`, `check_file_delete()`
- **Timeouts**: `check_command_timeout()` — auto-detects longer timeouts for build/test commands
- **Code smell**: `check_workspace_path()` has redundant nested try/except blocks (lines 99-143)

#### `approval.py` — Approval System
- **ApprovalDecision**: APPROVE, REJECT, MODIFY, CANCEL, EXPLAIN
- **ApprovalRequest**: prompt, level, context, options
- **Callback support**: `register_callback()` for programmatic approval
- **Auto-approve**: Configurable via `auto_approve` flag
- **Interactive mode**: Uses `input()` for CLI prompts
- **Missing**: No ASK/MODIFY/EXPLAIN callbacks for commands (only `approve_plan` and `approve_command` have callbacks)

#### `memory.py` — Memory System
- **TaskEntry**: id, request, plan, steps, result, status, timestamp, elapsed, errors, memories
- **ProjectMemory**: project_name, workspace_dir, notes, conventions, dependencies, learned
- **MemoryStore**: File-based JSON storage for tasks, projects, global memory
- **Search**: `search_memories()` — cross-references tasks and projects
- **Memory facade**: Combines store, project cache, task management

#### `analyzer.py` — Project Analyzer
- **AnalysisResult**: Comprehensive project metadata (languages, frameworks, dependencies, entry points, config files, test files, git info, build system, test command)
- **Detection**: Languages (by extension), frameworks (by config files), dependencies (go.mod, requirements.txt, pyproject.toml, package.json), git status
- **Convention extraction**: Framework detection, build system identification

#### `agent.py` — Agent Loop (Phase 1)
- **AgentStatus**: IDLE, PLANNING, AWAITING_APPROVAL, EXECUTING, TESTING, FIXING, COMPLETED, FAILED, CANCELLED
- **AgentConfig**: max_retries=3, retry_delay=2.0, command_timeout=60, auto_approve=False
- **Workflow**: `run()` → PLAN → ASK → CODE → TEST → FIX → REPORT → MEMORY
- **Step execution**: `_execute_step()` dispatches by `action_type` to `_tool_*` methods
- **Test/fix loop**: `_fix()` — bounded retry with plan regeneration
- **Missing stages**: UNDERSTAND, ANALYZE, DECIDE, OBSERVE, EVALUATE, REASON

#### `cli.py` — CLI Entry Point
- **Commands**: `run`, `plan`, `analyze`, `config`, `memory`
- **Mock provider**: `MockModelProvider` class defined inline
- **`_build_model_manager()`**: Only registers OpenAI or Mock (Anthropic not wired up)

#### `logger.py` — Structured Logging
- **Stage enum**: PLAN, ASK, CODE, TEST, FIX, SUCCESS, ERROR, INFO, WARN
- **Colored output**: colorama-based with stage-specific colors
- **Emojis**: Stage-specific emoji prefixes

### 2. Tests (tests/)

| Test File | Tests | Coverage |
|-----------|-------|----------|
| `test_agent.py` | 5 | Agent creation, run, memory save |
| `test_approval.py` | 5 | Approval decisions, auto-approve, callbacks |
| `test_config.py` | 4 | Config defaults, env overrides |
| `test_logger.py` | 5 | Logger creation, levels, stages, colors |
| `test_memory.py` | 18 | TaskEntry, ProjectMemory, MemoryStore, Memory |
| `test_model.py` | 14 | Types, ModelManager, OpenAI, Anthropic providers |
| `test_planner.py` | 7 | PlanStep, Plan, Planner parsing |
| `test_security.py` | 7 | Permission checking, command safety, timeouts |
| `test_tools.py` | 13 | All tools, ToolRegistry |
| **Total** | **78** | (81 total with anthropic tests now passing) |

---

## What's Missing for Phase 2

### 1. Task State Abstraction
- **Current**: `TaskEntry` in memory.py tracks basic task metadata
- **Missing**: A comprehensive `TaskState` that captures the full autonomous loop context:
  - Current objective, current step, observations, decisions, retry counts
  - Completion criteria evaluation
  - Next proposed action
  - Full serialization for checkpointing

### 2. Decision/Reasoning Layer
- **Current**: The agent has hardcoded stages (`_plan`, `_ask`, `_code`, `_test`, `_fix`, `_report`)
- **Missing**: A `DecisionEngine` that dynamically determines the next action based on:
  - Current state, observations, errors, available tools, completion criteria
  - Returns structured `Decision` objects (action, tool, arguments, confidence, risk, requires_approval)

### 3. Observation Layer
- **Current**: Tool results are consumed immediately but not stored as structured observations
- **Missing**: An `Observation` system that captures and catalogs all action results:
  - File created, file modified, command succeeded/failed, test passed/failed
  - Observations feed back into task state for reasoning

### 4. Evaluation Layer
- **Current**: Binary success/failure after step execution
- **Missing**: An `Evaluator` that assesses whether actions moved the task toward completion:
  - SUCCESS, PROGRESS, FAILURE, NEEDS_INFORMATION, NEEDS_USER_APPROVAL, BLOCKED

### 5. Dynamic Loop
- **Current**: Hardcoded sequence (PLAN → ASK → CODE → TEST → FIX → REPORT)
- **Missing**: A loop that calls `DECIDE → ACT → OBSERVE → EVALUATE → DECIDE AGAIN`
  - The decision engine determines whether to plan, execute, test, fix, or report

### 6. Extended Logger Stages
- **Current**: PLAN, ASK, CODE, TEST, FIX, SUCCESS, ERROR, INFO, WARN
- **Missing**: UNDERSTAND, ANALYZE, DECIDE, OBSERVE, EVALUATE, REASON

### 7. Approval System Gaps
- Missing MODIFY, CANCEL, EXPLAIN handling for command-level approvals
- No structured approval data flow (only callback-based)

### 8. Test Coverage Gaps
- No tests for dynamic decision-making
- No tests for observation → evaluation → re-decision loop
- No tests for retry limit enforcement
- No tests for cancellation
- No tests for evaluation outcomes
