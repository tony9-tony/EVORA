# EVORA — PHASE 3 ANALYSIS

## Status
- **Commit**: `a041714` (`feat: implement autonomous agent engine`)
- **Tests**: 186/186 passing
- **Phase 2**: Complete

---

## 1. Current State of Memory

### Existing Memory Stack (`evora/memory.py`)

| Class | Role | Status |
|-------|------|--------|
| `TaskEntry` | Per-task record: request, plan, steps, result, errors, memories | Phase 1 — basic, no TaskState integration |
| `ProjectMemory` | Per-project: notes, conventions, dependencies, entry_points, test_commands, learned | Phase 1 — minimal, missing many required fields |
| `MemoryStore` | JSON file-based storage: tasks/, projects/, global/ dirs | Phase 1 — functional but no secret filtering, no retrieval scoring |
| `Memory` | Facade: `store`, `project` property, `create_task`, `save_task`, `load_task`, `search` | Phase 1 — thin wrapper, no long-term memory abstraction |

### What the autonomous agent does with memory today (`evora/autonomous.py`)

- **On task start**: `_save_memory(state)` — creates/loads TaskEntry, syncs from TaskState
- **On task end**: `_save_memory(state)` — finishes TaskEntry, calls `memory.save_project()`
- **Does NOT**: retrieve relevant memory before/during execution, update project memory mid-run, remember decisions/outcomes as long-term memory

### CLI memory usage (`evora/cli.py`)

- `evora memory` — lists recent TaskEntry records only (no project memory, no long-term memory, no forget)
- No `remember`, `forget`, or `what do you remember` commands

---

## 2. What Can Be Reused

### Reusably solid
| Component | Reused as-is | Notes |
|-----------|-------------|-------|
| `MemoryStore` directory structure (tasks/, projects/, global/ → add longterm/) | ✓ | JSON file-based, replaceable |
| `TaskEntry` dataclass + to_dict/from_dict pattern | ✓ | Extend with Phase 2 TaskState fields |
| `ProjectMemory` dataclass pattern | ✓ | Extend with missing fields (languages, frameworks, etc.) |
| `Memory` facade class | ✓ | Extend with retrieval, long-term, identity |
| `PermissionLevel` enum (SAFE, ASK, DANGEROUS) | ✓ | Reused in identity/authority checks |
| `Config` / `load_config()` | ✓ | Add `creator_file` path, identity config |
| `PermissionManager` workspace restriction | ✓ | Reuse for memory isolation |
| Logger `Stage` enum | ✓ | Add MEMORY stage if needed |

### Patterns to follow
- Dataclass + `to_dict()` / `from_dict()` (as in `TaskState`, `Observation`, `Decision`)
- JSON file storage with `_safe_name()` sanitization
- Mock model provider pattern from `test_autonomous.py` / `test_agent.py`
- `run_async()` helper pattern in tests

---

## 3. What Must Change

### Missing entirely (new modules required)
| Gap | Required |
|-----|----------|
| **Creator identity system** | No identity concept exists. Need `Identity`/`IdentityStore`, `AuthorityLevel` enum (CREATOR, ADMIN, USER, GUEST), protected config file |
| **Long-term memory abstraction** | Only "global" key-value exists. Need a proper `LongTermMemory` category with importance scoring, pinning, recency tracking |
| **Memory retrieval** | `search_memories()` does a flat text search. Need a `MemoryRetriever` with relevance scoring (recency, importance, project-match, goal-match) |
| **Secret/sensitive-value filtering** | No filtering at all. Need `MemoryFilter` / `SecretScanner` to block API keys, passwords, tokens from being stored |
| **User-controlled memory operations** | No `remember`, `forget`, `what do you remember` commands. Need `MemoryService` with `remember()`, `forget()`, `list_memories()` |
| **Project memory isolation** | `ProjectMemory` is keyed by name but there's no enforcement that memories from Project A don't leak into Project B during retrieval |
| **Decision/outcome archival** | Phase 2 `TaskState` collects decisions and test_results, but they're not persisted as long-term memory after task completion |
| **Pre-action memory retrieval** | The autonomous loop doesn't call any retrieval before acting — it only saves at the end |

### Must be modified
| File | Change needed |
|------|---------------|
| `evora/memory.py` | Add `LongTermMemory` dataclass, extend `ProjectMemory` fields, add `MemoryRetriever`, `MemoryService`, secret filtering, `IdentityStore` |
| `evora/autonomous.py` | Call memory retrieval before acting; archive task outcome (decisions, observations, test results) as long-term memory after completion |
| `evora/cli.py` | Add `evora remember`, `evora forget`, `evora whoami` subcommands; enhance `evora memory` to show project + long-term memory |
| `evora/config.py` | Add `creator_identity_file` / `identity_dir` path |
| `evora/task.py` | Optionally bridge `TaskState` ↔ `TaskEntry` for seamless memory persistence |

### Files NOT touched (per Phase 2 plan and rules)
- `evora/model.py`, `evora/tools.py`, `evora/security.py`, `evora/approval.py`, `evora/analyzer.py`, `evora/planner.py` — no changes needed
- `requirements.txt`, `pyproject.toml` — no new external dependencies (JSON + stdlib)
- PesaTrack — must never be touched

---

## 4. Test Audit

| Test File | Tests | Status |
|-----------|-------|--------|
| `test_memory.py` | 13 | Tests basic TaskEntry/ProjectMemory/MemoryStore — must not break |
| `test_autonomous.py` | 15 | Uses `Memory` in agent construction — integration tests must still pass |
| `test_agent.py` | 5 | Uses `Memory` — must not break |
| All other test files | 153 | No memory dependency |

**Total: 186 tests** — all must continue passing after Phase 3.

---

## 5. Architecture Observations

### Current Memory Flow (Phase 2)
```
AutonomousAgent.run(request)
  → _save_memory(state)          # creates TaskEntry at start
  → decision_engine.decide_next(state)
  → _act(state, decision)
  → _observe(state, result)
  → _evaluate(state, observation)
  → _reason(state, decision, eval_result, observations)
  → ... (loop)
  → _report(state)
  → _save_memory(state)          # finishes TaskEntry at end
```

### Target Memory Flow (Phase 3 design)
```
AutonomousAgent.run(request)
  → identity_service.get_current_identity()   # who is running?
  → memory_service.retrieve_relevant(goal, project, types=[...])
    # inject relevant memory into TaskState.project_context
  → task_state = TaskState(...)
  → memory_service.save_task(task_state)       # task memory at start
  → [autonomous loop: decide → act → observe → evaluate → reason]
  → memory_service.archive_task_outcome(task_state)
    # decisions, errors, test_results → long-term memory (with secret filtering)
  → memory_service.update_project_memory(task_state)
  → memory_service.save_task(task_state)       # task memory at end
```

---

## 6. Key Design Considerations

1. **Identity must not be a chat message**: Creator identity must be stored in a separate protected file (e.g. `~/.evora/identity.json`) with a clear schema, not parsed from conversational text.

2. **Memory must be modular**: The `Memory` facade class is the integration point. New capabilities (retrieval, filtering, identity) should be layered on top, not rewrite the store.

3. **No new heavy dependencies**: Stick to JSON files + stdlib. SQLite is acceptable if JSON proves too simple, but JSON is preferred for simplicity.

4. **Secret filtering is critical**: Any memory archive path must filter API keys, passwords, tokens before persisting. A regex-based `SecretScanner` is sufficient for Phase 3.

5. **Project isolation**: Memory retrieval must be scoped to the current workspace. The `project_name` in `Memory` already provides a key; retrieval must enforce this boundary.

6. **Backward compatibility**: All 186 existing tests must pass. The `Memory` class must remain constructible with `Memory(memory_dir, project_name)` and support `store.list_tasks()`, `project.add_note()`, `project.add_learned()`, etc.
