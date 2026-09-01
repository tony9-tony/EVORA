# EVORA — PHASE 3 IMPLEMENTATION PLAN

## Overview

Implement a proper memory system with creator identity, long-term memory,
retrieval, user-controlled operations, and secret filtering.

All new code lives in `evora/` modules. Backward compatibility is preserved
(186 existing tests must still pass).

---

## Implementation Order

### Step 1: Identity & Authority System

**Files**: `evora/identity.py` (new), `evora/config.py` (modify), `tests/test_identity.py` (new)

**Dataclasses / Enums**:

```python
class AuthorityLevel(str, Enum):
    CREATOR = "creator"   # Full system control (config, memory mgmt, policy, self-improvement)
    ADMIN = "admin"       # Project-scoped admin (memory, permissions, but not global config)
    USER = "user"         # Standard user — can request tasks, limited memory ops
    GUEST = "guest"       # Read-only — view memory, no modifications

@dataclass
class Identity:
    id: str                    # UUID or username
    name: str                  # Display name
    authority: AuthorityLevel
    created_at: str
    # No password storage here — identity is local/config-based in Phase 3
    # Authentication integration is future work

@dataclass
class AuthorityRule:
    """A rule that defines what someone at this level can do."""
    action: str                 # e.g. "clear_project_memory", "set_creator", "modify_config"
    requires: AuthorityLevel   # Minimum level required
```

**Classes**:

```python
class IdentityStore:
    """Protected storage for identities. Uses JSON behind an abstraction."""

    def __init__(self, identity_dir: str): ...
    def load_identity(self, identity_id: str) -> Optional[Identity]: ...
    def save_identity(self, identity: Identity) -> None: ...
    def get_current(self) -> Identity: ...        # reads from protected config
    def set_current(self, identity: Identity) -> None: ...  # CREATOR-only
    def get_creator() -> Identity: ...            # default/first CREATOR identity
    def is_authorized(self, identity: Identity, action: str) -> bool: ...
    def list_identities() -> list[Identity]: ...

class IdentityService:
    """Runtime identity + authority checks for the agent loop."""

    def __init__(self, store: IdentityStore, logger: Logger): ...
    def current_identity(self) -> Identity: ...
    def check_authority(self, action: str, level: AuthorityLevel) -> bool: ...
    def require_authority(self, action: str) -> None: ...  # raises if unauthorized
```

**Config changes**: Add `identity_dir` to `Config` (defaults to `~/.evora/identity`).

**Secret**: Do NOT hardcode username. Identity file is config-controlled.

---

### Step 2: Long-Term Memory

**File**: `evora/memory.py` (extend existing)

**New dataclass**:

```python
@dataclass
class LongTermMemoryEntry:
    id: str                      # UUID
    memory_type: str            # "preference", "decision", "learning", "instruction"
    content: str                # The actual memory content
    created_at: float           # unix timestamp
    importance: float           # 0.0 – 1.0
    last_accessed: float        # unix timestamp
    access_count: int           # how many times retrieved
    project: Optional[str] = None    # scoped to project or None for global
    pinned: bool = False        # if True, never forget / always retrieve
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict: ...
    @classmethod
    def from_dict(cls, data: dict) -> "LongTermMemoryEntry": ...
```

**Extension to `MemoryStore`**:

```python
class MemoryStore:
    # ... existing methods unchanged ...

    # New subdirectory: longterm/
    def _longterm_dir(self) -> Path: ...

    def save_ltm_entry(self, entry: LongTermMemoryEntry) -> None: ...
    def load_ltm_entry(self, entry_id: str) -> Optional[LongTermMemoryEntry]: ...
    def list_ltm_entries(self, project: Optional[str] = None, limit: int = 100) -> list[LongTermMemoryEntry]: ...
    def delete_ltm_entry(self, entry_id: str) -> bool: ...
    def search_ltm(self, query: str, project: Optional[str] = None, limit: int = 20) -> list[LongTermMemoryEntry]: ...
```

**Extension to `ProjectMemory`** — add missing fields from the plan:

```python
@dataclass
class ProjectMemory:
    # ... existing fields ...
    languages: list[str] = field(default_factory=list)       # e.g. ["Python", "YAML"]
    frameworks: list[str] = field(default_factory=list)     # e.g. ["FastAPI", "Docker"]
    important_files: list[str] = field(default_factory=list)
    build_commands: list[str] = field(default_factory=list)
    architecture_notes: list[str] = field(default_factory=list)
    known_issues: list[str] = field(default_factory=list)
    previous_completed_tasks: list[str] = field(default_factory=list)
    important_technical_decisions: list[str] = field(default_factory=list)
    # test_commands, entry_points already exist
```

---

### Step 3: Secret Filtering

**File**: `evora/memory.py` (add `MemoryFilter`)

```python
class MemoryFilter:
    """Filters sensitive data before persisting to memory."""

    SENSITIVE_PATTERNS = [
        # API keys
        re.compile(r"(sk-[a-zA-Z0-9]{20,})"),
        re- pattern(r"(api[_-]?key\s*[:=]\s*'[^']+'|\"[^\"]+\")", re.IGNORECASE),
        # Passwords
        re.compile(r"(password\s*[:=]\s*'[^']+'|\"[^\"]+\")", re.IGNORECASE),
        re.compile(r"(passwd\s*[:=]\s*'[^']+'|\"[^\"]+\")", re.IGNORECASE),
        # Tokens
        re.compile(r"(token\s*[:=]\s*'[^']+'|\"[^\"]+\")", re.IGNORECASE),
        # Secrets
        re.compile(r"(secret\s*[:=]\s*'[^']+'|\"[^\"]+\")", re.IGNORECASE),
        # Private keys
        re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
        re.compile(r"(aws_access_key_id|aws_secret_access_key)", re.IGNORECASE),
        re.compile(r"(GH[pousr]_[A-Za-z0-9]{36})", re.IGNORECASE),
        re.compile(r"(github_pat_[A-Za-z0-9_]{22,})"),
    ]

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Replace secrets with [REDACTED]."""
        redacted = text
        for pattern in cls.SENSITIVE_PATTERNS:
            redacted = pattern.sub("[REDACTED]", redacted)
        return redacted

    @classmethod
    def contains_secrets(cls, text: str) -> bool:
        """Check if text contains any sensitive patterns."""
        return any(p.search(text) for p in cls.SENSITIVE_PATTERNS)
```

All `save` paths must call `MemoryFilter.sanitize()` before writing.

---

### Step 4: Memory Retrieval

**File**: `evora/memory.py` (add `MemoryRetriever`)

```python
@dataclass
class RetrievalResult:
    entry: LongTermMemoryEntry
    score: float

class MemoryRetriever:
    """Selects relevant memories based on query criteria."""

    def __init__(self, store: MemoryStore, logger: Optional[Logger] = None):
        self.store = store
        self.logger = logger

    def retrieve(
        self,
        goal: str = "",
        project: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        limit: int = 10,
        include_pinned: bool = True,
    ) -> list[RetrievalResult]:
        """Retrieve relevant memories with relevance scoring.

        Scoring factors (simple, no vector DB):
        - Keyword overlap with goal (30%)
        - Project match (25%)
        - Type relevance (20%)
        - Recency — updated_at (15%)
        - Importance field (10%)
        - Pinned entries always returned (top)
        """

    def _score(
        self,
        entry: LongTermMemoryEntry,
        goal: str,
        project: Optional[str],
        memory_types: Optional[list[str]],
    ) -> float:
        """Compute relevance score 0.0–1.0."""
```

---

### Step 5: Memory Service (User-Controlled Operations)

**File**: `evora/memory.py` (add `MemoryService`)

```python
class MemoryService:
    """High-level user-controlled memory operations.

    Bridges task memory, project memory, long-term memory, and identity.
    """

    def __init__(
        self,
        memory: Memory,          # existing facade
        store: MemoryStore,      # underlying store
        identity: IdentityService,
        logger: Logger,
    ): ...

    # --- remember ---
    def remember(
        self,
        content: str,
        memory_type: str = "preference",
        importance: float = 0.5,
        project: Optional[str] = None,
        tags: Optional[list[str]] = None,
        pinned: bool = False,
    ) -> LongTermMemoryEntry:
        """Explicitly store a memory (creator/admin only for some types)."""

    # --- forget ---
    def forget(self, entry_id: str) -> bool:
        """Delete a long-term memory entry by ID."""

    def forget_all(self, memory_type: Optional[str] = None, project: Optional[str] = None) -> int:
        """Bulk-delete memory entries."""

    # --- what do you remember ---
    def list_memories(
        self,
        project: Optional[str] = None,
        memory_type: Optional[str] = None,
        limit: int = 50,
    ) -> list[LongTermMemoryEntry]: ...

    # --- task lifecycle ---
    def archive_task_outcome(self, state: "TaskState") -> None:
        """Save task decisions, errors, test_results as long-term memory.
        Applies secret filtering."""

    def retrieve_relevant(self, goal: str, project: Optional[str] = None) -> list[RetrievalResult]:
        """Wrapper around MemoryRetriever.retrieve()."""

    # --- project memory management ---
    def update_project_memory(self, state: "TaskState") -> None:
        """Update project memory with task learnings (without leaking other projects)."""

    def clear_task_memory(self) -> int:
        """Delete all task entries for the current project."""

    def clear_project_memory(self) -> bool:
        """Delete all project + long-term memory for the current project."""
```

---

### Step 6: Integrate with AutonomousAgent

**File**: `evora/autonomous.py` (modify)

Add `memory_service: MemoryService` and `identity_service: IdentityService` to `AutonomousAgent.__init__`.

**Changes to the `run()` loop**:

```python
async def run(self, request: str, project_context: Optional[dict] = None) -> str:
    start_time = time.time()

    # 1. Check identity & authority (fail fast if unauthorized)
    # 2. Retrieve relevant long-term memory before acting
    relevant = self.memory_service.retrieve_relevant(state.goal, project=state.workspace)
    if relevant:
        state.project_context["relevant_memories"] = [r.to_dict() for r in relevant]
        self.logger.memory(f"Retrieved {len(relevant)} relevant memories")

    # 3. Create task state (unchanged from current)
    state = TaskState(...)

    # 4. Existing loop (unchanged)
    while not state.is_complete and not state.is_failed and not state.is_cancelled:
        decision = self.decision_engine.decide_next(state)
        ...

    # 5. After loop: archive outcome to long-term memory
    # 6. Update project memory
    # 7. Save task memory (existing _save_memory)
    state.elapsed = time.time() - start_time
    return self._report(state)
```

**Backward compatibility**: `AutonomousAgent.__init__` gets new optional params (`memory_service`, `identity_service`). If not provided, they default to `None` and memory features are gracefully skipped (existing tests pass).

---

### Step 7: CLI Integration

**File**: `evora/cli.py` (modify)

New subcommands:

```
evora remember "Always use pytest-cov for coverage" --type preference --importance 0.8
evora forget <entry_id>
evora memories [--type preference] [--project myproject]   # alias: "what do you remember"
evora whoami                                               # show current identity + authority
evora identity set --name "Tony" --authority creator      # CREATOR only
evora memory --long-term                                  # extended memory view
```

The existing `evora memory` command is enhanced to also list long-term entries.

CLI construction changes:
- Add `IdentityStore` and `IdentityService` to `async_run()`
- Add `MemoryService` wrapping the existing `Memory` instance
- Pass both to `AutonomousAgent`

---

### Step 8: Tests

**New test files** (following existing mock patterns):

| File | Tests | Covers |
|------|-------|--------|
| `tests/test_identity.py` | ~12 | AuthorityLevel enum, Identity creation, IdentityStore save/load, identity set/get current, is_authorized checks, unauthorized admin rejection, creator protection (no username comparison) |
| `tests/test_memory_phase3.py` | ~15 | LongTermMemoryEntry CRUD, MemoryStore long-term methods, MemoryFilter secret sanitization (API keys, passwords, tokens, private keys), MemoryRetriever scoring, MemoryService remember/forget/list, project isolation |
| `tests/test_autonomous_memory.py` | ~10 | Integration: agent retrieves memory before acting, archives outcome after, saves long-term memory, respects authority |

**Test principles**:
- Use existing mock model provider pattern
- Use `tmp_path` for isolated storage
- No real API keys needed
- All 186 existing tests must still pass

---

## Files to Create

| File | Purpose |
|------|---------|
| `evora/identity.py` | AuthorityLevel, Identity, IdentityStore, IdentityService |
| `tests/test_identity.py` | Identity & authority tests |
| `tests/test_memory_phase3.py` | Long-term memory, filtering, retrieval, service tests |
| `tests/test_autonomous_memory.py` | Integration tests for memory + autonomous agent |
| `docs/PHASE3_ANALYSIS.md` | This analysis |

## Files to Modify

| File | Changes |
|------|---------|
| `evora/memory.py` | Add LongTermMemoryEntry, extend ProjectMemory, add MemoryFilter, MemoryRetriever, MemoryService, extend MemoryStore |
| `evora/config.py` | Add identity_dir to Config + load_config |
| `evora/autonomous.py` | Add IdentityService + MemoryService integration, retrieval before acting, archive after completion |
| `evora/cli.py` | Add `remember`, `forget`, `memories`, `whoami`, `identity` subcommands; wire up new services |

## Files NOT Modified (backward compatibility)

- `evora/model.py`, `evora/tools.py`, `evora/security.py`, `evora/approval.py`
- `evora/analyzer.py`, `evora/planner.py`, `evora/agent.py`, `evora/logger.py`
- `requirements.txt`, `pyproject.toml`
- All existing test files
- PesaTrack (never touched)

---

## Timeline

1. **Identity & Authority** — 30 min
2. **Long-term memory dataclass + store** — 30 min
3. **Secret filtering** — 20 min
4. **Memory retrieval** — 30 min
5. **Memory service (user ops)** — 30 min
6. **Autonomous agent integration** — 30 min
7. **CLI integration** — 30 min
8. **Tests** — 60 min
9. **Verification** — 30 min

**Total estimated time**: ~4.5 hours

---

## Test Coverage

| Test Area | Test File | Tests |
|-----------|-----------|-------|
| Task memory persistence | `test_memory_phase3.py` | `test_save_and_load_ltm_entry`, `test_list_ltm_entries`, `test_delete_ltm_entry` |
| Project memory separation | `test_memory_phase3.py` | `test_project_isolation_prevents_leak`, `test_project_memory_extended_fields` |
| Long-term memory persistence | `test_memory_phase3.py` | `test_long_term_memory_save_load`, `test_ltm_restart_persistence` |
| Memory retrieval | `test_memory_phase3.py` | `test_retriever_scores_relevance`, `test_retriever_pinned_always_returned`, `test_retriever_project_filter` |
| Remember | `test_memory_phase3.py` | `test_remember_creates_entry`, `test_remember_assigns_importance` |
| Forget | `test_memory_phase3.py` | `test_forget_deletes_entry`, `test_forget_all_by_type` |
| Creator identity | `test_identity.py` | `test_creator_identity_is_protected`, `test_no_username_comparison` |
| Authority levels | `test_identity.py` | `test_creator_can_do_all`, `test_guest_cannot_do_admin`, `test_user_cannot_do_creator` |
| Unauthorized admin action rejection | `test_identity.py` | `test_unauthorized_action_raises`, `test_guest_cannot_clear_memory` |
| Secret filtering | `test_memory_phase3.py` | `test_api_key_redacted`, `test_password_redacted`, `test_token_redacted`, `test_private_key_redacted`, `test_github_token_redacted` |
| Restart persistence | `test_memory_phase3.py` | `test_ltm_survives_restart`, `test_task_memory_survives_restart` |
| Integration with autonomous task completion | `test_autonomous_memory.py` | `test_archive_task_outcome`, `test_retrieve_before_acting`, `test_save_long_term_after_completion` |
| No regression of Phase 1/2 | All existing tests | 186 tests must still pass |
