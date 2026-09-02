# EVORA — PHASE 6: SELF-IMPROVEMENT SYSTEM

## Overview

Phase 6 adds a controlled self-improvement system that allows EVORA to analyze
its own codebase, identify weaknesses, propose improvements, and apply them under
strict CREATOR authority and approval workflows.

All new code lives in `evora/self_improve.py`. Backward compatibility is preserved
(336 existing tests still pass; 51 new Phase 6 tests added).

## New Files

| File | Purpose |
|------|---------|
| `evora/self_improve.py` | Core self-improvement module (722 lines) |
| `tests/test_self_improve.py` | 51 tests for Phase 6 functionality |
| `docs/PHASE6.md` | This document |

## Modified Files

| File | Change |
|------|--------|
| `evora/tools.py` | Added `SelfImproveTool` registration in `ToolRegistry` |
| `evora/cli.py` | Added `evora status` command; wired `ToolRegistry` to use identity_service and approval_system |

---

## Architecture

```
ImprovementPlanner → ImprovementProposal → approval → ChangeValidator → ImprovementHistory
```

### Classes

| Class | Responsibility |
|-------|---------------|
| `ImprovementStatus` | Enum: PENDING, APPROVED, RUNNING, SUCCESS, FAILED, REJECTED |
| `ImprovementProposal` | Dataclass describing a proposed change (title, description, files, benefit, risk) |
| `ImprovementRecord` | Full record with proposal + status + validation results + history_id |
| `ImprovementHistory` | Append-only persistent store for improvement records (JSON files) |
| `ChangeValidator` | Validates paths (workspace boundary), scans for secrets, runs pre/post validation |
| `ImprovementPlanner` | Scans `evora/` and `tests/` for weaknesses (TODOs, bare excepts, long functions) |
| `SelfImproveTool` | Tool exposing `analyze`, `propose`, `apply`, `history` actions |

---

## Actions

### `analyze`
Scan the codebase for weaknesses. Returns a report of files scanned,
TODOs/FIXMEs found, bare except blocks, and long functions (>30 lines).

### `propose`
Generate improvement proposals based on self-analysis. Returns a list
of `ImprovementProposal` objects with titles, descriptions, and risk assessments.

### `apply` (CREATOR only)
Apply a specific improvement to a file. Requires:
1. CREATOR authority (`identity_service.require_authority("enable_self_modification")`)
2. Approval system available
3. File path within workspace
4. No secrets in new content
5. `old_string` found exactly once in file

Flow:
1. Pre-validation (path + secret scan)
2. Approval prompt (unless auto-approve)
3. Apply change
4. Post-validation (syntax check)
5. Test suite run
6. Rollback on any failure

### `history`
List all improvement records with status icons and summary statistics.

---

## Safety Boundaries

- **CREATOR authority required** for `apply` action
- **Workspace boundary enforced**: `ChangeValidator.validate_file_path()` raises `PermissionError` for paths outside workspace
- **Secret scanning** before and after changes (API keys, passwords, tokens)
- **Rollback on failure**: post-validation or test failures automatically revert the change
- **Immutable history**: append-only store, records never deleted
- **No silent self-modification**: every change requires explicit approval

---

## CLI Integration

### `evora status`
Displays system status including:
- Workspace and Python version
- Current identity and authority level
- Creator identity (if configured)
- Available model providers (OpenAI, Anthropic, Ollama)
- Memory count
- Phase 6 improvement statistics (total proposals, success rate, status breakdown)

Example output:
```
============================================================
  EVORA System Status
============================================================

  Workspace:      C:\Users\nic\Desktop\EVORA PROJECT
  Python:         3.13.14

  Identity:
    Current:       Guest
    Authority:     guest
    Creator:       Not configured

  Model Providers:
    OpenAI:        Not installed
    Anthropic:     Not installed
    Ollama:        Available
    Config model:  llama3
    Provider:      auto-select

  Memory:
    Memories:      0
    Directory:     .evora/memory

  Self-Improvement (Phase 6):
    Total proposals: 0
    Success rate:    0%
```

---

## Test Results

### Phase 6 Tests (51 new tests)

| Category | Tests | Status |
|----------|-------|--------|
| `TestImprovementStatus` | 2 | All pass |
| `TestImprovementProposal` | 4 | All pass |
| `TestImprovementRecord` | 4 | All pass |
| `TestImprovementHistory` | 8 | All pass |
| `TestChangeValidator` | 8 | All pass |
| `TestImprovementPlanner` | 5 | All pass |
| `TestSelfImproveTool` | 20 | All pass |

### Full Suite

```
387 passed, 3 warnings in 51.19s
```

---

## Configuration

No new configuration required. Phase 6 uses existing:
- `workspace_dir` from config
- `identity_dir` for CREATOR authority checks
- `approval_system` for interactive/non-interactive approval flows

---

## Future Enhancements

- Integration with LLM for automated weakness analysis (currently rule-based)
- Batch proposal application with dependency ordering
- Rollback history for undoing multiple changes
- Diff preview before applying changes
- Metrics dashboard for improvement trends
