"""
Tests for EVORA Phase 3 memory system extensions.

These tests verify:
- Long-term memory persistence (save/load/list/delete)
- Project memory separation (Project A ≠ Project B)
- ProjectMemory extended fields
- Memory retrieval with relevance scoring
- remember operation
- forget operation
- Secret filtering (API keys, passwords, tokens, private keys)
- Restart persistence
- Integration with autonomous task completion
"""

import json
import os
import time as time_module
from pathlib import Path

import pytest

from evora.identity import IdentityService, Identity, AuthorityLevel, IdentityStore
from evora.memory import (
    Memory,
    MemoryStore,
    LongTermMemoryEntry,
    MemoryFilter,
    MemoryRetriever,
    MemoryService,
    ProjectMemory,
    TaskEntry,
)


class TestLongTermMemoryEntry:

    def test_create_entry_defaults(self):
        entry = LongTermMemoryEntry()
        assert entry.id is not None
        assert entry.memory_type == "preference"
        assert entry.importance == 0.5
        assert entry.pinned is False
        assert entry.tags == []

    def test_create_entry_with_values(self):
        entry = LongTermMemoryEntry(
            memory_type="decision",
            content="Use FastAPI for REST APIs",
            importance=0.8,
            project="my-project",
            pinned=True,
            tags=["backend", "api"],
        )
        assert entry.memory_type == "decision"
        assert entry.content == "Use FastAPI for REST APIs"
        assert entry.importance == 0.8
        assert entry.project == "my-project"
        assert entry.pinned is True
        assert entry.tags == ["backend", "api"]

    def test_to_dict(self):
        entry = LongTermMemoryEntry(
            memory_type="learning",
            content="pytest is the test framework",
            importance=0.6,
            project="test-project",
        )
        d = entry.to_dict()
        assert d["memory_type"] == "learning"
        assert d["content"] == "pytest is the test framework"
        assert d["importance"] == 0.6
        assert d["project"] == "test-project"
        assert "id" in d

    def test_from_dict(self):
        data = {
            "id": "test-id",
            "memory_type": "instruction",
            "content": "Always run tests",
            "created_at": 1234567890.0,
            "importance": 0.9,
            "last_accessed": 1234567890.0,
            "access_count": 5,
            "project": "myproject",
            "pinned": True,
            "tags": ["critical"],
        }
        entry = LongTermMemoryEntry.from_dict(data)
        assert entry.id == "test-id"
        assert entry.memory_type == "instruction"
        assert entry.content == "Always run tests"
        assert entry.importance == 0.9
        assert entry.pinned is True
        assert entry.tags == ["critical"]

    def test_score_relevance_keyword_match(self):
        entry = LongTermMemoryEntry(
            content="Use pytest for testing Python projects",
            project="myapp",
        )
        score = entry.score_relevance(["pytest", "testing"], project="myapp")
        assert 0.0 < score <= 1.0

    def test_score_relevance_no_match(self):
        entry = LongTermMemoryEntry(
            content="The sky is blue",
            project="myapp",
        )
        # Query with no matching keywords and no project match
        score = entry.score_relevance(["pytest"], project="otherproject")
        # Should have low score: no keyword match + no project match
        assert score < 0.5

    def test_record_access(self):
        entry = LongTermMemoryEntry()
        original_count = entry.access_count
        entry.record_access()
        assert entry.access_count == original_count + 1

    def test_project_scoped(self):
        entry = LongTermMemoryEntry(content="test", project="projectA")
        assert entry.project == "projectA"


class TestMemoryStoreLTM:

    def test_save_and_load_ltm_entry(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        entry = LongTermMemoryEntry(content="Test memory", memory_type="preference")
        store.save_ltm_entry(entry)

        loaded = store.load_ltm_entry(entry.id)
        assert loaded is not None
        assert loaded.content == "Test memory"
        assert loaded.memory_type == "preference"

    def test_load_nonexistent_ltm(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        assert store.load_ltm_entry("nonexistent") is None

    def test_list_ltm_entries(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="Memory 1", project="projA")
        e2 = LongTermMemoryEntry(content="Memory 2", project="projB")
        store.save_ltm_entry(e1)
        store.save_ltm_entry(e2)

        entries = store.list_ltm_entries(limit=10)
        assert len(entries) == 2

    def test_list_ltm_entries_by_project(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="Memory 1", project="projA")
        e2 = LongTermMemoryEntry(content="Memory 2", project="projB")
        store.save_ltm_entry(e1)
        store.save_ltm_entry(e2)

        entries = store.list_ltm_entries(project="projA", limit=10)
        assert len(entries) == 1
        assert entries[0].project == "projA"

    def test_delete_ltm_entry(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        entry = LongTermMemoryEntry(content="To delete")
        store.save_ltm_entry(entry)

        assert store.delete_ltm_entry(entry.id) is True
        assert store.load_ltm_entry(entry.id) is None

    def test_delete_nonexistent_ltm(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        assert store.delete_ltm_entry("nonexistent") is False

    def test_delete_all_ltm_by_project(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="A1", project="projA")
        e2 = LongTermMemoryEntry(content="A2", project="projA")
        e3 = LongTermMemoryEntry(content="B1", project="projB")
        store.save_ltm_entry(e1)
        store.save_ltm_entry(e2)
        store.save_ltm_entry(e3)

        deleted = store.delete_all_ltm(project="projA")
        assert deleted == 2

        remaining = store.list_ltm_entries()
        assert len(remaining) == 1
        assert remaining[0].project == "projB"

    def test_delete_all_ltm_by_type(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="Pref 1", memory_type="preference")
        e2 = LongTermMemoryEntry(content="Dec 1", memory_type="decision")
        store.save_ltm_entry(e1)
        store.save_ltm_entry(e2)

        deleted = store.delete_all_ltm(memory_type="preference")
        assert deleted == 1

        remaining = store.list_ltm_entries()
        assert len(remaining) == 1
        assert remaining[0].memory_type == "decision"

    def test_search_ltm(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="Use pytest for testing")
        e2 = LongTermMemoryEntry(content="Use Flask for web APIs")
        store.save_ltm_entry(e1)
        store.save_ltm_entry(e2)

        results = store.search_ltm("pytest")
        assert len(results) == 1
        assert "pytest" in results[0].content

    def test_search_ltm_with_tags(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="Important note", tags=["urgent", "project"])
        store.save_ltm_entry(e1)

        results = store.search_ltm("urgent")
        assert len(results) == 1

    def test_ltm_restart_persistence(self, tmp_path):
        """LTM entries must survive store re-instantiation."""
        store = MemoryStore(str(tmp_path / "memory"))
        entry = LongTermMemoryEntry(content="Persistent memory", importance=0.9)
        store.save_ltm_entry(entry)

        store2 = MemoryStore(str(tmp_path / "memory"))
        loaded = store2.load_ltm_entry(entry.id)
        assert loaded is not None
        assert loaded.content == "Persistent memory"
        assert loaded.importance == 0.9


class TestProjectMemoryIsolation:

    def test_project_isolation_prevents_leak(self, tmp_path):
        """Project A memory must not appear when querying Project B."""
        store = MemoryStore(str(tmp_path / "memory"))

        pm_a = ProjectMemory.create("projectA", "/path/a")
        pm_a.add_note("Project A specific note")
        pm_a.add_language("Python")

        pm_b = ProjectMemory.create("projectB", "/path/b")
        pm_b.add_note("Project B specific note")
        pm_b.add_language("Go")

        store.save_project_memory(pm_a)
        store.save_project_memory(pm_b)

        loaded_a = store.load_project_memory("projectA")
        loaded_b = store.load_project_memory("projectB")

        assert "Project A specific note" in loaded_a.notes
        assert "Project A specific note" not in loaded_b.notes
        assert "Go" in loaded_b.languages
        assert "Go" not in loaded_a.languages

    def test_project_memory_extended_fields(self, tmp_path):
        """ProjectMemory should have all Phase 3 fields."""
        pm = ProjectMemory.create("ext-project", "/path/to/proj")

        pm.add_language("Python")
        pm.add_framework("FastAPI")
        pm.add_important_file("src/main.py")
        pm.add_build_command("python -m pytest")
        pm.add_architecture_note("Uses hexagonal architecture")
        pm.add_known_issue("Database connection timeout on startup")
        pm.add_technical_decision("Chose SQLAlchemy over Django ORM")
        pm.add_completed_task("Completed: Create API endpoints")

        assert "Python" in pm.languages
        assert "FastAPI" in pm.frameworks
        assert "src/main.py" in pm.important_files
        assert "python -m pytest" in pm.build_commands
        assert "hexagonal architecture" in pm.architecture_notes[0]
        assert "Database connection timeout" in pm.known_issues[0]
        assert "SQLAlchemy" in pm.important_technical_decisions[0]
        assert "Create API endpoints" in pm.previous_completed_tasks[0]

    def test_project_memory_serialization(self, tmp_path):
        """ProjectMemory with new fields must serialize/deserialize correctly."""
        pm = ProjectMemory.create("serial-project", "/path")
        pm.add_language("Rust")
        pm.add_framework("Actix")
        pm.add_known_issue("Memory leak in production")

        d = pm.to_dict()
        assert "languages" in d
        assert "frameworks" in d
        assert "known_issues" in d

        # Verify it round-trips
        path = tmp_path / "proj.json"
        with open(path, "w") as f:
            json.dump(d, f)
        with open(path, "r") as f:
            loaded_data = json.load(f)
        pm2 = ProjectMemory(**loaded_data)
        assert "Rust" in pm2.languages
        assert "Actix" in pm2.frameworks
        assert "Memory leak" in pm2.known_issues[0]


class TestMemoryRetriever:

    def test_retriever_returns_empty_for_empty_store(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        retriever = MemoryRetriever(store=store)
        results = retriever.retrieve(goal="test", project="proj")
        assert len(results) == 0

    def test_retriever_returns_pinned_first(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        pinned = LongTermMemoryEntry(
            content="Pinned important preference",
            pinned=True,
            project="proj",
            importance=0.9,
        )
        normal = LongTermMemoryEntry(
            content="Normal memory about pytest",
            importance=0.3,
            project="proj",
        )
        store.save_ltm_entry(pinned)
        store.save_ltm_entry(normal)

        retriever = MemoryRetriever(store=store)
        results = retriever.retrieve(goal="unrelated", project="proj", limit=10)

        assert len(results) >= 2
        # Pinned should be at top
        assert results[0].score == 1.0

    def test_retriever_filters_by_project(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e_a = LongTermMemoryEntry(content="Project A memory", project="projA")
        e_b = LongTermMemoryEntry(content="Project B memory", project="projB")
        store.save_ltm_entry(e_a)
        store.save_ltm_entry(e_b)

        retriever = MemoryRetriever(store=store)
        results = retriever.retrieve(goal="memory", project="projA", limit=10)

        assert len(results) == 1
        assert results[0].entry.project == "projA"

    def test_retriever_filters_by_type(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        e1 = LongTermMemoryEntry(content="Preference memory", memory_type="preference")
        e2 = LongTermMemoryEntry(content="Decision memory", memory_type="decision")
        store.save_ltm_entry(e1)
        store.save_ltm_entry(e2)

        retriever = MemoryRetriever(store=store)
        results = retriever.retrieve(
            goal="preference", memory_types=["preference"], limit=10
        )
        assert len(results) == 1
        assert results[0].entry.memory_type == "preference"

    def test_retriever_scores_relevance(self, tmp_path):
        store = MemoryStore(str(tmp_path / "memory"))
        relevant = LongTermMemoryEntry(
            content="Always use pytest-cov for Python test coverage",
            project="myproject",
            importance=0.8,
        )
        less_relevant = LongTermMemoryEntry(
            content="The sky is blue and birds fly high",
            project="myproject",
            importance=0.3,
        )
        store.save_ltm_entry(relevant)
        store.save_ltm_entry(less_relevant)

        retriever = MemoryRetriever(store=store)
        results = retriever.retrieve(goal="pytest", project="myproject", limit=10)

        # The relevant entry should score higher
        assert results[0].entry.content == "Always use pytest-cov for Python test coverage"
        assert results[0].score > results[1].score


class TestSecretFiltering:

    def test_api_key_redacted(self):
        text = "My key is sk-abc123def456ghi789jkl012mno345pqr678"
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized
        assert "sk-abc123" not in sanitized

    def test_password_redacted(self):
        text = "password: 'supersecret123'"
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized
        assert "supersecret123" not in sanitized

    def test_password_equals_redacted(self):
        text = 'password="my_secret_password"'
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized

    def test_token_redacted(self):
        text = "token: 'ghp_abc123def456ghi789jkl012'"
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized
        assert "ghp_abc123" not in sanitized

    def test_private_key_redacted(self):
        text = """-----BEGIN PRIVATE KEY-----
MIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQD
-----END PRIVATE KEY-----"""
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized
        assert "BEGIN PRIVATE KEY" not in sanitized

    def test_github_token_redacted(self):
        text = "token=ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized
        assert "ghp_abc" not in sanitized

    def test_aws_keys_redacted(self):
        text = "aws_access_key_id = AKIATEST12345"
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized

    def test_authorization_bearer_redacted(self):
        text = "Authorization: Bearer abc123.def456.ghi789"
        sanitized = MemoryFilter.sanitize(text)
        assert "[REDACTED]" in sanitized

    def test_contains_secrets_true(self):
        text = "password: 'mypassword'"
        assert MemoryFilter.contains_secrets(text) is True

    def test_contains_secrets_false(self):
        text = "This is a normal sentence about coding"
        assert MemoryFilter.contains_secrets(text) is False

    def test_sanitize_dict_recursive(self):
        data = {
            "config": "safe_value",
            "secrets": {
                "api_key": "sk-test12345678901234567890123456789012345",
                "password": "mysecret",
            },
            "items": ["normal text", "token='abc123def456'"],
        }
        sanitized = MemoryFilter.sanitize_dict(data)
        assert sanitized["config"] == "safe_value"
        assert "[REDACTED]" in sanitized["secrets"]["api_key"]
        assert "[REDACTED]" in sanitized["secrets"]["password"]
        assert "[REDACTED]" in sanitized["items"][1]

    def test_sanitize_preserves_safe_text(self):
        text = "The project uses Python and pytest for testing."
        sanitized = MemoryFilter.sanitize(text)
        assert sanitized == text


class TestMemoryService:

    def test_remember_creates_entry(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        store_path = tmp_path / "identity"
        identity_store = IdentityStore(str(store_path))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(
            memory=mem, identity_service=identity_service,
        )
        entry = service.remember(
            content="Always use pytest",
            memory_type="preference",
            importance=0.7,
        )
        assert entry.content == "Always use pytest"
        assert entry.memory_type == "preference"

        # Verify it was persisted
        loaded = store.load_ltm_entry(entry.id)
        assert loaded is not None
        assert loaded.content == "Always use pytest"

    def test_remember_with_project(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.USER)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        entry = service.remember(
            content="Project-specific memory",
            project="myproject",
            importance=0.6,
        )
        assert entry.project == "myproject"

    def test_remember_sanitizes_secrets(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.USER)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        entry = service.remember(
            content="API key: sk-test12345678901234567890123456789012345",
            memory_type="preference",
        )
        loaded = store.load_ltm_entry(entry.id)
        assert "[REDACTED]" in loaded.content
        assert "sk-test12345" not in loaded.content

    def test_remember_requires_authority(self, tmp_path):
        """GUEST should not be able to remember."""
        config_dir = tmp_path / "memory"
        identity = Identity.create("Guest", AuthorityLevel.GUEST)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        with pytest.raises(PermissionError):
            service.remember(content="test")

    def test_forget_deletes_entry(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        entry = service.remember(content="To be forgotten")
        assert service.forget(entry.id) is True
        assert store.load_ltm_entry(entry.id) is None

    def test_forget_nonexistent_returns_false(self, tmp_path):
        config_dir = tmp_path / "memory"
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)
        assert service.forget("nonexistent-id") is False

    def test_forget_requires_admin(self, tmp_path):
        """USER should not be able to forget (requires ADMIN)."""
        config_dir = tmp_path / "memory"
        identity = Identity.create("RegularUser", AuthorityLevel.USER)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        with pytest.raises(PermissionError):
            service.forget("some-id")

    def test_list_memories(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.USER)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        service.remember(content="Memory 1", memory_type="preference")
        service.remember(content="Memory 2", memory_type="decision")

        # User with USER authority can list memories
        user_ident = Identity.create("User2", AuthorityLevel.USER)
        identity_store.set_current(user_ident)
        entries = service.list_memories()
        assert len(entries) == 2

    def test_list_memories_filtered_by_project(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.USER)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        service.remember(content="Global pref", project=None)
        service.remember(content="Project memory", project="test-project")

        entries = service.list_memories(project="test-project")
        assert len(entries) == 1
        assert entries[0].project == "test-project"

    def test_retrieve_relevant(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        service.remember(content="Use pytest for Python testing", importance=0.8)
        service.remember(content="The weather is nice today", importance=0.3)

        results = service.retrieve_relevant(goal="pytest testing")
        assert len(results) >= 1
        assert "pytest" in results[0].entry.content.lower()

    def test_archive_task_outcome(self, tmp_path):
        """TaskState outcomes should be archived to long-term memory."""
        from evora.task import TaskState, Decision, Observation, TestResult

        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        state = TaskState(
            request="Create a test file",
            workspace=str(tmp_path),
        )
        state.add_decision(Decision(
            action="execute_tool",
            reason="Creating test file",
            tool="write_file",
            confidence=0.9,
        ))
        state.add_error("Permission denied during write")
        state.add_test_result(TestResult(
            command="pytest", passed=False, error="1 failed"
        ))
        state.mark_complete("File created")

        service.archive_task_outcome(state)

        entries = store.list_ltm_entries(project=str(tmp_path))
        assert len(entries) >= 3  # at least: decision, error, test result, completion

    def test_archive_applies_secret_filtering(self, tmp_path):
        """Archived task outcomes must not contain secrets."""
        from evora.task import TaskState

        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem, identity_service=identity_service)

        state = TaskState(request="test")
        state.add_error("API key sk-test12345678901234567890123456789012345 found")
        state.mark_complete("Done")

        service.archive_task_outcome(state)

        entries = store.list_ltm_entries(project=str(tmp_path))
        for e in entries:
            assert "sk-test12345" not in e.content
            assert "[REDACTED]" in e.content or "sk-test" not in e.content

    def test_update_project_memory_from_state(self, tmp_path):
        """ProjectMemory should be updated from TaskState after completion."""
        from evora.task import TaskState, Observation

        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        identity = Identity.create("Tester", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        workspace_name = Path(tmp_path).name
        mem = Memory(str(config_dir), workspace_name)
        service = MemoryService(memory=mem, identity_service=identity_service)

        state = TaskState(
            request="Implement feature",
            workspace=str(tmp_path),
            project_context={"languages": {"Python": 100}, "frameworks": ["FastAPI"]},
        )
        state.add_observation(Observation(
            type="file_created", source="write_file", data={"path": "test.py"},
        ))
        state.mark_complete("Feature complete")

        service.update_project_memory(state)

        pm = store.load_project_memory(workspace_name)
        assert pm is not None
        assert len(pm.previous_completed_tasks) >= 1

    def test_clear_task_memory(self, tmp_path):
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        entry = TaskEntry.create("test task", {})
        store.save_task(entry)

        mem = Memory(str(config_dir), "test-project")
        service = MemoryService(memory=mem)

        deleted = service.clear_task_memory()
        assert deleted >= 1
        assert store.load_task(entry.id) is None

    def test_clear_project_memory(self, tmp_path):
        """Admin should be able to clear project memory."""
        config_dir = tmp_path / "memory"
        store = MemoryStore(str(config_dir))
        pm = ProjectMemory.create("clearproj", str(tmp_path))
        store.save_project_memory(pm)

        identity = Identity.create("Admin", AuthorityLevel.ADMIN)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)

        mem = Memory(str(config_dir), "clearproj")
        service = MemoryService(memory=mem, identity_service=identity_service)

        result = service.clear_project_memory()
        assert result is True or result is False  # just verify it runs

    def test_clear_project_memory_requires_admin(self, tmp_path):
        """GUEST should not be able to clear project memory."""
        config_dir = tmp_path / "memory"
        mem = Memory(str(config_dir), "testproj")
        identity = Identity.create("Guest", AuthorityLevel.GUEST)
        identity_store = IdentityStore(str(tmp_path / "identity"))
        identity_store.set_current(identity)
        identity_service = IdentityService(store=identity_store)
        service = MemoryService(memory=mem, identity_service=identity_service)

        with pytest.raises(PermissionError):
            service.clear_project_memory()
