"""
Tests for the EVORA memory system.
"""

import json
import os
from pathlib import Path

import pytest

from evora.memory import Memory, TaskEntry, ProjectMemory, MemoryStore


class TestTaskEntry:

    def test_create(self):
        entry = TaskEntry.create("Do something", {"title": "Plan"})
        assert entry.id is not None
        assert entry.request == "Do something"
        assert entry.status == "pending"

    def test_update_status(self):
        entry = TaskEntry.create("Do something", {})
        entry.update_status("running")
        assert entry.status == "running"

    def test_add_step(self):
        entry = TaskEntry.create("Do something", {})
        entry.add_step("step1", "result", "completed")
        assert len(entry.steps) == 1
        assert entry.steps[0]["status"] == "completed"

    def test_add_memory(self):
        entry = TaskEntry.create("Do something", {})
        entry.add_memory("learned something")
        assert "learned something" in entry.memories

    def test_finish(self):
        entry = TaskEntry.create("Do something", {})
        entry.finish("Done!")
        assert entry.status == "completed"
        assert entry.result == "Done!"

    def test_fail(self):
        entry = TaskEntry.create("Do something", {})
        entry.fail("Error occurred")
        assert entry.status == "failed"


class TestProjectMemory:

    def test_create(self):
        pm = ProjectMemory.create("test-project", "/tmp/test")
        assert pm.project_name == "test-project"
        assert pm.conventions == {}

    def test_add_note(self):
        pm = ProjectMemory.create("test-project", "/tmp/test")
        pm.add_note("Uses Python")
        assert "Uses Python" in pm.notes

    def test_add_learned(self):
        pm = ProjectMemory.create("test-project", "/tmp/test")
        pm.add_learned("pytest for testing")
        assert "pytest for testing" in pm.learned


class TestMemoryStore:

    def test_save_and_load_task(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        entry = TaskEntry.create("Test request", {"title": "Test"})
        store.save_task(entry)
        loaded = store.load_task(entry.id)
        assert loaded is not None
        assert loaded.request == "Test request"
        assert loaded.status == "pending"

    def test_list_tasks(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        entry1 = TaskEntry.create("Request 1", {})
        entry2 = TaskEntry.create("Request 2", {})
        store.save_task(entry1)
        store.save_task(entry2)
        tasks = store.list_tasks(limit=10)
        assert len(tasks) == 2

    def test_load_nonexistent_task(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        result = store.load_task("nonexistent-id")
        assert result is None

    def test_save_and_load_project(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        pm = ProjectMemory.create("test-project", "/tmp/test")
        store.save_project_memory(pm)
        loaded = store.load_project_memory("test-project")
        assert loaded is not None
        assert loaded.project_name == "test-project"

    def test_save_and_load_global(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        store.save_global_memory("key1", {"data": "value"})
        result = store.load_global_memory("key1")
        assert result == {"data": "value"}

    def test_search_memories(self, tmp_path):
        store = MemoryStore(str(tmp_path))
        entry = TaskEntry.create("Create a login page", {})
        store.save_task(entry)
        results = store.search_memories("login")
        assert len(results) >= 1


class TestMemory:

    def test_memory_creation(self, tmp_path):
        mem = Memory(str(tmp_path), "test-project")
        assert mem.project_name == "test-project"

    def test_project_property(self, tmp_path):
        mem = Memory(str(tmp_path), "test-project-2")
        pm = mem.project
        assert pm.project_name == "test-project-2"

    def test_create_task(self, tmp_path):
        mem = Memory(str(tmp_path), "test-project-3")
        entry = mem.create_task("Test task", {"title": "Plan"})
        assert entry.request == "Test task"
        assert entry.id is not None
