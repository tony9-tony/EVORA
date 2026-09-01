"""
Tests for EVORA Phase 3 memory integration with the autonomous agent.

These tests verify:
- Agent retrieves relevant memory before acting
- Agent archives task outcome to long-term memory after completion
- Agent updates project memory after completion
- Memory service integration doesn't break existing autonomous behavior
- Identity service is wired into the agent
- Memory is persisted across restarts
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from evora.agent import Agent, AgentConfig
from evora.approval import ApprovalSystem, ApprovalDecision
from evora.autonomous import AutonomousAgent, AutonomousConfig
from evora.decision import DecisionEngine
from evora.evaluation import Evaluator
from evora.identity import IdentityService, IdentityStore, Identity, AuthorityLevel
from evora.logger import Logger
from evora.memory import Memory, MemoryService, MemoryStore, LongTermMemoryEntry
from evora.model import ModelManager, ModelResponse, Usage
from evora.planner import Planner
from evora.security import PermissionManager
from evora.tools import ToolRegistry, ToolResult
from evora.autonomous import AutonomousAgent as Phase2Agent


def run_async(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class MockModelProvider:
    """Mock model that returns configurable responses."""

    def __init__(self, responses=None):
        self.responses = responses or []
        self._call_idx = 0
        self.calls = []

    def name(self):
        return "mock"

    def model(self):
        return "mock-model"

    async def chat(self, request):
        self.calls.append(request)
        if self._call_idx < len(self.responses):
            content = self.responses[self._call_idx]
            self._call_idx += 1
            return ModelResponse(
                content=content,
                provider="mock",
                model="mock-model",
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )
        return ModelResponse(
            content=json.dumps({
                "title": "Mock Plan",
                "description": "Auto-generated mock plan",
                "steps": [{
                    "id": "step-1",
                    "name": "Create file",
                    "description": "Create a test file",
                    "action_type": "create_file",
                    "action_args": {"path": "output.txt", "content": "Hello"},
                    "depends_on": [],
                    "estimated_effort": "low",
                }],
            }),
            provider="mock",
            model="mock-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def chat_stream(self, request):
        yield await self.chat(request)

    def close(self):
        pass


class MockToolRegistry:
    """Mock tool registry that records calls and returns configurable results."""

    def __init__(self, results=None, defaults=None):
        self.results = results or {}
        self.defaults = defaults or {}
        self.call_log = []

    def get_specs(self):
        return []

    def list(self):
        return []

    def get(self, name):
        return None

    async def execute(self, name: str, **kwargs):
        self.call_log.append({"tool": name, "args": kwargs})
        key = name
        if key in self.defaults:
            return self.defaults[key]
        if key in self.results and self.results[key]:
            return self.results[key].pop(0)
        return ToolResult(success=True, output=f"Mock result for {name}", data={"returncode": 0})


class TestAutonomousMemoryIntegration:
    """Integration tests for memory + autonomous agent."""

    def _make_agent(self, tmp_path, config=None):
        logger = Logger("test_mem", "error")
        security = PermissionManager(
            workspace_dir=str(tmp_path),
            ask_approvals=False,
            allowed_cmds=[],
        )

        memory_dir = str(tmp_path / "memory")
        identity_dir = str(tmp_path / "identity")

        # Set up a creator identity
        identity_store = IdentityStore(identity_dir)
        creator = Identity.create_creator("TestCreator")
        identity_store.set_current(creator)
        identity_service = IdentityService(store=identity_store)

        memory = Memory(memory_dir, project_name=Path(tmp_path).name)
        memory_service = memory.get_memory_service(
            identity_service=identity_service, logger=logger
        )

        manager = ModelManager(logger)
        manager.register("mock", MockModelProvider())
        planner = Planner(manager, logger)
        approval = ApprovalSystem(logger=logger, auto_approve=True)
        tools = MockToolRegistry()
        analyzer = MagicMock()
        analyzer.analyze.return_value = MagicMock()
        analyzer.analyze.return_value.to_dict.return_value = {
            "project_name": "test-project",
            "workspace": str(tmp_path),
            "languages": {"Python": 100.0},
            "frameworks": [],
            "test_command": "python -m pytest",
        }

        agent = AutonomousAgent(
            model_manager=manager,
            planner=planner,
            approval=approval,
            tools=tools,
            memory=memory,
            security=security,
            logger=logger,
            analyzer=analyzer,
            config=config or AutonomousConfig(
                max_retries=3,
                retry_delay=0.01,
                auto_approve=True,
                max_iterations=10,
            ),
            identity_service=identity_service,
            memory_service=memory_service,
        )
        return agent, memory, memory_service, identity_service

    def test_agent_with_memory_service_completes(self, tmp_path):
        """Agent with memory_service should still complete tasks."""
        from pathlib import Path

        agent, memory, svc, ident = self._make_agent(tmp_path)
        report = run_async(agent.run("Create a test file"))
        assert "EVORA AUTONOMOUS TASK REPORT" in report
        assert "COMPLETED" in report

    def test_agent_archives_outcome_to_ltm(self, tmp_path):
        """After completion, task outcome should be archived to long-term memory."""
        from pathlib import Path

        agent, memory, svc, ident = self._make_agent(tmp_path)
        run_async(agent.run("Create a file"))

        # Check that LTM entries were created
        ltm_entries = memory.store.list_ltm_entries(project=str(tmp_path))
        assert len(ltm_entries) > 0, "Task outcome should be archived to LTM"

    def test_agent_updates_project_memory(self, tmp_path):
        """After completion, project memory should be updated."""
        from pathlib import Path

        agent, memory, svc, ident = self._make_agent(tmp_path)
        run_async(agent.run("Create a file"))

        workspace_name = Path(tmp_path).name
        pm = memory.store.load_project_memory(workspace_name)
        # Even if the file didn't exist before, the agent should create it
        assert pm is not None

    def test_agent_retrieves_memory_before_acting(self, tmp_path):
        """Agent should retrieve relevant memory and inject it into project_context."""
        from pathlib import Path

        agent, memory, svc, ident = self._make_agent(tmp_path)

        # Store a relevant memory before running
        svc.remember(
            content="This project uses pytest for testing",
            memory_type="learning",
            importance=0.8,
            project=str(tmp_path),
        )

        run_async(agent.run("Create a file"))

        # Check the state was updated with relevant memories
        # The last state should have relevant_memories in project_context
        assert hasattr(agent, "_current_state")
        if agent._current_state:
            ctx = agent._current_state.project_context
            # The relevant_memories key should be present
            assert "relevant_memories" in ctx

    def test_agent_archive_applies_secret_filtering(self, tmp_path):
        """Archived memory entries must not contain secrets."""
        from pathlib import Path

        agent, memory, svc, ident = self._make_agent(tmp_path)
        # Store a memory with a secret
        svc.remember(
            content="API key: sk-test12345678901234567890",
            memory_type="preference",
            project=str(tmp_path),
        )

        run_async(agent.run("Create a file"))

        ltm_entries = memory.store.list_ltm_entries(project=str(tmp_path))
        for entry in ltm_entries:
            assert "sk-test12345" not in entry.content

    def test_agent_memory_persistence_across_restart(self, tmp_path):
        """Memory should persist and be retrievable after agent restart."""
        from pathlib import Path

        agent, memory, svc, ident = self._make_agent(tmp_path)
        svc.remember(
            content="Use FastAPI for REST APIs",
            memory_type="decision",
            importance=0.9,
            project=str(tmp_path),
        )

        # Simulate restart: create new Memory/MemoryService from same dir
        memory2 = Memory(str(tmp_path / "memory"), project_name=Path(tmp_path).name)
        entries = memory2.store.list_ltm_entries(project=str(tmp_path))
        assert len(entries) >= 1
        assert any("FastAPI" in e.content for e in entries)

    def test_memory_service_optional(self, tmp_path):
        """Agent without memory_service should still work (backward compat)."""
        from pathlib import Path

        logger = Logger("test_compat", "error")
        security = PermissionManager(workspace_dir=str(tmp_path), ask_approvals=False)
        memory = Memory(str(tmp_path / "memory"), "test-project")
        manager = ModelManager(logger)
        manager.register("mock", MockModelProvider())
        planner = Planner(manager, logger)
        approval = ApprovalSystem(logger=logger, auto_approve=True)
        tools = MockToolRegistry()
        analyzer = MagicMock()
        analyzer.analyze.return_value = MagicMock()
        analyzer.analyze.return_value.to_dict.return_value = {}

        agent = AutonomousAgent(
            model_manager=manager,
            planner=planner,
            approval=approval,
            tools=tools,
            memory=memory,
            security=security,
            logger=logger,
            analyzer=analyzer,
            config=AutonomousConfig(max_retries=3, retry_delay=0.01, auto_approve=True, max_iterations=10),
            # No identity_service or memory_service — should work via defaults
        )

        report = run_async(agent.run("Create a test file"))
        assert "EVORA AUTONOMOUS TASK REPORT" in report

    def test_phase1_agent_still_works_with_memory(self, tmp_path):
        """Phase 1 Agent should still function with the extended Memory class."""
        from pathlib import Path

        logger = Logger("test_p1", "error")
        security = PermissionManager(workspace_dir=str(tmp_path), ask_approvals=False)
        memory = Memory(str(tmp_path / "memory"), "test-project")
        manager = ModelManager(logger)

        class MockProvider:
            def name(self):
                return "mock"

            def model(self):
                return "mock-model"

            async def chat(self, request):
                return ModelResponse(
                    content='{"title": "Test Plan", "description": "test", '
                            '"steps": [{"id": "s1", "name": "Create file", '
                            '"description": "desc", "action_type": "create_file", '
                            '"action_args": {"path": "test.txt", "content": "hello"}, '
                            '"depends_on": [], "estimated_effort": "low"}]}',
                    provider="mock",
                    model="mock-model",
                    usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
                )

            async def chat_stream(self, request):
                yield await self.chat(request)

            def close(self):
                pass

        manager.register("mock", MockProvider())
        planner = Planner(manager, logger)
        approval = ApprovalSystem(logger=logger, auto_approve=True)
        tools = ToolRegistry(security, logger)

        agent = Agent(
            model_manager=manager,
            plan=planner,
            approval=approval,
            tools=tools,
            memory=memory,
            security=security,
            logger=logger,
            config=AgentConfig(max_retries=1, retry_delay=0.1, auto_approve=True),
            analyzer=MagicMock(),
        )

        # Mock analyzer
        agent.analyzer = MagicMock()
        agent.analyzer.analyze.return_value = MagicMock()
        agent.analyzer.analyze.return_value.to_dict.return_value = {
            "languages": {"Python": 100.0},
            "frameworks": [],
        }

        report = run_async(agent.run("Create a test file"))
        assert "EVORA TASK REPORT" in report
