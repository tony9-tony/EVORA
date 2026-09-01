"""
Tests for the EVORA agent system.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evora.agent import Agent, AgentConfig, AgentStatus
from evora.approval import ApprovalSystem, ApprovalDecision
from evora.logger import Logger
from evora.memory import Memory
from evora.model import ModelManager, ModelResponse, Usage
from evora.planner import Planner
from evora.security import PermissionManager
from evora.tools import ToolRegistry


def run_async(coro):
    import asyncio
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@pytest.fixture
def setup_agent(tmp_path):
    logger = Logger("test_agent", "error")
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
                content='{"title": "Test Plan", "description": "test", "steps": [{"id": "s1", "name": "Create file", "description": "desc", "action_type": "create_file", "action_args": {"path": "test.txt", "content": "hello"}, "depends_on": [], "estimated_effort": "low"}]}',
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
    config = AgentConfig(max_retries=1, retry_delay=0.1)

    return {
        "logger": logger,
        "security": security,
        "memory": memory,
        "manager": manager,
        "planner": planner,
        "approval": approval,
        "tools": tools,
        "config": config,
    }


class TestAgentConfig:

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_retries == 3
        assert cfg.command_timeout == 60
        assert cfg.auto_approve is False


class TestAgentStatus:

    def test_status_values(self):
        assert AgentStatus.IDLE.value == "idle"
        assert AgentStatus.PLANNING.value == "planning"
        assert AgentStatus.EXECUTING.value == "executing"
        assert AgentStatus.COMPLETED.value == "completed"


class TestAgent:

    def test_agent_creation(self, setup_agent):
        s = setup_agent
        agent = Agent(
            model_manager=s["manager"],
            plan=s["planner"],
            approval=s["approval"],
            tools=s["tools"],
            memory=s["memory"],
            security=s["security"],
            logger=s["logger"],
            config=s["config"],
        )
        assert agent.status == AgentStatus.IDLE

    def test_agent_run(self, tmp_path, setup_agent):
        s = setup_agent
        agent = Agent(
            model_manager=s["manager"],
            plan=s["planner"],
            approval=s["approval"],
            tools=s["tools"],
            memory=s["memory"],
            security=s["security"],
            logger=s["logger"],
            config=s["config"],
        )

        report = run_async(agent.run("Create a test file"))
        assert "EVORA TASK REPORT" in report

    def test_agent_memory_save(self, tmp_path, setup_agent):
        s = setup_agent
        agent = Agent(
            model_manager=s["manager"],
            plan=s["planner"],
            approval=s["approval"],
            tools=s["tools"],
            memory=s["memory"],
            security=s["security"],
            logger=s["logger"],
            config=s["config"],
        )
        run_async(agent.run("Create a test file"))

        tasks = s["memory"].store.list_tasks(limit=5)
        assert len(tasks) >= 1
