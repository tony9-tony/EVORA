"""
Tests for the EVORA planner module.
"""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from evora.planner import Planner, Plan, PlanStep
from evora.model import ModelManager, ChatRequest, Message, Role, ModelResponse, Usage
from evora.logger import Logger


class TestPlanStep:

    def test_create_step(self):
        step = PlanStep(
            id="step-1",
            name="Create file",
            description="Create a test file",
            action_type="create_file",
            action_args={"path": "test.txt", "content": "hello"},
        )
        assert step.id == "step-1"
        assert step.name == "Create file"
        assert step.action_type == "create_file"
        assert step.action_args["path"] == "test.txt"
        assert step.depends_on == []
        assert step.estimated_effort == "low"

    def test_to_dict(self):
        step = PlanStep(
            id="step-1",
            name="Test",
            description="desc",
            action_type="create_file",
            action_args={"path": "test.txt"},
        )
        d = step.to_dict()
        assert d["id"] == "step-1"
        assert d["name"] == "Test"
        assert d["action_type"] == "create_file"


class TestPlan:

    def test_create_plan(self):
        plan = Plan(title="Test Plan", description="A test plan", steps=[])
        assert plan.title == "Test Plan"
        assert plan.total_steps() == 0

    def test_to_dict(self):
        step = PlanStep(id="s1", name="test", description="desc", action_type="create_file")
        plan = Plan(title="Test", description="desc", steps=[step])
        d = plan.to_dict()
        assert d["title"] == "Test"
        assert len(d["steps"]) == 1


class TestPlanner:

    def test_parse_valid_plan(self):
        logger = Logger("test", "error")
        manager = ModelManager(logger)
        planner = Planner(manager, logger)

        mock_response = json.dumps({
            "title": "Test Plan",
            "description": "A test plan",
            "steps": [
                {
                    "id": "step-1",
                    "name": "Create file",
                    "description": "Create test.txt",
                    "action_type": "create_file",
                    "action_args": {"path": "test.txt", "content": "hello"},
                    "depends_on": [],
                    "estimated_effort": "low",
                },
                {
                    "id": "step-2",
                    "name": "Run tests",
                    "description": "Run test suite",
                    "action_type": "run_tests",
                    "action_args": {"command": "pytest"},
                    "depends_on": ["step-1"],
                    "estimated_effort": "medium",
                }
            ]
        })

        plan = planner._parse_plan(mock_response)
        assert plan.title == "Test Plan"
        assert len(plan.steps) == 2
        assert plan.steps[0].id == "step-1"
        assert plan.steps[1].action_type == "run_tests"

    def test_parse_invalid_json(self):
        logger = Logger("test", "error")
        manager = ModelManager(logger)
        planner = Planner(manager, logger)

        plan = planner._parse_plan("not json at all")
        assert plan.title == "Fallback Plan"
        assert len(plan.steps) >= 1

    def test_parse_with_markdown(self):
        logger = Logger("test", "error")
        manager = ModelManager(logger)
        planner = Planner(manager, logger)

        mock_json = json.dumps({
            "title": "Plan",
            "description": "desc",
            "steps": [{
                "id": "s1",
                "name": "test",
                "description": "desc",
                "action_type": "create_file",
                "action_args": {"path": "test.txt"},
                "depends_on": [],
                "estimated_effort": "low",
            }]
        })
        wrapped = f"```json\n{mock_json}\n```"
        plan = planner._parse_plan(wrapped)
        assert plan.title == "Plan"
        assert len(plan.steps) == 1

    def test_format_plan(self):
        logger = Logger("test", "error")
        manager = ModelManager(logger)
        planner = Planner(manager, logger)

        plan = Plan(
            title="Test Plan",
            description="A description",
            steps=[PlanStep(id="s1", name="Test Step", description="desc", action_type="create_file")]
        )
        formatted = planner.format_plan(plan)
        assert "Test Plan" in formatted
        assert "Test Step" in formatted
