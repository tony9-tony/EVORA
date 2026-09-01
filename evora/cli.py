#!/usr/bin/env python3
"""
EVORA CLI - Command-line interface for the EVORA AI coding engine.

Usage:
    evora run "Create a Python script that prints hello world"
    evora run --auto-approve "Add a test for the auth module"
    evora plan "Build a REST API with FastAPI"
    evora analyze
    evora config
    evora memory
"""

import argparse
import asyncio
import sys
from pathlib import Path

from evora.analyzer import ProjectAnalyzer
from evora.approval import ApprovalSystem
from evora.config import load_config
from evora.logger import Logger
from evora.memory import Memory
from evora.model import ModelManager, OpenAIProvider
from evora.planner import Planner
from evora.agent import Agent, AgentConfig
from evora.autonomous import AutonomousAgent, AutonomousConfig
from evora.security import PermissionManager
from evora.tools import ToolRegistry


def _build_model_manager(config, logger):
    """Initialize model manager with available providers."""
    manager = ModelManager(logger)

    openai_key = config.api_key or config.providers.get("openai", {}).api_key if config.providers else ""
    if openai_key:
        provider = OpenAIProvider(
            api_key=openai_key,
            model=config.model,
            base_url=config.base_url,
        )
        manager.register("openai", provider)
    else:
        logger.warn("No API key found. Using mock model provider.")
        manager.register("mock", MockModelProvider())

    return manager


class MockModelProvider:
    """Simple mock provider for testing without API keys."""

    def name(self):
        return "mock"

    def model(self):
        return "mock-model"

    async def chat(self, request):
        import json
        from evora.model import ModelResponse, Usage

        last_user = ""
        for m in reversed(request.messages):
            from evora.model import Role
            if m.role == Role.USER:
                last_user = m.content[:200]
                break

        if "create_file" in last_user.lower() or "plan" in str(request.messages).lower():
            return ModelResponse(
                content=json.dumps({
                    "title": "Mock Plan",
                    "description": "A mock plan for testing",
                    "steps": [{
                        "id": "step-1",
                        "name": "Create file",
                        "description": "Create a test file",
                        "action_type": "create_file",
                        "action_args": {"path": "hello.py", "content": 'print("Hello from EVORA!")'},
                        "depends_on": [],
                        "estimated_effort": "low",
                    }]
                }),
                provider="mock",
                model="mock-model",
                usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            )

        return ModelResponse(
            content=f"[Mock response] You said: {last_user[:100]}",
            provider="mock",
            model="mock-model",
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

    async def chat_stream(self, request):
        resp = await self.chat(request)
        yield resp

    def close(self):
        pass


def cmd_analyze(args):
    """Analyze the current project."""
    config = load_config()
    logger = Logger("evora", config.log_level, config.log_file)
    workspace = args.workspace or config.workspace_dir

    analyzer = ProjectAnalyzer(workspace, logger)
    result = analyzer.analyze()

    print(f"\n{'=' * 60}")
    print(f"  Project Analysis: {result.project_name}")
    print(f"{'=' * 60}")
    print(f"\nWorkspace: {result.workspace}")
    print(f"Languages: {', '.join(f'{k} ({v:.0f}%)' for k, v in sorted(result.languages.items(), key=lambda x: -x[1])) if result.languages else 'None detected'}")
    print(f"Frameworks: {', '.join(result.frameworks) if result.frameworks else 'None detected'}")
    print(f"Build system: {result.build_system or 'Unknown'}")
    print(f"Test command: {result.test_command or 'Not detected'}")
    print(f"Entry points: {', '.join(result.entry_points) if result.entry_points else 'None'}")
    print(f"Config files: {len(result.config_files)} found")
    print(f"Test files: {len(result.test_files)}")
    print(f"Has git: {'Yes' if result.has_git else 'No'}")
    print(f"Git branch: {result.git_branch or 'N/A'}")
    print(f"File count: {result.file_count}")
    print(f"Size: {result.estimated_size / 1024:.1f} KB")
    print(f"\n{'=' * 60}\n")


def cmd_config(args):
    """Display/load configuration."""
    config = load_config()
    print(f"EVORA Configuration")
    print(f"{'=' * 50}")
    print(f"  Workspace:  {config.workspace_dir}")
    print(f"  Model:      {config.model}")
    print(f"  Base URL:   {config.base_url}")
    print(f"  Log level:  {config.log_level}")
    print(f"  Log file:   {config.log_file}")
    print(f"  Memory dir: {config.memory_dir}")
    print(f"  API key:    {'SET' if config.api_key else 'NOT SET'}")
    print(f"\n  Providers: {', '.join(config.providers.keys()) if config.providers else 'default only'}")
    print(f"\n  Permissions:")
    print(f"    Allow file write: {config.permissions.allow_file_write}")
    print(f"    Allow cmd exec:   {config.permissions.allow_cmd_exec}")
    print(f"    Allowed cmds:     {config.permissions.allowed_cmds or 'all'}")
    print(f"\n{'=' * 50}")


def cmd_memory(args):
    """View task and project memory."""
    config = load_config()
    memory = Memory(config.memory_dir, project_name=Path(config.workspace_dir).name)

    tasks = memory.store.list_tasks(limit=20)

    if not tasks:
        print("No tasks found in memory.")
        return

    print(f"\n{'=' * 60}")
    print(f"  EVORA Memory - Recent Tasks")
    print(f"{'=' * 60}\n")

    for t in tasks:
        status = t.get("status", "unknown")
        ts = t.get("timestamp", "")[:19]
        req = t.get("request", "")[:80]
        steps = len(t.get("steps", []))
        errors = len(t.get("errors", []))

        status_icon = {"completed": "[OK]", "failed": "[FAIL]", "running": "[...]", "pending": "[...]"}
        icon = status_icon.get(status, "[?]")

        print(f"  {icon} [{status}] {ts}")
        print(f"       Request: {req}")
        print(f"       Steps: {steps}, Errors: {errors}")
        print()

    print(f"{'=' * 60}\n")


async def async_run(args):
    """Run the EVORA autonomous agent on a task."""
    config = load_config()
    logger = Logger("evora", config.log_level, config.log_file)
    workspace = args.workspace or config.workspace_dir

    security = PermissionManager(
        workspace_dir=workspace,
        allow_file_write=config.permissions.allow_file_write,
        allow_cmd_exec=config.permissions.allow_cmd_exec,
        allowed_cmds=config.permissions.allowed_cmds,
        ask_approvals=not args.auto_approve,
    )

    manager = _build_model_manager(config, logger)
    memory = Memory(config.memory_dir, project_name=Path(workspace).name)
    analyzer = ProjectAnalyzer(workspace, logger)

    approval = ApprovalSystem(
        logger=logger,
        auto_approve=args.auto_approve,
    )

    security.add_approval_callback(approval.approve_command)

    result = analyzer.analyze()

    planner = Planner(manager, logger)
    tools = ToolRegistry(security, logger)

    agent_config = AutonomousConfig(
        max_retries=getattr(args, 'max_retries', 3),
        retry_delay=1.0,
        command_timeout=getattr(args, 'timeout', 60),
        auto_approve=args.auto_approve,
        max_iterations=50,
    )

    agent = AutonomousAgent(
        model_manager=manager,
        planner=planner,
        approval=approval,
        tools=tools,
        memory=memory,
        security=security,
        logger=logger,
        analyzer=analyzer,
        config=agent_config,
    )

    try:
        report = await agent.run(args.request, result.to_dict())
        print(report)
        return 0
    except KeyboardInterrupt:
        await agent.stop()
        print("\n[EVORA] Task cancelled by user.")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        print(f"\n[EVORA] Fatal error: {e}")
        return 1
    finally:
        manager.close()


async def async_plan(args):
    """Generate a plan without executing."""
    config = load_config()
    logger = Logger("evora", config.log_level, config.log_file)
    workspace = args.workspace or config.workspace_dir

    manager = _build_model_manager(config, logger)
    planner = Planner(manager, logger)
    analyzer = ProjectAnalyzer(workspace, logger)

    result = analyzer.analyze()
    plan = await planner.create_plan(args.request, result.to_dict())

    print(planner.format_plan(plan))
    manager.close()


def main():
    parser = argparse.ArgumentParser(
        prog="evora",
        description="EVORA - AI Software Development Engine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  evora run "Create a Python script that prints hello world"
  evora run --auto-approve "Add tests to the auth module"
  evora plan "Build a REST API with FastAPI"
  evora analyze
  evora config
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run EVORA on a task")
    run_parser.add_argument("request", type=str, help="The task description")
    run_parser.add_argument("--auto-approve", action="store_true", help="Skip approval prompts")
    run_parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")
    run_parser.add_argument("--timeout", type=int, default=60, help="Command timeout in seconds")
    run_parser.add_argument("--max-retries", type=int, default=3, help="Max retry attempts")

    plan_parser = subparsers.add_parser("plan", help="Generate a plan without executing")
    plan_parser.add_argument("request", type=str, help="The task description")
    plan_parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze the current project")
    analyze_parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    subparsers.add_parser("config", help="Show/load configuration")
    subparsers.add_parser("memory", help="View task and project memory")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    if args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "config":
        cmd_config(args)
    elif args.command == "memory":
        cmd_memory(args)
    elif args.command == "plan":
        asyncio.run(async_plan(args))
    elif args.command == "run":
        exit_code = asyncio.run(async_run(args))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
