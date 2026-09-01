#!/usr/bin/env python3
"""
EVORA CLI - Command-line interface for the EVORA AI coding engine.

Usage:
    evora run "Create a Python script that prints hello world"
    evora run --auto-approve "Add a test for the auth module"
    evora plan "Build a REST API with FastAPI"
    evora analyze
    evora config
    evora memory [--type tasks|long-term|all]
    evora remember "Use pytest for testing" --type preference --importance 0.8
    evora forget <memory_id>
    evora memories "What test framework"
    evora whoami
    evora identity set --name "Alice" --authority creator
    evora identity list
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from evora.analyzer import ProjectAnalyzer
from evora.approval import ApprovalSystem
from evora.config import load_config, ProviderConfig
from evora.identity import IdentityService, AuthorityLevel, Identity
from evora.logger import Logger
from evora.memory import Memory, MemoryService, LongTermMemoryEntry, MemoryFilter
from evora.model import ModelManager, OpenAIProvider, AnthropicProvider, OllamaProvider, _has_openai, _has_anthropic, Message, Role, ChatRequest, ModelResponse
from evora.planner import Planner
from evora.agent import Agent, AgentConfig
from evora.autonomous import AutonomousAgent, AutonomousConfig
from evora.security import PermissionManager
from evora.tools import ToolRegistry


def _build_model_manager(config, logger, provider_override=None):
    """Initialize model manager with available providers.

    Provider selection priority:
      1. Explicitly requested provider (config.provider / EVORA_PROVIDER env / --provider flag)
      2. openai   (requires API key)
      3. ollama   (no API key required; local)
      4. anthropic (requires API key)
      5. mock     (fallback for offline/testing)
    """
    manager = ModelManager(logger)
    providers = config.providers or {}
    requested = (provider_override or config.provider or "").strip().lower()

    # OpenAI (requires API key)
    if _has_openai:
        openai_pc = providers.get("openai")
        openai_key = (
            config.api_key
            or (openai_pc.api_key if openai_pc else "")
            or os.environ.get("EVORA_OPENAI_API_KEY", "")
        )
        if openai_key:
            try:
                pc = openai_pc or ProviderConfig(name="openai", model="gpt-4o", base_url="https://api.openai.com/v1")
                manager.register("openai", OpenAIProvider(
                    api_key=openai_key,
                    model=pc.model or "gpt-4o",
                    base_url=pc.base_url or "https://api.openai.com/v1",
                    timeout=pc.timeout,
                ))
            except ImportError as e:
                logger.warn(f"OpenAI provider unavailable: {e}")

    # Ollama (no API key required; local provider)
    ollama_pc = providers.get("ollama")
    ollama_model = (ollama_pc.model if ollama_pc else None) or OllamaProvider.DEFAULT_MODEL
    ollama_url = (ollama_pc.base_url if ollama_pc else None) or OllamaProvider.DEFAULT_BASE_URL
    ollama_timeout = (ollama_pc.timeout if ollama_pc else OllamaProvider.DEFAULT_TIMEOUT)
    try:
        manager.register("ollama", OllamaProvider(
            model=ollama_model,
            base_url=ollama_url,
            timeout=ollama_timeout,
        ))
    except ImportError as e:
        logger.warn(f"Ollama provider unavailable: {e}")

    # Anthropic (requires API key)
    if _has_anthropic:
        anthropic_pc = providers.get("anthropic")
        anthropic_key = (anthropic_pc.api_key if anthropic_pc else "") or os.environ.get("EVORA_ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                pc = anthropic_pc or ProviderConfig(name="anthropic", model="claude-3-5-sonnet-20241022", base_url="https://api.anthropic.com")
                manager.register("anthropic", AnthropicProvider(
                    api_key=anthropic_key,
                    model=pc.model or "claude-3-5-sonnet-20241022",
                    base_url=pc.base_url or "https://api.anthropic.com",
                    timeout=pc.timeout,
                ))
            except ImportError as e:
                logger.warn(f"Anthropic provider unavailable: {e}")

    # Select active provider
    available = manager.list_providers()
    if requested and requested in available:
        manager.set_active(requested)
    elif "openai" in available:
        manager.set_active("openai")
    elif "ollama" in available:
        manager.set_active("ollama")
    elif "anthropic" in available:
        manager.set_active("anthropic")
    else:
        logger.warn("No API key found and no local provider available. Using mock model provider.")
        manager.register("mock", MockModelProvider())

    active = manager.active
    if active:
        logger.info(f"Active model provider: {active.name()} ({active.model()})")
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

    view_type = getattr(args, "type", None) or "all"

    if view_type in ("all", "long-term", "ltm"):
        _cmd_memory_longterm(config, memory)

    if view_type in ("all", "tasks"):
        _cmd_memory_tasks(memory)


def _cmd_memory_tasks(memory):
    """Display recent task entries."""
    tasks = memory.store.list_tasks(limit=20)

    print(f"\n{'=' * 60}")
    print(f"  EVORA Memory - Recent Tasks")
    print(f"{'=' * 60}\n")

    if not tasks:
        print("No tasks found in memory.")
        return

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


def _cmd_memory_longterm(config, memory):
    """Display long-term memory entries."""
    entries = memory.store.list_ltm_entries(project=Path(config.workspace_dir).name, limit=50)

    print(f"\n{'=' * 60}")
    print(f"  EVORA Memory - Long-Term Knowledge")
    print(f"{'=' * 60}\n")

    if not entries:
        print("No long-term memories found.")
    else:
        for e in entries:
            pinned_str = " [PINNED]" if e.pinned else ""
            print(f"  [{e.memory_type}]{pinned_str} (importance: {e.importance:.1f}) {e.id[:8]}")
            print(f"    Content: {e.content[:200]}")
            if e.tags:
                print(f"    Tags: {', '.join(e.tags)}")
            print(f"    Project: {e.project or 'global'}")
            print()

    print(f"{'=' * 60}\n")


def cmd_remember(args):
    """Store a memory explicitly."""
    config = load_config()
    identity_service = IdentityService(identity_dir=config.identity_dir)
    memory_service = MemoryService(
        memory=Memory(config.memory_dir, project_name=Path(config.workspace_dir).name),
        identity_service=identity_service,
    )

    try:
        entry = memory_service.remember(
            content=args.content,
            memory_type=args.type,
            importance=args.importance,
            project=args.project if args.project != "global" else None,
            tags=args.tags.split(",") if args.tags else [],
            pinned=args.pinned,
        )
        print(f"Memory stored: id={entry.id[:12]} type={entry.memory_type} project={entry.project or 'global'}")
    except PermissionError as e:
        print(f"[DENIED] {e}")
        sys.exit(1)


def cmd_forget(args):
    """Delete a memory entry."""
    config = load_config()
    identity_service = IdentityService(identity_dir=config.identity_dir)
    memory_service = MemoryService(
        memory=Memory(config.memory_dir, project_name=Path(config.workspace_dir).name),
        identity_service=identity_service,
    )

    try:
        if args.all:
            deleted = memory_service.forget_all(
                memory_type=args.type if args.type != "all" else None,
                project=Path(config.workspace_dir).name if not args.global_only else None,
            )
            print(f"Deleted {deleted} memory entries.")
        else:
            success = memory_service.forget(args.id)
            if success:
                print(f"Memory deleted: {args.id}")
            else:
                print(f"Memory not found: {args.id}")
    except PermissionError as e:
        print(f"[DENIED] {e}")
        sys.exit(1)


def cmd_memories(args):
    """List and search memories ('what do you remember')."""
    config = load_config()
    identity_service = IdentityService(identity_dir=config.identity_dir)
    memory_service = MemoryService(
        memory=Memory(config.memory_dir, project_name=Path(config.workspace_dir).name),
        identity_service=identity_service,
    )

    try:
        if args.query:
            results = memory_service.search_memories(
                query=args.query,
                project=Path(config.workspace_dir).name if not args.global_only else None,
                limit=args.limit,
            )
            print(f"\nFound {len(results)} matching memories:\n")
            for e in results:
                pinned_str = " [PINNED]" if e.pinned else ""
                print(f"  [{e.memory_type}]{pinned_str} {e.content[:200]}")
                print(f"    id={e.id[:12]} project={e.project or 'global'}")
                print()
        else:
            entries = memory_service.list_memories(
                project=Path(config.workspace_dir).name if not args.global_only else None,
                memory_type=args.type if args.type != "all" else None,
                limit=args.limit,
            )
            print(f"\nFound {len(entries)} memories:\n")
            for e in entries:
                pinned_str = " [PINNED]" if e.pinned else ""
                print(f"  [{e.memory_type}]{pinned_str} (importance: {e.importance:.1f}) {e.id[:12]}")
                print(f"    Content: {e.content[:200]}")
                if e.tags:
                    print(f"    Tags: {', '.join(e.tags)}")
                print(f"    Project: {e.project or 'global'}")
                print()
    except PermissionError as e:
        print(f"[DENIED] {e}")
        sys.exit(1)


def cmd_whoami(args):
    """Show current identity and authority."""
    config = load_config()
    identity_service = IdentityService(identity_dir=config.identity_dir)

    identity = identity_service.current_identity()
    creator = identity_service.get_creator()

    print(f"\n{'=' * 60}")
    print(f"  EVORA Identity")
    print(f"{'=' * 60}\n")
    print(f"  Current identity: {identity.name}")
    print(f"  Authority level:  {identity.authority.value}")
    print(f"  ID:               {identity.id}")
    print(f"  Created:          {identity.created_at}")
    if creator and creator.id != identity.id:
        print(f"\n  Creator identity: {creator.name}")
        print(f"  Creator ID:       {creator.id}")
    print(f"\n{'=' * 60}\n")


def cmd_identity_set(args):
    """Set identity for the current session (CREATOR only, or first-time bootstrap)."""
    config = load_config()
    identity_service = IdentityService(identity_dir=config.identity_dir)

    authority = AuthorityLevel(args.authority)

    # Bootstrap: if no creator exists and user is trying to set creator,
    # allow first-time setup without authorization
    if authority == AuthorityLevel.CREATOR and not identity_service.store.get_creator():
        identity = Identity.create(name=args.name, authority=authority)
        identity_service.store.bootstrap_creator(args.name)
        print(f"Bootstrap: creator identity set: {identity.name} ({authority.value})")
        return

    try:
        identity_service.require_authority("change_identity")
    except PermissionError as e:
        print(f"[DENIED] {e}")
        sys.exit(1)

    identity = Identity.create(name=args.name, authority=authority)
    identity_service.store.save_identity(identity)
    identity_service.store.set_current(identity)

    if authority == AuthorityLevel.CREATOR:
        identity_service.store.set_creator(identity)

    print(f"Identity set: {identity.name} ({authority.value})")


def cmd_identity(args):
    """Handle identity subcommands."""
    config = load_config()

    if getattr(args, "identity_cmd", None) == "set":
        cmd_identity_set(args)
    elif getattr(args, "identity_cmd", None) == "list":
        identity_service = IdentityService(identity_dir=config.identity_dir)

        # Bootstrap mode: if no identities exist yet, show setup instructions
        identities = identity_service.store.list_identities()
        if not identities:
            print("No identities configured.\n")
            print("Run: evora identity set --name <name> --authority <level>")
            print("      (first setup does not require existing creator)")
            return

        try:
            identity_service.require_authority("change_identity")
        except PermissionError as e:
            print(f"[DENIED] {e}")
            sys.exit(1)

        current = identity_service.store.get_current()
        print(f"\n{'=' * 60}")
        print(f"  Known Identities")
        print(f"{'=' * 60}\n")
        for ident in identities:
            marker = " *" if ident.id == current.id else ""
            print(f"  {ident.name}{marker} ({ident.authority.value}) id={ident.id[:12]} created={ident.created_at[:19]}")
        print(f"\n{'=' * 60}\n")
    else:
        print("Usage: evora identity set --name <name> --authority <level>")
        print("       evora identity list")


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

    manager = _build_model_manager(config, logger, provider_override=getattr(args, "provider", None))
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

    # Phase 3: Set up identity and memory services
    identity_service = IdentityService(identity_dir=config.identity_dir, logger=logger)
    memory_service = memory.get_memory_service(identity_service=identity_service, logger=logger)

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
        identity_service=identity_service,
        memory_service=memory_service,
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

    manager = _build_model_manager(config, logger, provider_override=getattr(args, "provider", None))
    planner = Planner(manager, logger)
    analyzer = ProjectAnalyzer(workspace, logger)

    result = analyzer.analyze()
    plan = await planner.create_plan(args.request, result.to_dict())

    print(planner.format_plan(plan))
    manager.close()


async def _chat_turn(manager, messages, user_input, memory_service, workspace_name, logger):
    """Process one chat turn: append user message, retrieve memory context, call model, append response."""
    messages.append(Message(role=Role.USER, content=user_input))
    request_messages = list(messages)

    context = ""
    try:
        if memory_service is not None:
            relevant = memory_service.retrieve_relevant(
                goal=user_input,
                project=workspace_name,
                limit=5,
            )
            if relevant:
                context = "\n".join([f"- {r.entry.content[:200]}" for r in relevant[:5]])
    except Exception as e:
        logger.debug(f"Memory retrieval skipped: {e}")

    if context:
        request_messages.insert(-1, Message(role=Role.SYSTEM, content=f"Relevant memories:\n{context}"))

    request = ChatRequest(messages=request_messages, max_tokens=4096, temperature=0.7)
    response = await manager.chat(request)
    messages.append(Message(role=Role.ASSISTANT, content=response.content))
    return response


def async_chat(args):
    """Launch the EVORA chat web UI."""
    from evora.chat_server import start_chat_server
    config = load_config()
    logger = Logger("evora", config.log_level, config.log_file)
    provider_override = getattr(args, "provider", None)
    start_chat_server(config=config, logger=logger, provider_override=provider_override)
    return 0


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
    run_parser.add_argument("--provider", type=str, default=None, help="Model provider (openai, ollama, anthropic)")

    chat_parser = subparsers.add_parser("chat", help="Interactive chat with EVORA")
    chat_parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")
    chat_parser.add_argument("--provider", type=str, default=None, help="Model provider (openai, ollama, anthropic)")

    plan_parser = subparsers.add_parser("plan", help="Generate a plan without executing")
    plan_parser.add_argument("request", type=str, help="The task description")
    plan_parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")
    plan_parser.add_argument("--provider", type=str, default=None, help="Model provider (openai, ollama, anthropic)")

    analyze_parser = subparsers.add_parser("analyze", help="Analyze the current project")
    analyze_parser.add_argument("--workspace", type=str, default=None, help="Workspace directory")

    subparsers.add_parser("config", help="Show/load configuration")
    memory_parser = subparsers.add_parser("memory", help="View task and project memory")
    memory_parser.add_argument("--type", type=str, default="all",
                               choices=["all", "tasks", "long-term", "ltm"],
                               help="View tasks, long-term, or all memory")

    # Phase 3: User-controlled memory commands
    remember_parser = subparsers.add_parser("remember", help="Store a memory explicitly")
    remember_parser.add_argument("content", type=str, help="The memory content to store")
    remember_parser.add_argument("--type", type=str, default="preference",
                                  choices=["preference", "decision", "learning", "instruction"],
                                  help="Memory type (default: preference)")
    remember_parser.add_argument("--importance", type=float, default=0.5,
                                  help="Importance score 0.0-1.0 (default: 0.5)")
    remember_parser.add_argument("--project", type=str, default=None,
                                  help="Project scope (default: project-relative or 'global')")
    remember_parser.add_argument("--tags", type=str, default=None,
                                  help="Comma-separated tags")
    remember_parser.add_argument("--pinned", action="store_true",
                                  help="Pin this memory (always retrieved)")

    forget_parser = subparsers.add_parser("forget", help="Delete a memory entry")
    forget_parser.add_argument("id", type=str, nargs="?", default=None,
                                help="Memory entry ID to delete")
    forget_parser.add_argument("--all", action="store_true",
                                help="Delete all matching entries")
    forget_parser.add_argument("--type", type=str, default="all",
                                help="Filter by memory type (use with --all)")
    forget_parser.add_argument("--global-only", action="store_true",
                              help="Match only global (non-project) memories")

    memories_parser = subparsers.add_parser("memories", aliases=["recall"],
                                             help="List/search memories ('what do you remember')")
    memories_parser.add_argument("query", type=str, nargs="?", default=None,
                                  help="Search query (if omitted, list all)")
    memories_parser.add_argument("--type", type=str, default="all",
                                  help="Filter by memory type")
    memories_parser.add_argument("--limit", type=int, default=50,
                                  help="Maximum results to show")
    memories_parser.add_argument("--global-only", action="store_true",
                                  help="Search only global (non-project) memories")

    whoami_parser = subparsers.add_parser("whoami", help="Show current identity and authority")

    identity_parser = subparsers.add_parser("identity", help="Identity management (CREATOR only)")
    identity_sub = identity_parser.add_subparsers(dest="identity_cmd", help="Identity subcommands")
    identity_set = identity_sub.add_parser("set", help="Set current identity")
    identity_set.add_argument("--name", type=str, required=True, help="Identity name")
    identity_set.add_argument("--authority", type=str, required=True,
                               choices=["creator", "admin", "user", "guest"],
                               help="Authority level")
    identity_list = identity_sub.add_parser("list", help="List all known identities")

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
    elif args.command == "remember":
        cmd_remember(args)
    elif args.command == "forget":
        cmd_forget(args)
    elif args.command in ("memories", "recall"):
        cmd_memories(args)
    elif args.command == "whoami":
        cmd_whoami(args)
    elif args.command == "identity":
        cmd_identity(args)
    elif args.command == "plan":
        asyncio.run(async_plan(args))
    elif args.command == "chat":
        exit_code = async_chat(args)
        sys.exit(exit_code)
    elif args.command == "run":
        exit_code = asyncio.run(async_run(args))
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
