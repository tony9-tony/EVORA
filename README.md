# EVORA - AI Software Development Engine

EVORA is an AI software engineering agent that can analyze codebases, create plans, write code, run tests, and iterate on fixes autonomously.

## Installation

```bash
pip install -e .
```

## Quick Start

```bash
# Analyze a project
evora analyze

# Generate a plan for a task (without executing)
evora plan "Create a FastAPI REST API with user authentication"

# Run EVORA on a task
evora run "Write a Python script that prints hello world"

# Run with auto-approval (skip prompts for all commands)
evora run --auto-approve "Add unit tests to the utils module"

# View configuration
evora config

# View task and project memory
evora memory
```

## Commands

| Command    | Description                                                  |
|------------|--------------------------------------------------------------|
| `run`      | Execute a task - analyzes project, plans, codes, tests, fixes|
| `plan`     | Generate a plan without executing it                         |
| `analyze`  | Analyze the current project structure and dependencies        |
| `config`   | Show current configuration                                   |
| `memory`   | View task and project memory                                 |

## Features

### Agent Loop
EVORA follows a structured workflow:
1. **ANALYZE** - Inspects project structure, languages, frameworks
2. **PLAN** - Creates a step-by-step implementation plan
3. **CODE** - Implements changes using file and command tools
4. **TEST** - Runs tests and validates changes
5. **FIX** - Iterates on failures until tests pass
6. **REPORT** - Summarizes what was done

### Tools
EVORA has built-in tools for software engineering:
- **File Operations**: Read, write, edit, list, and search files
- **Command Execution**: Run shell commands with permission management
- **Directory Creation**: Create directories as needed

### Security
Commands are classified by risk level:
- **SAFE** - Read-only operations (ls, cat, git status) - auto-approved
- **ASK** - Operations needing confirmation (pip install, file writes outside workspace)
- **DANGEROUS** - Destructive operations (rm -rf, git push) - blocked by default

### Model Support
- OpenAI (GPT-4, o1, etc.)
- Anthropic (Claude 3, Claude 3.5 Sonnet)
- Configurable via environment variables or config file

### Memory
EVORA maintains memory for:
- **Tasks** - Completed task history and steps
- **Projects** - Per-project notes and learned conventions
- **Global** - Cross-project knowledge and preferences

## Configuration

Configuration can be set via:
1. Environment variables (`EVORA_API_KEY`, `EVORA_MODEL`, etc.)
2. `~/.evora/config.json`
3. `.evora.json` in the project directory

```json
{
  "api_key": "sk-...",
  "model": "gpt-4-turbo-preview",
  "provider": "openai",
  "permission_level": "ask",
  "auto_approve": false,
  "timeout": 30,
  "max_iterations": 10
}
```

## Testing

```bash
python -m pytest tests/ -v
```

## License

MIT