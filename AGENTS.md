# AGENTS.md

Instructions for working with the EVORA codebase.

## Commands

### Install (development)
```
pip install -e .
```

### Run tests
```
python -m pytest tests/ -v
```

### Lint / Type Check
No separate linter is configured. Use pytest for verification.

### Run the CLI
```
evora --help          # list commands
python -m evora --help  # alternative if evora not on PATH
```

### Bootstrap creator identity (first time only)
```
python -m evora.bootstrap_creator
```
