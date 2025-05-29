# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Lint/Test Commands
- Installation: `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Build: `python -m build`
- Lint: `ruff check .` (when implemented)
- Format: `ruff format .` (when implemented)
- Tests: `pytest` (when implemented)

## Code Style Guidelines
- Imports: Group by standard library, third-party, local; sort alphabetically
- Types: Use type hints (Optional[Type], List[str], Dict[str, Any])
- Naming: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_SNAKE_CASE
- Docstrings: Google-style with Args, Returns, Raises sections
- Formatting: 4-space indentation, ~88 char line length, dedent multiline strings
- Logging: Use `logging` with `rich` handlers; proper log levels (INFO, DEBUG)
- Error handling: Specific exceptions with informative messages
- Code organization: Modular structure by technique type, consistent patterns