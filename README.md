# The Council

Query multiple AI CLI coding assistants in parallel and compare their responses.

## Default Tools

- **Claude** (Claude Code CLI)
- **Gemini** (Gemini CLI with gemini-3-pro-preview)
- **Codex** (OpenAI Codex CLI)
- **Cursor Agent** (optional backend via `--use-cursor`)

## Install

```bash
make install
```

## Usage

```bash
# Default: read-only mode (can run gh/bash, no file writes)
council "Review this code for bugs"

# With project context
council -C /path/to/project "Analyze the architecture"

# Permission modes
council --locked "Explain this code"        # No shell, read-only
council --read-only "Review this PR"        # Default: shell OK, no writes
council --yolo "Fix all the tests"          # Unrestricted

# Use cursor-agent as backend for all tools
council --use-cursor "Compare error handling approaches"

# Pick specific tools
council -t claude,gemini "Should I use Redis or Memcached?"

# Health check
council doctor
```

## Permission Modes

| Mode | Flag | Shell | File Writes |
|------|------|-------|-------------|
| locked | `--locked` | No | No |
| read-only | `--read-only` (default) | Yes | No |
| yolo | `--yolo` | Yes | Yes |

## Requirements

At least one of: `claude`, `gemini`, `codex`, `agent` (cursor-agent)
