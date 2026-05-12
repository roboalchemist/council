# council

Single-file Python CLI that queries multiple AI coding assistants (Claude, Gemini, Codex, Cursor Agent) in **parallel** and consolidates their responses.

## Project layout

```
council.py                         # Entire application — single file
pyproject.toml                     # Package config: hatchling build, click dep
Makefile                           # Install target: uv tool install
schemas/convergence-findings.json  # JSON schema for `converge` command output
.github/workflows/bump-tap.yml     # Bumps roboalchemist/homebrew-tap on release
.gitea/workflows/bump-tap.yml      # Bumps roboalchemist/homebrew-private on release
```

No tests directory — no test suite exists yet.

## Architecture

Single-file async Python app. Entry point: `main()` → `cli()` (Click group).

### CLI commands

| Command | Description |
|---------|-------------|
| `ask` | Default. Query all tools in parallel; print consolidated results |
| `list` | Show which tools are installed |
| `doctor` | Health check: installed, API keys, auth probe |
| `converge` | Iterative review loop: council → triage → fix → repeat |
| `add` | **Stub — not implemented** |

`main()` inserts `ask` as the subcommand when the user calls `council <prompt>` directly (no subcommand).

### Tools supported

| Name | Binary | Default model | Notes |
|------|--------|--------------|-------|
| `claude` | `claude` | (claude's own default) | Prompt via stdin; uses `--output-format json` for schema mode |
| `gemini` | `gemini` | `gemini-3-pro-preview` | Prompt via `-p` arg |
| `codex` | `codex` | (codex's own default) | Prompt via positional arg; schema via `--output-schema` tempfile |
| `cursor-agent` | `agent` | `opus-4.6` | Optional; also acts as substitute backend via `--use-cursor` |

Override defaults with `--MODEL-model` flags or `~/.council.toml` (see [Model selection](#model-selection)).

### Permission modes

| Mode | Flag | Claude | Gemini | Codex |
|------|------|--------|--------|-------|
| locked | `--locked` | (no extra flags) | (no extra flags) | `-s read-only` |
| read-only | default | `--dangerously-skip-permissions` | `--approval-mode auto_edit --allowed-tools ShellTool(gh *)` | `-a never -s read-only` |
| yolo | `--yolo` | `--dangerously-skip-permissions` | `--yolo` | `-a never -s danger-full-access` |

### Parallelism

`asyncio.create_subprocess_exec` — all tools run simultaneously. Results gathered via `asyncio.gather`.

## Model selection

Models change frequently. Override per-invocation with flags, or set persistent defaults in `~/.council.toml`.

### CLI flags

```bash
council --gemini-model gemini-3.1-pro "review this"
council --claude-model claude-opus-4-7 --codex-model gpt-5.5-pro "review this"
```

Available on both `ask` and `converge`. When a model is overridden, its name appears in the output header: `## GEMINI [gemini-3.1-pro] ✓ (33s)`.

### Config file (`~/.council.toml`)

```toml
# ~/.council.toml — persistent defaults, overridden by CLI flags
[models]
claude       = "claude-opus-4-7"
gemini       = "gemini-3.1-pro"
codex        = "gpt-5.5-pro"
cursor-agent = "opus-4.7"
```

Priority: CLI flag > `~/.council.toml` > hardcoded default (`DEFAULT_MODELS` in `council.py`).

`tomllib` (Python 3.11+ stdlib) is used for parsing — no new dependencies.

## Key implementation details

- **Claude nested sessions**: unsets `CLAUDECODE` env var so `claude -p` can run inside an existing Claude Code session.
- **Codex EAGAIN fix**: stdin set to `DEVNULL` (not inherited) to prevent EAGAIN on non-blocking stdin.
- **Claude JSON schema**: uses `--output-format json` (wraps in envelope with `.result`) instead of `--json-schema` (triggers agent mode that prints "Done.").
- **Gemini JSON stripping**: `_extract_json_from_text()` removes markdown fences and surrounding prose.
- **Post-processing**: when `--json-schema` is set, any non-JSON response is piped through `claude haiku` with `--json-schema` to normalize it.
- **Convergence fixes**: applied in reverse line order so earlier line numbers stay valid.

## Convergence command

`council converge <artifact>` runs an iterative review loop:

1. Each round: format prompt with artifact reference → run council → parse JSON findings
2. Triage by agreement: unanimous (all tools agree) / majority (≥2) / minority (1)
3. Apply unanimous fixes to file via line replacement
4. Repeat until `clean_threshold` consecutive clean rounds or `max_rounds` hit

Default schema: `schemas/convergence-findings.json`

```json
{
  "findings": [{"line": int, "claim": str, "verdict": "ACCURATE|INACCURATE|UNVERIFIABLE|NO_CITATION", "evidence": str, "fix": str}],
  "clean": bool
}
```

Worklog at `/tmp/council-convergence.log` by default.

## Install & release

```bash
make install          # uv tool install . --force
make uninstall
make reinstall        # uninstall + clean cache + install
```

Distributed via two Homebrew taps:
- **Public** (`roboalchemist/tap` on GitHub): formula at `Formula/council.rb`
- **Private** (`roboalchemist/homebrew-private` on Gitea): formula at `Formula/council.rb`

CI auto-bumps both taps on `release: published` events. GitHub workflow also supports `workflow_dispatch` with a `tag` input for manual bumps.

Version is in `pyproject.toml`. Tag format: `vX.Y.Z`.

## Dependencies

- **Runtime**: `click>=8.0` (CLI framework), Python stdlib (`asyncio`, `json`, `subprocess`, `hashlib`, `pathlib`, `dataclasses`)
- **Build**: `hatchling`
- **Install tool**: `uv`
- **No test framework** — no tests exist

## Image evaluation (diagrams, screenshots)

`--image <file>` attaches one or more images for visual evaluation. Each tool handles it differently:

| Tool | Mechanism |
|------|-----------|
| Codex | Native `-i <path>` flag (before `exec`) |
| Claude | `--add-dir <image-dir>` + prompt instruction to use its Read tool |
| Gemini | Prompt instruction to read the file (uses ShellTool) |
| cursor-agent | Prompt instruction |

```bash
# Evaluate an engineering diagram
council --image /tmp/vparrot-diagram.png \
  "Evaluate this architecture diagram. Are all arrows directionally correct? \
   Are there any overlapping or confusing edges? Does it clearly show the data flow?"

# Compare a rendered diagram against a spec
council --image /tmp/diagram.png \
  "This diagram should show X calling Y which writes to Z. \
   Does it match? What's wrong or missing?"

# Multiple images (e.g., before/after)
council --image /tmp/v1.png --image /tmp/v2.png \
  "The first image is the old diagram, the second is the revised version. \
   Is the revision an improvement? What issues remain?"
```

**Typical engineering diagram loop:**
1. Generate diagram: `kroki convert diagram.mmd -t mermaid -o /tmp/diagram.png`
2. Council evaluates: `council --image /tmp/diagram.png "Review for visual correctness"`
3. Fix source, re-render, repeat

**PATH note**: The homebrew formula at `/opt/homebrew/bin/council` takes precedence over the uv-installed `~/.local/bin/council`. After `make install`, use `~/.local/bin/council` until a new homebrew release is cut via `/brew-bump`.

## Common patterns

```bash
# Run against a project directory
council -C /path/to/project "Review the auth module"

# Structured JSON output
council --yolo --json-schema schemas/convergence-findings.json "Find bugs"

# Convergence review
council converge --dry-run --max-rounds 5 doc.md

# Specific tools only
council -t claude,gemini "What's the best approach here?"

# Pipe input
cat file.py | council "Review this code"
```
