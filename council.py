#!/usr/bin/env python3
"""
Council - Query multiple AI CLI coding assistants and consolidate their responses.

This tool sends the same prompt to multiple AI assistants (Claude, Gemini, and Codex
by default) in parallel, then presents a consolidated view of their responses for
alternative viewpoints.
"""

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import click


@dataclass
class ToolResult:
    """Result from an AI tool query."""

    name: str
    output: str
    error: Optional[str]
    duration: float
    success: bool


# Tool substitution: cursor-agent can stand in for any of these tools
TOOL_NAMES = ["claude", "gemini", "codex", "cursor-agent"]
DEFAULT_TOOLS = ["claude", "gemini", "codex"]

# cursor-agent model mappings when substituting for other tools
CURSOR_AGENT_MODELS = {
    "claude": "opus-4.6",
    "gemini": "gemini-3-pro",
    "codex": "gpt-5.2-codex",
}

# Environment variables to check for auth per tool
TOOL_AUTH_CHECKS = {
    "claude": {
        "env_keys": [],
        "auth_cmd": ["claude", "-p", "reply: ok"],
    },
    "gemini": {
        "env_keys": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "auth_cmd": ["gemini", "-m", "gemini-3-pro-preview", "-p", "reply: ok"],
    },
    "codex": {
        "env_keys": ["OPENAI_API_KEY"],
        "auth_cmd": None,
    },
    "cursor-agent": {
        "env_keys": [],
        "auth_cmd": ["agent", "status"],
    },
}


def _prepend_dirs_to_prompt(prompt: str, add_dirs: list[str] | None) -> str:
    """Prepend additional context directories to a prompt (for tools without native dir support)."""
    if not add_dirs:
        return prompt
    dirs_str = ", ".join(add_dirs)
    return f"Additional context directories: {dirs_str}. Read files from these paths as needed.\n\n{prompt}"


def build_tool_command(
    name: str, prompt: str, mode: str = "read-only", use_cursor: bool = False,
    add_dirs: list[str] | None = None,
    json_schema: str = None,
) -> list[str]:
    """Build command for a specific tool.

    Args:
        name: Tool name (claude, gemini, codex, cursor-agent).
        prompt: The prompt to send.
        mode: Permission mode:
            - "locked": No shell access. Tools can only read files and analyze.
            - "read-only": (default) Full tool usage (bash, gh, file reads) but
              no file writes.
            - "yolo": Unrestricted. Can read, write, execute anything.
        use_cursor: When True, use cursor-agent as a substitute with the
            appropriate model for the requested tool.
    """
    # Append schema instruction to prompt for tools that don't support native schema
    if json_schema and name in ("gemini", "cursor-agent") or (json_schema and use_cursor):
        prompt = (
            prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON matching this schema:\n"
            + json_schema
        )

    # For cursor-agent (native or substitute), prepend dirs to prompt
    if use_cursor and name != "cursor-agent":
        model = CURSOR_AGENT_MODELS.get(name, "gemini-3-pro")
        cmd = ["agent", "--print", "--trust", "--model", model]
        if mode == "yolo":
            cmd.extend(["--force", "--approve-mcps"])
        elif mode == "read-only":
            cmd.extend(["--force", "--approve-mcps", "--mode=ask"])
        else:
            cmd.append("--mode=ask")
        effective_prompt = _prepend_dirs_to_prompt(prompt, add_dirs)
        cmd.append(effective_prompt)
        return cmd

    if name == "claude":
        cmd = ["claude", "-p"]
        if mode in ("yolo", "read-only"):
            cmd.append("--dangerously-skip-permissions")
        if add_dirs:
            for d in add_dirs:
                cmd.extend(["--add-dir", d])
        if json_schema:
            cmd.extend(["--json-schema", json_schema])
        cmd.append(prompt)
        return cmd
    elif name == "gemini":
        if mode == "yolo":
            cmd = ["gemini", "-m", "gemini-3-pro-preview", "--yolo"]
        elif mode == "read-only":
            cmd = [
                "gemini", "-m", "gemini-3-pro-preview",
                "--approval-mode", "auto_edit",
                "--allowed-tools", "ShellTool(gh *)",
            ]
        else:
            cmd = ["gemini", "-m", "gemini-3-pro-preview"]
        if add_dirs:
            cmd.extend(["--include-directories", ",".join(add_dirs)])
        cmd.extend(["-p", prompt])
        return cmd
    elif name == "codex":
        if mode == "yolo":
            cmd = ["codex", "-a", "never", "-s", "danger-full-access"]
        elif mode == "read-only":
            cmd = ["codex", "-a", "never", "-s", "read-only"]
        else:
            cmd = ["codex", "-s", "read-only"]
        cmd.append("exec")
        if add_dirs:
            for d in add_dirs:
                cmd.extend(["--add-dir", d])
        if json_schema:
            schema_hash = hashlib.md5(json_schema.encode()).hexdigest()[:12]
            schema_path = f"/tmp/council-schema-{schema_hash}.json"
            with open(schema_path, "w") as f:
                f.write(json_schema)
            cmd.extend(["--output-schema", schema_path])
        cmd.append(prompt)
        return cmd
    elif name == "cursor-agent":
        cmd = ["agent", "--print", "--trust", "--model", "opus-4.6"]
        if mode == "yolo":
            cmd.extend(["--force", "--approve-mcps"])
        elif mode == "read-only":
            cmd.extend(["--force", "--approve-mcps", "--mode=ask"])
        else:
            cmd.append("--mode=ask")
        effective_prompt = _prepend_dirs_to_prompt(prompt, add_dirs)
        cmd.append(effective_prompt)
        return cmd
    else:
        raise ValueError(f"Unknown tool: {name}")


async def run_tool(name: str, command: list[str], cwd: str, timeout: int = 600) -> ToolResult:
    """Run an AI tool command and capture its output."""
    start_time = time.time()

    try:
        # Claude Code blocks nested sessions via CLAUDECODE env var — unset it
        env = None
        if name == "claude" and "CLAUDECODE" in os.environ:
            env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(
            process.communicate(), timeout=timeout
        )

        duration = time.time() - start_time
        output = stdout.decode("utf-8", errors="replace").strip()
        error = stderr.decode("utf-8", errors="replace").strip() if stderr else None

        return ToolResult(
            name=name,
            output=output,
            error=error if process.returncode != 0 else None,
            duration=duration,
            success=process.returncode == 0,
        )

    except asyncio.TimeoutError:
        duration = time.time() - start_time
        return ToolResult(
            name=name,
            output="",
            error=f"Timed out after {timeout} seconds",
            duration=duration,
            success=False,
        )
    except Exception as e:
        duration = time.time() - start_time
        return ToolResult(
            name=name,
            output="",
            error=str(e),
            duration=duration,
            success=False,
        )


async def post_process_result(result: ToolResult, schema_str: str) -> ToolResult:
    """Post-process a tool result to ensure it matches the required JSON schema.

    Tries json.loads first. If the output isn't valid JSON, pipes through
    claude haiku to extract and restructure into the required format.
    """
    if not result.success or not result.output:
        return result

    # Already valid JSON? Return as-is.
    try:
        json.loads(result.output)
        return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Use haiku to extract JSON from freeform output
    try:
        extract_prompt = (
            f"Extract and restructure this into the required JSON format:\n{result.output}"
        )
        cmd = [
            "claude", "-p", "--model", "haiku", "--json-schema", schema_str,
            extract_prompt,
        ]

        # Unset CLAUDECODE env var to allow nested claude invocation
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )

        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
        normalized = stdout.decode("utf-8", errors="replace").strip()

        if process.returncode == 0 and normalized:
            return ToolResult(
                name=result.name,
                output=normalized,
                error=result.error,
                duration=result.duration,
                success=True,
            )
    except Exception:
        pass  # Fall through to return original result

    return result


def check_tool_availability() -> dict[str, bool]:
    """Check which AI tools are available on the system."""
    availability = {}
    for name in TOOL_NAMES:
        binary_name = "agent" if name == "cursor-agent" else name
        availability[name] = shutil.which(binary_name) is not None
    return availability


def format_output(results: list[ToolResult], show_timing: bool = True, tool_count: int = 0) -> str:
    """Format the consolidated output from all tools."""
    separator = "=" * 80
    lines = []

    lines.append(separator)
    lines.append(f"COUNCIL RESULTS ({tool_count} tools queried)")
    lines.append(separator)
    lines.append("")

    for result in results:
        status = "✓" if result.success else "✗"
        timing = f" ({result.duration:.1f}s)" if show_timing else ""
        lines.append(f"## {result.name.upper()} {status}{timing}")
        lines.append("-" * 40)

        if result.success and result.output:
            lines.append(result.output)
        elif result.error:
            lines.append(f"[ERROR] {result.error}")
        else:
            lines.append("[No output]")

        lines.append("")
        lines.append("")

    lines.append(separator)
    successful = sum(1 for r in results if r.success)
    lines.append(f"Summary: {successful}/{len(results)} tools responded successfully")

    return "\n".join(lines)


async def run_council(
    prompt: str,
    path: str,
    tools: Optional[list[str]] = None,
    timeout: int = 600,
    show_timing: bool = True,
    mode: str = "read-only",
    use_cursor: bool = False,
    add_dirs: list[str] | None = None,
    json_schema: str = None,
    raw: bool = False,
) -> str | list[ToolResult]:
    """Run the prompt against all specified AI tools in parallel.

    When raw=True, returns list[ToolResult] instead of formatted string.
    """
    available = check_tool_availability()

    if tools is None:
        tools = list(DEFAULT_TOOLS)

    # When using cursor-agent as backend, we only need cursor-agent installed
    if use_cursor:
        if not available.get("cursor-agent"):
            msg = "Error: --use-cursor requires cursor-agent (agent) to be installed."
            return [] if raw else msg
    else:
        # Filter to available tools
        tools = [t for t in tools if available.get(t)]

    tools_to_run = []
    for tool in tools:
        if tool not in TOOL_NAMES:
            click.echo(f"Warning: Unknown tool '{tool}'", err=True)
        elif not use_cursor and not available.get(tool):
            click.echo(f"Warning: Tool '{tool}' not found in PATH", err=True)
        else:
            tools_to_run.append(tool)

    if not tools_to_run:
        msg = "Error: No AI tools available. Install claude, gemini, or codex."
        return [] if raw else msg

    commands = {
        name: build_tool_command(name, prompt, mode=mode, use_cursor=use_cursor, add_dirs=add_dirs, json_schema=json_schema)
        for name in tools_to_run
    }

    # Label with model info when using cursor substitution
    display_names = {}
    for name in tools_to_run:
        if use_cursor and name != "cursor-agent":
            model = CURSOR_AGENT_MODELS.get(name, "unknown")
            display_names[name] = f"{name} (via cursor-agent/{model})"
        else:
            display_names[name] = name

    tasks = [run_tool(name, cmd, cwd=path, timeout=timeout) for name, cmd in commands.items()]

    click.echo(f"Querying {len(tasks)} AI tools in parallel (path: {path})...")
    if use_cursor:
        click.echo("Using cursor-agent as backend for all tools.")
    sys.stdout.flush()

    results = await asyncio.gather(*tasks)

    # Post-process results to ensure valid JSON schema compliance
    # Native tools (claude, codex) should return JSON but may not always,
    # so post-process ALL tools when json_schema is set
    if json_schema:
        processed = []
        for r in results:
            if r.success and r.output:
                try:
                    json.loads(r.output)
                except (json.JSONDecodeError, ValueError):
                    # Output isn't valid JSON — run through haiku post-processing
                    r = await post_process_result(r, json_schema)
            processed.append(r)
        results = processed

    if raw:
        return list(results)

    return format_output(list(results), show_timing=show_timing, tool_count=len(tasks))


@click.group()
def cli():
    """Query multiple AI CLI tools and consolidate their responses."""
    pass


@cli.command("ask")
@click.argument("prompt", required=False)
@click.option(
    "--path",
    "-C",
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    default=None,
    help="Path to working directory for context (default: current directory)",
)
@click.option(
    "--tools",
    "-t",
    help="Comma-separated list of tools to query (default: claude,gemini,codex). Options: claude,gemini,codex,cursor-agent",
)
@click.option(
    "--timeout",
    default=600,
    show_default=True,
    help="Timeout in seconds for each tool",
)
@click.option(
    "--no-timing",
    is_flag=True,
    help="Hide timing information in output",
)
@click.option(
    "--yolo",
    is_flag=True,
    help="Unrestricted: skip all permission prompts, allow reads, writes, and execution.",
)
@click.option(
    "--read-only",
    is_flag=True,
    help="(Default) Full tool usage (gh, bash, file reads) but no file modifications.",
)
@click.option(
    "--locked",
    is_flag=True,
    help="Most restrictive: read files and analyze only. No shell, no writes.",
)
@click.option(
    "--use-cursor",
    is_flag=True,
    help="Use cursor-agent as backend for all tools (routes to appropriate models).",
)
@click.option(
    "--add-dir",
    multiple=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Additional directory for context (repeatable). Mapped to each tool's native flag.",
)
@click.option(
    "--json-schema",
    default=None,
    help="JSON schema for structured output. Accepts inline JSON string or path to .json file.",
)
def ask(prompt, path, tools, timeout, no_timing, yolo, read_only, locked, use_cursor, add_dir, json_schema):
    """Ask all AI tools a question and compare their responses.

    \b
    Examples:
      council ask "What is the best way to handle errors in Python?"
      council ask -C /path/to/project "Analyze the architecture"
      council ask --tools claude,gemini "Review this code for security issues"
      council ask --use-cursor "Compare approaches to error handling"
      cat file.py | council ask "Review this code"
    """
    if prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()
        else:
            raise click.UsageError("No prompt provided. Pass as argument or pipe via stdin.")

    if not prompt:
        raise click.UsageError("Empty prompt provided.")

    if path is None:
        path = os.getcwd()

    tool_list = None
    if tools:
        tool_list = [t.strip().lower() for t in tools.split(",")]

    # Resolve --json-schema: file path or inline string
    schema_str = None
    if json_schema:
        if json_schema.endswith(".json") and os.path.isfile(json_schema):
            with open(json_schema) as f:
                schema_str = f.read().strip()
        else:
            schema_str = json_schema
        # Validate it's valid JSON
        try:
            json.loads(schema_str)
        except (json.JSONDecodeError, ValueError) as e:
            raise click.UsageError(f"Invalid JSON schema: {e}")

    flag_count = sum([yolo, read_only, locked])
    if flag_count > 1:
        raise click.UsageError("Only one of --yolo, --read-only, --locked can be used.")
    mode = "yolo" if yolo else "locked" if locked else "read-only"

    result = asyncio.run(
        run_council(
            prompt=prompt,
            path=path,
            tools=tool_list,
            timeout=timeout,
            show_timing=not no_timing,
            mode=mode,
            use_cursor=use_cursor,
            add_dirs=list(add_dir) if add_dir else None,
            json_schema=schema_str,
        )
    )

    click.echo(result)


@cli.command("list")
def list_tools():
    """List available AI tools and their installation status."""
    available = check_tool_availability()
    click.echo("Available AI tools:")
    for tool, avail in available.items():
        default = " (default)" if tool in DEFAULT_TOOLS else ""
        status = click.style("✓ installed", fg="green") if avail else click.style("✗ not found", fg="red")
        click.echo(f"  {tool}: {status}{default}")


@cli.command("doctor")
def doctor():
    """Check installation, authentication, and API key status for all tools."""
    available = check_tool_availability()

    click.echo("Council Doctor")
    click.echo("=" * 50)
    click.echo()

    for tool in TOOL_NAMES:
        default_marker = " (default)" if tool in DEFAULT_TOOLS else ""
        click.echo(f"## {tool}{default_marker}")

        # 1. Installed?
        binary_name = "agent" if tool == "cursor-agent" else tool
        binary_path = shutil.which(binary_name)
        if binary_path:
            click.echo(f"  Installed: {click.style('✓', fg='green')} ({binary_path})")
        else:
            click.echo(f"  Installed: {click.style('✗ not found', fg='red')}")
            click.echo()
            continue

        # 2. API keys present?
        auth_info = TOOL_AUTH_CHECKS.get(tool, {})
        env_keys = auth_info.get("env_keys", [])
        if env_keys:
            found_keys = []
            missing_keys = []
            for key in env_keys:
                val = os.environ.get(key)
                if val:
                    masked = val[:8] + "..." + val[-4:] if len(val) > 16 else val[:4] + "..."
                    found_keys.append(f"{key}={masked}")
                else:
                    missing_keys.append(key)
            if found_keys:
                click.echo(f"  API Key:   {click.style('✓', fg='green')} {', '.join(found_keys)}")
            if missing_keys:
                click.echo(f"  API Key:   {click.style('✗ missing', fg='yellow')} {', '.join(missing_keys)}")
            if not found_keys and not missing_keys:
                click.echo(f"  API Key:   {click.style('- no keys configured', fg='yellow')}")
        else:
            click.echo(f"  API Key:   {click.style('-', fg='blue')} uses OAuth/login (no env key)")

        # 3. Auth test (quick probe)
        auth_cmd = auth_info.get("auth_cmd")
        if auth_cmd:
            try:
                # Unset CLAUDECODE so claude can run nested inside another Claude session
                probe_env = None
                if tool == "claude" and "CLAUDECODE" in os.environ:
                    probe_env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
                result = subprocess.run(
                    auth_cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=os.getcwd(),
                    env=probe_env,
                )
                if result.returncode == 0:
                    click.echo(f"  Auth Test: {click.style('✓ responding', fg='green')}")
                else:
                    err_snippet = (result.stderr or result.stdout or "unknown error")[:80]
                    click.echo(f"  Auth Test: {click.style('✗ failed', fg='red')} ({err_snippet})")
            except subprocess.TimeoutExpired:
                click.echo(f"  Auth Test: {click.style('✗ timed out', fg='red')}")
            except Exception as e:
                click.echo(f"  Auth Test: {click.style('✗ error', fg='red')} ({e})")
        else:
            click.echo(f"  Auth Test: {click.style('-', fg='blue')} skipped")

        click.echo()

    # Summary
    click.echo("=" * 50)
    ready = [t for t in DEFAULT_TOOLS if available.get(t)]
    click.echo(f"Ready: {len(ready)}/{len(DEFAULT_TOOLS)} default tools available")
    if not available.get("cursor-agent"):
        click.echo("Note: cursor-agent not installed (optional --use-cursor backend)")
    else:
        click.echo(f"cursor-agent: available as --use-cursor backend")


DEFAULT_CONVERGE_PROMPT = (
    "Review the following document for factual accuracy, citations, and correctness. "
    "For each issue found, provide the line number, the claim, your verdict, evidence, "
    "and a suggested fix. If everything is clean, set clean=true with an empty findings array.\n\n"
    "{artifact}"
)


def triage_findings(all_findings: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    """Group findings across tools by (line, verdict) and classify agreement.

    Returns (unanimous, majority, minority) lists of finding dicts.
    """
    # Collect all individual findings with their source count
    # Key: (line, verdict) -> list of finding dicts from different tools
    grouped: dict[tuple, list[dict]] = {}
    tool_count = len(all_findings)

    for findings_obj in all_findings:
        for finding in findings_obj.get("findings", []):
            line = finding.get("line")
            verdict = finding.get("verdict")
            if line is None or verdict is None:
                continue
            key = (line, verdict)
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(finding)

    unanimous = []
    majority = []
    minority = []

    for key, findings_list in grouped.items():
        count = len(findings_list)
        # Pick the first finding with a fix as representative
        representative = findings_list[0]
        for f in findings_list:
            if f.get("fix"):
                representative = f
                break

        if count >= tool_count and tool_count > 0:
            unanimous.append(representative)
        elif count >= 2:
            majority.append(representative)
        else:
            minority.append(representative)

    return unanimous, majority, minority


def apply_fixes(fixes: list[dict], artifact_path: str) -> None:
    """Apply unanimous fixes to the artifact file via line replacement.

    Fixes are applied in reverse line order to preserve line numbers.
    """
    path = Path(artifact_path)
    lines = path.read_text().splitlines(keepends=True)

    # Sort by line number descending to preserve positions
    sorted_fixes = sorted(fixes, key=lambda f: f.get("line", 0), reverse=True)

    for fix in sorted_fixes:
        fix_text = fix.get("fix", "")
        line_num = fix.get("line", 0)
        if not fix_text or line_num < 1 or line_num > len(lines):
            continue
        # Preserve the original line ending
        original = lines[line_num - 1]
        ending = "\n" if original.endswith("\n") else ""
        lines[line_num - 1] = fix_text + ending

    path.write_text("".join(lines))


async def run_convergence(
    artifact: str,
    prompt_template: str,
    schema_str: str,
    max_rounds: int,
    clean_threshold: int,
    path: str,
    tools: Optional[list[str]],
    timeout: int,
    mode: str,
    use_cursor: bool,
    add_dirs: list[str] | None,
    log_path: str,
    dry_run: bool,
) -> None:
    """Run the convergence loop: council reviews, triage, fix, repeat."""
    log_file = Path(log_path)

    def log(msg: str) -> None:
        click.echo(msg)
        with log_file.open("a") as f:
            f.write(msg + "\n")

    log(f"=== Convergence started: {artifact} ===")
    log(f"max_rounds={max_rounds}, clean_threshold={clean_threshold}, dry_run={dry_run}")

    clean_count = 0
    for round_num in range(1, max_rounds + 1):
        log(f"\n--- Round {round_num}/{max_rounds} ---")

        # 1. Read artifact contents
        artifact_content = Path(artifact).read_text()

        # 2. Format prompt with artifact content
        formatted_prompt = prompt_template.format(artifact=artifact_content)

        # 3. Run council with raw=True to get ToolResults
        raw_results = await run_council(
            formatted_prompt, path, tools, timeout,
            show_timing=True, mode=mode, use_cursor=use_cursor,
            json_schema=schema_str, add_dirs=add_dirs, raw=True,
        )

        # 4. Parse structured findings from each tool result
        all_findings = []
        for result in raw_results:
            if not result.success:
                log(f"  WARNING: {result.name} failed: {result.error}")
                continue
            try:
                findings = json.loads(result.output)
                all_findings.append(findings)
                finding_count = len(findings.get("findings", []))
                log(f"  {result.name}: {'CLEAN' if findings.get('clean') else f'{finding_count} findings'} ({result.duration:.1f}s)")
            except json.JSONDecodeError:
                log(f"  WARNING: {result.name} returned non-JSON, skipping")

        if not all_findings:
            log(f"[Round {round_num}] No parseable results, skipping")
            continue

        # 5. Check if all clean
        if all(f.get("clean", False) for f in all_findings):
            clean_count += 1
            log(f"[Round {round_num}] CLEAN (streak: {clean_count}/{clean_threshold})")
            if clean_count >= clean_threshold:
                log("STEADY STATE — converged")
                return
            continue

        # 6. Reset clean streak, triage findings
        clean_count = 0
        unanimous, majority, minority = triage_findings(all_findings)

        # 7. Apply unanimous fixes (or show in dry-run)
        if unanimous and not dry_run:
            apply_fixes(unanimous, artifact)
            log(f"[Round {round_num}] FIXED {len(unanimous)} unanimous issues")
        elif unanimous and dry_run:
            log(f"[Round {round_num}] DRY-RUN: would fix {len(unanimous)} unanimous issues")
            for fix in unanimous:
                log(f"  L{fix.get('line')}: {fix.get('verdict')} — {fix.get('claim', '')[:80]}")

        log(f"[Round {round_num}] {len(majority)} majority, {len(minority)} minority (noted)")

    if clean_count < clean_threshold:
        log(f"MAX ROUNDS ({max_rounds}) reached, not converged")


@cli.command("converge")
@click.argument("artifact", type=click.Path(exists=True, dir_okay=False, resolve_path=True))
@click.option("--prompt", default=None, help="Review prompt template with {artifact} placeholder.")
@click.option("--prompt-file", type=click.Path(exists=True, resolve_path=True), default=None, help="File containing review prompt template.")
@click.option(
    "--schema", "schema_path", default=None,
    help="JSON schema file path (default: schemas/convergence-findings.json relative to council install dir).",
)
@click.option("--schema-file", type=click.Path(exists=True, resolve_path=True), default=None, help="Alias for --schema.")
@click.option("--max-rounds", default=3, show_default=True, help="Maximum convergence rounds.")
@click.option("--clean-threshold", default=2, show_default=True, help="Consecutive clean rounds for steady state.")
@click.option(
    "--add-dir", multiple=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Additional directory for context (repeatable).",
)
@click.option("--tools", "-t", help="Comma-separated list of tools to query.")
@click.option("--timeout", default=600, show_default=True, help="Timeout in seconds for each tool.")
@click.option("--yolo", is_flag=True, help="Unrestricted mode.")
@click.option("--read-only", is_flag=True, help="Read-only mode (default).")
@click.option("--locked", is_flag=True, help="Most restrictive mode.")
@click.option("--use-cursor", is_flag=True, help="Use cursor-agent as backend.")
@click.option("--log", "log_path", default="/tmp/council-convergence.log", show_default=True, help="Convergence worklog path.")
@click.option("--dry-run", is_flag=True, help="Print what would change without applying.")
def converge(artifact, prompt, prompt_file, schema_path, schema_file, max_rounds, clean_threshold,
             add_dir, tools, timeout, yolo, read_only, locked, use_cursor, log_path, dry_run):
    """Run iterative convergence review on an artifact file.

    \b
    The council reviews ARTIFACT each round, triages findings by agreement level,
    applies unanimous fixes, and repeats until convergence or max rounds.

    \b
    Examples:
      council converge doc.md
      council converge --dry-run --max-rounds 5 paper.md
      council converge --prompt "Check {artifact} for typos" notes.txt
    """
    # Resolve prompt template
    if prompt and prompt_file:
        raise click.UsageError("Only one of --prompt or --prompt-file can be used.")
    if prompt_file:
        prompt_template = Path(prompt_file).read_text()
    elif prompt:
        prompt_template = prompt
    else:
        prompt_template = DEFAULT_CONVERGE_PROMPT

    if "{artifact}" not in prompt_template:
        raise click.UsageError("Prompt template must contain {artifact} placeholder.")

    # Resolve schema
    resolved_schema = schema_file or schema_path
    if resolved_schema is None:
        # Default: schemas/convergence-findings.json relative to this file
        default_schema = Path(__file__).parent / "schemas" / "convergence-findings.json"
        if default_schema.exists():
            resolved_schema = str(default_schema)
        else:
            raise click.UsageError(f"Default schema not found: {default_schema}")

    schema_str = Path(resolved_schema).read_text().strip()
    try:
        json.loads(schema_str)
    except (json.JSONDecodeError, ValueError) as e:
        raise click.UsageError(f"Invalid JSON schema: {e}")

    # Mode
    flag_count = sum([yolo, read_only, locked])
    if flag_count > 1:
        raise click.UsageError("Only one of --yolo, --read-only, --locked can be used.")
    mode = "yolo" if yolo else "locked" if locked else "read-only"

    tool_list = None
    if tools:
        tool_list = [t.strip().lower() for t in tools.split(",")]

    path = os.getcwd()

    asyncio.run(run_convergence(
        artifact=artifact,
        prompt_template=prompt_template,
        schema_str=schema_str,
        max_rounds=max_rounds,
        clean_threshold=clean_threshold,
        path=path,
        tools=tool_list,
        timeout=timeout,
        mode=mode,
        use_cursor=use_cursor,
        add_dirs=list(add_dir) if add_dir else None,
        log_path=log_path,
        dry_run=dry_run,
    ))


@cli.command("add")
@click.argument("name")
@click.argument("command", nargs=-1, required=True)
def add_tool(name, command):
    """Add a custom AI tool to the council.

    \b
    Example:
      council add my-tool my-ai-cli --prompt
    """
    click.echo(f"Would add tool '{name}' with command: {' '.join(command)}")
    click.echo("(Custom tool persistence not yet implemented)")


def main():
    """Entry point that defaults to 'ask' when no subcommand given."""
    if len(sys.argv) > 1 and sys.argv[1] not in [
        "ask", "list", "add", "doctor", "converge", "--help", "-h"
    ]:
        sys.argv.insert(1, "ask")
    elif len(sys.argv) == 1:
        pass

    cli()


if __name__ == "__main__":
    main()
