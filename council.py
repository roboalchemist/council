#!/usr/bin/env python3
"""
Council - Query multiple AI CLI coding assistants and consolidate their responses.

This tool sends the same prompt to multiple AI assistants (Claude, Gemini, and Codex
by default) in parallel, then presents a consolidated view of their responses for
alternative viewpoints.
"""

import asyncio
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
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
    "claude": "claude-4-opus",
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


def build_tool_command(
    name: str, prompt: str, mode: str = "read-only", use_cursor: bool = False
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
    if use_cursor and name != "cursor-agent":
        model = CURSOR_AGENT_MODELS.get(name, "gemini-3-pro")
        cmd = ["agent", "--print", "--model", model]
        if mode == "yolo":
            cmd.extend(["--force", "--approve-mcps"])
        elif mode == "read-only":
            cmd.extend(["--force", "--approve-mcps", "--mode=ask"])
        else:
            cmd.append("--mode=ask")
        cmd.append(prompt)
        return cmd

    if name == "claude":
        cmd = ["claude", "-p"]
        if mode in ("yolo", "read-only"):
            cmd.append("--dangerously-skip-permissions")
        cmd.append(prompt)
        return cmd
    elif name == "gemini":
        if mode == "yolo":
            return ["gemini", "-m", "gemini-3-pro-preview", "--yolo", "-p", prompt]
        elif mode == "read-only":
            return [
                "gemini", "-m", "gemini-3-pro-preview",
                "--approval-mode", "auto_edit",
                "--allowed-tools", "ShellTool(gh *)",
                "-p", prompt,
            ]
        else:
            return ["gemini", "-m", "gemini-3-pro-preview", "-p", prompt]
    elif name == "codex":
        if mode == "yolo":
            cmd = ["codex", "-a", "never", "-s", "danger-full-access", "exec", prompt]
        elif mode == "read-only":
            cmd = ["codex", "-a", "never", "-s", "read-only", "exec", prompt]
        else:
            cmd = ["codex", "-s", "read-only", "exec", prompt]
        return cmd
    elif name == "cursor-agent":
        cmd = ["agent", "--print", "--model", "gemini-3-pro"]
        if mode == "yolo":
            cmd.extend(["--force", "--approve-mcps"])
        elif mode == "read-only":
            cmd.extend(["--force", "--approve-mcps", "--mode=ask"])
        else:
            cmd.append("--mode=ask")
        cmd.append(prompt)
        return cmd
    else:
        raise ValueError(f"Unknown tool: {name}")


async def run_tool(name: str, command: list[str], cwd: str, timeout: int = 600) -> ToolResult:
    """Run an AI tool command and capture its output."""
    start_time = time.time()

    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
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
) -> str:
    """Run the prompt against all specified AI tools in parallel."""
    available = check_tool_availability()

    if tools is None:
        tools = list(DEFAULT_TOOLS)

    # When using cursor-agent as backend, we only need cursor-agent installed
    if use_cursor:
        if not available.get("cursor-agent"):
            return "Error: --use-cursor requires cursor-agent (agent) to be installed."
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
        return "Error: No AI tools available. Install claude, gemini, or codex."

    commands = {
        name: build_tool_command(name, prompt, mode=mode, use_cursor=use_cursor)
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
def ask(prompt, path, tools, timeout, no_timing, yolo, read_only, locked, use_cursor):
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
                result = subprocess.run(
                    auth_cmd,
                    capture_output=True,
                    text=True,
                    timeout=15,
                    cwd=os.getcwd(),
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
        "ask", "list", "add", "doctor", "--help", "-h"
    ]:
        sys.argv.insert(1, "ask")
    elif len(sys.argv) == 1:
        pass

    cli()


if __name__ == "__main__":
    main()
