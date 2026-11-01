#!/usr/bin/env python3
"""Shared non-interactive Claude Code execution helpers."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Callable, Mapping

from validation import common as validation_common
from transcripts import record_transcript


REASONING_EFFORTS = ("low", "medium", "high", "xhigh", "max")
WEB_SEARCH_MODES = ("disabled", "live")
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_REASONING_EFFORT = "high"
# Default model and effort per pipeline stage: cheaper models extract and
# screen; the strongest model works wherever the product is mathematics.
STAGE_MODEL_DEFAULTS = {
    "metadata": ("claude-haiku-5-5", "low"),
    "analyze": ("claude-sonnet-5-5", "high"),
    "triage": ("claude-sonnet-5-5", "medium"),
    "literature": ("claude-sonnet-5-5", "medium"),
    "solve": (DEFAULT_MODEL, DEFAULT_REASONING_EFFORT),
    "review": (DEFAULT_MODEL, DEFAULT_REASONING_EFFORT),
    "write": (DEFAULT_MODEL, DEFAULT_REASONING_EFFORT),
}
DEFAULT_VALIDATION_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent / "prompts" / "validate-output.md"
)
VALIDATION_PYTHON_PLACEHOLDER = "{{PYTHON}}"
AGENT_LAUNCH_INTERVAL_SECONDS = 1.0
AGENT_POLL_INTERVAL_SECONDS = 0.5
AGENT_STOP_GRACE_SECONDS = 10.0
WORKBENCH_DATABASE_ENV = "LOOSE_ENDS_WORKBENCH_DATABASE"
LAUNCH_GATE_ENV = "LOOSE_ENDS_AGENT_LAUNCH_GATE"
CONFIG_DIR_ENV = "CLAUDE_CONFIG_DIR"
WORKSPACE_TOOLS = ("Read", "Write", "Edit", "Glob", "Grep", "Bash")
WEB_TOOLS = ("WebSearch", "WebFetch")
# Bash runs inside the OS sandbox: writes only in the workspace, no network.
# The file tools are confined separately: acceptEdits approves edits only
# inside the working directory, and the read block covers Read/Glob/Grep.
SANDBOX_SETTINGS = {
    "sandbox": {
        "enabled": True,
        "failIfUnavailable": True,
        "autoAllowBashIfSandboxed": True,
        "allowUnsandboxedCommands": False,
        "network": {"allowedDomains": []},
    },
    "permissions": {"blockReadsOutsideWorkingDirectories": True},
}
# A parent Claude Code session exports these; some override the flags below.
# Runs must behave the same whether the workbench starts from a terminal or
# from inside a session.  ANTHROPIC_API_KEY would silently replace the
# subscription login with per-token API billing.
STRIPPED_ENVIRONMENT = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_MODEL",
    "CLAUDECODE",
    "CLAUDE_EFFORT",
    "CLAUDE_PID",
    "CLAUDE_PLUGIN_DATA",
}
STRIPPED_ENVIRONMENT_PREFIX = "CLAUDE_CODE_"
KEPT_ENVIRONMENT = {"CLAUDE_CODE_OAUTH_TOKEN"}
QUOTA_PHRASES = ("usage limit", "hit your limit", "credit balance is too low")
_LAUNCH_LOCK = threading.Lock()
_next_launch_at = 0.0


class AgentError(RuntimeError):
    """A local Claude Code invocation or its workspace could not be used."""


def iter_events(events_path: Path):
    """Yield the JSON object events of a stream-json log, skipping noise."""
    with events_path.open(encoding="utf-8", errors="replace") as events:
        for line in events:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                yield event


def quota_error(events_path: Path) -> str | None:
    """Recognize exhausted subscription or API quota in Claude Code events."""
    for event in iter_events(events_path):
        if event.get("type") == "rate_limit_event":
            info = event.get("rate_limit_info")
            if isinstance(info, dict) and info.get("status") == "rejected":
                window = info.get("rateLimitType") or "unknown"
                message = f"Claude usage limit reached ({window} window)"
                resets = info.get("resetsAt")
                if isinstance(resets, (int, float)):
                    message += "; resets at " + time.strftime(
                        "%Y-%m-%d %H:%M %Z", time.localtime(resets)
                    )
                return message
        elif event.get("type") == "result" and event.get("is_error"):
            errors = event.get("errors")
            text = " ".join(
                [str(event.get("result") or "")]
                + [str(item) for item in errors if isinstance(item, str)]
                if isinstance(errors, list)
                else [str(event.get("result") or "")]
            ).strip()
            if event.get("api_error_status") == 429 or any(
                phrase in text.casefold() for phrase in QUOTA_PHRASES
            ):
                return text or "Claude API returned HTTP 429"
    return None


def pause_queue_for_quota(message: str) -> None:
    database = os.environ.get(WORKBENCH_DATABASE_ENV)
    if database:
        from workbench_store import WorkbenchStore

        WorkbenchStore(Path(database)).pause_for_quota(message)


@dataclass(frozen=True)
class ModelOptions:
    model: str | None = None
    reasoning_effort: str | None = None
    max_budget_usd: float | None = None


DEFAULT_MODEL_OPTIONS = ModelOptions()


def inherit_model_options(
    primary: ModelOptions,
    secondary: ModelOptions,
) -> ModelOptions:
    """Fill unset secondary-run options (for example a critic's) from the primary."""
    return ModelOptions(
        secondary.model or primary.model,
        secondary.reasoning_effort or primary.reasoning_effort,
        secondary.max_budget_usd or primary.max_budget_usd,
    )


@dataclass(frozen=True)
class OutputValidator:
    """One task-specific validator and its authoritative dynamic inputs."""

    source: Path
    validate: Callable[..., validation_common.ValidationReport]
    expectations: Mapping[str, object]


def validated_result(
    report: validation_common.ValidationReport,
) -> dict[str, object]:
    """Return a checked result without relying on assertions at call sites."""
    if not report.valid or report.result is None:
        raise AgentError("validator accepted no structured result")
    return report.result


def configure_utf8_stdio() -> None:
    """Keep research titles and author names printable on non-UTF-8 stdio."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="backslashreplace")


def popen_session_options(*, new_process_group: bool = True) -> dict[str, object]:
    """Start children in their own session so a stop reaches all of them."""
    return {"start_new_session": True} if new_process_group else {}


def report_error(parser: argparse.ArgumentParser, error: BaseException) -> int:
    """Report a post-parse failure without printing irrelevant CLI usage."""
    print(f"{parser.prog}: error: {error}", file=sys.stderr)
    return 1


def positive_integer(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def positive_number(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return number


def add_prompt_arguments(
    parser: argparse.ArgumentParser,
    *,
    default_template: Path,
    task: str,
    prefix: str = "",
) -> None:
    """Add consistent user-direction and low-level prompt-template flags."""
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}prompt",
        dest=f"{destination_prefix}prompt",
        metavar="TEXT",
        help=f"additional instruction for the {task}",
    )
    parser.add_argument(
        f"--{option_prefix}prompt-template",
        dest=f"{destination_prefix}prompt_template",
        type=Path,
        default=default_template,
        metavar="FILE",
        help=(
            f"replace the complete low-level {task} prompt template "
            f"(default: {default_template})"
        ),
    )


def with_user_prompt(
    template: str,
    instruction: str | None,
    *,
    task: str,
    option_name: str = "--prompt",
) -> str:
    """Append an explicit user direction without replacing core safeguards."""
    if instruction is None:
        return template
    instruction = instruction.strip()
    if not instruction:
        raise AgentError(f"{option_name} must be nonempty")
    return (
        template.rstrip()
        + "\n\n# Additional user direction\n\n"
        + f"The user explicitly requested this direction for the {task}:\n\n"
        + f"<user_instruction>\n{instruction}\n</user_instruction>\n\n"
        + "Follow it throughout this run while preserving the task's output "
        + "contract, validation requirements, and mathematical accuracy.\n"
    )


def stage_model_defaults(stage: str) -> dict[str, str]:
    """Keyword defaults for add_model_arguments for one pipeline stage."""
    model, effort = STAGE_MODEL_DEFAULTS[stage]
    return {"default_model": model, "default_reasoning_effort": effort}


def add_model_arguments(
    parser: argparse.ArgumentParser,
    *,
    prefix: str = "",
    default_model: str = DEFAULT_MODEL,
    default_reasoning_effort: str = DEFAULT_REASONING_EFFORT,
) -> None:
    """Add consistent model, effort, budget, and executable options.

    A prefixed group (for example the critic of a solve run) inherits the
    primary run's values unless set explicitly.
    """
    if default_reasoning_effort not in REASONING_EFFORTS:
        raise ValueError(f"invalid default reasoning effort: {default_reasoning_effort}")
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    inherit = "inherit the primary run"
    parser.add_argument(
        f"--{option_prefix}model",
        dest=f"{destination_prefix}model",
        default=None if prefix else default_model,
        metavar="MODEL",
        help=(
            "Claude model ID or alias; for example, claude-sonnet-5-5 or sonnet "
            f"(default: {inherit if prefix else default_model})"
        ),
    )
    parser.add_argument(
        f"--{option_prefix}reasoning-effort",
        dest=f"{destination_prefix}reasoning_effort",
        default=None if prefix else default_reasoning_effort,
        choices=REASONING_EFFORTS,
        metavar="LEVEL",
        help=(
            "effort level: " + ", ".join(REASONING_EFFORTS)
            + f" (default: {inherit if prefix else default_reasoning_effort})"
        ),
    )
    parser.add_argument(
        f"--{option_prefix}max-budget-usd",
        dest=f"{destination_prefix}max_budget_usd",
        type=positive_number,
        default=None,
        metavar="USD",
        help=(
            "stop one Claude invocation once its API-equivalent cost reaches "
            f"this amount (default: {inherit if prefix else 'no limit'})"
        ),
    )


def add_web_search_argument(
    parser: argparse.ArgumentParser,
    *,
    default: str,
    prefix: str = "",
) -> None:
    """Add a scoped option for Claude's WebSearch and WebFetch tools."""
    if default not in WEB_SEARCH_MODES:
        raise ValueError(f"invalid default web-search mode: {default}")
    option_prefix = f"{prefix}-" if prefix else ""
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    parser.add_argument(
        f"--{option_prefix}web-search",
        dest=f"{destination_prefix}web_search",
        choices=WEB_SEARCH_MODES,
        default=None if prefix else default,
        metavar="MODE",
        help=(
            "Claude WebSearch and WebFetch tools: disabled or live "
            f"(default: {'inherit the primary run' if prefix else default}); "
            "shell commands never get network access"
        ),
    )


def model_options_from_args(
    args: argparse.Namespace,
    *,
    prefix: str = "",
) -> ModelOptions:
    destination_prefix = f"{prefix.replace('-', '_')}_" if prefix else ""
    return ModelOptions(
        model=getattr(args, f"{destination_prefix}model"),
        reasoning_effort=getattr(args, f"{destination_prefix}reasoning_effort"),
        max_budget_usd=getattr(args, f"{destination_prefix}max_budget_usd"),
    )


def semantic_config_digest(
    prompt: str,
    schema_text: str,
    options: ModelOptions,
    *,
    web_search: str = "disabled",
    validation_source: Path | None = None,
) -> str:
    """Identify the inputs that change a run's meaning (not its budget)."""
    payload = {
        "model": options.model,
        "prompt": prompt,
        "reasoning_effort": options.reasoning_effort,
        "schema": schema_text,
        "web_search": web_search,
    }
    if validation_source is not None:
        payload["validation"] = validation_code_digest(validation_source)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validation_code_digest(source: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        Path(validation_common.__file__).resolve(),
        source.resolve(),
        DEFAULT_VALIDATION_PROMPT_PATH.resolve(),
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stage_output_validator(workspace: Path, validator: OutputValidator) -> Path:
    """Stage only the selected checker, shared primitives, and expectations."""
    directory = workspace / "validation"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        Path(validation_common.__file__).resolve(),
        directory / "common.py",
    )
    shutil.copyfile(validator.source.resolve(), directory / "validate.py")
    (directory / "__init__.py").write_text("", encoding="utf-8")
    (directory / validation_common.EXPECTATIONS_FILENAME).write_text(
        json.dumps(
            validator.expectations,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return directory


def with_validation_instructions(prompt: str) -> str:
    """Append the shared validation prompt, naming this driver's Python.

    macOS has no ``python`` command and its ``/usr/bin/python3`` can be too
    old for the staged validator, so the prompt names the exact interpreter.
    """
    try:
        instructions = DEFAULT_VALIDATION_PROMPT_PATH.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise AgentError(
            f"could not read validation prompt {DEFAULT_VALIDATION_PROMPT_PATH}: {exc}"
        ) from exc
    instructions = instructions.replace(
        VALIDATION_PYTHON_PLACEHOLDER, shlex.quote(sys.executable)
    )
    return prompt.rstrip() + "\n\n" + instructions.strip() + "\n"


def resolve_agent_executable(value: str) -> str:
    executable = shutil.which(value)
    if executable is None:
        raise AgentError(
            f"could not find the Claude Code executable {value!r} on PATH"
        )
    return executable


def read_agent_version(claude: str) -> str:
    try:
        completed = subprocess.run(
            [claude, "--version"],
            capture_output=True,
            text=True,
            check=False,
            env=agent_subprocess_environment(),
        )
    except OSError as exc:
        raise AgentError(f"could not run {claude!r}: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or f"exit status {completed.returncode}"
        raise AgentError(f"could not query the Claude Code version: {detail}")
    return completed.stdout.strip()


def agent_subprocess_environment(
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Copy the environment without variables that change Claude's behavior."""
    source = os.environ if environment is None else environment
    return {
        key: value
        for key, value in source.items()
        if key in KEPT_ENVIRONMENT
        or (
            key not in STRIPPED_ENVIRONMENT
            and not key.startswith(STRIPPED_ENVIRONMENT_PREFIX)
        )
    }


def agent_config_dir(environment: Mapping[str, str] | None = None) -> Path:
    """Resolve the Claude Code account directory a run will use."""
    environment = os.environ if environment is None else environment
    configured = environment.get(CONFIG_DIR_ENV)
    path = Path(configured).expanduser() if configured else Path.home() / ".claude"
    return path.resolve()


def agent_config_warnings(environment: Mapping[str, str] | None = None) -> list[str]:
    """Check account selection without starting Claude or changing its files."""
    environment = os.environ if environment is None else environment
    try:
        path = agent_config_dir(environment)
    except (OSError, ValueError) as exc:
        return [f"Could not inspect the Claude config directory: {exc}"]
    if path.is_dir():
        return []
    return [
        f"Claude config directory is not an existing directory: {path}. "
        f"Set {CONFIG_DIR_ENV} in the task's model settings or in the CLI "
        "environment to a logged-in account directory, or run `claude` once "
        "to log in."
    ]


def report_agent_config_warnings() -> None:
    for warning in agent_config_warnings():
        print(f"Warning: {warning}", file=sys.stderr)


def wait_for_launch_slot(interval: float) -> None:
    """Space process startups while allowing already-started runs to overlap."""
    if interval <= 0:
        return
    shared_gate = os.environ.get(LAUNCH_GATE_ENV)
    if shared_gate:
        _wait_for_shared_launch_slot(Path(shared_gate), interval)
        return
    global _next_launch_at
    with _LAUNCH_LOCK:
        delay = max(0.0, _next_launch_at - time.monotonic())
        if delay:
            time.sleep(delay)
        _next_launch_at = time.monotonic() + interval


def _wait_for_shared_launch_slot(path: Path, interval: float) -> None:
    """Coordinate the startup interval across workbench worker processes."""
    import fcntl

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="ascii") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            lock.seek(0)
            raw = lock.read().strip()
            try:
                next_launch = float(raw)
            except ValueError:
                next_launch = 0.0
            delay = max(0.0, next_launch - time.time())
            if delay:
                time.sleep(delay)
            lock.seek(0)
            lock.truncate()
            lock.write(f"{time.time() + interval:.9f}")
            lock.flush()
            os.fsync(lock.fileno())
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


class ResultWatcher:
    """Follow a stream-json log incrementally and keep its last result event.

    Long research turns write megabytes of events; reading only the new
    complete lines keeps each poll proportional to fresh output.
    """

    def __init__(self, events_path: Path):
        self.events_path = events_path
        self.offset = 0
        self.result: dict | None = None

    def poll(self) -> dict | None:
        try:
            with self.events_path.open("rb") as source:
                source.seek(self.offset)
                for line in source:
                    if not line.endswith(b"\n"):
                        break
                    self.offset += len(line)
                    if b'"result"' not in line:
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(event, dict) and event.get("type") == "result":
                        self.result = event
        except OSError:
            pass
        return self.result


def turn_is_complete(
    result_event: dict | None,
    result_path: Path,
    *,
    model_writes_result: bool,
) -> bool:
    """Return whether Claude finished the turn and its result is usable."""
    if result_event is None or result_event.get("is_error"):
        return False
    if not model_writes_result:
        return isinstance(result_event.get("structured_output"), dict)
    try:
        with result_path.open(encoding="utf-8") as source:
            json.load(source)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return False
    return True


def logged_turn_is_complete(events_path: Path, result_path: Path) -> bool:
    """Return whether a saved turn finished and left a valid model-written result."""
    return turn_is_complete(
        ResultWatcher(events_path).poll(),
        result_path,
        model_writes_result=True,
    )


def _stop_agent_process(process: subprocess.Popen) -> None:
    """Ask a Claude process group to stop, then escalate if necessary."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
    except (OSError, ProcessLookupError):
        pass
    for escalate in (process.terminate, process.kill, None):
        try:
            process.wait(timeout=AGENT_STOP_GRACE_SECONDS)
            return
        except subprocess.TimeoutExpired:
            pass
        if escalate is None:
            return
        try:
            escalate()
        except OSError:
            pass


def _run_agent_process(
    command: list[str],
    *,
    prompt: str,
    workspace: Path,
    environment: dict[str, str],
    events,
    log,
    is_complete: Callable[[], bool],
    timeout_seconds: float | None,
    completion_grace_seconds: float | None,
) -> tuple[subprocess.CompletedProcess, bool, bool]:
    """Run Claude while detecting completed turns with lingering tools."""
    process = subprocess.Popen(
        command,
        cwd=workspace,
        env=environment,
        stdin=subprocess.PIPE,
        stdout=events,
        stderr=log,
        text=True,
        encoding="utf-8",
        **popen_session_options(),
    )
    if process.stdin is None:
        _stop_agent_process(process)
        raise AgentError("could not open the Claude prompt input")
    try:
        process.stdin.write(prompt)
    except BrokenPipeError:
        pass
    finally:
        try:
            process.stdin.close()
        except BrokenPipeError:
            pass
    started_at = time.monotonic()
    completed_at: float | None = None
    stopped_after_completion = False
    timed_out = False
    try:
        while process.poll() is None:
            now = time.monotonic()
            if is_complete():
                if completed_at is None:
                    completed_at = now
                elif (
                    completion_grace_seconds is not None
                    and now - completed_at >= completion_grace_seconds
                ):
                    log.write(
                        "\nDriver: structured turn completed but Claude did "
                        "not exit within the grace period; stopping its "
                        "process group.\n"
                    )
                    log.flush()
                    stopped_after_completion = True
                    _stop_agent_process(process)
                    break
            if (
                timeout_seconds is not None
                and completed_at is None
                and now - started_at >= timeout_seconds
            ):
                log.write(
                    f"\nDriver: Claude exceeded the {timeout_seconds:g}-second "
                    "wall-clock timeout; stopping its process group.\n"
                )
                log.flush()
                timed_out = True
                _stop_agent_process(process)
                break
            time.sleep(AGENT_POLL_INTERVAL_SECONDS)
    except BaseException:
        _stop_agent_process(process)
        raise
    return (
        subprocess.CompletedProcess(command, process.returncode),
        stopped_after_completion,
        timed_out,
    )


def build_agent_command(
    *,
    claude: str,
    schema_path: Path,
    options: ModelOptions,
    web_search: str = "disabled",
    model_writes_result: bool = False,
) -> list[str]:
    """Build a headless, sandboxed, configuration-isolated Claude command.

    The prompt goes on standard input: ``--tools`` takes a variable number
    of values and would consume a trailing positional prompt.
    """
    if web_search not in WEB_SEARCH_MODES:
        raise AgentError(
            "web search must be one of " + ", ".join(WEB_SEARCH_MODES)
        )
    web_tools = WEB_TOOLS if web_search == "live" else ()
    command = [
        claude,
        "--print",
        "--safe-mode",
        "--strict-mcp-config",
        "--no-session-persistence",
        "--output-format=stream-json",
        "--verbose",
        "--exclude-dynamic-system-prompt-sections",
        "--permission-mode=acceptEdits",
        "--permission-prompts=none",
        "--settings=" + json.dumps(SANDBOX_SETTINGS, separators=(",", ":")),
        "--tools=" + ",".join((*WORKSPACE_TOOLS, *web_tools)),
        "--allowedTools=" + ",".join(("Glob", "Grep", *web_tools)),
    ]
    if not model_writes_result:
        try:
            schema = json.loads(schema_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise AgentError(f"could not read result schema {schema_path}: {exc}") from exc
        command.append(
            "--json-schema=" + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
        )
    if options.model is not None:
        command.append(f"--model={options.model}")
    if options.reasoning_effort is not None:
        command.append(f"--effort={options.reasoning_effort}")
    if options.max_budget_usd is not None:
        command.append(f"--max-budget-usd={options.max_budget_usd:g}")
    return command


def _result_failure(result_event: dict) -> str:
    errors = result_event.get("errors")
    detail = "; ".join(
        str(item) for item in errors if isinstance(item, str)
    ) if isinstance(errors, list) else ""
    detail = detail or str(result_event.get("result") or "").strip()
    subtype = result_event.get("subtype") or "error"
    return f"{subtype}: {detail}" if detail else str(subtype)


def remove_agent_scratch(workspace: Path) -> None:
    """Delete the project-state folder Claude Code creates in its cwd."""
    shutil.rmtree(workspace / ".claude", ignore_errors=True)


@record_transcript
def run_structured_agent(
    *,
    claude: str,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    result_filename: str = "agent-result.json",
    events_filename: str = "events.jsonl",
    log_filename: str = "run.log",
    options: ModelOptions = DEFAULT_MODEL_OPTIONS,
    web_search: str = "disabled",
    launch_interval: float = AGENT_LAUNCH_INTERVAL_SECONDS,
    timeout_seconds: float | None = None,
    completion_grace_seconds: float | None = None,
    model_writes_result: bool = False,
) -> Path:
    """Run one structured Claude turn and return its result path.

    With ``model_writes_result`` the model writes the result file itself;
    otherwise Claude validates against ``--json-schema`` and the driver
    writes the reported ``structured_output``.
    """
    workspace = workspace.resolve()
    result_path = workspace / result_filename
    events_path = workspace / events_filename
    log_path = workspace / log_filename
    events_path.write_text("", encoding="utf-8")
    log_path.write_text("", encoding="utf-8")
    command = build_agent_command(
        claude=claude,
        schema_path=schema_path,
        options=options,
        web_search=web_search,
        model_writes_result=model_writes_result,
    )
    watcher = ResultWatcher(events_path)

    def is_complete() -> bool:
        return turn_is_complete(
            watcher.poll(),
            result_path,
            model_writes_result=model_writes_result,
        )

    wait_for_launch_slot(launch_interval)
    try:
        with (
            events_path.open("a", encoding="utf-8") as events,
            log_path.open("a", encoding="utf-8") as log,
        ):
            completed, _, timed_out = _run_agent_process(
                command,
                prompt=prompt,
                workspace=workspace,
                environment=agent_subprocess_environment(),
                events=events,
                log=log,
                is_complete=is_complete,
                timeout_seconds=timeout_seconds,
                completion_grace_seconds=completion_grace_seconds,
            )
    except OSError as exc:
        raise AgentError(
            f"could not start Claude Code; workspace preserved at "
            f"{workspace}: {exc}"
        ) from exc
    finally:
        remove_agent_scratch(workspace)

    quota = quota_error(events_path)
    if quota is not None:
        pause_queue_for_quota(quota)
        raise AgentError(
            f"Claude usage exhausted: {quota}; workspace preserved at {workspace}"
        )
    if timed_out:
        raise AgentError(
            f"Claude exceeded the {timeout_seconds:g}-second wall-clock "
            f"timeout; workspace preserved at {workspace}"
        )
    result_event = watcher.poll()
    if result_event is None:
        raise AgentError(
            f"Claude exited with status {completed.returncode} without a "
            f"final result; workspace preserved at {workspace}"
        )
    if result_event.get("is_error"):
        raise AgentError(
            f"Claude stopped with {_result_failure(result_event)}; "
            f"workspace preserved at {workspace}"
        )
    if not model_writes_result:
        structured = result_event.get("structured_output")
        if not isinstance(structured, dict):
            raise AgentError(
                f"Claude returned no structured output; workspace preserved "
                f"at {workspace}"
            )
        result_path.write_text(
            json.dumps(structured, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    return result_path


def _merge_repair_logs(workspace: Path) -> None:
    pairs = (
        ("repair-events.jsonl", "events.jsonl", "repair turn events"),
        ("repair-run.log", "run.log", "repair turn log"),
    )
    for source_name, destination_name, heading in pairs:
        source = workspace / source_name
        if not source.is_file():
            continue
        contents = source.read_text(encoding="utf-8", errors="replace")
        with (workspace / destination_name).open("a", encoding="utf-8") as output:
            output.write(f"\n--- {heading} ---\n")
            output.write(contents)
        source.unlink()


def run_validated_agent(
    *,
    claude: str,
    workspace: Path,
    prompt: str,
    schema_path: Path,
    validator: OutputValidator,
    options: ModelOptions = DEFAULT_MODEL_OPTIONS,
    web_search: str = "disabled",
    launch_interval: float = AGENT_LAUNCH_INTERVAL_SECONDS,
    timeout_seconds: float | None = None,
    completion_grace_seconds: float | None = None,
    repair_turns: int = 1,
) -> validation_common.ValidationReport:
    """Run Claude with in-turn validation and authoritative host rechecking."""
    workspace = workspace.resolve()
    try:
        result_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AgentError(f"could not read result schema for validation: {exc}") from exc
    effective_validator = OutputValidator(
        validator.source,
        validator.validate,
        {**validator.expectations, "result_schema": result_schema},
    )
    stage_output_validator(workspace, effective_validator)
    run_structured_agent(
        claude=claude,
        workspace=workspace,
        prompt=with_validation_instructions(prompt),
        schema_path=schema_path,
        options=options,
        web_search=web_search,
        launch_interval=launch_interval,
        timeout_seconds=timeout_seconds,
        completion_grace_seconds=completion_grace_seconds,
        model_writes_result=True,
    )
    report = effective_validator.validate(
        workspace=workspace,
        expectations=effective_validator.expectations,
    )
    for _ in range(repair_turns):
        if report.valid or not report.repairable:
            break
        repair_prompt = with_validation_instructions(
            "The previous Claude turn left generated output that failed the "
            "authoritative deterministic validator. Preserve all substantive "
            "work and staged inputs. Fix only the reported output-contract "
            "issues, then return a short completion note.\n\n"
            "Validation issues:\n"
            + "\n".join(f"- {issue.render()}" for issue in report.issues)
        )
        run_structured_agent(
            claude=claude,
            workspace=workspace,
            prompt=repair_prompt,
            schema_path=schema_path,
            events_filename="repair-events.jsonl",
            log_filename="repair-run.log",
            options=options,
            web_search=web_search,
            launch_interval=launch_interval,
            timeout_seconds=timeout_seconds,
            completion_grace_seconds=completion_grace_seconds,
            model_writes_result=True,
        )
        _merge_repair_logs(workspace)
        report = effective_validator.validate(
            workspace=workspace,
            expectations=effective_validator.expectations,
        )
    if not report.valid:
        raise AgentError("generated output failed validation:\n" + report.failure_message())
    return report
