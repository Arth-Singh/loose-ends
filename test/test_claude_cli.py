import argparse
import json
import os
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import support

support.use_resolved_temp_root()

import claude_cli


SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string"}},
    "required": ["status"],
    "additionalProperties": False,
}


def init_event(session="s1", model="claude-opus-5-5"):
    return {"type": "system", "subtype": "init", "session_id": session, "model": model}


def result_event(session="s1", **fields):
    # Real Claude Code output does not put "type" first in result events.
    return {
        "duration_ms": 1200,
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "session_id": session,
        "total_cost_usd": 0.02,
        **fields,
    }


def lines(*events):
    return "".join(json.dumps(event) + "\n" for event in events)


def option_value(command, name):
    prefix = f"--{name}="
    values = [argument[len(prefix):] for argument in command if argument.startswith(prefix)]
    return values[0] if len(values) == 1 else None


class CommandTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.schema_path = self.root / "schema.json"
        self.schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")

    def command(self, **overrides):
        values = {
            "claude": "claude",
            "schema_path": self.schema_path,
            "options": claude_cli.ModelOptions(),
        }
        values.update(overrides)
        return claude_cli.build_agent_command(**values)

    def test_runs_headless_and_isolated_from_user_configuration(self):
        command = self.command()
        self.assertEqual(command[:2], ["claude", "--print"])
        for flag in (
            "--safe-mode",
            "--strict-mcp-config",
            "--no-session-persistence",
            "--verbose",
            "--exclude-dynamic-system-prompt-sections",
        ):
            self.assertIn(flag, command)
        self.assertEqual(option_value(command, "output-format"), "stream-json")
        # The prompt goes on stdin: --tools would consume a positional prompt.
        self.assertTrue(all(argument.startswith("-") or argument == "claude" for argument in command))

    def test_confines_tools_to_the_workspace_without_network(self):
        command = self.command()
        self.assertEqual(option_value(command, "permission-mode"), "acceptEdits")
        self.assertEqual(option_value(command, "permission-prompts"), "none")
        settings = json.loads(option_value(command, "settings"))
        sandbox = settings["sandbox"]
        self.assertTrue(sandbox["enabled"])
        self.assertTrue(sandbox["failIfUnavailable"])
        self.assertFalse(sandbox["allowUnsandboxedCommands"])
        self.assertEqual(sandbox["network"]["allowedDomains"], [])
        self.assertTrue(settings["permissions"]["blockReadsOutsideWorkingDirectories"])
        self.assertEqual(
            option_value(command, "tools").split(","),
            ["Read", "Write", "Edit", "Glob", "Grep", "Bash"],
        )
        # A bare Write/Edit/Bash allow rule would bypass acceptEdits' cwd
        # confinement (verified: it let the Write tool write to $HOME).
        allowed = option_value(command, "allowedTools").split(",")
        self.assertEqual(allowed, ["Glob", "Grep"])

    def test_live_web_search_adds_only_the_web_tools(self):
        command = self.command(web_search="live")
        self.assertEqual(
            option_value(command, "tools").split(","),
            ["Read", "Write", "Edit", "Glob", "Grep", "Bash", "WebSearch", "WebFetch"],
        )
        self.assertEqual(
            option_value(command, "allowedTools").split(","),
            ["Glob", "Grep", "WebSearch", "WebFetch"],
        )
        settings = json.loads(option_value(command, "settings"))
        self.assertEqual(settings["sandbox"]["network"]["allowedDomains"], [])
        with self.assertRaisesRegex(claude_cli.AgentError, "web search must be one of"):
            self.command(web_search="indexed")

    def test_driver_written_results_use_the_json_schema(self):
        command = self.command()
        self.assertEqual(json.loads(option_value(command, "json-schema")), SCHEMA)
        model_owned = self.command(model_writes_result=True)
        self.assertIsNone(option_value(model_owned, "json-schema"))
        with self.assertRaisesRegex(claude_cli.AgentError, "could not read result schema"):
            self.command(schema_path=self.root / "missing.json")

    def test_model_options_become_flags(self):
        command = self.command(
            options=claude_cli.ModelOptions("claude-sonnet-5-5", "xhigh", 2.5)
        )
        self.assertEqual(option_value(command, "model"), "claude-sonnet-5-5")
        self.assertEqual(option_value(command, "effort"), "xhigh")
        self.assertEqual(option_value(command, "max-budget-usd"), "2.5")
        bare = self.command()
        for name in ("model", "effort", "max-budget-usd"):
            self.assertIsNone(option_value(bare, name))


class EnvironmentTests(unittest.TestCase):
    def test_strips_parent_session_and_api_key_variables(self):
        environment = claude_cli.agent_subprocess_environment({
            "PATH": "/usr/bin",
            "HOME": "/Users/test",
            "CLAUDE_CONFIG_DIR": "/accounts/work",
            "CLAUDE_CODE_OAUTH_TOKEN": "token",
            "ANTHROPIC_API_KEY": "secret",
            "ANTHROPIC_MODEL": "claude-haiku-5-5",
            "CLAUDECODE": "1",
            "CLAUDE_EFFORT": "max",
            "CLAUDE_PID": "123",
            "CLAUDE_PLUGIN_DATA": "/plugins",
            "CLAUDE_CODE_EFFORT_LEVEL": "max",
            "CLAUDE_CODE_MESSAGING_SOCKET": "/tmp/socket",
        })
        self.assertEqual(environment, {
            "PATH": "/usr/bin",
            "HOME": "/Users/test",
            "CLAUDE_CONFIG_DIR": "/accounts/work",
            "CLAUDE_CODE_OAUTH_TOKEN": "token",
        })

    def test_config_warnings_only_report_missing_directories(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.assertEqual(
                claude_cli.agent_config_warnings({claude_cli.CONFIG_DIR_ENV: str(root)}),
                [],
            )
            missing = root / "missing"
            warnings = claude_cli.agent_config_warnings({claude_cli.CONFIG_DIR_ENV: str(missing)})
            self.assertEqual(len(warnings), 1)
            self.assertIn("not an existing directory", warnings[0])
            self.assertIn(str(missing), warnings[0])
            with patch.object(claude_cli.Path, "home", return_value=root):
                self.assertEqual(claude_cli.agent_config_dir({}), root / ".claude")
                self.assertEqual(len(claude_cli.agent_config_warnings({})), 1)
                (root / ".claude").mkdir()
                self.assertEqual(claude_cli.agent_config_warnings({}), [])


class ArgumentTests(unittest.TestCase):
    def test_model_arguments_default_and_inherit(self):
        parser = argparse.ArgumentParser()
        claude_cli.add_model_arguments(
            parser,
            default_model="claude-haiku-5-5",
            default_reasoning_effort="low",
        )
        claude_cli.add_model_arguments(parser, prefix="review")
        args = parser.parse_args([])
        self.assertEqual(
            claude_cli.model_options_from_args(args),
            claude_cli.ModelOptions("claude-haiku-5-5", "low", None),
        )
        review = claude_cli.model_options_from_args(args, prefix="review")
        self.assertEqual(review, claude_cli.ModelOptions())
        primary = claude_cli.ModelOptions("claude-opus-5-5", "high", 3.0)
        self.assertEqual(claude_cli.inherit_model_options(primary, review), primary)
        self.assertEqual(
            claude_cli.inherit_model_options(
                primary, claude_cli.ModelOptions("claude-sonnet-5-5", None, None)
            ),
            claude_cli.ModelOptions("claude-sonnet-5-5", "high", 3.0),
        )
        args = parser.parse_args(["--max-budget-usd", "1.5", "--review-reasoning-effort", "max"])
        self.assertEqual(args.max_budget_usd, 1.5)
        self.assertEqual(args.review_reasoning_effort, "max")
        with redirect_stderr(StringIO()):
            for argv in (["--reasoning-effort", "ultra"], ["--max-budget-usd", "0"]):
                with self.subTest(argv=argv), self.assertRaises(SystemExit):
                    parser.parse_args(argv)

    def test_web_search_argument_offers_only_disabled_and_live(self):
        parser = argparse.ArgumentParser()
        claude_cli.add_web_search_argument(parser, default="live")
        self.assertEqual(parser.parse_args([]).web_search, "live")
        with redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            parser.parse_args(["--web-search", "indexed"])
        with self.assertRaises(ValueError):
            claude_cli.add_web_search_argument(argparse.ArgumentParser(), default="indexed")

    def test_digest_tracks_meaning_but_not_budget(self):
        base = claude_cli.ModelOptions("claude-opus-5-5", "high")
        digest = claude_cli.semantic_config_digest("p", "{}", base)
        self.assertEqual(
            digest,
            claude_cli.semantic_config_digest(
                "p", "{}", claude_cli.ModelOptions("claude-opus-5-5", "high", 9.0)
            ),
        )
        for changed in (
            claude_cli.semantic_config_digest("p", "{}", claude_cli.ModelOptions("claude-opus-5-5", "max")),
            claude_cli.semantic_config_digest("p", "{}", base, web_search="live"),
            claude_cli.semantic_config_digest("q", "{}", base),
        ):
            self.assertNotEqual(digest, changed)

    def test_shared_prompt_arguments_separate_direction_from_template(self):
        parser = argparse.ArgumentParser()
        default_template = Path("default-prompt.md")
        claude_cli.add_prompt_arguments(
            parser,
            default_template=default_template,
            task="test agent",
        )
        defaults = parser.parse_args([])
        self.assertIsNone(defaults.prompt)
        self.assertEqual(defaults.prompt_template, default_template)
        custom = parser.parse_args(
            ["--prompt", "Try the dual formulation", "--prompt-template", "custom.md"]
        )
        self.assertEqual(custom.prompt, "Try the dual formulation")
        self.assertEqual(custom.prompt_template, Path("custom.md"))
        rendered = claude_cli.with_user_prompt(
            "Core instructions.\n", custom.prompt, task="test agent"
        )
        self.assertTrue(rendered.startswith("Core instructions.\n"))
        self.assertIn("<user_instruction>", rendered)
        self.assertIn("Try the dual formulation", rendered)
        options = claude_cli.ModelOptions()
        self.assertNotEqual(
            claude_cli.semantic_config_digest("Core instructions.", "{}", options),
            claude_cli.semantic_config_digest(rendered, "{}", options),
        )
        with self.assertRaisesRegex(claude_cli.AgentError, "--prompt must be nonempty"):
            claude_cli.with_user_prompt("Core instructions.", "  ", task="test agent")

    def test_validation_prompt_names_this_interpreter(self):
        prompt = claude_cli.with_validation_instructions("Task.")
        self.assertIn(f"{shlex.quote(sys.executable)} -m validation.validate", prompt)
        self.assertNotIn("{{PYTHON}}", prompt)
        self.assertTrue(prompt.startswith("Task.\n\n"))

    def test_reports_post_parse_error_without_usage(self):
        parser = argparse.ArgumentParser(prog="test-tool.py")
        error_output = StringIO()
        with redirect_stderr(error_output):
            returncode = claude_cli.report_error(
                parser, claude_cli.AgentError("the operation failed")
            )
        self.assertEqual(returncode, 1)
        self.assertEqual(error_output.getvalue(), "test-tool.py: error: the operation failed\n")


class EventTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.events = self.root / "events.jsonl"

    def test_quota_detection_reads_only_limit_signals(self):
        rejected = {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "rejected", "rateLimitType": "five_hour", "resetsAt": 1793898000},
        }
        for event, expected in (
            (rejected, "five_hour"),
            ({"type": "rate_limit_event", "rate_limit_info": {"status": "allowed"}}, None),
            (result_event(is_error=True, subtype="error_during_execution", api_error_status=429), "429"),
            (result_event(is_error=True, result="Claude AI usage limit reached"), "usage limit"),
            (result_event(is_error=True, errors=["You've hit your limit"]), "hit your limit"),
            (result_event(result="The paper discusses a usage limit."), None),
            (result_event(is_error=True, subtype="error_max_budget_usd", errors=["Reached maximum budget"]), None),
            ({"type": "assistant", "message": {"content": [{"type": "text", "text": "usage limit"}]}}, None),
        ):
            with self.subTest(event=event):
                self.events.write_text("not json\n[]\n" + lines(event), encoding="utf-8")
                found = claude_cli.quota_error(self.events)
                if expected is None:
                    self.assertIsNone(found)
                else:
                    self.assertIn(expected, found)

    def test_result_watcher_reads_complete_lines_incrementally(self):
        watcher = claude_cli.ResultWatcher(self.events)
        self.assertIsNone(watcher.poll())
        partial = json.dumps(result_event())
        with self.events.open("w", encoding="utf-8") as stream:
            stream.write(lines(init_event()) + partial[:20])
        self.assertIsNone(watcher.poll())
        consumed = watcher.offset
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(partial[20:] + "\n")
        self.assertEqual(watcher.poll()["session_id"], "s1")
        self.assertGreater(watcher.offset, consumed)
        with self.events.open("a", encoding="utf-8") as stream:
            stream.write(lines(result_event(session="s2")))
        self.assertEqual(watcher.poll()["session_id"], "s2")

    def test_turn_completion_requires_a_usable_result(self):
        result_path = self.root / "agent-result.json"
        self.assertFalse(claude_cli.turn_is_complete(None, result_path, model_writes_result=False))
        self.assertFalse(claude_cli.turn_is_complete(
            result_event(is_error=True), result_path, model_writes_result=False
        ))
        self.assertFalse(claude_cli.turn_is_complete(result_event(), result_path, model_writes_result=False))
        self.assertTrue(claude_cli.turn_is_complete(
            result_event(structured_output={"status": "done"}), result_path, model_writes_result=False
        ))
        self.assertFalse(claude_cli.turn_is_complete(result_event(), result_path, model_writes_result=True))
        result_path.write_text("{", encoding="utf-8")
        self.assertFalse(claude_cli.turn_is_complete(result_event(), result_path, model_writes_result=True))
        result_path.write_text('{"status":"done"}\n', encoding="utf-8")
        self.assertTrue(claude_cli.turn_is_complete(result_event(), result_path, model_writes_result=True))
        self.events.write_text(lines(init_event(), result_event()), encoding="utf-8")
        self.assertTrue(claude_cli.logged_turn_is_complete(self.events, result_path))


class RunStructuredAgentTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.workspace = Path(temporary.name)
        self.schema_path = self.workspace / "schema.json"
        self.schema_path.write_text(json.dumps(SCHEMA), encoding="utf-8")
        launch = patch.object(claude_cli, "wait_for_launch_slot")
        launch.start()
        self.addCleanup(launch.stop)

    def run_with_events(self, events_text, *, returncode=0, **kwargs):
        def fake_process(command, **process_kwargs):
            (self.workspace / ".claude").mkdir(exist_ok=True)
            process_kwargs["events"].write(events_text)
            process_kwargs["events"].flush()
            self.command = command
            self.prompt = process_kwargs["prompt"]
            return subprocess.CompletedProcess(command, returncode), False, False

        with patch.object(claude_cli, "_run_agent_process", side_effect=fake_process):
            return claude_cli.run_structured_agent(
                claude="claude",
                workspace=self.workspace,
                prompt="Return a status.",
                schema_path=self.schema_path,
                **kwargs,
            )

    def test_driver_writes_structured_output_and_removes_scratch(self):
        path = self.run_with_events(
            lines(init_event(), result_event(structured_output={"status": "done"}))
        )
        self.assertEqual(path, self.workspace / "agent-result.json")
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "done"})
        self.assertEqual(self.prompt, "Return a status.")
        self.assertIsNotNone(option_value(self.command, "json-schema"))
        self.assertFalse((self.workspace / ".claude").exists())

    def test_reports_missing_result_error_result_and_missing_output(self):
        for events_text, message in (
            (lines(init_event()), "without a final result"),
            (lines(result_event(
                is_error=True, subtype="error_max_budget_usd",
                errors=["Reached maximum budget ($0.5)"],
            )), r"error_max_budget_usd: Reached maximum budget \(\$0.5\)"),
            (lines(result_event()), "no structured output"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(claude_cli.AgentError, message):
                    self.run_with_events(events_text, returncode=1)

    def test_model_written_result_skips_the_schema_flag(self):
        (self.workspace / "agent-result.json").write_text('{"status":"done"}\n', encoding="utf-8")
        path = self.run_with_events(lines(init_event(), result_event()), model_writes_result=True)
        self.assertIsNone(option_value(self.command, "json-schema"))
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"status": "done"})

    def test_quota_exhaustion_pauses_queue_and_preserves_pending_work(self):
        from workbench_store import WorkbenchStore

        store = WorkbenchStore(self.workspace / "workbench.sqlite3")
        rejected = {
            "type": "rate_limit_event",
            "rate_limit_info": {"status": "rejected", "rateLimitType": "seven_day"},
        }
        with patch.dict(os.environ, {claude_cli.WORKBENCH_DATABASE_ENV: str(store.database)}):
            with self.assertRaisesRegex(claude_cli.AgentError, "usage exhausted"):
                self.run_with_events(lines(init_event(), rejected), returncode=1)
        reopened = WorkbenchStore(store.database)
        settings = reopened.scheduler_settings()
        self.assertEqual(settings["queuePauseReason"], "quota")
        self.assertTrue(settings["queuePaused"])
        self.assertIn("seven_day", settings["quotaError"])
        self.assertIsNone(reopened.claim_next_run(set()))
        reopened.update_scheduler_settings(queue_paused=False)
        self.assertFalse(reopened.scheduler_settings()["queuePaused"])
        self.assertIsNone(reopened.scheduler_settings()["quotaError"])


class ProcessTests(unittest.TestCase):
    def run_process(self, process, *, is_complete, timeout_seconds, monotonic, log=None):
        log = log if log is not None else StringIO()
        with (
            patch.object(subprocess, "Popen", return_value=process),
            patch.object(claude_cli, "_stop_agent_process") as stop,
            patch.object(claude_cli.time, "monotonic", side_effect=monotonic),
            patch.object(claude_cli.time, "sleep"),
        ):
            outcome = claude_cli._run_agent_process(
                ["claude"],
                prompt="test prompt",
                workspace=Path("."),
                environment={},
                events=StringIO(),
                log=log,
                is_complete=is_complete,
                timeout_seconds=timeout_seconds,
                completion_grace_seconds=1,
            )
        return outcome, stop

    def test_stops_lingering_process_after_completed_turn(self):
        process = Mock()
        process.poll.return_value = None
        process.returncode = -1
        log = StringIO()
        (completed, stopped, timed_out), stop = self.run_process(
            process, is_complete=lambda: True, timeout_seconds=10, monotonic=[0, 0, 2], log=log
        )
        stop.assert_called_once_with(process)
        self.assertTrue(stopped)
        self.assertFalse(timed_out)
        self.assertEqual(completed.returncode, -1)
        self.assertIn("structured turn completed", log.getvalue())
        process.stdin.write.assert_called_once_with("test prompt")

    def test_stops_process_at_wall_clock_timeout(self):
        process = Mock()
        process.poll.return_value = None
        process.returncode = -1
        (_, stopped, timed_out), stop = self.run_process(
            process, is_complete=lambda: False, timeout_seconds=1, monotonic=[0, 2]
        )
        stop.assert_called_once_with(process)
        self.assertFalse(stopped)
        self.assertTrue(timed_out)

    def test_returns_normally_when_the_process_exits(self):
        process = Mock()
        process.poll.side_effect = [None, 0]
        process.returncode = 0
        (completed, stopped, timed_out), stop = self.run_process(
            process, is_complete=lambda: False, timeout_seconds=None, monotonic=[0, 0]
        )
        stop.assert_not_called()
        self.assertEqual(completed.returncode, 0)
        self.assertFalse(stopped)
        self.assertFalse(timed_out)


if __name__ == "__main__":
    unittest.main()
