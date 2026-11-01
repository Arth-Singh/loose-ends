from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import json
import os
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import support

support.use_resolved_temp_root()

import analyze_papers
import claude_cli
from analyze_papers import (
    AnalysisError,
    analyze_paper,
    discover_paper_directories,
    recover_complete_analysis,
    source_digest,
)


PROMPT_TEMPLATE = analyze_papers.DEFAULT_PROMPT_PATH.read_text(encoding="utf-8")
SCHEMA_TEXT = analyze_papers.DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8")


def make_paper(root: Path, name: str = "arXiv-1706.03762") -> Path:
    paper = root / name
    source = paper / "source"
    source.mkdir(parents=True)
    (paper / "paper.pdf").write_bytes(b"%PDF-test")
    (source / "main.tex").write_text(
        "\\section{Introduction}\nAn open problem remains.\n",
        encoding="utf-8",
    )
    return paper


def write_analysis_outputs(workspace: Path) -> None:
    (workspace / "summary.md").write_text(
        "# Summary\n\nA summary.\n",
        encoding="utf-8",
    )
    (workspace / "results.md").write_text(
        "# Results\n\n## R-001\n\nA result.\n",
        encoding="utf-8",
    )
    (workspace / "open-problems.md").write_text(
        "# Open Problems\n\n## OP-001: A question\n\nUnresolved.\n",
        encoding="utf-8",
    )
    (workspace / "agent-result.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "paper_title": "Test Paper",
                "paper_authors": ["Ada Lovelace", "Alan Turing"],
                "open_problems": [
                    {
                        "id": "OP-001",
                        "title": "A question",
                        "explicitness": "explicit",
                    }
                ],
                "warnings": [],
            }
        ),
        encoding="utf-8",
    )


def claude_events(session_id: str = "s1") -> str:
    events = (
        {
            "type": "system",
            "subtype": "init",
            "session_id": session_id,
            "model": claude_cli.DEFAULT_MODEL,
        },
        {
            "type": "result",
            "subtype": "success",
            "is_error": False,
            "session_id": session_id,
            "total_cost_usd": 0.01,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
    )
    return "".join(json.dumps(event) + "\n" for event in events)


def successful_agent_process(
    calls: list[dict],
    *,
    require_staged_inputs: bool = True,
):
    def run(command, *, prompt, workspace, events, is_complete, **kwargs):
        calls.append({"command": command, "prompt": prompt, "workspace": workspace})
        staged_paper = workspace / analyze_papers.STAGED_PAPER_DIRECTORY
        if (
            require_staged_inputs
            and not (staged_paper / "paper.pdf").is_file()
        ):
            raise AssertionError("paper.pdf was not staged inside the workspace")
        if (
            require_staged_inputs
            and not (staged_paper / "source" / "main.tex").is_file()
        ):
            raise AssertionError("source/main.tex was not staged inside the workspace")
        write_analysis_outputs(workspace)
        events.write(claude_events())
        events.flush()
        if not is_complete():
            raise AssertionError("the runner did not detect the completed turn")
        return subprocess.CompletedProcess(command, 0), False, False

    return run


def failed_agent_process(command, **kwargs):
    return subprocess.CompletedProcess(command, 2), False, False


class PaperDiscoveryTests(unittest.TestCase):
    def test_accepts_a_direct_paper_and_discovers_nested_papers(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = make_paper(root / "author-a", "arXiv-1706.03762")
            second = make_paper(root / "author-b", "arXiv-cs_9910024v2")

            direct = discover_paper_directories([first])
            nested = discover_paper_directories([root])

            self.assertEqual(direct, [first.resolve()])
            self.assertEqual(nested, sorted([first.resolve(), second.resolve()]))

    def test_discovers_non_arxiv_paper_directories(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            paper = make_paper(root / "publisher", "conference-paper-2025")

            self.assertEqual(discover_paper_directories([root]), [paper.resolve()])

    def test_rejects_a_directory_without_downloaded_papers(self):
        with TemporaryDirectory() as temporary:
            with self.assertRaises(AnalysisError):
                discover_paper_directories([Path(temporary)])


class PaperDigestTests(unittest.TestCase):
    def test_digest_changes_with_source_but_not_analysis(self):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            initial = source_digest(paper)

            analysis = paper / "analysis"
            analysis.mkdir()
            (analysis / "summary.md").write_text("generated", encoding="utf-8")
            self.assertEqual(source_digest(paper), initial)

            (paper / "source" / "main.tex").write_text(
                "changed",
                encoding="utf-8",
            )
            self.assertNotEqual(source_digest(paper), initial)


class AnalyzePaperTests(unittest.TestCase):
    @patch.object(claude_cli, "_run_agent_process")
    def test_writes_three_markdown_files_and_manifest(self, run):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            calls: list[dict] = []
            run.side_effect = successful_agent_process(calls)

            outcome = analyze_paper(
                paper,
                claude="claude",
                claude_version="claude-cli test",
                prompt_template=PROMPT_TEMPLATE,
                schema_path=analyze_papers.DEFAULT_SCHEMA_PATH,
                schema_text=SCHEMA_TEXT,
                launch_interval=0,
            )

            self.assertEqual(outcome.status, "analyzed")
            self.assertEqual(outcome.result_count, 1)
            self.assertEqual(outcome.open_problem_count, 1)
            self.assertIn("1 result, 1 open problem", outcome.message)
            analysis = paper / "analysis"
            for filename in analyze_papers.CONTENT_FILES:
                self.assertTrue((analysis / filename).is_file())
            self.assertTrue((analysis / "events.jsonl").is_file())
            self.assertTrue((analysis / "run.log").is_file())
            self.assertEqual(list(analysis.glob(".run-*")), [])

            manifest = json.loads(
                (analysis / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["paper_title"], "Test Paper")
            self.assertEqual(
                manifest["paper_authors"],
                ["Ada Lovelace", "Alan Turing"],
            )
            self.assertEqual(manifest["open_problems"][0]["id"], "OP-001")
            self.assertEqual(manifest["claude_version"], "claude-cli test")
            self.assertIsNone(manifest["requested_model"])
            self.assertIsNone(manifest["requested_reasoning_effort"])
            self.assertIsNone(manifest["requested_max_budget_usd"])

            self.assertEqual(len(calls), 1)
            call = calls[0]
            command = call["command"]
            self.assertEqual(command[0], "claude")
            self.assertEqual(call["workspace"].parent, analysis)
            self.assertTrue(call["workspace"].name.startswith(".run-"))
            self.assertIn("--print", command)
            self.assertIn("--safe-mode", command)
            self.assertIn("--strict-mcp-config", command)
            self.assertIn("--no-session-persistence", command)
            self.assertIn("--output-format=stream-json", command)
            self.assertIn("--permission-mode=acceptEdits", command)
            settings = json.loads(
                next(
                    argument.removeprefix("--settings=")
                    for argument in command
                    if argument.startswith("--settings=")
                )
            )
            self.assertTrue(settings["sandbox"]["enabled"])
            self.assertEqual(settings["sandbox"]["network"]["allowedDomains"], [])
            tools = next(
                argument.removeprefix("--tools=").split(",")
                for argument in command
                if argument.startswith("--tools=")
            )
            self.assertNotIn("WebSearch", tools)
            self.assertNotIn("WebFetch", tools)
            self.assertFalse(
                any(argument.startswith("--json-schema") for argument in command)
            )
            self.assertFalse(
                any(argument.startswith("--add-dir") for argument in command)
            )
            self.assertFalse(
                any(argument.startswith("--model") for argument in command)
            )
            self.assertIn(
                str(call["workspace"] / analyze_papers.STAGED_PAPER_DIRECTORY),
                call["prompt"],
            )
            self.assertIn("-m validation.validate", call["prompt"])

    @patch.object(claude_cli, "_run_agent_process")
    def test_passes_model_reasoning_effort_and_budget_to_claude(self, run):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            calls: list[dict] = []
            run.side_effect = successful_agent_process(calls)

            analyze_paper(
                paper,
                claude="claude",
                claude_version="claude-cli test",
                prompt_template=PROMPT_TEMPLATE,
                schema_path=analyze_papers.DEFAULT_SCHEMA_PATH,
                schema_text=SCHEMA_TEXT,
                options=claude_cli.ModelOptions(
                    model=claude_cli.DEFAULT_MODEL,
                    reasoning_effort="xhigh",
                    max_budget_usd=2.5,
                ),
                launch_interval=0,
            )

            command = calls[0]["command"]
            self.assertIn(f"--model={claude_cli.DEFAULT_MODEL}", command)
            self.assertIn("--effort=xhigh", command)
            self.assertIn("--max-budget-usd=2.5", command)
            manifest = json.loads(
                (paper / "analysis" / "manifest.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(manifest["requested_model"], claude_cli.DEFAULT_MODEL)
            self.assertEqual(manifest["requested_reasoning_effort"], "xhigh")
            self.assertEqual(manifest["requested_max_budget_usd"], 2.5)

    @patch.object(claude_cli, "_run_agent_process")
    def test_skips_a_current_analysis_and_reruns_after_source_change(self, run):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            calls: list[dict] = []
            run.side_effect = successful_agent_process(calls)
            arguments = {
                "claude": "claude",
                "claude_version": "claude-cli test",
                "prompt_template": PROMPT_TEMPLATE,
                "schema_path": analyze_papers.DEFAULT_SCHEMA_PATH,
                "schema_text": SCHEMA_TEXT,
                "launch_interval": 0,
            }

            first = analyze_paper(paper, **arguments)
            second = analyze_paper(paper, **arguments)
            (paper / "source" / "main.tex").write_text(
                "new version",
                encoding="utf-8",
            )
            third = analyze_paper(paper, **arguments)

            self.assertEqual(first.status, "analyzed")
            self.assertEqual(second.status, "current")
            self.assertEqual(second.result_count, 1)
            self.assertEqual(second.open_problem_count, 1)
            self.assertIn("1 result, 1 open problem", second.message)
            self.assertEqual(third.status, "analyzed")
            self.assertEqual(run.call_count, 2)

    @patch.object(claude_cli, "_run_agent_process")
    def test_failed_rerun_preserves_the_previous_analysis(self, run):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            calls: list[dict] = []
            run.side_effect = successful_agent_process(calls)
            arguments = {
                "claude": "claude",
                "claude_version": "claude-cli test",
                "prompt_template": PROMPT_TEMPLATE,
                "schema_path": analyze_papers.DEFAULT_SCHEMA_PATH,
                "schema_text": SCHEMA_TEXT,
                "launch_interval": 0,
            }
            analyze_paper(paper, **arguments)
            old_summary = (paper / "analysis" / "summary.md").read_text(
                encoding="utf-8"
            )
            old_manifest = (paper / "analysis" / "manifest.json").read_text(
                encoding="utf-8"
            )

            (paper / "source" / "main.tex").write_text(
                "changed",
                encoding="utf-8",
            )
            run.side_effect = failed_agent_process
            with self.assertRaisesRegex(AnalysisError, "workspace preserved"):
                analyze_paper(paper, **arguments)

            self.assertEqual(
                (paper / "analysis" / "summary.md").read_text(encoding="utf-8"),
                old_summary,
            )
            self.assertEqual(
                (paper / "analysis" / "manifest.json").read_text(
                    encoding="utf-8"
                ),
                old_manifest,
            )
            self.assertEqual(len(list((paper / "analysis").glob(".run-*"))), 1)

    @patch.object(claude_cli, "_run_agent_process")
    def test_cleanup_failure_does_not_reclassify_installed_analysis(self, run):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            calls: list[dict] = []
            run.side_effect = successful_agent_process(calls)
            real_rmtree = analyze_papers.shutil.rmtree

            def fail_workspace_cleanup(path, *args, **kwargs):
                if Path(path).name.startswith(".run-"):
                    raise PermissionError("sandbox-owned temporary directory")
                return real_rmtree(path, *args, **kwargs)

            with patch.object(
                analyze_papers.shutil,
                "rmtree",
                side_effect=fail_workspace_cleanup,
            ):
                outcome = analyze_paper(
                    paper,
                    claude="claude",
                    claude_version="claude-cli test",
                    prompt_template=PROMPT_TEMPLATE,
                    schema_path=analyze_papers.DEFAULT_SCHEMA_PATH,
                    schema_text=SCHEMA_TEXT,
                    launch_interval=0,
                )

            analysis = paper / "analysis"
            self.assertEqual(outcome.status, "analyzed")
            self.assertIn("temporary workspace preserved", outcome.message)
            self.assertTrue((analysis / "manifest.json").is_file())
            self.assertEqual(len(list(analysis.glob(".run-*"))), 1)
            self.assertIn(
                "Driver cleanup warning:",
                (analysis / "run.log").read_text(encoding="utf-8"),
            )
            real_rmtree(next(analysis.glob(".run-*")))

    def test_recovers_a_complete_run_without_removing_it(self):
        with TemporaryDirectory() as temporary:
            paper = make_paper(Path(temporary))
            analysis = paper / "analysis"
            workspace = analysis / ".run-preserved"
            workspace.mkdir(parents=True)
            write_analysis_outputs(workspace)
            (workspace / "events.jsonl").write_text(
                claude_events(),
                encoding="utf-8",
            )
            (workspace / "run.log").write_text("", encoding="utf-8")

            outcome = recover_complete_analysis(
                paper,
                claude_version="claude-cli test",
            )

            self.assertEqual(outcome.status, "recovered")
            self.assertEqual(outcome.result_count, 1)
            self.assertEqual(outcome.open_problem_count, 1)
            self.assertIn("1 result, 1 open problem", outcome.message)
            self.assertTrue(workspace.is_dir())
            manifest = json.loads(
                (analysis / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertTrue(manifest["recovered_without_config"])
            self.assertTrue(manifest["original_run_preserved"])
            self.assertEqual(
                manifest["recovered_from_run"],
                ".run-preserved",
            )
            self.assertEqual(manifest["claude_version"], "claude-cli test")
            for filename in analyze_papers.CONTENT_FILES:
                self.assertTrue((analysis / filename).is_file())
            self.assertEqual(
                (analysis / "events.jsonl").read_text(encoding="utf-8"),
                claude_events(),
            )


class MainOutputTests(unittest.TestCase):
    def test_dry_run_reports_stale_paper_without_resolving_claude(self):
        with (
            TemporaryDirectory() as temporary,
            patch.object(analyze_papers, "analysis_is_current", return_value=False),
            patch.object(analyze_papers, "resolve_agent_executable") as resolve,
            patch.object(analyze_papers, "analyze_paper") as analyze,
            patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": ""}),
            patch.object(claude_cli.Path, "home", return_value=Path(temporary)),
        ):
            paper = make_paper(Path(temporary))
            output = StringIO()

            error = StringIO()
            with redirect_stdout(output), redirect_stderr(error):
                returncode = analyze_papers.main([str(paper), "--dry-run"])

        self.assertEqual(returncode, 0)
        self.assertIn(f"Would analyze: {paper}", output.getvalue())
        self.assertIn("1 would run; 0 current", output.getvalue())
        self.assertIn("Claude config directory is not an existing directory", error.getvalue())
        self.assertIn(str(Path(temporary) / ".claude"), error.getvalue())
        resolve.assert_not_called()
        analyze.assert_not_called()

    def test_summary_includes_result_and_open_problem_totals(self):
        with (
            TemporaryDirectory() as temporary,
            patch.object(
                analyze_papers,
                "resolve_agent_executable",
                return_value="claude",
            ),
            patch.object(
                analyze_papers,
                "read_agent_version",
                return_value="claude-cli test",
            ),
            patch.object(analyze_papers, "analyze_paper") as analyze,
        ):
            paper = make_paper(Path(temporary))
            analyze.return_value = analyze_papers.AnalysisOutcome(
                paper,
                "current",
                "3 results, 2 open problems; analysis matches",
                result_count=3,
                open_problem_count=2,
            )
            output = StringIO()

            with redirect_stdout(output):
                returncode = analyze_papers.main([str(paper)])

        self.assertEqual(returncode, 0)
        self.assertIn(
            "Totals: 3 results, 2 open problems.",
            output.getvalue(),
        )


if __name__ == "__main__":
    unittest.main()
