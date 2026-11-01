import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import support

support.use_resolved_temp_root()

import claude_cli
import open_problem_common
from validation import common as validation_common
from validation import solver as solver_validation


def solver_result(claim_id: str = "C-001", artifacts=None) -> dict:
    return {
        "claimed_result_type": "partial_result",
        "summary": "A checkable lemma.",
        "external_sources": [],
        "checkable_claims": [
            {
                "id": claim_id,
                "type": "lemma",
                "statement": "The lemma holds.",
                "support": "A derivation is supplied.",
                "remaining_gap": "The main theorem remains.",
            }
        ],
        "prior_claim_dispositions": [],
        "artifacts": artifacts or [],
        "warnings": [],
    }


class ValidationTests(unittest.TestCase):
    def test_rejects_control_characters_in_markdown_and_nested_json(self):
        controls = [chr(code) for code in range(0xA0)
                    if code < 0x20 or code >= 0x7F]
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            markdown = workspace / "attempt.md"
            result_path = workspace / "agent-result.json"
            for control in controls:
                if control in "\t\n\r":
                    continue
                with self.subTest(code=f"U+{ord(control):04X}"):
                    text = f"# Attempt\n\n## C-001\n\n{control}arepsilon\n"
                    markdown.write_text(text, encoding="utf-8")
                    result = solver_result()
                    result["checkable_claims"][0]["support"] = f"{control}arepsilon"
                    result_path.write_text(json.dumps(result), encoding="utf-8")

                    report = solver_validation.validate(workspace=workspace, expectations={})

                    self.assertFalse(report.valid)
                    issues = [issue for issue in report.issues
                              if issue.code == "E_CONTROL_CHARACTER"]
                    self.assertEqual({issue.path for issue in issues}, {
                        "attempt.md", "agent-result.json#/checkable_claims/0/support",
                    })
                    self.assertTrue(all(f"U+{ord(control):04X}" in issue.message
                                        for issue in issues))
                    self.assertTrue(report.repairable)
                    with self.assertRaisesRegex(claude_cli.AgentError, "E_CONTROL_CHARACTER"):
                        open_problem_common.read_json(result_path)
                    with self.assertRaisesRegex(claude_cli.AgentError, "E_CONTROL_CHARACTER"):
                        open_problem_common.validate_markdown(markdown, description="attempt")

    def test_preserves_latex_backslashes_and_normal_text_whitespace(self):
        text = "# Attempt\r\n\r\n## C-001\r\n\r\n\t" + (
            r"\(\varepsilon + \theta + \beta + \frac{a}{b} + \rho\)"
        ) + "\n"
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            markdown = workspace / "attempt.md"
            markdown.write_bytes(text.encode("utf-8"))
            result = solver_result()
            result["checkable_claims"][0]["support"] = text
            result_path = workspace / "agent-result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")

            report = solver_validation.validate(workspace=workspace, expectations={})

            self.assertTrue(report.valid, report.failure_message())
            self.assertEqual(open_problem_common.read_json(result_path), result)
            self.assertIn(r"\varepsilon", open_problem_common.validate_markdown(
                markdown, description="attempt",
            ))

    def test_reports_control_characters_in_json_keys(self):
        reporter = validation_common.Reporter()
        validation_common.validate_control_characters(
            {"bad\vkey": "text"}, reporter, path="agent-result.json#",
        )
        self.assertEqual(len(reporter.issues), 1)
        self.assertIn("U+000B", reporter.issues[0].message)
        self.assertNotIn("\v", reporter.issues[0].render())

    def test_shared_validation_prompt_is_appended(self):
        template = claude_cli.DEFAULT_VALIDATION_PROMPT_PATH.read_text(
            encoding="utf-8"
        ).strip()
        self.assertIn("{{PYTHON}} -m validation.validate", template)
        python = shlex.quote(sys.executable)
        instructions = template.replace("{{PYTHON}}", python)

        combined = claude_cli.with_validation_instructions("Do the task.\n")

        self.assertEqual(combined, f"Do the task.\n\n{instructions}\n")
        self.assertIn(f"{python} -m validation.validate", combined)
        self.assertNotIn("{{PYTHON}}", combined)

    def test_stages_only_selected_checker_and_runs_without_arguments(self):
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "attempt.md").write_text(
                "# Attempt\n\n## C-001\n\nA derivation.\n",
                encoding="utf-8",
            )
            (workspace / "agent-result.json").write_text(
                json.dumps(solver_result()),
                encoding="utf-8",
            )
            validator = claude_cli.OutputValidator(
                Path(solver_validation.__file__).resolve(),
                solver_validation.validate,
                {},
            )
            directory = claude_cli.stage_output_validator(workspace, validator)

            self.assertEqual(
                {path.name for path in directory.iterdir()},
                {"__init__.py", "common.py", "validate.py", "expectations.json"},
            )
            completed = subprocess.run(
                [sys.executable, "-m", "validation.validate"],
                cwd=workspace,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            self.assertEqual(completed.stdout, "Validation passed.\n")

    def test_solver_rejects_loose_ids_and_markdown_artifact_links(self):
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "attempt.md").write_text(
                "# Attempt\n\n## C1\n\nA derivation.\n",
                encoding="utf-8",
            )
            (workspace / "agent-result.json").write_text(
                json.dumps(
                    solver_result(
                        "C1",
                        ["[Verifier](artifacts/verifier.py)"],
                    )
                ),
                encoding="utf-8",
            )

            report = solver_validation.validate(
                workspace=workspace,
                expectations={},
            )

            self.assertFalse(report.valid)
            self.assertIn("E_CLAIM_ID", report.failure_message())
            self.assertIn("E_ARTIFACT_PATH", report.failure_message())

    def test_solver_validates_prior_claim_dispositions(self):
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "attempt.md").write_text(
                "# Attempt\n\n## C-001\n\nA retained cumulative claim.\n",
                encoding="utf-8",
            )
            result = solver_result()
            result["prior_claim_dispositions"] = [
                {
                    "source_attempt": "attempt-001",
                    "source_claim_id": "C-002",
                    "disposition": "retained",
                    "current_claim_id": "C-001",
                    "explanation": "The earlier proof remains valid.",
                }
            ]
            (workspace / "agent-result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )

            valid = solver_validation.validate(
                workspace=workspace,
                expectations={"prior_claim_refs": ["attempt-001/C-002"]},
            )
            self.assertTrue(valid.valid, valid.failure_message())

            result["prior_claim_dispositions"][0]["source_claim_id"] = "C-999"
            result["prior_claim_dispositions"][0]["disposition"] = "refuted"
            (workspace / "agent-result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            invalid = solver_validation.validate(
                workspace=workspace,
                expectations={"prior_claim_refs": ["attempt-001/C-002"]},
            )
            self.assertFalse(invalid.valid)
            self.assertIn("E_PRIOR_CLAIM", invalid.failure_message())
            self.assertIn("E_PRIOR_CURRENT_CLAIM", invalid.failure_message())

    def test_staged_schema_rejects_unknown_result_fields(self):
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            (workspace / "attempt.md").write_text(
                "# Attempt\n\n## C-001\n\nA derivation.\n",
                encoding="utf-8",
            )
            result = solver_result()
            result["unexpected"] = True
            (workspace / "agent-result.json").write_text(
                json.dumps(result),
                encoding="utf-8",
            )
            schema = json.loads(
                (
                    Path(__file__).resolve().parents[1]
                    / "schemas"
                    / "open-problem-solution.schema.json"
                ).read_text(encoding="utf-8")
            )

            report = solver_validation.validate(
                workspace=workspace,
                expectations={"result_schema": schema},
            )

            self.assertFalse(report.valid)
            self.assertIn("E_SCHEMA_PROPERTY", report.failure_message())

    def test_model_owned_result_command_omits_json_schema(self):
        with TemporaryDirectory() as temporary:
            schema = Path(temporary) / "schema.json"
            missing = Path(temporary) / "missing.json"
            model_owned = claude_cli.build_agent_command(
                claude="claude",
                schema_path=missing,
                options=claude_cli.ModelOptions(),
                model_writes_result=True,
            )
            schema.write_text('{"type": "object"}\n', encoding="utf-8")
            driver_owned = claude_cli.build_agent_command(
                claude="claude",
                schema_path=schema,
                options=claude_cli.ModelOptions(),
            )
            with self.assertRaisesRegex(claude_cli.AgentError, "result schema"):
                claude_cli.build_agent_command(
                    claude="claude",
                    schema_path=missing,
                    options=claude_cli.ModelOptions(),
                )

        self.assertFalse(
            any(argument.startswith("--json-schema") for argument in model_owned)
        )
        self.assertIn('--json-schema={"type":"object"}', driver_owned)

    def test_authoritative_failure_gets_one_repair_turn(self):
        with TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            calls = []
            schema = workspace / "schema.json"
            schema.write_text('{"type":"object"}\n', encoding="utf-8")

            def fake_run(**kwargs):
                calls.append(kwargs)
                claim_id = "C1" if len(calls) == 1 else "C-001"
                (workspace / "attempt.md").write_text(
                    f"# Attempt\n\n## {claim_id}\n\nA derivation.\n",
                    encoding="utf-8",
                )
                (workspace / "agent-result.json").write_text(
                    json.dumps(solver_result(claim_id)),
                    encoding="utf-8",
                )
                return workspace / "agent-result.json"

            validator = claude_cli.OutputValidator(
                Path(solver_validation.__file__).resolve(),
                solver_validation.validate,
                {},
            )
            with patch.object(
                claude_cli,
                "run_structured_agent",
                side_effect=fake_run,
            ):
                report = claude_cli.run_validated_agent(
                    claude="claude",
                    workspace=workspace,
                    prompt="Do the task.",
                    schema_path=schema,
                    validator=validator,
                    launch_interval=0,
                )

            self.assertTrue(report.valid)
            self.assertEqual(len(calls), 2)
            self.assertIn("E_CLAIM_ID", calls[1]["prompt"])
            self.assertTrue(calls[0]["model_writes_result"])


if __name__ == "__main__":
    unittest.main()
