from pathlib import Path
import json
import os
import sqlite3
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import support

support.isolate()

import analyze_papers
import claude_cli
from transcripts import record_transcript, RUN_TRANSCRIPT_DIRECTORY, TRANSCRIPT_ENV
import solve_open_problems
import workbench
from workbench_store import WorkbenchStore
from workbench_tasks import _common_arguments, PlanError
from workbench_usage import path_key, result_usage, UsageIndex
import write_paper


OPUS = "claude-opus-5-5"
SONNET = "claude-sonnet-5-5"
COST = .25
# Claude Code reports uncached input, cache reads, and cache writes separately.
RAW_USAGE = {"input_tokens": 300, "cache_read_input_tokens": 600, "cache_creation_input_tokens": 100,
             "output_tokens": 100, "output_tokens_details": {"thinking_tokens": 40}}
USAGE = dict(input_tokens=1000, cached_input_tokens=600,
             cache_write_input_tokens=100, output_tokens=100, reasoning_output_tokens=40)


def events(session, usage=RAW_USAGE, *, cost=COST, model=None, subtype="success"):
    init = {"type": "system", "subtype": "init", "session_id": session}
    if model is not None:
        init["model"] = model
    result = [init, {"type": "assistant", "session_id": session,
                     "message": {"content": [{"type": "text", "text": "Working."}]}}]
    if usage is not None:
        # Like Claude Code, do not put "type" first in result events.
        completed = {"duration_ms": 1200, "type": "result", "subtype": subtype,
                     "is_error": subtype != "success", "session_id": session, "usage": usage}
        if cost is not None:
            completed["total_cost_usd"] = cost
        result.append(completed)
    return "".join(json.dumps(event, separators=(",", ":")) + "\n" for event in result)


class UsageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.papers = self.root / "papers"
        self.paper = self.papers / "paper"
        self.problem = self.paper / "OP-001"
        self.attempt = self.problem / "attempt-001"
        self.attempt.mkdir(parents=True)
        (self.paper / "paper.pdf").touch()
        self.manuscripts = self.root / "manuscripts"
        self.manuscripts.mkdir()
        self.store = WorkbenchStore(self.root / "state" / "workbench.sqlite3")
        self.index = UsageIndex(self.store, [self.papers], self.manuscripts)

    def context(self, directory=None, **overrides):
        directory = directory or self.attempt
        value = {"stage": "Solve", "model": OPUS,
                 "quality": 2, "scopes": sorted(self.index._scopes(directory / "events.jsonl"))}
        return {**value, **overrides}

    def source(self, name, text, context=None):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode())
        self.index.import_source(path, context or self.context())
        return path

    def total(self, *targets):
        return self.index.summary([{"kind": kind, "path": str(path)} for kind, path in
                                   (targets or [("paper", self.paper)])])

    def managed_run(self, *, started=100, finished=200, status="succeeded", action="solve"):
        job = self.store.create_job({}, {"action": action, "title": "Timing", "units": [{
            "label": "Problem", "argv": ["python", "solve.py", "--model", OPUS],
            "cwd": str(self.root), "targets": [{"kind": "problem", "path": str(self.problem)}],
        }]})
        run = job["runs"][0]
        self.store.update_run(run["id"], status=status, started_at=started, finished_at=finished)
        return run

    def archive(self, run, session, *, stage="Solve", model=OPUS, elapsed=None, directory=None):
        archive = Path(run["log_path"]).parent / RUN_TRANSCRIPT_DIRECTORY / ("turn-" + session)
        archive.mkdir(parents=True)
        (archive / "events").write_text(events(session, model=model))
        sources = {"events": str((directory or self.attempt) / "events.jsonl"),
                   "stage": stage, "model": model}
        if elapsed is not None:
            sources["elapsed_seconds"] = elapsed
        (archive / "source.json").write_text(json.dumps(sources))
        return archive

    def test_precise_elapsed_breakdown_and_run_remainder_are_counted_once(self):
        run = self.managed_run()
        self.archive(run, "solve", elapsed=30)
        self.archive(run, "review", stage="Review", model=SONNET, elapsed=20)
        self.index.refresh()
        # Installed copies enrich scope but do not add the same invocation twice.
        self.source("installed-copy", events("solve"))
        total = self.total(("paper", self.paper), ("problem", self.problem), ("attempt", self.attempt))
        self.assertEqual(total["elapsedSeconds"], 100)
        self.assertEqual(total["fallbackElapsedSeconds"], 50)
        self.assertEqual(total["invocations"], 2)
        stages = {row["stage"]: row for row in total["stages"]}
        self.assertEqual(stages["Solve"]["elapsedSeconds"], 30)
        self.assertEqual(stages["Review"]["elapsedSeconds"], 20)
        self.assertEqual(stages["Other run time"]["elapsedSeconds"], 50)
        self.assertTrue(stages["Other run time"]["timingOnly"])
        filtered = self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                      exclude_models=[SONNET, None])
        self.assertEqual(filtered["elapsedSeconds"], 30)
        self.assertEqual(filtered["elapsedFallbackRuns"], 0)

    def test_historical_mixed_models_show_unassigned_time_instead_of_inventing_a_split(self):
        run = self.managed_run()
        self.archive(run, "solve")
        self.archive(run, "review", stage="Review", model=SONNET)
        self.index.refresh()
        total = self.total()
        self.assertEqual(total["elapsedSeconds"], 100)
        self.assertEqual(total["missingElapsed"], 0)
        stages = {row["stage"]: row for row in total["stages"]}
        self.assertIsNone(stages["Solve"]["elapsedSeconds"])
        self.assertIsNone(stages["Review"]["elapsedSeconds"])
        self.assertEqual(stages["Run time · stage unavailable"]["elapsedSeconds"], 100)
        self.assertEqual(self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                           exclude_models=[None])["elapsedSeconds"], 100)
        filtered = self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                      exclude_models=[SONNET])
        self.assertIsNone(filtered["elapsedSeconds"])
        self.assertEqual(filtered["missingElapsed"], 1)

    def test_partial_timing_uses_only_the_remaining_run_time(self):
        run = self.managed_run()
        self.archive(run, "solve", elapsed=30)
        self.archive(run, "review", stage="Review", model=SONNET)
        self.index.refresh()
        total = self.total()
        self.assertEqual(total["elapsedSeconds"], 100)
        self.assertEqual(total["fallbackElapsedSeconds"], 70)
        self.assertEqual({row["stage"]: row["elapsedSeconds"] for row in total["stages"]},
                         {"Review": 70, "Solve": 30})
        filtered = self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                      exclude_models=[OPUS])
        self.assertEqual(filtered["elapsedSeconds"], 70)
        self.assertEqual(filtered["missingElapsed"], 0)

    def test_overlapping_run_durations_are_summed_and_survive_activity_removal(self):
        self.managed_run(started=100, finished=160, action="analyze")
        self.managed_run(started=120, finished=180, action="analyze")
        self.index.refresh()
        self.assertEqual(self.total()["elapsedSeconds"], 120)
        with self.store.connect() as db:
            db.execute("DELETE FROM runs")
            db.execute("DELETE FROM jobs")
        reopened = UsageIndex(self.store, [self.papers], self.manuscripts)
        with patch.object(Path, "open", side_effect=AssertionError("summary read logs")):
            result = reopened.summary([{"kind": "paper", "path": str(self.paper)}])
        self.assertEqual(result["elapsedSeconds"], 120)

    def test_running_elapsed_excludes_queue_waiting_and_uses_current_database_finish_time(self):
        running = self.managed_run(started=100, finished=None, status="running")
        self.managed_run(started=None, finished=None, status="queued")
        self.index.refresh()
        with patch("workbench_usage.time.time", return_value=130):
            self.assertEqual(self.total()["elapsedSeconds"], 30)
            self.assertEqual(self.total()["runningElapsed"], 1)
        self.store.update_run(running["id"], finished_at=140, status="succeeded")
        with patch("workbench_usage.time.time", return_value=180):
            self.assertEqual(self.total()["elapsedSeconds"], 40)
            self.assertEqual(self.total()["runningElapsed"], 0)

    def test_failed_launch_without_events_keeps_its_precise_timing(self):
        run = self.managed_run(finished=110, status="failed")
        folder = self.archive(run, "failed", elapsed=5)
        (folder / "events").unlink()
        self.index.refresh()
        result = self.total()
        self.assertEqual(result["invocations"], 1)
        self.assertEqual(result["missingUsage"], 1)
        self.assertEqual(result["elapsedSeconds"], 10)
        self.assertEqual({row["stage"]: row["elapsedSeconds"] for row in result["stages"]},
                         {"Solve": 5, "Other run time": 5})
        self.assertEqual(self.index.refresh(), 0)

    def test_successful_runs_filter_excludes_all_work_in_unsuccessful_and_active_runs(self):
        succeeded = self.managed_run(started=100, finished=160)
        self.archive(succeeded, "success", elapsed=30)
        for status in ("partial", "failed", "canceled", "interrupted", "running"):
            run = self.managed_run(status=status, finished=None if status == "running" else 200)
            self.archive(run, status, elapsed=20, model=SONNET, stage="Review")
        self.index.refresh()
        # An installed copy must not let a failed run back into the result.
        self.source("failed-installed-copy", events("failed"))
        targets = [{"kind": "paper", "path": str(self.paper)},
                   {"kind": "problem", "path": str(self.problem)}]
        total = self.index.summary(targets, successful_runs_only=True)
        self.assertEqual(total["invocations"], 1)
        self.assertEqual(total["tokens"], USAGE)
        self.assertAlmostEqual(total["estimatedCost"], COST)
        self.assertEqual(total["elapsedSeconds"], 60)
        self.assertTrue(total["successfulRunsOnly"])
        self.assertNotIn("Review", [stage["stage"] for stage in total["stages"]])
        # Keep model and overhead choices.
        self.assertEqual([item["model"] for item in total["modelBreakdown"]], [None, OPUS, SONNET])
        filtered = self.index.summary(targets, successful_runs_only=True,
                                      exclude_models=[OPUS, None])
        self.assertEqual(filtered["invocations"], 0)
        self.assertEqual(filtered["estimatedCost"], 0)
        self.assertEqual(filtered["elapsedSeconds"], 0)
        self.assertEqual(filtered["stages"], [])
        self.assertEqual(self.index.summary(targets)["invocations"], 6)

    def test_successful_filter_uses_live_status_then_retained_status_after_activity_deletion(self):
        run = self.managed_run(status="running", finished=None)
        self.archive(run, "live", elapsed=20)
        self.index.refresh()
        targets = [{"kind": "paper", "path": str(self.paper)}]
        self.assertEqual(self.index.summary(targets, successful_runs_only=True)["invocations"], 0)
        self.store.update_run(run["id"], status="succeeded", finished_at=150)
        self.assertEqual(self.index.summary(targets, successful_runs_only=True)["elapsedSeconds"], 50)
        self.index.refresh()
        with self.store.connect() as db:
            db.execute("DELETE FROM runs")
            db.execute("DELETE FROM jobs")
        reopened = UsageIndex(self.store, [self.papers], self.manuscripts)
        with patch.object(Path, "open", side_effect=AssertionError("summary read logs")):
            total = reopened.summary(targets, successful_runs_only=True)
        self.assertEqual(total["invocations"], 1)
        self.assertEqual(total["elapsedSeconds"], 50)

    def test_successful_filter_for_legacy_turns_and_successful_runs_with_missing_usage(self):
        self.source("complete", events("complete"))
        self.source("errored", events("errored", subtype="error_during_execution"))
        self.source("incomplete", events("incomplete", None))
        run = self.managed_run()
        folder = self.archive(run, "missing")
        (folder / "events").write_text(events("missing", None))
        self.index.refresh()
        total = self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                   successful_runs_only=True)
        self.assertEqual(total["invocations"], 2)
        self.assertEqual(total["missingUsage"], 1)
        self.assertEqual(total["elapsedSeconds"], 100)
        self.assertEqual(total["missingElapsed"], 1)  # Legacy completed turn has no timing.

    def test_successful_filter_keeps_successful_timing_only_runs(self):
        self.managed_run(started=100, finished=140, action="analyze")
        self.managed_run(status="failed", action="analyze")
        self.index.refresh()
        total = self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                   successful_runs_only=True)
        self.assertEqual(total["elapsedSeconds"], 40)
        self.assertIsNone(total["estimatedCost"])

    def test_manuscript_run_fallback_is_not_charged_to_source_problem_or_one_draft(self):
        run = self.managed_run(action="write")
        manuscript = self.manuscripts / "new-result"
        draft = manuscript / "draft-001"
        draft.mkdir(parents=True)
        self.store.append_run_output(run["id"], str(draft / "main.tex"))
        self.index.refresh()
        self.assertIsNone(self.total()["elapsedSeconds"])
        self.assertEqual(self.total(("manuscript", manuscript))["elapsedSeconds"], 100)
        self.assertIsNone(self.total(("draft", draft))["elapsedSeconds"])

    def test_a_partial_attempt_selection_does_not_inherit_a_multi_attempt_run(self):
        run = self.managed_run()
        self.archive(run, "first", elapsed=20)
        self.archive(run, "second", directory=self.problem / "attempt-002")
        self.index.refresh()
        self.assertEqual(self.total()["elapsedSeconds"], 100)
        self.assertEqual(self.total(("attempt", self.attempt))["elapsedSeconds"], 20)

    def test_codex_era_usage_database_keeps_tokens_and_indexes_new_runs(self):
        database = self.root / "old.sqlite3"
        with sqlite3.connect(database) as db:
            db.execute("""CREATE TABLE usage_invocations (id TEXT PRIMARY KEY, stage TEXT,
                model TEXT, tier TEXT, quality INTEGER, usage_json TEXT, completed INTEGER)""")
            db.execute("""CREATE TABLE usage_run_times (id TEXT PRIMARY KEY, started_at REAL,
                finished_at REAL, status TEXT NOT NULL, model TEXT, tier TEXT, stage TEXT)""")
            db.execute("INSERT INTO usage_invocations VALUES ('old','Solve','gpt-6.1-sol','standard',2,?,1)",
                       (json.dumps(USAGE),))
        db.close()
        self.store = WorkbenchStore(database)
        self.index = UsageIndex(self.store, [self.papers], self.manuscripts)
        with self.store.connect() as db:
            db.execute("INSERT INTO usage_scopes VALUES ('old','paper',?)", (path_key(self.paper),))
        self.managed_run(action="analyze")
        self.index.refresh()
        total = self.total()
        self.assertEqual(total["tokens"], USAGE)
        self.assertEqual(total["unpriced"], 1)  # Codex logs carry no reported cost.
        self.assertEqual(total["elapsedSeconds"], 100)
        self.assertEqual(total["missingElapsed"], 1)

    def test_union_of_parent_child_and_duplicate_logs_counts_each_turn_once(self):
        main = events("solve") + "\n--- repair turn events ---\n" + events("repair")
        self.source("installed/events.jsonl", main)
        self.source("archive/solve/events", events("solve"))
        self.source("archive/repair/events", events("repair"), self.context(stage="Solve · Repair"))
        total = self.total(("paper", self.paper), ("problem", self.problem), ("attempt", self.attempt))
        self.assertEqual(total["invocations"], 2)
        self.assertEqual(total["tokens"]["input_tokens"], 2000)
        self.assertEqual(total["tokens"]["output_tokens"], 200)
        self.assertEqual(total["tokens"]["reasoning_output_tokens"], 80)
        self.assertAlmostEqual(total["estimatedCost"], 2 * COST)

    def test_usage_events_are_found_with_any_key_order_or_json_spacing(self):
        spaced = "".join(json.dumps(json.loads(line)) + "\n" for line in events("spaced").splitlines())
        self.source("spaced", spaced)
        self.source("compact", events("compact"))
        total = self.total()
        self.assertEqual((total["invocations"], total["missingUsage"]), (2, 0))
        self.assertEqual(total["tokens"], {key: 2 * value for key, value in USAGE.items()})
        self.assertAlmostEqual(total["estimatedCost"], 2 * COST)

    def test_incremental_import_preserves_partial_lines_and_does_not_reread_unchanged_logs(self):
        path = self.source("live/events", events("live", None))
        self.assertEqual(self.total()["missingUsage"], 1)
        with patch.object(Path, "open", side_effect=AssertionError("read unchanged log")):
            self.assertEqual(self.index.import_source(path, self.context()), 0)
        completion = events("live")[len(events("live", None)):].encode()
        with path.open("ab") as stream:
            stream.write(completion[:20])
        self.index.import_source(path, self.context())
        self.assertEqual(self.total()["missingUsage"], 1)
        with path.open("ab") as stream:
            stream.write(completion[20:])
        self.index.import_source(path, self.context())
        self.assertEqual(self.total()["invocations"], 1)
        self.assertEqual(self.total()["missingUsage"], 0)
        self.assertEqual(self.total()["tokens"], USAGE)

    def test_replacement_retains_history_and_restart_uses_database(self):
        path = self.source("events", events("first"))
        path.write_bytes(events("second").encode())
        self.index.import_source(path, self.context())
        path.unlink()
        reopened = UsageIndex(self.store, [self.papers], self.manuscripts)
        with patch.object(Path, "open", side_effect=AssertionError("summary read filesystem")):
            total = reopened.summary([{"kind": "paper", "path": str(self.paper)}])
        self.assertEqual(total["invocations"], 2)

    def test_unpriced_and_missing_usage_remain_visible(self):
        self.source("custom", events("custom"), self.context(model="custom-model"))
        self.source("unpriced", events("unpriced", cost=None))
        self.source("failed", events("failed", None))
        self.source("bad", events("bad", {**RAW_USAGE, "output_tokens_details": {"thinking_tokens": 500}}))
        self.source("invalid", '{"type":"result","usage":broken}\n')
        total = self.total()
        self.assertEqual(total["invocations"], 5)
        self.assertEqual((total["priced"], total["unpriced"], total["missingUsage"]), (1, 1, 3))
        # Claude Code reports the cost of any model; no rate table is involved.
        self.assertAlmostEqual(total["estimatedCost"], COST)
        self.assertEqual(total["tokens"], {key: 2 * value for key, value in USAGE.items()})
        self.assertEqual(total["models"], [OPUS, "custom-model"])
        self.assertEqual(total["modelBreakdown"], [
            {"model": OPUS, "invocations": 4, "estimatedCost": None},
            {"model": "custom-model", "invocations": 1, "estimatedCost": COST},
        ])
        unpriced = self.index.summary([{"kind": "paper", "path": str(self.paper)}],
                                      exclude_models=["custom-model"])
        self.assertIsNone(unpriced["estimatedCost"])
        self.assertEqual((unpriced["unpriced"], unpriced["missingUsage"]), (1, 3))

    def test_session_init_model_overrides_the_requested_model(self):
        # A requested alias is stored as the model Claude Code reports at startup.
        self.source("alias", events("alias", model=OPUS), self.context(model="opus"))
        self.source("unreported", events("unreported"), self.context(model="sonnet"))
        total = self.total()
        self.assertEqual(total["models"], [OPUS, "sonnet"])
        targets = [{"kind": "paper", "path": str(self.paper)}]
        self.assertEqual(self.index.summary(targets, exclude_models=["opus"])["invocations"], 2)
        self.assertEqual(self.index.summary(targets, exclude_models=[OPUS])["models"], ["sonnet"])

    def test_managed_and_installed_backfill_fills_model_and_attempt_attribution(self):
        job = self.store.create_job({}, {"action": "solve", "title": "Solve", "units": [{
            "label": "Problem", "argv": ["python", "solve.py"], "cwd": str(self.root),
            "targets": [{"kind": "problem", "path": str(self.problem)}],
        }]})
        run = job["runs"][0]
        archive = Path(run["log_path"]).parent / RUN_TRANSCRIPT_DIRECTORY / "turn-1"
        archive.mkdir(parents=True)
        (archive / "source.json").write_text(json.dumps({
            "events": str(self.problem / ".solve-run-x" / "events.jsonl"), "stage": "Solve",
        }))
        (archive / "events").write_text(events("same"))
        (self.attempt / "events.jsonl").write_text(events("same"))
        (self.attempt / "manifest.json").write_text(json.dumps({"requested_model": OPUS}))
        self.assertGreater(self.index.refresh(), 0)
        total = self.total(("job", job["id"]))
        self.assertEqual(total["invocations"], 1)
        self.assertEqual(total["models"], [OPUS])
        self.assertEqual(self.total(("attempt", self.attempt))["invocations"], 1)
        self.assertEqual(self.index.refresh(), 0)
        with self.store.connect() as db:
            db.execute("DELETE FROM runs")
            db.execute("DELETE FROM jobs")
        self.assertEqual(self.total(("paper", self.paper))["invocations"], 1)

    def test_analysis_and_joint_manuscripts_are_not_charged_to_each_source_problem(self):
        analysis = self.paper / "analysis"
        analysis.mkdir()
        self.source("analysis", events("analysis"), self.context(directory=analysis, stage="Analyze"))
        draft = self.manuscripts / "joint" / "draft-001"
        draft.mkdir(parents=True)
        self.source("draft", events("write"), self.context(directory=draft, stage="Write"))
        self.assertEqual(self.total(("problem", self.problem))["invocations"], 0)
        self.assertEqual(self.total()["invocations"], 1)
        self.assertEqual(self.total(("manuscript", draft.parent), ("draft", draft))["invocations"], 1)

    def test_metadata_change_updates_without_changing_tokens(self):
        path = self.source("events", events("known"), self.context(model=None))
        self.assertEqual(self.total()["models"], [])
        self.index.import_source(path, self.context(stage="Review", quality=3))
        total = self.total()
        self.assertEqual(total["invocations"], 1)
        self.assertEqual(total["tokens"], USAGE)
        self.assertAlmostEqual(total["estimatedCost"], COST)
        self.assertEqual([stage["stage"] for stage in total["stages"]], ["Review"])
        self.assertEqual(total["modelBreakdown"], [{"model": OPUS, "invocations": 1, "estimatedCost": COST}])

    def test_model_filter_updates_totals_and_stages_without_losing_choices(self):
        self.source("opus", events("opus"))
        self.source("opus-duplicate", events("opus"))
        self.source("opus-second", events("opus-second"))
        self.source("sonnet", events("sonnet", cost=1.5), self.context(model=SONNET, stage="Review"))
        self.source("custom", events("custom", cost=None), self.context(model="custom-model"))
        self.source("missing", events("missing", None), self.context(model=None))
        targets = [{"kind": "paper", "path": str(self.paper)},
                   {"kind": "problem", "path": str(self.problem)}]
        breakdown = [
            {"model": None, "invocations": 1, "estimatedCost": None},
            {"model": OPUS, "invocations": 2, "estimatedCost": 2 * COST},
            {"model": SONNET, "invocations": 1, "estimatedCost": 1.5},
            {"model": "custom-model", "invocations": 1, "estimatedCost": None},
        ]

        sonnet = self.index.summary(targets, exclude_models=[OPUS, "custom-model", None])
        self.assertEqual(sonnet["invocations"], 1)
        self.assertEqual(sonnet["tokens"], USAGE)
        self.assertAlmostEqual(sonnet["estimatedCost"], 1.5)
        self.assertEqual([item["stage"] for item in sonnet["stages"]], ["Review"])
        self.assertEqual((sonnet["unpriced"], sonnet["missingUsage"]), (0, 0))
        self.assertTrue(sonnet["modelFilterApplied"])
        self.assertEqual(sonnet["modelBreakdown"], breakdown)

        opus = self.index.summary(targets, exclude_models=[SONNET, "custom-model", None])
        self.assertEqual(opus["invocations"], 2)
        self.assertEqual(opus["tokens"]["input_tokens"], 2000)
        self.assertAlmostEqual(opus["estimatedCost"], 2 * COST)
        self.assertEqual(opus["models"], [OPUS])

        unknown = self.index.summary(targets, exclude_models=[OPUS, SONNET])
        self.assertEqual((unknown["invocations"], unknown["unpriced"], unknown["missingUsage"]), (2, 1, 1))
        self.assertIsNone(unknown["estimatedCost"])

        empty = self.index.summary(targets, exclude_models=[OPUS, SONNET, "custom-model", None])
        self.assertEqual(empty["invocations"], 0)
        self.assertEqual(empty["estimatedCost"], 0)
        self.assertFalse(any(empty["tokens"].values()))
        self.assertEqual(empty["stages"], [])
        self.assertEqual(empty["modelBreakdown"], breakdown)
        everything = self.index.summary(targets)
        self.assertEqual(everything["invocations"], 5)
        self.assertAlmostEqual(everything["estimatedCost"], 2 * COST + 1.5)

    def test_large_selection_uses_database_union_without_expression_depth_limit(self):
        self.source("events", events("selected"))
        targets = [{"kind": "problem", "path": str(self.paper / f"OP-{n:04d}")} for n in range(2000)]
        targets.extend([{"kind": "paper", "path": str(self.paper)}] * 3)
        self.assertEqual(self.index.summary(targets)["invocations"], 1)

    def test_stopped_import_is_retried_on_restart(self):
        path = self.root / "events"
        path.write_bytes(events("restart").encode())
        self.index.stopping.set()
        self.assertEqual(self.index.import_source(path, self.context()), 0)
        self.index.stopping.clear()
        self.index.import_source(path, self.context())
        self.assertEqual(self.total()["invocations"], 1)

    def test_usage_request_rejects_invalid_or_outside_targets(self):
        app = object.__new__(workbench.WorkbenchApplication)
        app.allowed_roots = [self.papers, self.manuscripts]
        app.usage = self.index
        for body in ({}, {"targets": []}, {"targets": [None]},
                     {"targets": [{"kind": [], "path": str(self.paper)}]},
                     {"targets": [{"kind": "paper", "path": str(self.root / "outside")}]},
                     {"targets": [{"kind": "unknown", "path": str(self.paper)}]}):
            with self.subTest(body=body), self.assertRaises(PlanError):
                app.usage_summary(body)
        self.assertEqual(app.usage_summary({"targets": [{"kind": "paper", "path": str(self.paper)}]})["invocations"], 0)
        targets = [{"kind": "paper", "path": str(self.paper)}]
        for excluded in (OPUS, {}, [7], [[]], ["m"] * 1001):
            with self.subTest(excluded=excluded), self.assertRaises(PlanError):
                app.usage_summary({"targets": targets, "excludeModels": excluded})
        self.assertTrue(app.usage_summary({"targets": targets, "excludeModels": [None]})["modelFilterApplied"])
        for invalid in (None, "true", 1, [], {}):
            with self.subTest(successfulRunsOnly=invalid), self.assertRaises(PlanError):
                app.usage_summary({"targets": targets, "successfulRunsOnly": invalid})
        self.assertTrue(app.usage_summary({"targets": targets, "successfulRunsOnly": True})["successfulRunsOnly"])


class ClaudeAccountingTests(unittest.TestCase):
    def test_result_usage_counts_cache_subsets_and_reported_cost(self):
        event = {"type": "result", "usage": RAW_USAGE, "total_cost_usd": COST}
        self.assertEqual(result_usage(event), {**USAGE, "cost_usd": COST})
        self.assertEqual(result_usage({**event, "total_cost_usd": 1})["cost_usd"], 1.0)
        self.assertEqual(result_usage({"usage": {"input_tokens": 5, "output_tokens": 2}}), {
            "input_tokens": 5, "cached_input_tokens": 0, "cache_write_input_tokens": 0,
            "output_tokens": 2, "reasoning_output_tokens": 0,
        })
        for cost in (None, -1, float("nan"), float("inf"), "0.25", True):
            with self.subTest(cost=cost):
                self.assertEqual(result_usage({**event, "total_cost_usd": cost}), USAGE)
        for usage in (None, [], {**RAW_USAGE, "input_tokens": True}, {**RAW_USAGE, "output_tokens": -1},
                      {**RAW_USAGE, "cache_read_input_tokens": 1.5},
                      {**RAW_USAGE, "output_tokens_details": {"thinking_tokens": 101}}):
            with self.subTest(usage=usage):
                self.assertIsNone(result_usage({"usage": usage, "total_cost_usd": COST}))

    def test_budget_option_reaches_cli_parsers_and_task_arguments(self):
        for module in (analyze_papers, solve_open_problems, write_paper):
            parser = module.build_parser()
            # Parser defaults/positional inputs differ; inspect the argument group.
            self.assertIn("--max-budget-usd", parser._option_string_actions)
        for module in (solve_open_problems, write_paper):
            self.assertIn("--review-max-budget-usd", module.build_parser()._option_string_actions)
        for value, expected in ((2.5, ["--max-budget-usd", "2.5"]), ("3", ["--max-budget-usd", "3"]),
                                (None, []), ("", [])):
            with self.subTest(value=value):
                argv = []
                _common_arguments(argv, {"maxBudgetUsd": value})
                self.assertEqual(argv, expected)
        for invalid in (0, -1, "abc", True, float("nan"), float("inf"), "inf", [], {}):
            with self.subTest(invalid=invalid), self.assertRaises(PlanError):
                _common_arguments([], {"maxBudgetUsd": invalid})

    def test_invocation_provenance_survives_failure(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            workspace = root / ".solve-run-example"
            workspace.mkdir()

            @record_transcript
            def invoke(**kwargs):
                (workspace / "events.jsonl").write_text(events("failed", None))
                raise RuntimeError("failure")

            with patch.dict(os.environ, {TRANSCRIPT_ENV: str(root / "archive")}):
                with self.assertRaises(RuntimeError):
                    invoke(workspace=workspace, prompt="p", options=claude_cli.ModelOptions(SONNET, "high"))
            folder = next((root / "archive").iterdir())
            metadata = json.loads((folder / "source.json").read_text())
            self.assertEqual(metadata["model"], SONNET)
            self.assertEqual(metadata["reasoning_effort"], "high")
            self.assertGreaterEqual(metadata["elapsed_seconds"], 0)
            self.assertTrue((folder / "events").is_file())


if __name__ == "__main__":
    unittest.main()
