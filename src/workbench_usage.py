"""Persistent, rebuildable token and cost accounting for saved Claude invocations.

Logs are imported in the background, never on a summary request. Claude
session IDs deduplicate archived, installed, and merged repair logs. The index retains
observed history even when a job or an installed log is removed or replaced.
"""

from collections import defaultdict
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import threading
import time

from transcripts import RUN_TRANSCRIPT_DIRECTORY, transcript_stage
from write_paper import resolve_manifest_path


TOKEN_FIELDS = (
    "input_tokens", "cached_input_tokens", "cache_write_input_tokens",
    "output_tokens", "reasoning_output_tokens",
)
# Parse only session-start and result lines. Claude Code does not put "type"
# first in result events, and other writers may add spaces after colons, so
# search anywhere in the line, tolerating JSON whitespace.
USAGE_EVENT_RE = re.compile(rb'"subtype"\s*:\s*"init"|"type"\s*:\s*"result"')
ENTITY_RE = re.compile(r"^(OP-\d+|attempt-\d+|draft-\d+)$")


def path_key(value):
    return os.path.normcase(str(resolve_manifest_path(str(value)))).replace("\\", "/").rstrip("/")


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def elapsed_value(value):
    return value if type(value) in (int, float) and math.isfinite(value) and value >= 0 else None


def _count(source, key):
    value = source.get(key, 0) if isinstance(source, dict) else 0
    return value if type(value) is int and value >= 0 else None


def result_usage(event):
    """Convert a Claude Code result event into stored counters and its cost.

    ``input_tokens`` counts all input, including cache reads and writes, so
    the cache counters are subsets of it. Thinking is part of output. The
    cost is Claude Code's own API-equivalent estimate for the invocation.
    """
    usage = event.get("usage")
    if not isinstance(usage, dict):
        return None
    uncached = _count(usage, "input_tokens")
    reads = _count(usage, "cache_read_input_tokens")
    writes = _count(usage, "cache_creation_input_tokens")
    output = _count(usage, "output_tokens")
    thinking = _count(usage.get("output_tokens_details"), "thinking_tokens")
    if None in (uncached, reads, writes, output, thinking) or thinking > output:
        return None
    result = {
        "input_tokens": uncached + reads + writes,
        "cached_input_tokens": reads,
        "cache_write_input_tokens": writes,
        "output_tokens": output,
        "reasoning_output_tokens": thinking,
    }
    cost = event.get("total_cost_usd")
    if type(cost) in (int, float) and math.isfinite(cost) and cost >= 0:
        result["cost_usd"] = float(cost)
    return result


class UsageIndex:
    def __init__(self, store, roots, manuscripts, hub=None):
        self.store = store
        self.roots = [Path(root).resolve() for root in roots]
        self.manuscripts = Path(manuscripts).resolve()
        self.hub = hub
        self.stopping = threading.Event()
        self.pending = threading.Event()
        self.thread = None
        self.indexing = True
        self.error = ""
        self.updated_at = None
        self._json_cache = {}
        self._run_cache = {}
        with store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS usage_sources (
                    path TEXT PRIMARY KEY, signature TEXT NOT NULL,
                    offset INTEGER NOT NULL, state_json TEXT NOT NULL,
                    tail TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS usage_invocations (
                    id TEXT PRIMARY KEY, stage TEXT NOT NULL, model TEXT,
                    quality INTEGER NOT NULL, usage_json TEXT,
                    completed INTEGER NOT NULL, elapsed_seconds REAL
                );
                CREATE TABLE IF NOT EXISTS usage_scopes (
                    invocation_id TEXT NOT NULL REFERENCES usage_invocations(id),
                    kind TEXT NOT NULL, path TEXT NOT NULL,
                    PRIMARY KEY (invocation_id, kind, path)
                );
                CREATE INDEX IF NOT EXISTS usage_scope_lookup ON usage_scopes(kind, path);
                CREATE TABLE IF NOT EXISTS usage_run_times (
                    id TEXT PRIMARY KEY, started_at REAL, finished_at REAL,
                    status TEXT NOT NULL, model TEXT, stage TEXT
                );
                CREATE TABLE IF NOT EXISTS usage_run_scopes (
                    run_id TEXT NOT NULL REFERENCES usage_run_times(id),
                    kind TEXT NOT NULL, path TEXT NOT NULL,
                    PRIMARY KEY (run_id, kind, path)
                );
                CREATE INDEX IF NOT EXISTS usage_run_scope_lookup ON usage_run_scopes(kind, path);
            """)
            columns = {row["name"] for row in db.execute("PRAGMA table_info(usage_invocations)")}
            if "elapsed_seconds" not in columns:
                db.execute("ALTER TABLE usage_invocations ADD COLUMN elapsed_seconds REAL")

    def start(self):
        self.thread = threading.Thread(target=self._loop, name="usage-index", daemon=True)
        self.thread.start()

    def close(self):
        self.stopping.set()
        self.pending.set()
        if self.thread:
            self.thread.join(timeout=5)

    def request_refresh(self):
        self.pending.set()

    def _loop(self):
        while not self.stopping.is_set():
            self.pending.clear()
            self.indexing = True
            try:
                self.refresh()
                self.error = ""
                self.updated_at = time.time()
            except Exception as exc:
                self.error = str(exc)
                logging.exception("Could not index Claude usage")
            finally:
                self.indexing = False
                if self.hub:
                    self.hub.publish("usage.changed")
            self.pending.wait(30)

    def _json(self, path):
        try:
            stat = path.stat()
        except OSError:
            return {}
        signature = stat.st_mtime_ns, stat.st_size
        cached = self._json_cache.get(path)
        if cached is None or cached[0] != signature:
            cached = signature, read_json(path)
            self._json_cache[path] = cached
        return cached[1]

    def _scopes(self, events):
        """Attribute by actual workspace, never all inputs of a batch job."""
        scopes = set()
        for directory in events.parents:
            if ENTITY_RE.fullmatch(directory.name):
                if directory.name.startswith("OP-"):
                    scopes.add(("problem", path_key(directory)))
                    scopes.add(("paper", path_key(directory.parent)))
                    break
                elif directory.name.startswith("attempt-"):
                    scopes.add(("attempt", path_key(directory)))
                else:
                    scopes.add(("draft", path_key(directory)))
                    scopes.add(("manuscript", path_key(directory.parent)))
                    break
            if directory.parent == self.manuscripts:
                scopes.add(("manuscript", path_key(directory)))
            if any(directory == root or root in directory.parents for root in self.roots):
                if (directory / "paper.pdf").is_file() or (directory / "source").is_dir():
                    scopes.add(("paper", path_key(directory)))
                    break
        return scopes

    def _installed_context(self, events):
        prefix = events.name.removesuffix("events.jsonl")
        if prefix == "repair-":
            prefix = ""
        manifest = self._json(events.parent / f"{prefix}manifest.json")
        if prefix == "review-" and events.parent.name.startswith("draft-"):
            manifest = self._json(events.parent / "paper-review-manifest.json")
        return {"stage": transcript_stage(events) or "Claude",
                "model": manifest.get("requested_model"),
                "quality": 2, "scopes": sorted(self._scopes(events))}

    def _managed_context(self, sources, run):
        events = resolve_manifest_path(sources["events"])
        scopes = self._scopes(events)
        scopes.update({("run", run["id"]), ("job", run["job_id"])})
        stage = sources.get("stage") or transcript_stage(events) or "Claude"
        argv = json.loads(run["argv_json"])
        review = stage.startswith(("Review", "Paper review")) and run["action"] in {"solve", "write", "revise"}
        prefix = "--review-" if review else "--"

        def option(flag):
            for index, arg in enumerate(argv):
                if arg == flag and index + 1 < len(argv):
                    return argv[index + 1]
                if arg.startswith(flag + "="):
                    return arg[len(flag) + 1:]
            return None

        model = sources.get("model") or option(prefix + "model") or option("--model")
        # A review command naming an attempt is precise even before installation.
        targets = json.loads(run["targets_json"])
        if len(targets) == 1 and targets[0].get("kind") in {"attempt", "draft"}:
            scopes.add((targets[0]["kind"], path_key(targets[0]["path"])))
        return {"stage": stage, "model": model,
                "quality": 3 if sources.get("model") else 1, "scopes": sorted(scopes),
                "elapsed_seconds": elapsed_value(sources.get("elapsed_seconds"))}

    def _store_run_times(self, runs):
        """Retain run timestamps independently of Activity history."""
        pending = {run["id"]: tuple(run) for run in runs
                   if self._run_cache.get(run["id"]) != tuple(run)}
        if not pending:
            return
        with self.store.connect() as db:
            for run in runs:
                if run["id"] not in pending:
                    continue
                argv = json.loads(run["argv_json"])

                def option(name):
                    for index, arg in enumerate(argv):
                        if arg == name and index + 1 < len(argv):
                            return argv[index + 1]
                        if arg.startswith(name + "="):
                            return arg[len(name) + 1:]
                    return None

                model = option("--model")
                if option("--review-model") not in (None, model):
                    model = None
                stage = {"analyze": "Analyze", "metadata": "Extract metadata",
                         "literature": "Literature review", "review": "Review"}.get(run["action"])
                if run["action"] == "triage" and "--solve" not in argv:
                    stage = "Triage"
                if run["action"] == "solve" and option("--review") == "none":
                    stage = "Solve"
                scopes = {("run", run["id"]), ("job", run["job_id"])}
                if run["action"] not in {"write", "revise"}:
                    for target in json.loads(run["targets_json"]):
                        if target.get("kind") in {"paper", "problem", "attempt", "draft"}:
                            path = resolve_manifest_path(target["path"])
                            scopes.add((target["kind"], path_key(path)))
                            scopes.update(self._scopes(path / "events.jsonl"))
                else:
                    for target in json.loads(run["targets_json"]):
                        if target.get("kind") == "draft":
                            scopes.add(("manuscript", path_key(resolve_manifest_path(target["path"]).parent)))
                # Output ancestors identify manuscripts without charging their
                # source problems, or a whole multi-draft run to a single draft.
                for output in json.loads(run["outputs_json"]):
                    path = resolve_manifest_path(output)
                    scopes.update((kind, value) for kind, value in self._scopes(path / "events.jsonl")
                                  if kind in {"paper", "problem", "manuscript"})
                db.execute("""INSERT INTO usage_run_times
                    (id, started_at, finished_at, status, model, stage) VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET started_at=excluded.started_at,
                        finished_at=excluded.finished_at, status=excluded.status,
                        model=excluded.model, stage=excluded.stage
                    WHERE started_at IS NOT excluded.started_at OR finished_at IS NOT excluded.finished_at
                        OR status IS NOT excluded.status OR model IS NOT excluded.model
                        OR stage IS NOT excluded.stage""",
                    (run["id"], run["started_at"], run["finished_at"], run["status"], model, stage))
                db.executemany("INSERT OR IGNORE INTO usage_run_scopes VALUES (?, ?, ?)",
                               [(run["id"], kind, value) for kind, value in scopes])
        self._run_cache.update(pending)

    def refresh(self):
        """Discover sources; import only new/changed files, in short transactions."""
        with self.store.connect() as db:
            runs = db.execute("""SELECT runs.id, runs.job_id, runs.argv_json,
                runs.targets_json, runs.outputs_json, runs.log_path, runs.started_at,
                runs.finished_at, runs.status, jobs.action FROM runs
                JOIN jobs ON jobs.id = runs.job_id""").fetchall()
            known = {row["path"]: dict(row) for row in db.execute("SELECT * FROM usage_sources")}
        changed = 0
        self._store_run_times(runs)
        for run in runs:
            if self.stopping.is_set():
                return changed
            root = resolve_manifest_path(run["log_path"]).parent / RUN_TRANSCRIPT_DIRECTORY
            for folder in root.glob("turn-*"):
                sources = self._json(folder / "source.json")
                if not isinstance(sources.get("events"), str):
                    continue
                events = folder / "events"
                if not events.is_file():
                    events = resolve_manifest_path(sources["events"])
                    if not any(events.is_relative_to(root) or events.is_relative_to(path)
                               for path in [*self.roots, self.manuscripts]):
                        continue
                context = self._managed_context(sources, run)
                if context["elapsed_seconds"] is not None:
                    try:
                        empty = events.stat().st_size == 0
                    except OSError:
                        empty = True
                    if empty:
                        changed += self._import_empty_invocation(folder, context)
                        continue
                changed += self.import_source(events, context, known)
        # Installed logs also cover CLI runs predating managed transcript archives.
        for root in dict.fromkeys([*self.roots, self.manuscripts]):
            for directory, children, files in os.walk(root):
                if self.stopping.is_set():
                    return changed
                children[:] = [name for name in children if name not in {
                    "source", "artifacts", "figures", "code", "paper-input", ".git", ".loose-ends",
                } and not name.startswith((".attempt-install-", ".review-install-", ".draft-install-"))]
                for name in files:
                    if name.endswith("events.jsonl"):
                        events = Path(directory) / name
                        changed += self.import_source(events, self._installed_context(events), known)
        return changed

    def _import_empty_invocation(self, folder, context):
        """A failed launch can have precise timing without any Claude events."""
        identity = "invocation:" + path_key(folder)
        with self.store.connect() as db:
            changed = db.execute("""INSERT INTO usage_invocations
                (id, stage, model, quality, usage_json, completed, elapsed_seconds)
                VALUES (?, ?, ?, ?, NULL, 0, ?)
                ON CONFLICT(id) DO UPDATE SET elapsed_seconds=excluded.elapsed_seconds,
                    model=excluded.model
                WHERE elapsed_seconds IS NOT excluded.elapsed_seconds OR model IS NOT excluded.model""",
                (identity, context["stage"], context["model"],
                 context["quality"], context["elapsed_seconds"])).rowcount
            db.executemany("INSERT OR IGNORE INTO usage_scopes VALUES (?, ?, ?)",
                           [(identity, kind, value) for kind, value in context["scopes"]])
        return changed

    def import_source(self, events, context, known=None):
        """Incrementally consume complete lines; reset safely on replacement."""
        key = path_key(events)
        try:
            stat = events.stat()
        except OSError:
            return 0
        context_json = json.dumps(context, sort_keys=True)
        signature = json.dumps([stat.st_size, stat.st_mtime_ns, stat.st_ino, context_json])
        if known is None:
            with self.store.connect() as db:
                row = db.execute("SELECT * FROM usage_sources WHERE path = ?", (key,)).fetchone()
                prior = dict(row) if row else None
        else:
            prior = known.get(key)
        if prior and prior["signature"] == signature:
            return 0
        offset, state = 0, {}
        records = {}
        try:
            with events.open("rb") as stream:
                if prior:
                    old = json.loads(prior["signature"])
                    if old[0] < stat.st_size and old[2:] == [stat.st_ino, context_json]:
                        stream.seek(max(0, prior["offset"] - 128))
                        tail = stream.read(min(128, prior["offset"]))
                        if hashlib.sha256(tail).hexdigest() == prior["tail"]:
                            offset = prior["offset"]
                            state = json.loads(prior["state_json"])
                stream.seek(offset)
                while not self.stopping.is_set():
                    line = stream.readline()
                    if not line or not line.endswith(b"\n"):
                        break
                    offset = stream.tell()
                    if not USAGE_EVENT_RE.search(line):
                        continue
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    session = event.get("session_id")
                    identity = (f"session:{session}" if isinstance(session, str) and session
                                else f"source:{key}")
                    if event.get("type") == "system" and event.get("subtype") == "init":
                        model = event.get("model") if isinstance(event.get("model"), str) else None
                        state = {"session": identity, "model": model}
                        records.setdefault(identity, (None, False, model))
                    elif event.get("type") == "result":
                        model = state.get("model") if state.get("session") == identity else None
                        completed = event.get("subtype") == "success" and not event.get("is_error")
                        records[identity] = (result_usage(event), completed, model)
                if not records and not state and stat.st_size:
                    records[f"source:{key}"] = (None, False, None)
                stream.seek(max(0, offset - 128))
                tail_digest = hashlib.sha256(stream.read(min(128, offset))).hexdigest()
        except OSError:
            return 0
        if self.stopping.is_set():
            return 0  # Do not mark an interrupted import as fully consumed.
        with self.store.connect() as db:
            for identity, (usage, completed, model) in records.items():
                db.execute("""INSERT INTO usage_invocations
                    (id, stage, model, quality, usage_json, completed)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                    stage = CASE WHEN (excluded.quality >= quality AND stage NOT LIKE '%Repair%')
                        OR excluded.stage LIKE '%Repair%'
                        THEN excluded.stage ELSE stage END,
                    model = CASE WHEN excluded.model IS NOT NULL AND
                        (model IS NULL OR excluded.quality >= quality) THEN excluded.model ELSE model END,
                    quality = MAX(quality, excluded.quality),
                    usage_json = COALESCE(excluded.usage_json, usage_json),
                    completed = MAX(completed, excluded.completed)
                    """, (identity, context["stage"], model or context.get("model"),
                          context["quality"], json.dumps(usage) if usage is not None else None, int(completed)))
                db.executemany("INSERT OR IGNORE INTO usage_scopes VALUES (?, ?, ?)",
                               [(identity, kind, path) for kind, path in context["scopes"]])
            duration = elapsed_value(context.get("elapsed_seconds"))
            if duration is not None and len(records) == 1:
                db.execute("UPDATE usage_invocations SET elapsed_seconds = ? WHERE id = ?",
                           (duration, next(iter(records))))
            # New scopes (e.g. a job's installed artifact) apply to earlier turns
            # too when context changes; a reset above reparses the whole source.
            db.execute("INSERT OR REPLACE INTO usage_sources VALUES (?, ?, ?, ?, ?)",
                       (key, signature, offset, json.dumps(state), tail_digest))
        if known is not None:
            known[key] = {"signature": signature, "offset": offset,
                          "state_json": json.dumps(state), "tail": tail_digest}
        return 1

    @staticmethod
    def _run_timing_rows(db, selected, parameters):
        if not selected:
            return [], []
        cte = "WITH selected(kind, path) AS (VALUES " + ",".join("(?,?)" for _ in selected) + """),
            matched AS (SELECT invocation_id FROM usage_scopes JOIN selected USING (kind, path)),
            direct AS (SELECT run_id FROM usage_run_scopes JOIN selected USING (kind, path)),
            matching_runs AS (SELECT run_id AS id FROM direct UNION
                SELECT path AS id FROM usage_scopes WHERE kind='run' AND invocation_id IN matched)
            """
        runs = db.execute(cte + """SELECT t.id, t.model, t.stage,
            COALESCE(r.started_at, t.started_at) AS started_at,
            COALESCE(r.finished_at, t.finished_at) AS finished_at,
            CASE WHEN r.id IS NULL AND t.status IN ('starting','running','cancel_requested')
                THEN 'interrupted' ELSE COALESCE(r.status, t.status) END AS status,
            t.id IN (SELECT run_id FROM direct) AS direct_match
            FROM usage_run_times t JOIN matching_runs m ON m.id=t.id
            LEFT JOIN runs r ON r.id=t.id""", parameters).fetchall()
        members = db.execute(cte + """SELECT i.*, s.path AS run_id FROM usage_invocations i
            JOIN usage_scopes s ON s.invocation_id=i.id AND s.kind='run'
            JOIN matching_runs m ON m.id=s.path""", parameters).fetchall()
        return runs, members

    @staticmethod
    def _add_run_elapsed(total, stages, original_rows, rows, runs, members, excluded):
        """Use each run's remainder once, without inventing model/stage splits."""
        grouped = defaultdict(list)
        for member in members:
            grouped[member["run_id"]].append(member)
        original_ids = {row["id"] for row in original_rows}
        selected_ids = {row["id"] for row in rows}
        covered = set()
        covered_stages = defaultdict(set)
        model_hints = set()
        now = time.time()
        for run in runs:
            invocations = grouped[run["id"]]
            if not run["direct_match"] and (not invocations or not all(
                item["id"] in original_ids for item in invocations
            )):
                continue  # A single attempt/draft cannot inherit its whole multi-round run.
            untimed = [item for item in invocations if item["elapsed_seconds"] is None]
            models = {item["model"] for item in untimed}
            names = {item["stage"] for item in untimed}
            model = next(iter(models)) if len(models) == 1 else None if untimed else run["model"]
            stage = next(iter(names)) if len(names) == 1 else run["stage"] if not untimed else None
            if invocations and not untimed:
                model, stage = None, "Other run time"
            start = elapsed_value(run["started_at"])
            end = elapsed_value(run["finished_at"])
            running = end is None and run["status"] in {"starting", "running", "cancel_requested"}
            if running:
                end = now
            if start is None or end is None or end < start:
                if start is not None and not invocations and model not in excluded:
                    total["missingElapsedRuns"] += 1
                continue
            precise = sum(item["elapsed_seconds"] or 0 for item in invocations)
            remainder = max(0.0, end - start - precise)
            if not remainder and invocations and not untimed:
                continue
            if len(models) <= 1:
                model_hints.add(model)
            if (not models and model in excluded) or any(name in excluded for name in models):
                continue  # A mixed-model remainder cannot be assigned to one selected model.
            name = stage or "Run time · stage unavailable"
            if name not in stages:
                stages[name] = {"stage": name, **UsageIndex._totals([]), "timingOnly": True}
            for destination in (total, stages[name]):
                destination["elapsedSeconds"] = (destination["elapsedSeconds"] or 0) + remainder
                destination["fallbackElapsedSeconds"] += remainder
                destination["elapsedFallbackRuns"] += 1
                destination["runningElapsed"] += int(running)
            ids = {item["id"] for item in untimed} & selected_ids
            total["missingElapsed"] -= len(ids - covered)
            covered.update(ids)
            if stage:
                stages[name]["missingElapsed"] -= len(ids - covered_stages[name])
                covered_stages[name].update(ids)
        return model_hints

    def summary(self, targets, *, exclude_models=(), successful_runs_only=False):
        """All accounting on this path reads SQLite only."""
        selected = set()
        for target in targets:
            kind, value = target["kind"], target["path"]
            value = value if kind in {"job", "run"} else path_key(value)
            selected.add((kind, value))
        parameters = [value for pair in sorted(selected) for value in pair]
        with self.store.connect() as db:
            rows = db.execute("WITH selected(kind, path) AS (VALUES "
                + ",".join("(?,?)" for _ in selected) + """ )
                SELECT i.*, COALESCE((
                    SELECT MAX(CASE WHEN COALESCE(r.status, t.status)='succeeded' THEN 1 ELSE 0 END)
                    FROM usage_scopes s LEFT JOIN runs r ON r.id=s.path
                    LEFT JOIN usage_run_times t ON t.id=s.path
                    WHERE s.invocation_id=i.id AND s.kind='run'
                ), i.completed) AS successful
                FROM usage_invocations i WHERE i.id IN (
                    SELECT invocation_id FROM usage_scopes JOIN selected USING (kind, path)
                )""", parameters).fetchall() if selected else []
            run_times, run_members = self._run_timing_rows(db, selected, parameters)
        all_rows = rows
        if successful_runs_only:
            rows = [row for row in rows if row["successful"]]
            run_times = [run for run in run_times if run["status"] == "succeeded"]
        eligible_rows = rows
        excluded = set(exclude_models)
        rows = [row for row in rows if row["model"] not in excluded]
        total = self._totals(rows)
        if (excluded or successful_runs_only) and not rows:
            total["estimatedCost"] = 0.0
        stage_rows = defaultdict(list)
        for row in rows:
            stage_rows[row["stage"]].append(row)
        stages = {stage: {"stage": stage, **self._totals(values)} for stage, values in stage_rows.items()}
        hints = self._add_run_elapsed(total, stages, eligible_rows, rows, run_times, run_members, excluded)
        if (excluded or successful_runs_only) and not rows and total["elapsedSeconds"] is None:
            total["elapsedSeconds"] = 0.0
        if not rows and total["elapsedFallbackRuns"]:
            total["estimatedCost"] = None
        # Keep excluded models visible so the popup can include them again.
        by_model = defaultdict(list)
        for row in all_rows:
            by_model[row["model"]].append(row)
        for model in hints:
            by_model.setdefault(model, [])
        breakdown = [
            {"model": model, "invocations": len(values),
             "estimatedCost": self._totals(values)["estimatedCost"]}
            for model, values in sorted(by_model.items(), key=lambda item: item[0] or "")
        ]
        return {**total, "stages": [value for _, value in sorted(stages.items())],
                "indexing": self.indexing, "indexError": self.error, "updatedAt": self.updated_at,
                "modelBreakdown": breakdown,
                "modelFilterApplied": bool(excluded),
                "successfulRunsOnly": successful_runs_only,
                "assumptions": "Cost as reported by Claude Code for each invocation: an "
                "API-equivalent list-price estimate that includes cache reads and writes, "
                "thinking, and web search. Subscription usage is not billed per token."}

    @staticmethod
    def _totals(rows):
        counters = {field: 0 for field in TOKEN_FIELDS}
        available = {field: 0 for field in TOKEN_FIELDS}
        missing = unpriced = priced = 0
        cost = 0.0
        durations = [row["elapsed_seconds"] for row in rows if row["elapsed_seconds"] is not None]
        models = set()
        for row in rows:
            if row["model"]:
                models.add(row["model"])
            usage = json.loads(row["usage_json"]) if row["usage_json"] else None
            if usage is None:
                missing += 1
                continue
            for field in TOKEN_FIELDS:
                if field in usage:
                    counters[field] += usage[field]
                    available[field] += 1
            if "cost_usd" in usage:
                cost += usage["cost_usd"]
                priced += 1
            else:
                unpriced += 1
        return {"tokens": counters, "reported": available, "invocations": len(rows),
                "missingUsage": missing, "unpriced": unpriced, "priced": priced,
                "estimatedCost": cost if priced else None, "models": sorted(models),
                "elapsedSeconds": sum(durations) if durations else None,
                "missingElapsed": len(rows) - len(durations), "missingElapsedRuns": 0,
                "fallbackElapsedSeconds": 0.0, "elapsedFallbackRuns": 0, "runningElapsed": 0}
