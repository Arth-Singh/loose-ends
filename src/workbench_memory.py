#!/usr/bin/env python3
"""Per-worker memory accounting and limits for workbench workers.

Limits use cgroups v2 on Linux. Other platforms run workers without a limit.
"""

from __future__ import annotations

import hashlib
import mmap
import os
from pathlib import Path
import sys


GB_BYTES = 1024 ** 3
QUEUE_JOB_ENV = "LOOSE_ENDS_QUEUE_JOB"
CGROUP_ROOT_ENV = "LOOSE_ENDS_CGROUP_ROOT"
CGROUP_BACKEND = "linux_cgroup_v2"
WORKER_CGROUP_FILES = (
    "cgroup.procs",
    "memory.current",
    "memory.max",
    "memory.swap.current",
    "memory.swap.max",
)


def _memory_enforcement_supported() -> bool:
    return sys.platform.startswith("linux")


def physical_memory_bytes() -> int | None:
    """Return installed physical memory without adding a dependency."""
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    return int(page_size * page_count)


def resolved_limit_bytes(settings: dict, physical_bytes: int | None) -> int | None:
    """Resolve a persisted percent/GB/unlimited policy to total allocation."""
    memory = settings.get("memoryLimit", {"mode": "percent", "value": 50})
    mode = memory["mode"]
    if mode == "unlimited":
        return None
    value = float(memory["value"])
    if mode == "gb":
        return max(1, round(value * GB_BYTES))
    if mode == "percent" and physical_bytes is not None:
        return max(1, round(physical_bytes * value / 100.0))
    return None


def per_worker_limit_bytes(
    settings: dict,
    physical_bytes: int | None,
) -> int | None:
    """Divide the configured total allocation across maximum worker slots."""
    allocation = resolved_limit_bytes(settings, physical_bytes)
    if allocation is None:
        return None
    worker_limit = max(1, int(settings.get("workerLimit", 1)))
    return max(1, allocation // worker_limit)


def _database_digest(database: Path) -> str:
    identity = str(database.resolve())
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]


def _linux_cgroup_parent() -> Path:
    mount = Path("/sys/fs/cgroup")
    if not (mount / "cgroup.controllers").is_file():
        raise OSError("cgroup v2 is not mounted at /sys/fs/cgroup")
    override = os.environ.get(CGROUP_ROOT_ENV)
    if override:
        parent = Path(override).resolve()
    else:
        try:
            lines = Path("/proc/self/cgroup").read_text(encoding="ascii").splitlines()
        except OSError as exc:
            raise OSError(f"could not read /proc/self/cgroup: {exc}") from exc
        relative = next(
            (line[3:] for line in lines if line.startswith("0::")),
            None,
        )
        if relative is None:
            raise OSError("this process is not in a cgroup v2 hierarchy")
        parent = (mount / relative.lstrip("/")).resolve()
    try:
        parent.relative_to(mount.resolve())
    except ValueError as exc:
        raise OSError(f"cgroup root must be beneath {mount}") from exc
    if not (parent / "cgroup.controllers").is_file():
        raise OSError(f"{parent} is not a cgroup v2 directory")
    return parent


def _linux_queue_cgroup(database: Path, parent: Path) -> Path:
    return parent / f"loose-ends-workers-{_database_digest(database)}"


def _linux_worker_cgroup(database: Path, parent: Path, run_id: str) -> Path:
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    return _linux_queue_cgroup(database, parent) / f"worker-{run_digest}"


def _read_cgroup_value(cgroup: Path, name: str) -> str:
    return (cgroup / name).read_text(encoding="ascii").strip()


class QueueMemoryController:
    """Own one independently limited cgroup per worker run."""

    def __init__(self, database: Path):
        self.database = database
        self.physical_bytes = physical_memory_bytes()
        self.available = False
        self.error: str | None = None
        self.backend: str | None = None
        self._containers: dict[str, Path] = {}
        self._observed_peak = 0
        if not _memory_enforcement_supported():
            return
        try:
            self._initialize_linux()
        except OSError as exc:
            self.error = (
                "Linux cgroup v2 memory enforcement is unavailable: "
                f"{exc}. Run the workbench in a delegated cgroup with the "
                "memory controller enabled."
            )

    @staticmethod
    def _enable_linux_memory_children(cgroup: Path) -> None:
        controllers = _read_cgroup_value(cgroup, "cgroup.controllers").split()
        subtree_control = _read_cgroup_value(cgroup, "cgroup.subtree_control").split()
        if "memory" in subtree_control:
            return
        if "memory" not in controllers:
            raise OSError(f"the memory controller is not available in {cgroup}")
        try:
            (cgroup / "cgroup.subtree_control").write_text(
                "+memory\n", encoding="ascii"
            )
        except OSError as exc:
            raise OSError(
                f"could not enable the memory controller in {cgroup}: {exc}"
            ) from exc

    def _initialize_linux(self) -> None:
        parent = _linux_cgroup_parent()
        self._enable_linux_memory_children(parent)
        manager = _linux_queue_cgroup(self.database, parent)
        try:
            manager.mkdir(exist_ok=True)
        except OSError as exc:
            raise OSError(
                f"could not create worker cgroup manager {manager}: {exc}"
            ) from exc
        self._enable_linux_memory_children(manager)
        self._cgroup_parent = parent
        self._cgroup = manager
        for child in manager.iterdir():
            if not child.name.startswith("worker-") or not child.is_dir():
                continue
            try:
                if not _read_cgroup_value(child, "cgroup.procs").split():
                    child.rmdir()
            except OSError:
                continue
        self.backend = CGROUP_BACKEND
        self.available = True

    def close(self) -> None:
        self._containers.clear()

    def _container_path(self, run_id: str) -> Path:
        return _linux_worker_cgroup(self.database, self._cgroup_parent, run_id)

    def _open_existing_container(self, run_id: str) -> Path | None:
        cgroup = self._container_path(run_id)
        return cgroup if cgroup.is_dir() else None

    def _create_container(self, run_id: str) -> Path:
        cgroup = self._container_path(run_id)
        try:
            cgroup.mkdir(exist_ok=True)
        except OSError as exc:
            raise OSError(f"could not create worker cgroup {cgroup}: {exc}") from exc
        missing = [name for name in WORKER_CGROUP_FILES if not (cgroup / name).is_file()]
        if missing:
            raise OSError(
                f"worker cgroup is missing required files: {', '.join(missing)}"
            )
        return cgroup

    @staticmethod
    def _container_limit(container: Path) -> int | None:
        value = _read_cgroup_value(container, "memory.max")
        return None if value == "max" else int(value)

    @staticmethod
    def _set_container_limit(container: Path, limit_bytes: int | None) -> None:
        value = "max" if limit_bytes is None else str(limit_bytes)
        swap_value = "max" if limit_bytes is None else "0"
        (container / "memory.swap.max").write_text(swap_value + "\n", encoding="ascii")
        (container / "memory.max").write_text(value + "\n", encoding="ascii")

    @staticmethod
    def _container_process_ids(container: Path) -> list[int]:
        return [int(value) for value in _read_cgroup_value(container, "cgroup.procs").split()]

    @staticmethod
    def _container_current_bytes(container: Path) -> int:
        memory = int(_read_cgroup_value(container, "memory.current"))
        swap = int(_read_cgroup_value(container, "memory.swap.current"))
        return memory + swap

    @staticmethod
    def _container_limit_matches(
        container: Path,
        desired: int | None,
        applied: int | None,
    ) -> bool:
        swap = _read_cgroup_value(container, "memory.swap.max")
        expected_swap = "max" if desired is None else "0"
        return desired == applied and swap == expected_swap

    @staticmethod
    def _normalize_limit(desired: int | None) -> int | None:
        if desired is None:
            return None
        page_size = mmap.PAGESIZE
        return ((desired + page_size - 1) // page_size) * page_size

    def run_has_processes(self, run_id: str) -> bool | None:
        """Return whether a run's cgroup is nonempty, if knowable."""
        if not self.available:
            return None
        try:
            container = self._containers.get(run_id)
            if container is None:
                container = self._open_existing_container(run_id)
                if container is None:
                    return False
                self._containers[run_id] = container
            alive = bool(self._container_process_ids(container))
            self.error = None
            return alive
        except OSError as exc:
            self.error = str(exc)
            return None

    def _retire_finished_containers(self, active_run_ids: set[str]) -> None:
        for run_id, container in list(self._containers.items()):
            if run_id in active_run_ids or self._container_process_ids(container):
                continue
            try:
                container.rmdir()
            except OSError:
                continue
            del self._containers[run_id]

    def prepare_run(self, run_id: str, settings: dict) -> str | None:
        """Create and configure a worker's cgroup before it starts."""
        desired = self._normalize_limit(
            per_worker_limit_bytes(settings, self.physical_bytes)
        )
        if not self.available:
            if desired is None or not _memory_enforcement_supported():
                return None
            raise OSError(self.error or "worker memory enforcement is unavailable")
        try:
            container = self._containers.get(run_id)
            if container is None:
                container = self._open_existing_container(run_id)
            if container is None:
                container = self._create_container(run_id)
            self._containers[run_id] = container
            if not self._container_limit_matches(
                container,
                desired,
                self._container_limit(container),
            ):
                self._set_container_limit(container, desired)
            self.error = None
            return str(container)
        except OSError as exc:
            self.error = str(exc)
            self.available = False
            raise

    def reconcile(
        self,
        settings: dict,
        active_run_ids: set[str] | None = None,
    ) -> dict:
        active_run_ids = set(active_run_ids or ())
        allocation = resolved_limit_bytes(settings, self.physical_bytes)
        desired = self._normalize_limit(
            per_worker_limit_bytes(settings, self.physical_bytes)
        )
        snapshot = {
            "available": self.available,
            "backend": self.backend,
            "physicalBytes": self.physical_bytes,
            "allocationBytes": allocation,
            "resolvedBytes": desired,
            "appliedBytes": None,
            "currentBytes": None,
            "peakBytes": self._observed_peak or None,
            "pending": False,
            "error": self.error,
            "warning": None,
        }
        if not self.available:
            if not _memory_enforcement_supported():
                platform = "macOS" if sys.platform == "darwin" else "this platform"
                snapshot["warning"] = (
                    f"Worker memory limits are unavailable on {platform}; "
                    "workers will run without a memory limit."
                )
            elif desired is not None:
                snapshot["pending"] = True
            return snapshot
        try:
            for run_id in active_run_ids:
                if run_id in self._containers:
                    continue
                container = self._open_existing_container(run_id)
                if container is not None:
                    self._containers[run_id] = container
            self._retire_finished_containers(active_run_ids)

            total_current = 0
            pending = False
            applied_values: list[int] = []
            for container in self._containers.values():
                current = self._container_current_bytes(container)
                total_current += current
                applied = self._container_limit(container)
                matches = self._container_limit_matches(container, desired, applied)
                if desired is None:
                    if not matches:
                        self._set_container_limit(container, None)
                    applied = None
                elif applied is not None and desired > applied:
                    self._set_container_limit(container, desired)
                    applied = desired
                elif desired < current and (applied is None or desired < applied):
                    pending = True
                elif not matches:
                    self._set_container_limit(container, desired)
                    applied = desired
                if applied is not None:
                    applied_values.append(applied)

            self._observed_peak = max(self._observed_peak, total_current)
            self.error = None
            snapshot.update(
                appliedBytes=(max(applied_values) if applied_values else desired),
                currentBytes=total_current,
                peakBytes=self._observed_peak or None,
                pending=pending,
                error=None,
            )
        except OSError as exc:
            self.error = str(exc)
            snapshot["error"] = self.error
            snapshot["pending"] = desired is not None
        return snapshot

def join_queue_job_from_environment() -> None:
    """Join the configured worker cgroup before launching child processes."""
    name = os.environ.get(QUEUE_JOB_ENV)
    if not name or not _memory_enforcement_supported():
        return
    procs = Path(name) / "cgroup.procs"
    if not procs.is_file():
        raise OSError(f"worker cgroup is unavailable: {Path(name)}")
    procs.write_text(f"{os.getpid()}\n", encoding="ascii")
