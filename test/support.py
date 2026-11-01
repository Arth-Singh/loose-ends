"""Shared setup for the unittest modules."""

import atexit
import os
from pathlib import Path
import shutil
import tempfile

CLAUDE_STUB_EXIT_CODE = 97
_CLAUDE_STUB = """#!/bin/sh
echo "tests must not run the real Claude Code; pass a fake --claude executable" >&2
exit {code}
""".format(code=CLAUDE_STUB_EXIT_CODE)
_stub_directory: Path | None = None


def isolate() -> None:
    """Prepare the process environment every test module relies on.

    Temporary directories use their canonical path: macOS puts the default
    temporary directory under /var, a symlink to /private/var, and code under
    test resolves paths, so expectations built from the unresolved root would
    compare two spellings of one directory.

    A failing ``claude`` goes first on PATH, so a test that reaches the real
    CLI fails on every machine instead of spending subscription quota
    wherever Claude Code is installed.
    """
    global _stub_directory
    tempfile.tempdir = os.path.realpath(tempfile.gettempdir())
    if _stub_directory is not None:
        return
    _stub_directory = Path(tempfile.mkdtemp(prefix="loose-ends-claude-stub-"))
    atexit.register(shutil.rmtree, _stub_directory, ignore_errors=True)
    stub = _stub_directory / "claude"
    stub.write_text(_CLAUDE_STUB, encoding="utf-8")
    stub.chmod(0o755)
    os.environ["PATH"] = os.pathsep.join(
        [str(_stub_directory), os.environ.get("PATH", "")]
    )
