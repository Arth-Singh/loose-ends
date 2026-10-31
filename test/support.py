"""Shared setup for the unittest modules."""

import os
import tempfile


def use_resolved_temp_root() -> None:
    """Create temporary directories under their canonical path.

    macOS puts the default temporary directory under /var, a symlink to
    /private/var. Code under test resolves paths, so expectations built from
    the unresolved root would compare two spellings of one directory.
    """
    tempfile.tempdir = os.path.realpath(tempfile.gettempdir())
