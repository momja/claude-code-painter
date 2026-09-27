import stat
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent


@pytest.fixture
def fake_claude(tmp_path: Path) -> Path:
    """An executable `claude` that runs tests/fake_claude.py with this interpreter."""
    exe = tmp_path / "claude"
    exe.write_text(f"#!/bin/sh\nexec {sys.executable} {HERE / 'fake_claude.py'} \"$@\"\n")
    exe.chmod(exe.stat().st_mode | stat.S_IXUSR)
    return exe
