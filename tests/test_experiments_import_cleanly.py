"""Every experiment program must parse, and must import without hardware.

Two failure modes this catches, both of which actually happened during the
repository migration:

* an automated import rewrite landing inside a multi-line parenthesised import,
  which leaves a file that no longer parses but that nothing imports;
* a module-level path or device lookup that raises when the hardware or an
  environment variable is absent -- configuration errors belong at run time.

The public repo has the same pair of guards; this repo went without them through
the whole migration, which is why several rewrites were only caught by hand.
"""

import ast
import importlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
ROBOT_FREE_ENV = "LEROBOT_TELEOPERATOR_SO101_WEBCAM_ROBOT_FREE_IMPORT"


def tracked_python_files():
    listed = subprocess.run(["git", "-C", str(REPO), "ls-files", "*.py"],
                            capture_output=True, text=True, check=True)
    return [REPO / line for line in listed.stdout.splitlines()]


TRACKED = tracked_python_files()
#: Programs, as `import <name>` sees them: pyproject puts experiments/ and
#: experiments/classifier/ on the path, so they are imported by bare name.
PROGRAM_DIRS = [REPO / "experiments", REPO / "experiments" / "classifier"]


def test_there_are_files_to_check():
    """Guards the guards: an empty file list passes everything below."""
    assert len(TRACKED) >= 50, f"only {len(TRACKED)} tracked .py files"


@pytest.mark.parametrize("path", TRACKED, ids=lambda p: str(p.relative_to(REPO)))
def test_every_tracked_file_parses(path):
    ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def program_modules():
    seen = set()
    for directory in PROGRAM_DIRS:
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.py")):
            if path.stem not in seen:
                seen.add(path.stem)
                yield directory, path.stem


PROGRAMS = list(program_modules())


def test_there_are_programs_to_check():
    assert len(PROGRAMS) >= 20, f"only {len(PROGRAMS)} programs"


@pytest.mark.parametrize("directory,name", PROGRAMS, ids=lambda v: getattr(v, "name", v))
def test_program_imports_without_hardware(directory, name, monkeypatch):
    monkeypatch.setenv(ROBOT_FREE_ENV, "1")
    monkeypatch.syspath_prepend(str(directory))
    for module in [m for m in sys.modules if m == name]:
        del sys.modules[module]
    try:
        importlib.import_module(name)
    except ImportError as exc:
        # A missing optional third-party driver is an environment fact, not a
        # defect in the program. A missing *first-party* module is a bad rewrite.
        missing = getattr(exc, "name", "") or ""
        if missing.split(".")[0] in {"ir_force", "lerobot_teleoperator_so101_webcam",
                                     "webcam_input", "pressurevision_integration"}:
            raise
        pytest.skip(f"optional dependency unavailable: {exc}")
