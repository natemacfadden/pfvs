# =============================================================================
#    Copyright (C) 2026  Liam McAllister Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.
# =============================================================================
#
# -----------------------------------------------------------------------------
# Description:  The scripts in examples/ run as documented.
# -----------------------------------------------------------------------------

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"
CYTOOLS_AVAILABLE = importlib.util.find_spec("cytools") is not None


def _run(script, *args, cwd):
    env = dict(os.environ, MPLBACKEND="Agg")
    res = subprocess.run([sys.executable, str(EXAMPLES / script), *args], cwd=cwd, env=env,
                         capture_output=True, text=True, timeout=600)
    assert res.returncode == 0, res.stdout + res.stderr
    return res.stdout


def test_manwe_example(tmp_path):
    out = _run("manwe.py", "--plot", cwd=tmp_path)
    assert "Manwe: found" in out
    assert (tmp_path / "manwe.png").exists()


@pytest.mark.skipif(not CYTOOLS_AVAILABLE, reason="requires CYTools")
def test_h11_11_example(tmp_path):
    out = _run("h11_11.py", "--n-p", "10000", cwd=tmp_path)
    assert "coni PFVs" in out
