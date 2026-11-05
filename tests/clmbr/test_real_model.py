"""Opt-in: import the pinned release dictionary from a local copy (``OMOPMAP_CLMBR_DIR``).

Skipped unless the variable names a directory holding a byte-identical copy of
the pinned ``dictionary.msgpack``. Only counts and hashes are asserted; no code
is printed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from omopmap.cli import app

pytestmark = pytest.mark.real_model

_DIR = os.environ.get("OMOPMAP_CLMBR_DIR")


@pytest.mark.skipif(not _DIR, reason="OMOPMAP_CLMBR_DIR is not set")
def test_real_import_counts(tmp_path: Path) -> None:
    assert _DIR is not None
    out = tmp_path / "kdl"
    runner = CliRunner()
    result = runner.invoke(app, ["clmbr", "import", "--model-dir", _DIR, "--out", str(out)])
    assert result.exit_code == 0, result.stderr
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["entry_type_counts"] == {
        "code": 25667,
        "numeric": 11183,
        "text": 2961,
        "unused": 25725,
    }
    assert manifest["total"] == 25749 and manifest["care_site_excluded"] == 396
    assert manifest["total"] + manifest["care_site_excluded"] == 26145
    assert manifest["revision"] == "a0f2b36ef3976faf032bfbce23030cb640b5e97f"
    assert manifest["file_sha256"] == (
        "481ddac70a37e79bbfadde1411ad674544a161e3c1948149989cd159a34203ee"
    )
    assert manifest["concept_list_sha256"] == (
        "71f396167f40096c9ff11dc1b762746e946a7ddaec195f11a813a341bffedbc1"
    )
    verify = runner.invoke(app, ["clmbr", "verify", "--manifest", str(out / "manifest.json")])
    assert verify.exit_code == 0, verify.stderr
