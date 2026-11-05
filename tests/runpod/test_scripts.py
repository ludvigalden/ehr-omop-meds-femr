"""Hermetic tests for the rented-GPU job folder.

These tests keep the job scripts from drifting away from the CLI: syntax
(``sh -n``) and, for every ``omopmap`` invocation in the scripts, the
subcommand path and every referenced flag are checked against the CLI's own
command tree (the click parameters ``--help`` is rendered from), so the
check does not depend on how help text is styled for the current terminal.
No GPU, no model downloads, no network.
"""

from __future__ import annotations

import shlex
import subprocess
from pathlib import Path

import typer
from typer.core import TyperGroup

from omopmap.cli import app

REPO_ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = REPO_ROOT / "jobs" / "runpod"
SCRIPTS = ("setup.sh", "run_all.sh", "real_model_smoke.sh")


def _script(name: str) -> Path:
    path = JOBS_DIR / name
    assert path.is_file(), f"missing job script: {path}"
    return path


def _logical_lines(text: str) -> list[str]:
    """Join backslash-newline continuations and return whole logical lines."""
    return text.replace("\\\n", " ").splitlines()


def _omopmap_invocations(script: Path) -> list[tuple[str, ...]]:
    """Extract every omopmap invocation as a token tuple.

    A logical line counts when the token stream contains the token
    ``omopmap``; the tokens after it up to the first ``--`` flag form the
    subcommand path, the rest are arguments and flags.
    """
    found: list[tuple[str, ...]] = []
    for line in _logical_lines(script.read_text(encoding="utf-8")):
        try:
            tokens = shlex.split(line)
        except ValueError:
            continue
        if "omopmap" not in tokens:
            continue
        tail = tokens[tokens.index("omopmap") + 1 :]
        if not tail:
            continue
        found.append(tuple(tail))
    assert found, f"no omopmap invocations parsed from {script}"
    return found


def _command_flags(path: tuple[str, ...]) -> set[str]:
    """Resolve a subcommand path in the CLI tree and return every flag it offers."""
    command = typer.main.get_command(app)
    for name in path:
        assert isinstance(command, TyperGroup), (
            f"'omopmap {' '.join(path)}': {command.name} has no subcommands"
        )
        sub = command.commands.get(name)
        assert sub is not None, f"'omopmap {' '.join(path)}': no subcommand {name!r}"
        command = sub
    return {
        opt
        for param in command.params
        for opt in (*param.opts, *param.secondary_opts)
        if opt.startswith("--")
    }


class TestScriptSyntax:
    def test_setup_passes_sh_n(self) -> None:
        subprocess.run(["sh", "-n", str(_script("setup.sh"))], check=True)

    def test_run_all_passes_sh_n(self) -> None:
        subprocess.run(["sh", "-n", str(_script("run_all.sh"))], check=True)

    def test_real_model_smoke_passes_sh_n(self) -> None:
        subprocess.run(["sh", "-n", str(_script("real_model_smoke.sh"))], check=True)


class TestCliDrift:
    def test_every_subcommand_and_flag_exists(self) -> None:
        for script in ("run_all.sh", "real_model_smoke.sh"):
            for invocation in _omopmap_invocations(_script(script)):
                first_flag = next(
                    (i for i, t in enumerate(invocation) if t.startswith("--")),
                    len(invocation),
                )
                path = invocation[:first_flag]
                offered = _command_flags(path)
                for token in invocation[first_flag:]:
                    if not token.startswith("--"):
                        continue
                    flag = token.split("=", 1)[0]
                    assert flag in offered, (
                        f"{script}: flag {flag} not offered by 'omopmap {' '.join(path)}'"
                    )

    def test_gate_precedes_bulk_index_in_run_all(self) -> None:
        text = _script("run_all.sh").read_text(encoding="utf-8")
        gate = text.find("omopmap gate")
        index = text.find("omopmap retrieve index")
        assert gate != -1 and index != -1
        assert gate < index, "the gate step must run before bulk embedding"

    def test_required_env_vars_are_named(self) -> None:
        text = _script("run_all.sh").read_text(encoding="utf-8")
        for var in ("OMOPMAP_CONCEPTS", "OMOPMAP_QUERIES"):
            assert f'"${{{var}:?' in text, f"{var} is not a required env var"
