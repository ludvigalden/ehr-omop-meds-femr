"""The committed vocabulary schema and module parse, and emitted documents follow them.

The hermetic check reads the node declarations of ``vocab/schema.kdl`` with the
KDL parser and compares every emitted node, and every child it carries, with
the children those declarations bind.
"""

from __future__ import annotations

from pathlib import Path

from omopmap.clmbr.importer import build_concept_set, enrich
from omopmap.clmbr.kdl import parse_kdl, render_files
from omopmap.clmbr.model import load_dictionary
from omopmap.clmbr.source import ResolvedDictionary
from omopmap.meds.convert import read_table
from tests.clmbr.dictionaries import sha256, write_layouts

VOCAB = Path(__file__).resolve().parents[2] / "vocab"
SCHEMA = VOCAB / "schema.kdl"


def _schema_nodes() -> dict[str, set[str]]:
    """``{node name: names of the children its field bindings consume}``."""
    (root,) = parse_kdl(SCHEMA.read_text(encoding="utf-8")).nodes
    assert root.name == "schema"
    declared: dict[str, set[str]] = {}
    for node in root.children:
        if node.name != "node":
            continue
        children: set[str] = set()
        for binding in node.children:
            if binding.name == "field":
                child = binding.properties.get("child")
                if isinstance(child, str):
                    children.add(child)
                for block in binding.children:  # `from { child NAME ... }`
                    for source in block.children:
                        if source.name in ("child", "children"):
                            children.add(str(source.args[0]))
            elif binding.name == "children":
                children.add(str(binding.args[0]))
        declared[str(node.args[0])] = children
    return declared


def _emitted(tmp_path: Path) -> dict[str, str]:
    path = write_layouts(tmp_path)["flat"]
    athena = tmp_path / "CONCEPT.csv"
    athena.write_text(
        "concept_id\tdomain_id\tvocabulary_id\tstandard_concept\tconcept_code\n"
        "2000000001\tCondition\tDEMOCOND\tS\tC001\n"
        "2000000002\tCondition\tDEMOCOND\t\tC002\n"
    )
    concepts = enrich(build_concept_set(load_dictionary(path)), read_table(athena))
    source = ResolvedDictionary(
        path=path,
        kind="local",
        repo="demo-org/demo-model",
        revision="0123456789abcdef0123456789abcdef01234567",
        filename=path.name,
        sha256=sha256(path),
    )
    return render_files(concepts, source, sha256(athena))


def test_module_points_at_schema() -> None:
    module = parse_kdl((VOCAB / "module.kdl").read_text(encoding="utf-8"))
    (root,) = module.nodes
    assert root.name == "module"
    children = {c.name: c for c in root.children}
    assert children["schema"].args == ["schema.kdl"]
    assert (VOCAB / str(children["schema"].args[0])).is_file()


def test_emitted_documents_follow_schema(tmp_path: Path) -> None:
    declared = _schema_nodes()
    for name, text in _emitted(tmp_path).items():
        document = parse_kdl(text)
        assert document.nodes[0].name == "context", name
        for node in document.nodes:
            assert node.name in declared, (name, node.name)
            assert {c.name for c in node.children} <= declared[node.name], (name, node.name)
