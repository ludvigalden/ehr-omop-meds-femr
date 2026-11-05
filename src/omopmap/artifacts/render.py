"""Render the contract tables in ``context/reference/artifacts.md`` from code.

Each generated block sits between ``<!-- contract:<name> -->`` and
``<!-- /contract:<name> -->`` markers; ``render_blocks`` returns the expected
body of every block and a drift test asserts the document carries exactly
those bodies.
"""

from __future__ import annotations

import re
from typing import Final

from pydantic import BaseModel

from omopmap.artifacts.contract import CODE_PATTERN, CONTRACT_VERSION, MEDS_VERSION
from omopmap.artifacts.models import (
    ArtifactManifest,
    ClmbrManifest,
    ConceptSetSummary,
    CoverageReport,
    DatasetMetadata,
    FileRecord,
    InputRecord,
    LabelSetSummary,
    MedsExtractSummary,
    Producer,
    TaskDefinition,
    UnrecognisedCode,
    VocabularyCoverage,
)
from omopmap.artifacts.schemas import ALL_TABLES, TableSpec

_BLOCK_RE: Final = re.compile(
    r"<!-- contract:(?P<name>[a-z_]+) -->\n(?P<body>.*?)<!-- /contract:(?P=name) -->",
    re.DOTALL,
)

_MODELS: Final[tuple[type[BaseModel], ...]] = (
    ArtifactManifest,
    Producer,
    FileRecord,
    InputRecord,
    MedsExtractSummary,
    LabelSetSummary,
    ConceptSetSummary,
    DatasetMetadata,
    TaskDefinition,
    ClmbrManifest,
    CoverageReport,
    VocabularyCoverage,
    UnrecognisedCode,
)


_ALIASES: Final = (
    (r"\b(Sha256|Timestamp|SemVer|RelPath|TaskName|VocabularyCode)\b", "str"),
    (r"\bSplit\b", '"train" | "tuning" | "held_out"'),
    (r"\bLabelType\b", '"boolean" | "integer" | "float" | "categorical"'),
    (r"\bEntryType\b", '"code" | "numeric" | "text" | "unused"'),
    (r"\bArtifactType\b", '"meds_extract" | "label_set" | "clmbr_concept_set"'),
    (r"\bArtifactSummary\b", "MedsExtractSummary | LabelSetSummary | ConceptSetSummary"),
    (r"\bLiteral\[([^\]]*)\]", r"\1"),
)


def _annotation_text(model: type[BaseModel], name: str) -> str:
    """The field's annotation as written in ``models.py``, with aliases expanded."""
    for klass in model.__mro__:
        raw: object = getattr(klass, "__annotations__", {}).get(name)
        if isinstance(raw, str):
            text = raw
            for pattern, replacement in _ALIASES:
                text = re.sub(pattern, replacement, text)
            return text
    raise TypeError(f"{model.__name__}.{name} has no string annotation")


_ARROW_NAMES: Final = {"float": "float32", "double": "float64"}


def arrow_type_name(dtype: object) -> str:
    """Arrow type as written in the contract (``float32`` / ``float64`` spelled out)."""
    text = str(dtype)
    return _ARROW_NAMES.get(text, text)


def _escape(text: str) -> str:
    return text.replace("|", "\\|")


def table_markdown(spec: TableSpec) -> str:
    """Markdown table for one Parquet schema."""
    lines = [
        "| Column | Arrow type | Nullable | Semantics |",
        "|---|---|---|---|",
    ]
    for c in spec.columns:
        lines.append(
            f"| `{c.name}` | `{arrow_type_name(c.dtype)}` | {'yes' if c.nullable else 'no'} "
            f"| {_escape(c.semantics)} |"
        )
    return "\n".join(lines) + "\n"


def model_markdown(model: type[BaseModel]) -> str:
    """Markdown table for one JSON model."""
    lines = [
        "| Field | Type | Meaning |",
        "|---|---|---|",
    ]
    for name, info in model.model_fields.items():
        type_text = _escape(_annotation_text(model, name))
        lines.append(f"| `{name}` | `{type_text}` | {_escape(info.description or '')} |")
    return "\n".join(lines) + "\n"


def _model_key(model: type[BaseModel]) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", model.__name__).lower()


def render_blocks() -> dict[str, str]:
    """Expected body of every generated block, keyed by block name."""
    blocks = {
        "versions": (
            f"Contract version `{CONTRACT_VERSION}`; MEDS `{MEDS_VERSION}`; "
            f"code pattern `{CODE_PATTERN}`.\n"
        )
    }
    blocks.update({spec.name: table_markdown(spec) for spec in ALL_TABLES})
    blocks.update({_model_key(m): model_markdown(m) for m in _MODELS})
    return blocks


def document_blocks(text: str) -> dict[str, str]:
    """The generated blocks present in a document, keyed by name."""
    found: dict[str, str] = {}
    for match in _BLOCK_RE.finditer(text):
        name = match.group("name")
        if name in found:
            raise ValueError(f"contract block {name} appears twice")
        found[name] = match.group("body")
    return found


def rewrite_document(text: str) -> str:
    """Replace every generated block's body with the rendered one."""
    blocks = render_blocks()

    def _sub(match: re.Match[str]) -> str:
        name = match.group("name")
        return f"<!-- contract:{name} -->\n{blocks[name]}<!-- /contract:{name} -->"

    return _BLOCK_RE.sub(_sub, text)
