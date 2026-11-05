"""Parquet I/O pinned to the contract schemas."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import cast

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from omopmap.artifacts.errors import MissingFileError, SchemaMismatchError
from omopmap.artifacts.schemas import TableSpec

# The pyarrow stubs leave some parameter types (``IO[Unknown]``) unparameterised;
# these narrowed aliases confine that to one place.
_read_schema = cast(
    "Callable[[Path], pa.Schema]",
    pq.read_schema,  # pyright: ignore[reportUnknownMemberType]
)
_read_metadata = cast(
    "Callable[[Path], pq.FileMetaData]",
    pq.read_metadata,  # pyright: ignore[reportUnknownMemberType]
)
_write_table = cast(
    "Callable[[pa.Table, Path], None]",
    pq.write_table,  # pyright: ignore[reportUnknownMemberType]
)
_read_table = cast(
    "Callable[[Path], pa.Table]",
    pq.read_table,  # pyright: ignore[reportUnknownMemberType]
)


def read_table(path: Path) -> pa.Table:
    """Read a Parquet file as an Arrow table, exactly as stored."""
    return _read_table(path)


def write_table(table: pa.Table, path: Path) -> None:
    """Write an Arrow table as Parquet, exactly as given (no contract cast)."""
    _write_table(table, path)


def schema_columns(schema: pa.Schema) -> list[tuple[str, pa.DataType, bool]]:
    """``(name, type, nullable)`` for every field of a schema."""
    columns: list[tuple[str, pa.DataType, bool]] = []
    for name in schema.names:
        field = cast("pa.Field[pa.DataType]", schema.field(name))  # pyright: ignore[reportUnknownMemberType]
        columns.append((name, field.type, field.nullable))
    return columns


def parquet_rows(path: Path) -> int:
    """Row count from a Parquet footer."""
    return _read_metadata(path).num_rows


def to_contract_table(frame: pl.DataFrame, spec: TableSpec) -> pa.Table:
    """Convert a frame to an Arrow table with exactly ``spec``'s columns, order and types.

    Polars writes ``large_string``; the contract types are the MEDS ones
    (``string``), so every artifact goes through this cast before it is written.
    """
    if set(frame.columns) != set(spec.names):
        raise SchemaMismatchError(
            f"{spec.name}: columns {sorted(frame.columns)} differ from {sorted(spec.names)}"
        )
    table = frame.select(list(spec.names)).to_arrow()
    try:
        return table.cast(spec.schema)
    except (pa.ArrowInvalid, pa.ArrowTypeError, ValueError) as exc:
        raise SchemaMismatchError(
            f"{spec.name}: cannot cast to the contract schema: {exc}"
        ) from exc


def write_parquet(frame: pl.DataFrame, path: Path, spec: TableSpec) -> Path:
    """Write ``frame`` as ``spec``-conformant Parquet; parent directories are created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_table(to_contract_table(frame, spec), path)
    return path


def check_parquet_schema(path: Path, spec: TableSpec, label: str) -> None:
    """Fail unless ``path`` has exactly ``spec``'s column names, order, Arrow types and
    nullability flags (shown as ``(name, type, nullable)`` in the error)."""
    if not path.is_file():
        raise MissingFileError(f"{label} is missing")
    schema = _read_schema(path)
    expected = spec.schema
    # Parquet writes nested types with compliant child names (``list<element: …>``);
    # Arrow type equality ignores child field names, so compare with ``equals``.
    same = len(schema.names) == len(expected.names) and schema.equals(
        expected, check_metadata=False
    )
    if not same:
        actual = [(n, str(t), nullable) for n, t, nullable in schema_columns(schema)]
        wanted = [(c.name, str(c.dtype), c.nullable) for c in spec.columns]
        raise SchemaMismatchError(f"{label}: schema {actual} differs from contract {wanted}")


def read_contract_parquet(path: Path, spec: TableSpec, label: str) -> pl.DataFrame:
    """Schema-check then read a contract Parquet file."""
    check_parquet_schema(path, spec, label)
    return pl.read_parquet(path)
