"""Read a verified MEDS extract back into one event frame."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from omopmap.artifacts.contract import DATA_DIR
from omopmap.artifacts.io import read_contract_parquet
from omopmap.artifacts.models import ArtifactManifest
from omopmap.artifacts.schemas import MEDS_DATA
from omopmap.artifacts.validate import validate_artifact
from omopmap.meds.events import as_event_frame, empty_events


def read_extract(root: Path) -> tuple[ArtifactManifest, pl.DataFrame]:
    """Verify ``root`` against its manifest and the contract, then read every shard.

    Fails closed (``ArtifactError``) on any violation; an unverified extract is
    never read.
    """
    manifest = validate_artifact(root)
    shards = [
        read_contract_parquet(root / record.path, MEDS_DATA, record.path)
        for record in manifest.files
        if record.path.startswith(f"{DATA_DIR}/")
    ]
    events = pl.concat(shards) if shards else empty_events()
    return manifest, as_event_frame(events)
