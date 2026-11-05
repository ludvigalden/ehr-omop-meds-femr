"""Contract constants: versions, split names, reserved codes and on-disk names.

These values are part of the artifact contract documented in
``context/reference/artifacts.md``; changing any of them is a contract change
governed by the versioning rules there.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from omopmap.artifacts.errors import ManifestError

#: Semantic version of the artifact contract.
CONTRACT_VERSION: Final = "1.0.0"

#: MEDS release whose schemas this contract follows.
MEDS_VERSION: Final = "0.3.3"

#: MEDS split names, in canonical order.
SPLITS: Final = ("train", "tuning", "held_out")
type Split = Literal["train", "tuning", "held_out"]

#: MEDS reserved codes that carry no vocabulary prefix.
BIRTH_CODE: Final = "MEDS_BIRTH"
DEATH_CODE: Final = "MEDS_DEATH"
RESERVED_CODES: Final = frozenset({BIRTH_CODE, DEATH_CODE})

#: A vocabulary code string is ``<vocabulary_id>/<concept_code>``: split on the
#: first ``/``; both parts non-empty, without leading or trailing whitespace and
#: without line breaks. The concept code may itself contain ``/``.
CODE_PATTERN: Final = r"^[^/\s](?:[^/\n]*[^/\s])?/\S(?:.*\S)?$"
_CODE_RE: Final = re.compile(CODE_PATTERN)

#: The vocabulary id the CLMBR dictionary uses for site-local care-site ids.
CARE_SITE_VOCABULARY: Final = "CARE_SITE"

# On-disk names.
MANIFEST_NAME: Final = "artifact_manifest.json"
DATA_DIR: Final = "data"
METADATA_DIR: Final = "metadata"
DATASET_JSON: Final = "metadata/dataset.json"
CODES_PARQUET: Final = "metadata/codes.parquet"
SUBJECT_SPLITS_PARQUET: Final = "metadata/subject_splits.parquet"
LABELS_PARQUET: Final = "labels.parquet"
TASK_JSON: Final = "task.json"
CONCEPT_SET_PARQUET: Final = "concept_set.parquet"
SHARD_NAME_PATTERN: Final = r"^(0|[1-9][0-9]*)\.parquet$"

_SEMVER_RE: Final = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")


def is_vocabulary_code(code: str) -> bool:
    """Return True when ``code`` is a well-formed ``<vocabulary_id>/<concept_code>``."""
    return _CODE_RE.match(code) is not None


def is_valid_data_code(code: str) -> bool:
    """Return True for a code allowed in MEDS data: a vocabulary code or a reserved code."""
    return code in RESERVED_CODES or is_vocabulary_code(code)


def split_code(code: str) -> tuple[str, str]:
    """Split a vocabulary code string into ``(vocabulary_id, concept_code)``."""
    if not is_vocabulary_code(code):
        raise ValueError(f"not a <vocabulary_id>/<concept_code> string: {code!r}")
    vocabulary_id, concept_code = code.split("/", 1)
    return vocabulary_id, concept_code


def parse_version(version: str) -> tuple[int, int, int]:
    """Parse a ``MAJOR.MINOR.PATCH`` string."""
    match = _SEMVER_RE.match(version)
    if match is None:
        raise ManifestError(f"contract version {version!r} is not MAJOR.MINOR.PATCH")
    major, minor, patch = match.groups()
    return int(major), int(minor), int(patch)


def check_compatible(version: str) -> None:
    """Fail closed unless ``version`` is readable by this contract implementation.

    Readable means the same MAJOR and a MINOR no newer than ours.
    """
    major, minor, _ = parse_version(version)
    our_major, our_minor, _ = parse_version(CONTRACT_VERSION)
    if major != our_major:
        raise ManifestError(
            f"artifact contract {version} has a different major version than {CONTRACT_VERSION}"
        )
    if minor > our_minor:
        raise ManifestError(
            f"artifact contract {version} is newer than {CONTRACT_VERSION}; upgrade omopmap"
        )
