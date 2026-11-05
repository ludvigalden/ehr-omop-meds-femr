"""Seeded synthetic MEDS cohort generator.

Every code is either invented (``DEMO*`` vocabularies) or an OMOP-generated
code (``Gender/F``, ``Visit/ER``); no real clinical vocabulary content and no
descriptions are produced. The code list marks which codes a synthetic CLMBR
dictionary "knows" (:func:`synthetic_dictionary_codes`), so dictionary-coverage
code can be exercised on a cohort whose share of recognised codes is strictly
between 0 and 1. ``DEMOSRC`` stands in for local source codes that have no
standard concept and are never recognised.

Determinism: each subject is generated from its own generator seeded with
``(seed, subject_id)``, so a subject's events do not depend on how many other
subjects are generated, and the same configuration always yields the same
events (and, through the writer, the same bytes).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Final

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from omopmap.artifacts.contract import BIRTH_CODE, DEATH_CODE
from omopmap.meds.events import EVENT_SCHEMA
from omopmap.meds.splits import SplitConfig


@dataclass(frozen=True, slots=True)
class SynthCode:
    """One code of the synthetic code list."""

    code: str
    kind: str  # "condition" | "drug" | "lab" | "procedure" | "source" | "observation"
    recognised: bool


def _family(prefix: str, letter: str, kind: str, size: int, recognised: int) -> list[SynthCode]:
    return [
        SynthCode(f"{prefix}/{letter}{i:03d}", kind, i <= recognised) for i in range(1, size + 1)
    ]


#: OMOP-generated demographic and visit codes (recognised by the synthetic dictionary).
SEX_CODES: Final = ("Gender/F", "Gender/M")
VISIT_ER: Final = "Visit/ER"
VISIT_IP: Final = "Visit/IP"
VISIT_OP: Final = "Visit/OP"
VISIT_CODES: Final = (VISIT_ER, VISIT_IP, VISIT_OP)

#: Invented code families, most frequent first; the first ``recognised`` of each
#: family are in the synthetic dictionary.
CODE_LIST: Final = (
    *_family("DEMOCOND", "C", "condition", 40, 25),
    *_family("DEMODRUG", "D", "drug", 30, 15),
    *_family("DEMOLAB", "L", "lab", 12, 8),
    *_family("DEMOPROC", "P", "procedure", 20, 10),
    *_family("DEMOSRC", "S", "source", 20, 0),
    *_family("DEMOOBS", "O", "observation", 5, 2),
)

#: Invented text values for observation events.
OBSERVATION_TEXT: Final = ("positive", "negative", "indeterminate")


def synthetic_dictionary_codes() -> frozenset[str]:
    """The codes a synthetic CLMBR dictionary recognises (for coverage tests)."""
    recognised = {c.code for c in CODE_LIST if c.recognised}
    return frozenset({*recognised, *SEX_CODES, *VISIT_CODES})


class SynthConfig(BaseModel):
    """Parameters of a synthetic cohort."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    subjects: int = Field(ge=1, le=1_000_000)
    seed: int = Field(ge=0)
    as_of: date = Field(default=date(2024, 12, 31), description="Last possible event date.")
    min_age_years: int = Field(default=18, ge=0)
    max_age_years: int = Field(default=90, ge=1)
    observation_years: int = Field(default=5, ge=1, description="History window before as_of.")
    mean_visits: float = Field(default=4.0, gt=0.0)
    max_visits: int = Field(default=20, ge=1)
    readmission_rate: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Chance an ER visit is followed by IP."
    )
    death_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    first_subject_id: int = Field(default=1, ge=0)

    @model_validator(mode="after")
    def _ages(self) -> SynthConfig:
        if self.max_age_years <= self.min_age_years:
            raise ValueError("max_age_years must exceed min_age_years")
        return self

    def sha256(self) -> str:
        """SHA-256 of the canonical JSON of this configuration (the manifest input)."""
        canonical = self.model_dump_json(round_trip=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _zipf_weights(size: int, exponent: float = 1.1) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    ranks = np.arange(1, size + 1, dtype=np.float64)
    weights = 1.0 / ranks**exponent
    return weights / weights.sum()


_BY_KIND: Final = {
    kind: tuple(c.code for c in CODE_LIST if c.kind == kind)
    for kind in ("condition", "drug", "lab", "procedure", "source", "observation")
}
_WEIGHTS: Final = {kind: _zipf_weights(len(codes)) for kind, codes in _BY_KIND.items()}
_LAB_INDEX: Final = {code: i for i, code in enumerate(_BY_KIND["lab"])}

type Row = tuple[int, datetime | None, str, float | None, str | None]


class _Subject:
    """Generates one subject's rows from its own seeded generator."""

    def __init__(self, config: SynthConfig, subject_id: int) -> None:
        self.config = config
        self.subject_id = subject_id
        self.rng = np.random.default_rng([config.seed, subject_id])
        self.rows: list[Row] = []

    def _pick(self, kind: str, count: int) -> list[str]:
        codes = _BY_KIND[kind]
        chosen = self.rng.choice(len(codes), size=count, p=_WEIGHTS[kind])
        return [codes[int(i)] for i in chosen]

    def _minutes(self, low: int, high: int) -> timedelta:
        return timedelta(minutes=int(self.rng.integers(low, high)))

    def _add(
        self, time: datetime | None, code: str, value: float | None = None, text: str | None = None
    ) -> None:
        self.rows.append((self.subject_id, time, code, value, text))

    def _visit(self, start: datetime, code: str) -> None:
        self._add(start, code)
        for c in self._pick("condition", int(self.rng.integers(1, 4))):
            self._add(start + self._minutes(5, 120), c)
        if self.rng.random() < 0.3:
            for c in self._pick("source", 1):
                self._add(start + self._minutes(5, 120), c)
        for c in self._pick("lab", int(self.rng.integers(0, 4))):
            index = _LAB_INDEX[c]
            mean = 10.0 * (index + 1)
            value = round(float(self.rng.normal(mean, mean * 0.1)), 2)
            self._add(start + self._minutes(10, 240), c, value=value)
        for c in self._pick("drug", int(self.rng.integers(0, 3))):
            self._add(start + self._minutes(30, 360), c)
        if code != VISIT_OP and self.rng.random() < 0.5:
            for c in self._pick("procedure", 1):
                self._add(start + self._minutes(30, 480), c)
        if self.rng.random() < 0.2:
            for c in self._pick("observation", 1):
                text = OBSERVATION_TEXT[int(self.rng.integers(0, len(OBSERVATION_TEXT)))]
                self._add(start + self._minutes(10, 240), c, text=text)

    def generate(self) -> list[Row]:
        cfg = self.config
        end = datetime.combine(cfg.as_of, datetime.min.time())
        age_days = int(self.rng.integers(cfg.min_age_years * 365, cfg.max_age_years * 365))
        birth = (end - timedelta(days=age_days)).replace(hour=0, minute=0)
        self._add(None, SEX_CODES[int(self.rng.integers(0, 2))])
        self._add(birth, BIRTH_CODE)

        window_start = max(
            birth + timedelta(days=1), end - timedelta(days=365 * cfg.observation_years)
        )
        span_minutes = max(int((end - window_start).total_seconds() // 60), 1)
        visits = min(max(int(self.rng.poisson(cfg.mean_visits)), 1), cfg.max_visits)
        starts = sorted(
            window_start + timedelta(minutes=int(m))
            for m in self.rng.integers(0, span_minutes, size=visits)
        )
        visit_codes = (VISIT_ER, VISIT_OP, VISIT_IP)
        last = birth
        for start in starts:
            code = visit_codes[int(self.rng.choice(3, p=[0.4, 0.45, 0.15]))]
            self._visit(start, code)
            if code == VISIT_ER and self.rng.random() < cfg.readmission_rate:
                follow = start + timedelta(days=int(self.rng.integers(1, 31)))
                if follow < end:
                    self._visit(follow, VISIT_IP)
        last = max((t for _, t, _, _, _ in self.rows if t is not None), default=birth)
        if self.rng.random() < cfg.death_rate:
            death = last + self._minutes(60, 60 * 24 * 60)
            self._add(
                min(max(death, last + timedelta(minutes=1)), end + timedelta(hours=23)), DEATH_CODE
            )
        return self.rows


def generate_events(config: SynthConfig) -> pl.DataFrame:
    """The cohort's events as an event frame (unsorted; the writer sorts)."""
    rows: list[Row] = []
    for offset in range(config.subjects):
        rows.extend(_Subject(config, config.first_subject_id + offset).generate())
    return pl.DataFrame(rows, schema=EVENT_SCHEMA, orient="row")


def synth_created_at(config: SynthConfig) -> str:
    """The deterministic ``created_at`` of a synthetic extract: ``as_of`` at midnight UTC."""
    return f"{config.as_of.isoformat()}T00:00:00Z"


def default_split_config(seed: int) -> SplitConfig:
    """Split configuration for a synthetic cohort: 80/10/10, seeded like the cohort."""
    return SplitConfig(seed=seed)
