"""Build the committed synthetic test fixtures, byte-stably.

Every string here is invented. Concept names follow the pattern
"demo <category> <greek letter>" (e.g. "demo finding alpha") and the
Swedish source terms are common-language words or invented compounds;
nothing is copied from SNOMED, LOINC, RxNorm, Athena or any other
licensed source. The few rows carrying real vocabulary_id values
(SNOMED, LOINC, RxNorm) use clearly fake codes (e.g. 900000001) and
invented names, so eligibility filters can be tested without shipping
any licensed vocabulary text.

Reruns reproduce the committed files byte for byte: generation is a
pure function of the constants in this file (no clock, no RNG, no
locale).

Run from the repository root:

    python tests/fixtures/build_fixtures.py

or regenerate programmatically with ``main(target_dir)``.
"""

from __future__ import annotations

from pathlib import Path

CONCEPT_HEADER = (
    "concept_id\tconcept_name\tdomain_id\tvocabulary_id\tconcept_class_id\t"
    "standard_concept\tconcept_code\tvalid_start_date\tvalid_end_date\tinvalid_reason"
)
SOURCE_HEADER = "source_term\tconcept_id\tsplit"

CONCEPTS_PATH = Path("concepts.csv")
SOURCE_TERMS_PATH = Path("source_terms.csv")

GREEK = (
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
)

# Each demo category carries its own invented vocabulary_id, domain and
# concept_class, mirroring the OMOP column semantics without borrowing
# any real vocabulary content.
DEMO_CATEGORIES = (
    ("DEMOCOND", "Condition", "demo disorder", "DC"),
    ("DEMODRUG", "Drug", "demo drug product", "DD"),
    ("DEMOPROC", "Procedure", "demo procedure", "DP"),
    ("DEMOLAB", "Measurement", "demo lab test", "DL"),
    ("DEMOVOCAB", "Observation", "demo finding", "DO"),
)

VARIANT_CATEGORIES = (
    ("DEMOCOND", "Condition", "demo disorder", "DC"),
    ("DEMOLAB", "Measurement", "demo lab test", "DL"),
    ("DEMOPROC", "Procedure", "demo procedure", "DP"),
    ("DEMOVOCAB", "Observation", "demo finding", "DO"),
)

VARIANT_SUFFIXES = (
    ("acute demo phase", "S"),
    ("chronic demo phase", "C"),
    ("severity index", ""),
)

# Rows under real vocabulary_id names: invented names, clearly fake codes.
REAL_VOCAB_ROWS = (
    ("demo snomed finding alpha", "Condition", "SNOMED", "demo disorder", "S", "900000001"),
    ("demo snomed procedure beta", "Procedure", "SNOMED", "demo procedure", "S", "900000002"),
    ("demo loinc measurement gamma", "Measurement", "LOINC", "demo lab test", "S", "90000-1"),
    ("demo rxnorm drug delta", "Drug", "RxNorm", "demo drug product", "S", "9000001"),
)

# Units, values and type rows: OMOP-generated vocabulary shapes with
# invented names.
SUPPORT_ROWS = (
    ("demo unit millidemo", "Unit", "DEMOVOCAB", "demo unit", "S", "DEMO-U-01"),
    ("demo unit kilodemo", "Unit", "DEMOVOCAB", "demo unit", "S", "DEMO-U-02"),
    ("demo unit microdemo", "Unit", "DEMOVOCAB", "demo unit", "", "DEMO-U-03"),
    ("demo value low", "Meas Value", "DEMOVOCAB", "demo value", "S", "DEMO-V-01"),
    ("demo value high", "Meas Value", "DEMOVOCAB", "demo value", "S", "DEMO-V-02"),
    ("demo value borderline", "Meas Value", "DEMOVOCAB", "demo value", "C", "DEMO-V-03"),
    ("demo type concept inpatient", "Type Concept", "DEMOVOCAB", "demo type", "S", "DEMO-T-01"),
    ("demo type concept outpatient", "Type Concept", "DEMOVOCAB", "demo type", "S", "DEMO-T-02"),
    ("demo provider role primary", "Metadata", "DEMOVOCAB", "demo metadata", "S", "DEMO-M-01"),
    ("demo provider role secondary", "Metadata", "DEMOVOCAB", "demo metadata", "", "DEMO-M-02"),
    ("demo observation type qualifier", "Metadata", "DEMOVOCAB", "demo metadata", "C", "DEMO-M-03"),
    ("demo special form standard", "Metadata", "DEMOVOCAB", "demo metadata", "S", "DEMO-M-04"),
    ("demo special form extended", "Metadata", "DEMOVOCAB", "demo metadata", "S", "DEMO-M-05"),
    ("demo special form legacy", "Metadata", "DEMOVOCAB", "demo metadata", "", "DEMO-M-06"),
    ("demo relationship primary", "Metadata", "DEMOVOCAB", "demo metadata", "S", "DEMO-M-07"),
    ("demo relationship secondary", "Metadata", "DEMOVOCAB", "demo metadata", "C", "DEMO-M-08"),
)

VALID_START = "2020-01-01"
VALID_END = "2099-12-31"
INVALID_END = "2023-12-31"

# Swedish source terms: common-language clinical words and invented
# compounds only. Each references a generated concept_name; the split is
# assigned deterministically from the row index.
SOURCE_TERMS: tuple[tuple[str, str], ...] = (
    ("smärta i bröstet", "demo finding alpha"),
    ("bröstsmärta vid ansträngning", "demo finding beta"),
    ("feber", "demo disorder alpha"),
    ("huvudvärk", "demo disorder gamma"),
    ("ont i magen", "demo disorder delta"),
    ("kräkningar", "demo disorder epsilon"),
    ("yrhet", "demo disorder zeta"),
    ("andnöd", "demo disorder eta"),
    ("hjärtklappning", "demo finding theta"),
    ("hosta", "demo disorder iota"),
    ("svullnad i benen", "demo finding kappa"),
    ("utslag på armen", "demo finding lambda"),
    ("ryggsmärta", "demo disorder nu"),
    ("ledvärk", "demo disorder xi"),
    ("muskelsmärtor", "demo disorder omicron"),
    ("trötthet", "demo disorder pi"),
    ("viktminskning", "demo finding rho"),
    ("frossa", "demo disorder sigma"),
    ("svettningar på natten", "demo finding tau"),
    ("kval i magen", "demo disorder upsilon"),
    ("halsont", "demo disorder beta"),
    ("öronvärk", "demo disorder gamma"),
    ("tandvärk", "demo disorder delta"),
    ("dimsyn", "demo disorder epsilon"),
    ("domningar i handen", "demo finding zeta"),
    ("kramper i vaden", "demo disorder eta"),
    ("blodtryck", "demo lab test alpha"),
    ("blodsocker", "demo lab test beta"),
    ("puls", "demo lab test gamma"),
    ("kroppstemperatur", "demo lab test delta"),
    ("syresättning", "demo lab test epsilon"),
    ("hemoglobin", "demo lab test zeta"),
    ("tvättkälla", "demo lab test eta"),
    ("blodprov", "demo procedure alpha"),
    ("röntgen", "demo procedure beta"),
    ("datortomografi", "demo procedure gamma"),
    ("ekokardiografi", "demo procedure delta"),
    ("elektrokardiogram", "demo procedure epsilon"),
    ("demovärk i armen", "demo finding alpha"),
    ("kvidfeber", "demo disorder alpha"),
    ("stelbleka i nacken", "demo disorder theta"),
    ("lufsfeber med frossa", "demo disorder iota"),
    ("vindkramper", "demo disorder kappa"),
    ("provvärk vid belastning", "demo disorder lambda"),
    ("skuggvärk i bröstet", "demo finding mu"),
    ("blekhosta på natten", "demo disorder nu"),
    ("kylsvett", "demo finding xi"),
    ("rytmvind i bröstet", "demo finding omicron"),
    ("stelvind", "demo disorder pi"),
    ("kvidandning", "demo disorder rho"),
    ("skrivarvärk i handen", "demo finding sigma"),
    ("lufsande andning", "demo finding tau"),
    ("blekhet", "demo finding upsilon"),
    ("vindfeber", "demo disorder alpha"),
    ("kvidpuls", "demo lab test gamma"),
    ("luftpuls i handleden", "demo lab test delta"),
    ("provsvett", "demo finding epsilon"),
    ("stelvärk i axeln", "demo finding theta"),
    ("kvidsmärta i benet", "demo disorder zeta"),
    ("lufsvärk i bröstet", "demo finding beta"),
)


def _demo_rows() -> list[tuple[str, str, str, str, str, str]]:
    """Demo-vocabulary rows.

    Returns tuples of (name, domain, vocabulary_id, concept_class_id,
    concept_code, standard_concept); validity is derived in ``concepts``.
    """
    rows: list[tuple[str, str, str, str, str, str]] = []
    # Main pass: one standard concept per category per letter -> 100 rows.
    for letter in GREEK:
        for vocab, domain, concept_class, prefix in DEMO_CATEGORIES:
            rows.append(
                (
                    f"demo {_category_word(domain)} {letter}",
                    domain,
                    vocab,
                    concept_class,
                    f"{prefix}-{letter.upper()[:4]}-01",
                    "S",
                )
            )
    # Variant pass: three suffixes per letter over a letter-fixed
    # category -> 60 rows, mixing standard/classification/non-standard.
    for letter in GREEK:
        vocab, domain, concept_class, prefix = VARIANT_CATEGORIES[GREEK.index(letter) % 4]
        for variant_index, (suffix, standard) in enumerate(VARIANT_SUFFIXES):
            rows.append(
                (
                    f"demo {_category_word(domain)} {letter} {suffix}",
                    domain,
                    vocab,
                    concept_class,
                    f"{prefix}-{letter.upper()[:4]}-{variant_index + 2:02d}",
                    standard,
                )
            )
    # Third pass: one extended variant per letter, cycling suffixes and
    # categories -> 20 rows, all non-standard or classification-level.
    for letter in GREEK:
        suffix, standard = VARIANT_SUFFIXES[GREEK.index(letter) % 3]
        if standard == "S":
            standard = ""
        vocab, domain, concept_class, prefix = VARIANT_CATEGORIES[(GREEK.index(letter) + 2) % 4]
        rows.append(
            (
                f"demo {_category_word(domain)} {letter} {suffix} extended",
                domain,
                vocab,
                concept_class,
                f"{prefix}-{letter.upper()[:4]}-05",
                standard,
            )
        )
    return rows


def _category_word(domain: str) -> str:
    return {
        "Condition": "disorder",
        "Drug": "medication",
        "Procedure": "procedure",
        "Measurement": "lab test",
        "Observation": "finding",
    }[domain]


def concepts() -> list[list[str]]:
    """Return the 200 concept rows as field lists, in concept_id order."""
    rows: list[list[str]] = []
    for name, domain, vocab, concept_class, code, standard in _demo_rows():
        rows.append([name, domain, vocab, concept_class, code, standard])
    for name, domain, vocab, concept_class, standard, code in REAL_VOCAB_ROWS:
        rows.append([name, domain, vocab, concept_class, code, standard])
    for name, domain, vocab, concept_class, standard, code in SUPPORT_ROWS:
        rows.append([name, domain, vocab, concept_class, code, standard])
    # Invalid_reason is derived from the row index, so the valid/invalid
    # mix stays deterministic.
    flagged: list[list[str]] = []
    for index, (name, domain, vocab, concept_class, code, standard) in enumerate(rows):
        invalid_reason = "D" if index % 7 == 3 else ""
        valid_end = INVALID_END if invalid_reason else VALID_END
        flagged.append(
            [
                str(index + 1),
                name,
                domain,
                vocab,
                concept_class,
                standard,
                code,
                VALID_START,
                valid_end,
                invalid_reason,
            ]
        )
    return flagged


def source_terms(concept_rows: list[list[str]]) -> list[list[str]]:
    """Return the 60 source-term rows referencing ``concept_rows``."""
    id_by_name = {row[1]: row[0] for row in concept_rows}
    out: list[list[str]] = []
    for index, (term, concept_name) in enumerate(SOURCE_TERMS):
        if index % 7 == 6:
            split = "heldout"
        elif index % 5 == 4:
            split = "dev"
        else:
            split = "train"
        out.append([term, id_by_name[concept_name], split])
    return out


def render(header: str, rows: list[list[str]]) -> bytes:
    """Render a header plus tab-separated rows as UTF-8 bytes with a
    trailing newline on every line."""
    lines = [header, *(("\t".join(row)) for row in rows)]
    return ("\n".join(lines) + "\n").encode("utf-8")


def main(target_dir: Path) -> list[Path]:
    """Write both fixture files into ``target_dir`` and return the paths."""
    rows = concepts()
    source_rows = source_terms(rows)
    target_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for path, header, table in (
        (CONCEPTS_PATH, CONCEPT_HEADER, rows),
        (SOURCE_TERMS_PATH, SOURCE_HEADER, source_rows),
    ):
        out = target_dir / path
        out.write_bytes(render(header, table))
        written.append(out)
    return written


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parent.parent.parent
    written = main(repo_root / "tests" / "fixtures")
    for path in written:
        print(path)
