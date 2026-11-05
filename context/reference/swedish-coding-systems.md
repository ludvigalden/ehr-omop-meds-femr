# Swedish coding systems

Swedish healthcare documents care with four national code systems: ICD-10-SE
for diagnoses, KVÅ for procedures, ATC (with national extensions) for
medications, and NPU for laboratory results. None of them is in the
vocabulary of a clinical foundation model trained on American records, so
every one of their codes reaches the model only through a mapping to a
standard vocabulary — or not at all. This page records the systems and
their OMOP mapping situation, as background for the mapping methods in
[retrieval-methods.md](retrieval-methods.md) and the coverage measurement in
[clmbr-concept-set.md](clmbr-concept-set.md).

The coverage statements below summarise what is publicly documented about
Swedish OMOP mapping work (the OHDSI Sweden efforts and published Swedish
OMOP implementations); they describe the general situation, not a result
of this repository.

## ICD-10-SE (diagnoses)

Sweden's mandatory diagnosis classification is ICD-10-SE, the Swedish
national version of WHO's ICD-10, maintained and annually updated by
Socialstyrelsen. Sweden extends the WHO hierarchy with fifth-character
"depth codes" (fördjupningskoder) — roughly 3,000 national extension codes
are in use — where international codes use digits. The fifth character
either specifies anatomy (in the musculoskeletal chapter, a letter selects
the side and region) or classifies the episode (first episode, recurrence,
primary or secondary diagnosis).

**OMOP mapping situation.** Athena has no dedicated ICD-10-SE vocabulary.
Core three- to four-character codes map well through the standard
ICD-10-to-SNOMED CT relationships (above 90% in published Swedish work);
the fifth-character extensions map poorly (below 50%), and the common
default is truncation to the parent code — `M16.1A` and `M16.1B` both
collapse to the target of `M16.1`, losing laterality and episode
information. Preserving the fifth character requires custom mappings to
pre-coordinated SNOMED CT concepts. Diagnoses route to the Condition
domain.

## KVÅ (procedures)

KVÅ (Klassifikation av vårdåtgärder) is Sweden's national procedure
classification, mandatory for national reporting since 2007. It merges
three groups totalling roughly 9,000 active codes: KKÅ, the surgical
codes (three letters plus two digits, based on the Nordic NOMESCO
classification, roughly 5,000 codes); KMÅ, the medical procedures (two
letters plus three digits, a purely national extension, roughly 3,000);
and administrative Z-codes (roughly 1,000).

**OMOP mapping situation.** The Nordic connection helps the surgical half:
NOMESCO-based codes reach SNOMED CT procedure concepts through
published Nordic mapping work (around 70% of KKÅ in published Swedish
efforts). KMÅ has no international counterpart and maps poorly; the
administrative Z-codes vary. Procedures route to the Procedure domain,
with value-bearing diagnostic procedures reaching Measurement and
administrative codes Observation. KVÅ is the weakest of the four systems
in standard coverage and the clearest case for a reviewed, semi-automated
mapping workflow.

## ATC (medications)

Sweden classifies pharmaceutical substances with the WHO ATC
classification. Standard ATC codes (first letter A through V) map well to
RxNorm, the OMOP standard drug vocabulary — above 95% in published work —
because the RxNorm Extension vocabulary carries the non-US drug mappings.

Two national wrinkles:

- **Y-codes.** The Swedish Dental and Pharmaceutical Benefits Agency
  (TLV) assigns Y-codes to medical consumables — ostomy and diabetes
  supplies, drug-administration equipment — which sit outside the WHO
  classification. Standard ATC parsers that expect A through V reject
  them. They route to Device Exposure or Observation, not Drug Exposure,
  and typically need custom concepts.
- **Granularity.** ATC classifies at the ingredient level. Some Swedish
  OMOP implementations instead use NPL-ID, the national product-level
  medication identifier maintained by Läkemedelsverket, when dose- or
  product-level granularity matters. ATC-to-RxNorm remains sufficient
  when only the drug class is needed.

Medications route to the Drug Exposure domain.

## NPU (laboratory results)

Sweden codes laboratory results with NPU (Nomenclature for Properties
and Units), the international metrological nomenclature, with national
SWE-codes for tests that lack an international NPU entry (maintained in
the Swedish NPU database by Equalis). NPU differs structurally from
LOINC, the OMOP standard for laboratory observations: an NPU code
pre-coordinates its unit of measure, while a LOINC code keeps the unit
separate and accepts many. One LOINC code can therefore correspond to
several NPU codes that differ only by unit, and no comprehensive
one-to-one NPU-to-LOINC mapping exists. Laboratory observations route to
the Measurement domain, with the unit from the NPU code carried in the
measurement's unit field.

## What this means for the pipeline

Three consequences shape this repository:

1. **Unmapped codes are kept, not dropped.** A code whose system has no
   standard mapping — a KVÅ medical code, an ICD-10-SE depth code, an
   ATC Y-code — keeps its source vocabulary and code in the MEDS extract
   (`<vocabulary_id>/<concept_code>`), never the OMOP placeholder
   concept 0, so `omopmap clmbr coverage` can count it
   ([artifacts.md](artifacts.md)).
2. **Coverage is measured per vocabulary.** The coverage report splits
   recognition by vocabulary id, so the share attributable to each source
   system is visible rather than averaged away
   ([clmbr-concept-set.md](clmbr-concept-set.md)).
3. **Mapping is a reviewed workflow, not a lookup.** The retrieval,
   adjudication and calibration tooling
   ([retrieval-methods.md](retrieval-methods.md),
   [human-adjudication.md](human-adjudication.md)) exists because the
   hard part — KVÅ medical procedures, Swedish depth codes — has no
   reliable automatic mapping to accept blindly.
