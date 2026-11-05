# Project context

This repository exists because of a concrete measurement problem that the
research behind it ran into: a clinical foundation model trained on
American records recognised fewer than 20% of the distinct codes in the
transformed Swedish records.
This page records that context — the research, its framing, where this
repository sits in it, and the compute environments the work ran in.

## The research

The research is the thesis **"Global Models, Local Friction:
Interoperability, Adaptation, and Deployment of Clinical Foundation
Models in Swedish Healthcare"** — short name **GMLF**, expansion
**"Global Models, Local Friction"**. Canonical record:
<https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-590690>.

The thesis work was carried out within the EmergAI research project at
Uppsala University and Uppsala University Hospital.

### Abstract

> What does it cost to bring an international clinical foundation model
> into Swedish emergency care? Foundation models are large AI systems
> that learn from the time-ordered records of millions of patients to
> anticipate what is likely to happen to a patient next. The clinical
> foundation models most relevant to this project have largely been
> trained on American records and expect each record in one international
> format — which Swedish hospitals, with their own code systems, do not
> use. This project approaches that gap as a feasibility study, using one
> such model (CLMBR-T-Base) as its anchor.
>
> The adaptation cost takes three forms. Terminological cost arises when
> Swedish code systems (ICD-10-SE, KVÅ, ATC, NPU) are mapped onto
> international standard vocabularies (SNOMED CT, RxNorm, LOINC).
> Distributional cost arises because Swedish care records a different mix
> and concentration of events than the American data the model learned
> from, so the model interprets even faithfully translated codes through
> expectations tuned elsewhere. Implementation cost is the organisational
> and regulatory distance to using the model in practice.
>
> To measure that cost, a data-processing chain was built to translate
> Swedish emergency department records into the format the model expects,
> recording at each step what survives the translation and what is lost. A
> decisive finding appears when the transformed Swedish records enter the
> model: fewer than 20% of distinct codes are recognised by the model's
> vocabulary. The model can still be trained further on Swedish data, a
> step that would make it more adapted to Swedish care. However, further
> training on real Swedish records was deferred because the severely
> limited vocabulary recognition made a full run unlikely to produce
> scientifically reliable performance estimates. Moreover, a proposal for
> integration with the standard hospital-record interface (FHIR) is
> specified but not tested. Our assessment is that, for this model, the
> total cost of adapting it was too large relative to what the model could
> offer. What this work leaves behind, however, is not tied to this
> particular model. Its main contributions are an infrastructure for
> standardising Swedish emergency department records, a vocabulary mapping
> with broad coverage, a proposal for integrating the model with the
> hospital-record interface, and a three-part framework for adaptation
> cost.

## Where this repository sits: the adaptation-cost framework

The abstract names three costs of bringing a clinical foundation model
into Swedish care, and this repository addresses one of them directly plus
the infrastructure all three rest on:

| Cost (thesis framing) | What it is | Where it lives |
|---|---|---|
| **Terminological cost** | Swedish code systems (ICD-10-SE, KVÅ, ATC, NPU) must land on the standard vocabularies the model knows (SNOMED CT, RxNorm, LOINC), and every code that lands on nothing is lost to the model | **This repository**: the retrieval and mapping methods ([retrieval-methods.md](retrieval-methods.md), [dictionary-eligibility.md](dictionary-eligibility.md), [fine-tuning.md](fine-tuning.md), [human-adjudication.md](human-adjudication.md)) |
| **Record-standardisation infrastructure** | The data-processing chain that turns Swedish records into the format the model expects | **This repository**: OMOP CDM → MEDS/FEMR artifacts ([artifacts.md](artifacts.md)), and the CLMBR dictionary importer and coverage instrument ([clmbr-concept-set.md](clmbr-concept-set.md)) |
| **Distributional cost** | The model interprets even faithfully translated codes through expectations tuned on American event mixes | [ehr-clmbr-training](https://github.com/ludvigalden/ehr-clmbr-training) (continued pretraining); context only here |
| **Implementation cost** | The organisational and regulatory distance to clinical use | Out of scope for both code repositories |

Every mapping method here serves the terminological side: each source term
or code that reaches no standard concept is a code the model will never
see, so the mapping's recall and the dictionary's coverage bound what
continued pretraining can recover.

## The headline coverage finding, and how this repository measures it

The finding that motivates the tooling: **fewer than 20% of distinct
codes are recognised by the model's vocabulary** when the transformed
Swedish records enter CLMBR-T-Base. It is quoted exactly as the abstract
states it.

`omopmap clmbr coverage` is the reproducible instrument for that
measurement. It takes a MEDS extract and the pinned CLMBR concept set,
verifies both against their hashes, and reports the share of the
extract's distinct codes that the concept set contains — overall, per
vocabulary id, and with the most frequent unrecognised codes. It
**measures the extract it is given**; it does not reproduce the thesis
figure, which came from records that are not part of this repository.
What it covers, why it lists only codes, and its exit codes are in
[clmbr-concept-set.md](clmbr-concept-set.md); the report schema is in
[artifacts.md](artifacts.md).

## Compute environments

Two environments did the research's compute, and they stay distinct:

- **The record-standardisation pipeline and the model runs** (OMOP ETL,
  MEDS/FEMR artifact production, CLMBR work) ran on **Bianca at UPPMAX**,
  "a research system dedicated to analysing sensitive personal data, or
  other types of sensitive data": a sensitive-data enclave without
  outbound network access at run time.
- **The vocabulary-mapping experiments** (embedding retrieval, encoder
  fine-tuning, the source experiments this repository's methods follow,
  recorded in [ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se))
  ran on rented GPUs at **Runpod**. They work on vocabulary strings only,
  never on patient records.

The two never shared a machine, and no mapping experiment saw patient
data.

Because the standardisation side ran inside a sensitive-data enclave, the
artifact-producing parts of this repository are built for that setting,
and the claims below are true of the code as written (not aspirational;
see [reproducibility.md](reproducibility.md) for the per-command detail):

- The CLMBR importer has a local route (`--model-dir`) that accepts a
  byte-identical copy of the pinned dictionary file and checks the same
  SHA-256 as the download route; it performs no network access
  ([clmbr-concept-set.md](clmbr-concept-set.md)).
- `omopmap artifacts verify`, the MEDS writer and converter, label
  building, dictionary building, `stats mcnemar`, `validate runs` and
  `metrics compute` read and write local files only.
- Dependencies are pinned in the committed `uv.lock`. Commands that load
  a model by name load the pinned revision through `transformers`; offline,
  that needs the revision already in the local Hugging Face cache with
  `HF_HUB_OFFLINE=1`. That load and the CLMBR importer's download route are
  the only network access in the code.
- Artifacts carry manifests of SHA-256 hashes, so an artifact can be
  produced in one environment and checked byte for byte on arrival in
  another.

## Sibling repositories

- [ehr-clmbr-training](https://github.com/ludvigalden/ehr-clmbr-training)
  — continued pretraining, task heads, evaluation and inference for the
  CLMBR pipeline. It is the **consumer** of this repository's outputs:
  it reads the MEDS/FEMR extracts, the CLMBR concept set, the label sets
  ([artifacts.md](artifacts.md) is the contract) and nothing else of this
  code. Its validated pipeline was exercised on synthetic data; it claims
  no results from training on real records, and this repository claims no
  CLMBR training results at all.
- [ludvigalden/vocabmod-se](https://github.com/ludvigalden/vocabmod-se) — the Swedish
  vocabulary-mapping experiments. The mapping-method material here
  (retrieval, fine-tuning, adjudication, statistics) is the rewritten,
  generalised form of what those experiments did, and the aggregate
  results in [results.md](results.md) are theirs.
