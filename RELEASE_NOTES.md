# Dataset snapshot

**Snapshot date:** 2026-10-13  
**Package date:** 2026-10-14

The snapshot contains 1,598 entities, 2,988 relations and 6,417 supporting claim records. It combines Wikipedia and SEC filing information for semiconductor supply-chain research. A total of 589 relations have support from both source types.

## Contents

- Entity and relation data in UTF-8 JSONL.
- Claim-to-document provenance and source URLs.
- Evidence-support scores and flags for time, scope and unresolved evidence.
- Record schemas and Python query/validation examples.
- LLM-assisted precision and ETO reference-coverage summaries.

## Data boundaries

Claim provenance retains identifiers, relation endpoints, source attribution and extraction-version metadata. It does not contain quotation text. The document index contains 129 records; 128 are referenced by active claims. Relation dates describe source observations, not necessarily the period when a business relationship was valid.

This package provides a graph snapshot rather than the construction pipeline. This release includes the knowledge graph, source references, schema documentation, evaluation summaries, and usage examples. Methodology and field definitions are documented in `README.md`, `schema/SCHEMA.md` and `evaluation/README.md`.
