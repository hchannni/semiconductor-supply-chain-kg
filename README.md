# Semiconductor Supply-Chain Knowledge Graph

A research dataset of semiconductor supply-chain relationships derived from Wikipedia and SEC filings. It connects companies to offerings, suppliers, customers, subsidiaries, partners, locations and supply-chain stages, with links to supporting source documents.

**Data snapshot:** 2026-10-13 · **Package date:** 2026-10-14

## Dataset overview

| Item | Count |
|---|---:|
| Entities | 1,598 |
| Relations | 2,988 |
| Supporting claim records | 6,417 |
| Relations supported by both Wikipedia and SEC | 589 (19.71%) |
| Source document records | 129 |

Collection covers 50 target companies. The graph contains 780 Company nodes in total, including companies mentioned as suppliers, subsidiaries and partners; coverage is not equally detailed for every company. The document index contains 50 Wikipedia entries and 79 SEC filings. Of these, 50 Wikipedia entries and 78 filings support active claims; one filing is retained as a document record only.

The graph was constructed using LLM extraction, entity and offering-family normalization, rule-based fusion, and targeted model-assisted source review. Relations retain their supporting claim and document identifiers. Confidence scores and review flags describe the available support and known uncertainties; they are not guarantees of correctness.

## Quick start

Python 3.9 or later is sufficient. All examples use the standard library and run offline, without an API key or database.

```sh
python3 examples/query_graph.py stats
python3 examples/query_graph.py company TSMC --relation PROVIDES --limit 5
python3 examples/query_graph.py company NVIDIA --relation SUPPLIES_TO --limit 5
python3 examples/query_graph.py trace edge_8e7037da59e45a32
python3 examples/validate_release.py
```

Run these commands from the repository root. Company search accepts a name, alias or exact entity ID and shows incoming and outgoing relations. The displayed subject–relation–object order gives the direction of each relation. Source URLs are printed but not fetched. Validation checks file integrity and graph references, not factual truth.

## File guide

The package contains graph data, source references, evaluation summaries and Python query examples. Article/filing bodies, quotation text, model-response caches, ETO benchmark tables and the full construction pipeline are outside the package. The examples read and validate the provided snapshot.

| Path | Contents |
|---|---|
| `data/canonical_entities.jsonl` | Entity IDs, names, types and aliases |
| `data/fused_edges.jsonl` | Relations, source references, confidence labels and conflict flags |
| `data/claim_provenance.jsonl` | Claim identifiers, resolved endpoints and source metadata |
| `data/documents.jsonl` | Document identifiers, source URLs and publication metadata |
| `data/edge_scores.jsonl` | Heuristic evidence-support scores and publisher groups |
| `data/edge_review_flags.jsonl` | Time, scope and evidence-verification flags |
| `data/statistics.json` | Dataset counts and distributions |
| `schema/SCHEMA.md` | Field definitions, relation directions and provenance joins |
| `schema/*.schema.json` | Record schemas for the six JSONL files |
| `examples/` | Query and validation scripts |
| `evaluation/` | Evaluation results, methods and scoring rules |
| `BUILD_METADATA.json` | Snapshot, format and repository information |
| `MANIFEST.json` | File sizes and SHA-256 checksums |
| `RELEASE_NOTES.md` | Snapshot contents and scope |
| `REUSE_STATUS.md` | Attribution and reuse terms |

JSONL stores one JSON object per line. Follow `edge.claim_ids → claim_provenance.document_id → documents.source_url` to locate the sources of a relation. The [schema documentation](schema/SCHEMA.md) provides the complete join definitions.

## Evaluation

| Measure | Result |
|---|---:|
| Strict evidence-support precision | 410/505 = 81.19% |
| Evidence-support precision including partial support | 424/505 = 83.96% |
| Strict ETO recall | 70/128 = 54.69% |
| Graded ETO recall (broader matches receive half credit) | 75.00% |
| ETO coverage including broader categories (full credit) | 122/128 = 95.31% |

Precision uses a relation/source-stratified sample of 505 relations; ETO covers 128 reference pairs across 28 target companies. Both evaluations use `openai/gpt-oss-120b`. See [evaluation details](evaluation/README.md) for methods and scoring rules.

## Usage notes

This dataset reflects information extracted from the collected Wikipedia articles and SEC filings and may contain errors or omissions. Source references, confidence labels, and review flags are provided to support interpretation. Confidence scores are heuristic rather than probabilities of correctness, and a missing relation does not imply that no real-world relationship exists. See the [evaluation documentation](evaluation/README.md) for methodology and scope.

## Attribution and reuse

Source URLs and available publication identifiers are provided in `data/documents.jsonl`. Dataset and example-code licenses have not yet been specified; see [reuse terms](REUSE_STATUS.md).

Repository: https://github.com/hchannni/semiconductor-supply-chain-kg
