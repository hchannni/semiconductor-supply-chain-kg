# Snapshot schema and provenance

Files are UTF-8 JSON Lines: one JSON object per line, not a single JSON array. The companion JSON Schema files validate individual records. The schemas describe the graph records distributed in this repository.

## Graph model

`canonical_entities.jsonl` contains nodes. `fused_edges.jsonl` contains directed labeled edges (PARTNERS_WITH is semantically symmetric). IDs are snapshot-local join keys; do not join different releases by name or assume unregistered IDs are stable across releases.

The seven node types are Company, Input, Product, Technology, Facility, Country and SupplyChainStage. Offering families use the existing Input/Product/Technology labels; there is no literal OfferingFamily type in this snapshot. Descriptions can contain parent categories, but these are not an additional formal taxonomy in this export.

## Relations

The signatures below are observed in this snapshot; they are not a claim of an exhaustive universal domain ontology.

| Relation | Observed endpoint types (subject → object) | Interpretation |
|---|---|---|
| `BELONGS_TO_STAGE` | Input → SupplyChainStage; Product → SupplyChainStage; Technology → SupplyChainStage | Offering → supply-chain stage classification. |
| `LOCATED_IN` | Company → Country; Facility → Country | Company headquarters or facility → country; not every business-operation location. |
| `OPERATES_IN` | Company → Country; Company → Facility | Company → country or facility in which it operates. |
| `PARTNERS_WITH` | Company → Company | Symmetric company partnership; stored in lexical entity-ID order, not a directed business hierarchy. |
| `PROVIDES` | Company → Input; Company → Product; Company → Technology | Subject company offers the object product, input, technology or service family; it does not necessarily manufacture it. |
| `SUBSIDIARY_OF` | Company → Company | Subsidiary company → parent company. Retained errors and time/scope uncertainty are possible. |
| `SUPPLIES_TO` | Company → Company | Supplier company → customer company. CUSTOMER_OF claims are reversed into this orientation. |
| `USES` | Company → Input; Company → Product; Company → Technology | Subject company uses the object input, product or technology; not an assertion that the company sells it. |

## Provenance joins

`edge.claim_ids[] → claim_provenance.claim_id → claim_provenance.document_id → documents.document_id → source_url`

`edge.subject/object → canonical_entities.entity_id`; `edge.sources[] → documents.document_id`.

Claim orientation can differ: CUSTOMER_OF(A,B) becomes SUPPLIES_TO(B,A), and PARTNERS_WITH endpoints are sorted. The examples validate these transformations. A claim-level document owner can differ from either endpoint. Original article/filing bodies and quotation strings are intentionally absent; these links locate the cited documents but do not independently verify factual correctness.

## Groups and confidence

SUPPORT means both source types support an aligned triple before possible conflict overrides. SINGLE_SOURCE means a single claim; DUPLICATE and COMPLEMENTARY distinguish same-source repeated vs different surface mentions of the same triple. COMPLEMENTARY does not mean that semantic complementarity was independently established. CONFLICT overrides the group label for a detected conflict. Count cross-source edges using source_types, not group_type.

The confidence policy limits repeated SEC evidence by issuer group, treats Wikipedia as one publisher group and downgrades specified unresolved conflicts/scope/history cases. Neither confidence labels nor numeric scores are calibrated probabilities, and sources are not proven independent. first_seen/last_seen are SEC observation dates, not relation validity intervals. Empty date strings and absent optional fields must not be interpreted as present-day validity.

## Fields

### `canonical_entities.jsonl`

| Field | JSON type | Required | Meaning |
|---|---|---|---|
| `aliases` | array | yes | Alternate labels retained by normalization; not guaranteed globally unique. |
| `canonical_name` | string | yes | Canonical display label; not a unique key. |
| `cik` | string | no | Optional SEC CIK string; retain leading zeroes. |
| `description` | string | no | Optional entity description, sometimes containing a parent category and family definition. |
| `entity_id` | string | yes | Stable identifier within this snapshot; join edges to this field. |
| `entity_type` | string | yes | Node class; offering families retain Input, Product or Technology classes. |
| `ticker` | string | no | Optional issuer ticker. |

### `claim_provenance.jsonl`

| Field | JSON type | Required | Meaning |
|---|---|---|---|
| `claim_id` | string | yes | Evidence-record identifier; quotation text is not included in this package. |
| `document_id` | string | yes | Document identifier used to join provenance. |
| `extractor_version` | string | yes | Recorded extraction/recovery version label. |
| `object_entity_id` | string | yes | Resolved claim object entity ID, before inverse/symmetric fusion orientation. |
| `prompt_version` | string | yes | Recorded prompt label or not_applicable for recovery. |
| `publication_date` | string | yes | Source observation date as recorded; empty string means unknown, not current validity. |
| `relation` | string | yes | Predicate; direction is described in the relation table. |
| `resolution_score` | number | yes | Resolver heuristic; not validated correctness probability. |
| `source` | string | yes | Claims use uppercase SEC/WIKIPEDIA; documents retain lowercase sec/wikipedia. |
| `subject_entity_id` | string | yes | Resolved claim subject entity ID, before inverse/symmetric fusion orientation. |

### `documents.jsonl`

| Field | JSON type | Required | Meaning |
|---|---|---|---|
| `accession_number` | string | no | SEC filing accession number. |
| `cik` | string | no | Optional SEC CIK string; retain leading zeroes. |
| `company` | string | no | SEC company display name. |
| `company_id` | ['null', 'string'] | yes | Document owner/subject hint; nullable and not necessarily every company mentioned in the document. |
| `content_hash` | string | no | Original collector content fingerprint; not a checksum for this public file. |
| `document_id` | string | yes | Document identifier used to join provenance. |
| `filing_date` | string | no | SEC filing date. |
| `filing_type` | string | no | SEC form type. |
| `page_title` | string | no | Wikipedia page title. |
| `publication_date` | string | yes | Source observation date as recorded; empty string means unknown, not current validity. |
| `retrieved_at` | string | yes | Recorded source retrieval timestamp. |
| `source` | string | yes | Claims use uppercase SEC/WIKIPEDIA; documents retain lowercase sec/wikipedia. |
| `source_url` | string | yes | Original source URL; availability is not checked by this package. |
| `ticker` | string | no | Optional issuer ticker. |

### `edge_review_flags.jsonl`

| Field | JSON type | Required | Meaning |
|---|---|---|---|
| `all_support_claims_verified` | boolean | yes | Review flag; false also includes claims not fully reviewed. |
| `edge_id` | string | yes | Identifier for a fused relation. |
| `unresolved_evidence_candidates` | integer | yes | Number of associated evidence candidates whose support was insufficient to add them to the relation. |
| `historical_context_flagged` | boolean | yes | The relation involves a historical event; do not assume it describes the current situation. |
| `independence_verified` | boolean | yes | Whether source independence was established, not assumed. |
| `time_or_scope_review_required` | boolean | yes | True when a retained review flag requires time/scope interpretation. |

### `edge_scores.jsonl`

| Field | JSON type | Required | Meaning |
|---|---|---|---|
| `confidence` | string | yes | Heuristic HIGH/MEDIUM/LOW label; not a calibrated correctness probability. |
| `confidence_score` | number | yes | Heuristic evidence-support score; not a calibrated factual probability. |
| `edge_id` | string | yes | Identifier for a fused relation. |
| `interpretation` | string | yes | Text explaining the score limitation. |
| `publisher_groups` | array | yes | Issuer-level SEC groups plus Wikipedia group; independence is not proven. |

### `fused_edges.jsonl`

| Field | JSON type | Required | Meaning |
|---|---|---|---|
| `claim_ids` | array | yes | Supporting claim_id values; join claim_provenance.jsonl. |
| `confidence` | string | yes | Heuristic HIGH/MEDIUM/LOW label; not a calibrated correctness probability. |
| `conflict` | boolean | yes | A limited rule-based conflict flag; false does not establish truth. |
| `conflict_type` | string | no | Optional attribute or relation conflict category. |
| `conflict_with` | array | no | Optional edge ID(s) involved in the conflict. |
| `edge_id` | string | yes | Identifier for a fused relation. |
| `first_seen` | string | no | Earliest supporting SEC claim publication date; not relation validity start. |
| `group_type` | string | yes | Fusion bookkeeping classification; see group definitions below. |
| `last_seen` | string | no | Latest supporting SEC claim publication date; not relation validity end. |
| `layer` | string | yes | core/extended export classification; extended is not a correctness judgement. |
| `object` | string | yes | Object entity_id. |
| `relation` | string | yes | Predicate; direction is described in the relation table. |
| `source_count` | integer | yes | Number of distinct supporting documents; not the number of source families. |
| `source_types` | array | yes | Supporting source families: SEC and/or WIKIPEDIA. |
| `sources` | array | yes | Unique supporting document_id values, not URLs. |
| `subject` | string | yes | Subject entity_id. |
