# KG pipeline: resolution, canonicalization, fusion, ETO evaluation (Person 2)

Person 2's implementation of the stages after extraction, so the graph can be rebuilt from **Person 1's
`wiki_claims.jsonl` + Person 2's `sec_claims.jsonl` (v3)** alone. The outputs keep Person 1's contracts
(`common/schemas.py`: CanonicalEntity, ResolvedClaim, FusedEdge), so his Neo4j loader reads them unchanged.
Person 3's artifacts are not needed. His numbers can still be cited as a baseline.

> **Note:** the six `.py` modules listed below are added in a separate commit. Colab paths in the notebook are placeholders (`YOUR_FOLDER`).

## Files

| File | Stage | LLM calls |
|---|---|---|
| `kg_common.py` | config, IO, Gemini call + cache, embeddings, company resolution, ID ledgers | – |
| `canonicalize_offerings.py` | offering mentions → product families (EDC define + canonicalize) | ~110 + ~20 merge |
| `build_kg.py` | entity resolution + fusion → `data/kg/*.jsonl` + build report | none |
| `eto_reference.py` | arranges ETO data into reference pairs (evaluation only) | none |
| `evaluate_eto.py` | semantic evaluation: LLM judge + embeddings, RQ4 variants, per-group recall, label sheet | ~85 (judge) |
| `audit_edges.py` | precision by relation / source / entity type, entity-recognition accuracy, confidence calibration | ~45 (judge) |
| `run_kg_pipeline.ipynb` | runs everything in Colab, pilot first | |

Gemini does extraction (SEC) and canonicalization; the **judge** is a different model family: Groq's free tier, `openai/gpt-oss-120b` (set in `kg_common.JUDGE`; any OpenAI-compatible endpoint works). Colab secrets: `GeminiAPIKey`, `GroqAPIKey`.
Groq's free tier allows about 200k tokens per day for this model. A full ETO judge pass uses most of that, so run the
edge check on another day. Cached replies make re-runs free.

All LLM replies are cached in `data/kg/cache/`; IDs are kept stable across re-runs by ledgers in `data/kg/`.

## Design

**Company resolution (deterministic).** Registry names, aliases, legal names and tickers, compared on normalised names
(no punctuation, no legal suffixes). Then a short reviewed list of former names and short forms (Hynix → SK Hynix,
Avago → Broadcom …), `Name (ABBR)` / `Name ("ABBR")` patterns (the abbreviation becomes an alias), trailing brackets
(`APPLIED MATERIALS, INC. (Santa Clara, CA)`), and a conservative fuzzy match (≥ 95) against registry names only.
Anything else becomes a provisional company (`company_1001+`) instead of being dropped. Generic mentions
("subcontractors", "our customers") are dropped with a logged reason. An SEC subsidiary statement whose child would
otherwise collapse into its parent (Arm Limited → Arm Holdings) keeps the child as its own legal entity.

**Offering canonicalization (LLM, ETO-blind).** Before this step, 90–98 % of Input/Product/Technology nodes have exactly
one edge (measured on our data). For each (company, mention) the LLM reads the evidence sentence and names the generic
**product family** ("TWINSCAN NXT:2100i" → "DUV immersion lithography systems"). It picks one of 37 broad categories
(our own list) and records whether the evidence was enough or outside knowledge was needed (`grounding`). Existing
families are retrieved by embedding similarity and shown in the prompt so they get reused (EDC's canonicalize step). A second pass
re-asks about any non-vague mention whose family only repeats its category name (e.g. "deposition equipment"),
so families stay one level below the category ("ALD systems", "electroplating systems"). It also re-checks mentions judged "not an offering"
under a narrower rule, because manufacturing services like wafer fabrication or testing are offerings. A verification
pass then shows the LLM each family with its mentions and moves misfits to a better family (EDC-style self-check).
A merge pass finally checks near-duplicate family names. Offerings resolve **many-to-one** to family entities. Every observed
mention is kept as an alias and on its Claim, so no detail is lost. Entity types follow Person 1's ontology: chip,
equipment, material, design and manufacturing-service families are `Input`; process and packaging technologies are
`Technology`; downstream systems, software and services are `Product`.

**Fusion.** One edge per (subject, relation, object). `CUSTOMER_OF(a,b)` is merged with `SUPPLIES_TO(b,a)`, and
`PARTNERS_WITH` is symmetric.
- group type: SUPPORT (both sources), DUPLICATE, COMPLEMENTARY (different mentions merged by resolution),
  SINGLE_SOURCE, CONFLICT;
- confidence: HIGH (two sources or ≥ 2 documents), MEDIUM, LOW (conflict losers);
- one headquarters per company: ranked by the filer's own SEC statement > number of documents > Wikipedia infobox,
  with losers kept and flagged;
- `first_seen` / `last_seen` are SEC filing dates (observations, not validity);
- core layer as defined in the ontology.

All outputs are validated against Person 1's pydantic models in every run.

**Numeric confidence.** Next to the HIGH/MEDIUM/LOW label (the only form the contract allows), every edge gets a score in
`edge_scores.jsonl`: a noisy-OR over supporting documents, with each document contributing
prior(source) × resolution score. Repeats within one document don't add up, independent documents do, and conflict
losers are penalised. `SOURCE_PRIOR` starts at SEC 0.83 (our audit) and Wikipedia 0.80 (placeholder). The build report
profiles labels and scores by source, relation, object type and layer. Whether the score means anything is tested in
two ways: against ETO (does ETO confirm high-score PROVIDES edges more often?) and against the human edge audit
(observed precision and Brier score for each score bin).

## Evaluation against ETO

**Reference.** ETO's organisation providers are linked to our companies by name (+3 hand-checked overrides). Scope is
limited to companies with a source document in our corpus. Years are merged, and each pair gets ETO's definition,
`is a type of` ancestors, stage, and market share. Pairs where ETO also lists a subtype for the same company are flagged
(`most_specific`). On the current data: **140 pairs for 30 of 62 corpus companies (105 most-specific); 116 of 140 are
tools**, mostly Applied Materials (31), Lam (22), KLA (19), ASML (9). ETO has no rows for fabless and analog companies
(Broadcom, Qualcomm, TI …), so they can't be scored.

**Matching (semantic, never by name).**
- *LLM judge.* Per company, the judge labels each ETO input exact / kg_more_specific / kg_more_general / related / none
  against the graph's offerings and flags matches that needed outside knowledge.
  - Reported: strict recall (exact or more specific), graded recall (½ for broader), text-grounded recall,
    most-specific-only recall, share-weighted recall and stage-level recall, with company-level bootstrap 95% CIs.
- *Embeddings.* The best cosine similarity between the ETO definition and the company's offerings is thresholded at τ.
- *Variants (RQ4).* Wikipedia, SEC, combined before fusion (raw mentions, company resolution only) and combined after
  fusion (families).
- *Precision side.* The share of the company's Input families that ETO confirms. Unconfirmed ones are listed for manual
  checking, because ETO is incomplete.

**Per-group results.** Recall broken down by ETO input type (tool / material / design), stage and most-specific flag,
for each source variant. Also: how much recall survives if the graph is filtered to HIGH (or score ≥ t) edges, and the
ETO-confirmed share by edge confidence and source.

**Robustness.** Judge replies must be valid JSON that answers every reference input. Otherwise they are retried, then
the offering list is split in half. Invalid replies are never cached. Inputs that still get no answer are marked
`judge_failed` and excluded from the recall denominators; the report shows how many pairs were judged.

**Validation.** `eto_label_sheet.csv` holds 60 pairs (20 judge matches, 20 near misses, 20 random). It includes the company's full offering list. Label them
yes (graph has it, same or more specific) / partial (only a broader offering) / no and save the file as `eto_label_sheet_labeled.csv`. `evaluate_eto.main(calibrate=True)` then reports
the judge's 3-class accuracy (match / broader / none), binary accuracy, Cohen's κ and precision/recall against you, and calibrates τ.

## Edge audit (everything ETO cannot check)

`audit_edges.py` draws 80 fused edges, 10 per relation (split Wikipedia / SEC / both). You label whether each edge is
correct, whether the subject and object entities were recognised correctly, and the error type. Reported, with Wilson
95% CIs:
- precision by relation, source, object entity type, confidence label and score bin;
- entity-recognition accuracy by type (companies, countries, product families …);
- error types;
- confidence calibration;
- measured per-source priors.

The judge model also checks about 600 edges against their evidence, which gives larger samples per group. Use those
numbers only if the judge's agreement with your labels (Cohen's κ) is acceptable.

## Leakage guards and limitations

**Leakage guards.**
1. The canonicalizer never reads ETO files; families are our own vocabulary.
2. `provision.csv` (the answers) is used only to build the reference.
3. No ETO ID or name is written into the graph (`eto_id` stays empty).
4. The judge only sees the company's ETO inputs at evaluation time.
5. The judge is a different model family (gpt-oss via Groq) from the canonicalizer and the extractors (Gemini, Qwen), to avoid self-preference.

**Limitations to state in the report.**
- The reference is equipment-heavy.
- ETO is incomplete, so recall is a lower bound and precision cannot be measured against it.
- The judge is an LLM (hence the human validation).
- The two sources were extracted by different LLMs and from different sections.
- The "before fusion" variant already includes company resolution, which is needed to know whose offerings they are.
