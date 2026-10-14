# Evaluation

The following results describe the dataset in this repository. Both evaluations use `openai/gpt-oss-120b` to measure evidence support and coverage of a fixed ETO reference.

## Results

| Measure | Calculation | Result |
|---|---|---:|
| Strict evidence-support precision | 410 explicitly supported / 505 sampled relations | 81.19% |
| Precision including partial support | (410 explicit + 14 partial) / 505 | 83.96% |
| Strict ETO recall | 70 exact or more-specific matches / 128 reference pairs | 54.69% |
| Graded ETO recall | (70 + 0.5 × 52 broader matches) / 128 | 75.00% |
| ETO coverage including broader categories | (70 + 52 broader matches) / 128 | 95.31% |

The precision labels are 410 explicit, 14 partial and 81 unsupported. ETO labels are 70 strict matches, 52 broader-only matches and 6 unmatched cases, with no unresolved cases. Graded recall gives broader matches half credit; broader-inclusive coverage gives them full credit.

## Evidence-support precision

The `edgecheck-v3` protocol uses relation/source-stratified sampling and asks the model whether the supplied evidence supports each relation. Model settings are temperature 0 and low reasoning effort; the provider is Groq. Strict precision accepts only `yes`; partial-inclusive precision also accepts `partly`, with full credit in that measure.

The reported precision is the unweighted proportion of accepted labels among the 505 sampled relations.

## ETO reference coverage

The `eto-pairwise-v3.1-qualification-validation` protocol uses 128 company-input reference pairs covering 28 of the 50 target companies. Reference pairs are evaluation-only. The model is called through OpenRouter with Groq as the fixed provider and fallback disabled, at temperature 0 and low reasoning effort.

The comparison distinguishes an offering's canonical name, parent category, definition, company-specific mentions and supporting claim text. Transparent equipment-name equivalence rules can recognize direct matches. Remaining candidates use individual LLM comparisons; quotation checks and required technical qualifiers constrain positive judgments. Token overlap orders candidates; there is no embedding-based second judge in this evaluation.

Strict recall counts an exact or more-specific match. Graded recall assigns half credit to a broader match. Full-credit broader coverage also accepts a parent category as a match. All 128 cases were processed.

## Result files

`metrics.json` contains aggregate results, relation-level precision breakdowns and protocol metadata. The full benchmark tables, row-level evidence quotations and judge caches are outside this dataset package. No evaluation API calls are needed to use the graph or run its examples.
