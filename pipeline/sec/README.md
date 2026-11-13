# SEC pipeline

Run `run_sec_pipeline.ipynb` top to bottom (written for Google Colab), or the scripts in order:

1. `fetch_sec_filings.py` downloads the newest 10-K / 20-F for each company in `company_metadata.csv`.
2. `split_sections.py` converts the HTML to text and cuts out Item 1 (10-K) / Item 4 (20-F).
3. `chunk_sections.py` splits sections into chunks of about 7,000 characters at sentence boundaries.
4. `extract_claims.py` sends chunks to Gemini, validates every claim against its quoted evidence sentence, and writes `sec_claims.jsonl`.
5. `make_audit_sheet.py` (optional) draws a seeded sample of claims into an Excel sheet for human labelling.

## Configuration

| Variable | Purpose |
|---|---|
| `KG_ROOT` | working folder for `data/` (defaults to `./workdir` next to the scripts) |
| `SEC_USER_AGENT` | SEC asks for a contact in the User-Agent, e.g. `"Jane Doe jane@example.com"` |
| Colab secret `GeminiAPIKey` | Gemini API key, read in `extract_claims.py` through `google.colab.userdata` |

Step 4 imports the team's shared schemas and validators from `person1export/` under `KG_ROOT`. That code was written
by a teammate and is not included here.
