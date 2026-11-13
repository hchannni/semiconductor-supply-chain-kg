# Pipeline (reference code)

Code used to build the SEC portion of release 2026-10-13, and the resolution / canonicalization / fusion stages
that turned the Wikipedia and SEC claims into the graph. It is kept for transparency and has not been re-run since that
release. Running it again needs API keys and will not reproduce the release byte-for-byte: the LLM outputs were cached
and the cache is not part of this repository.

| Folder | What it does |
|---|---|
| `sec/` | SEC EDGAR 10-K / 20-F filings → sections → chunks → LLM claim extraction and validation → audit sample |
| `fusion/` | Wikipedia + SEC claims → company resolution → offering canonicalization → fused edges → ETO evaluation |

Not included: intermediate data, model caches, ETO tables (CC BY-NC, downloaded at run time for evaluation only),
the Wikipedia extraction, and the shared ontology, schemas and validators written by a teammate (imported as
`person1export/`; see `sec/README.md`).

Author of this folder: Roope Laurila.
