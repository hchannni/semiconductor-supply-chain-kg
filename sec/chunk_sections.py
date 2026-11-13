"""
Step 3 of the SEC pipeline: section files -> chunks ready for the LLM extractor.

Reads data/interim/sec_sections/{document_id}_item{N}.txt (from split_sections.py)
and writes data/interim/sec_chunks.jsonl, one chunk per line:
  chunk_id, document_id, ticker, company, filing_type, filing_date, item,
  heading_hint, text

Design choices
  * Chunks break only at line ends, preferably after a line ending in sentence
    punctuation, so a sentence is (almost) never cut in half.
  * We do NOT assume one line == one paragraph (TSM filings break mid-sentence).
  * Junk lines (bare page numbers, "Table of Contents") are dropped.
  * heading_hint = the most recent short, heading-looking line, to give the LLM
    context like "Manufacturing". It is only a hint.
  * PHASE A extracts from the business-description items only. Risk factors are
    ~70% of the text but mostly hypothetical statements; add them in a second pass.
"""
import os
import json
import re
from pathlib import Path

ROOT = Path(os.environ.get("KG_ROOT", Path(__file__).resolve().parent / "workdir"))   # set KG_ROOT to your working folder
SEC_DIR = ROOT / "data/raw/sec"
SECTION_DIR = ROOT / "data/interim/sec_sections"
OUT_PATH = ROOT / "data/interim/sec_chunks.jsonl"

EXTRACT_ITEMS = {"10-K": ["1"], "20-F": ["4"]}     # PHASE A. Phase B would add 10-K "1A" and 20-F "3".
MIN_SECTION_CHARS = 20000                          # shorter than this = the splitter failed (e.g. ASML); skip, do not send junk to the LLM
# ---- per-company fallback (ASML, Intel): their filings do not use standard "Item" headings, so the section split fails.
# For them only, paragraphs about suppliers/customers/manufacturing are selected from the full converted text instead.
# Other companies are untouched, so their chunks, saved model replies and claim IDs stay exactly the same.
KEYWORD_FALLBACK_TICKERS = {"ASML", "INTC"}
TEXT_CACHE = str(ROOT / "data/interim/sec_text") + "/{doc_id}.v2.txt"      # written by split_sections.py
FALLBACK_MAX_CHARS = 120_000                       # per filing, about 17 chunks
STRONG_CUE = re.compile(r"(?i)\bsuppl(?:ier|iers|y|ies|ied)\b|\bcustomers?\b|\bfoundr(?:y|ies)\b|sole[- ]source|\bvendors?\b|"
                        r"\bpartners?(?:hip|ships)?\b|joint venture|\bsubsidiar")
WEAK_CUE = re.compile(r"(?i)manufactur|\bwafers?\b|purchas|\bsourc(?:e|es|ed|ing)\b|\bfabs?\b|assembl|packag|lithograph|\bEUV\b|\bacquir")
SKIP_PARAGRAPH = re.compile(r"(?i)contracts? with customers|performance obligation|ASC 606|IFRS 15|deferred revenue|"
                            r"forward[- ]looking statements|trademark")
MAX_CHARS = 7000                                   # ~1.7k tokens
HARD_CAP = 10000                                   # force a cut here even mid-paragraph

JUNK = re.compile(r"(?i)^(\d{1,3}|table of contents|index|part [ivx]+)$")
SENTENCE_END = re.compile(r"[.!?:;)\u201d\"']\s*$")


def clean_lines(text):
    return [l for l in (x.strip() for x in text.split("\n")) if l and not JUNK.match(l)]


def split_long_line(line):
    """A single line longer than HARD_CAP is split on sentence boundaries."""
    if len(line) <= HARD_CAP:
        return [line]
    parts, cur = [], ""
    for sent in re.split(r"(?<=[.!?])\s+", line):
        if cur and len(cur) + len(sent) > MAX_CHARS:
            parts.append(cur)
            cur = ""
        cur = (cur + " " + sent).strip()
    if cur:
        parts.append(cur)
    return parts


def looks_like_heading(line):
    return (len(line) <= 70 and line[0].isupper() and not re.search(r"[.,;:]$", line)
            and len(line.split()) <= 10)


def chunk_lines(lines):
    """Yield (heading_hint, text) chunks, never exceeding HARD_CAP characters."""
    flat = [p for l in lines for p in split_long_line(l)]
    cur, size, hint, cur_hint = [], 0, "", ""
    for line in flat:
        if cur and size + len(line) + 1 > HARD_CAP:      # adding this line would break the cap: flush first
            yield cur_hint, "\n".join(cur)
            cur, size = [], 0
        if looks_like_heading(line):
            hint = line
        if not cur:
            cur_hint = hint
        cur.append(line)
        size += len(line) + 1
        if size >= MAX_CHARS and SENTENCE_END.search(line):
            yield cur_hint, "\n".join(cur)
            cur, size = [], 0
    if cur:
        yield cur_hint, "\n".join(cur)


def keyword_section(text, max_chars=FALLBACK_MAX_CHARS):
    """Pick the paragraphs densest in supply-chain wording, each with its neighbours, back in document order."""
    lines = clean_lines(text)
    scored = []
    for i, l in enumerate(lines):
        if len(l) < 60 or SKIP_PARAGRAPH.search(l):
            continue
        s = 2 * len(STRONG_CUE.findall(l)) + len(WEAK_CUE.findall(l))
        if s:
            scored.append((s / len(l) ** 0.5, i))      # density: long boilerplate paragraphs do not win just by length
    keep, total = set(), 0
    for _, i in sorted(scored, key=lambda x: (-x[0], x[1])):
        if total >= max_chars:
            break
        for j in (i - 1, i, i + 1):                    # neighbours keep sentences that wrap across lines intact
            if 0 <= j < len(lines) and j not in keep and not SKIP_PARAGRAPH.search(lines[j]):
                keep.add(j)
                total += len(lines[j]) + 1
    return [lines[i] for i in sorted(keep)]


def write_chunks(out, r, item, lines):
    k, chars = 0, 0
    for hint, text in chunk_lines(lines):
        k += 1
        rec = {
            "chunk_id": f"{r['document_id']}_item{item}_c{k:03d}",
            "document_id": r["document_id"], "ticker": r["ticker"], "company": r["company"],
            "filing_type": r["filing_type"], "filing_date": r["filing_date"],
            "item": item, "heading_hint": hint, "text": text,
        }
        out.write(json.dumps(rec, ensure_ascii=False) + "\n")
        chars += len(text)
    return k, chars


def main():
    recs = [json.loads(l) for l in (SEC_DIR / "sec_documents.jsonl").read_text(encoding="utf-8").splitlines()]
    n_chunks, total_chars = 0, 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for r in recs:
            for item in EXTRACT_ITEMS.get(r["filing_type"], []):
                path = SECTION_DIR / f"{r['document_id']}_item{item}.txt"
                section_text = path.read_text(encoding="utf-8") if path.exists() else ""
                if len(section_text) < MIN_SECTION_CHARS:
                    cache = Path(TEXT_CACHE.format(doc_id=r["document_id"]))
                    if r["ticker"] in KEYWORD_FALLBACK_TICKERS and cache.exists():
                        lines = keyword_section(cache.read_text(encoding="utf-8"))
                        k, chars = write_chunks(out, r, "kw", lines)
                        n_chunks += k; total_chars += chars
                        print(f"{r['document_id']} {r['ticker']}: section split failed -> keyword fallback, {k} chunks")
                    else:
                        print(f"[skip] {r['document_id']} {r['ticker']} Item {item}: only {len(section_text)} chars, section split failed")
                    continue
                k, chars = write_chunks(out, r, item, clean_lines(section_text))
                n_chunks += k; total_chars += chars
                print(f"{r['document_id']} {r['ticker']} Item {item}: {k} chunks")
    print(f"\nTotal: {n_chunks} chunks, {total_chars:,} chars (~{total_chars // 4:,} tokens) -> {OUT_PATH}")


if __name__ == "__main__":
    main()
