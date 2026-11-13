"""
Step 2 of the SEC pipeline: filing HTML -> plain text -> relevant sections.

Run in Colab (Drive mounted) or locally. For every filing listed in
sec_documents.jsonl it:
  1. converts the HTML to text (cached to data/interim/sec_text/, since this is slow),
  2. finds "Item N" headings and cuts the filing into items,
  3. keeps only the items likely to describe suppliers/customers/partners,
  4. saves them to data/interim/sec_sections/{document_id}_item{N}.txt,
  5. prints a size report so you can sanity-check by eye.

Key trick: every "Item" heading appears at least twice (table of contents + real
section). For each item we keep the LONGEST span, because table-of-contents
entries are only a few characters long.
"""
import os
import json
import re
from pathlib import Path

ROOT = Path(os.environ.get("KG_ROOT", Path(__file__).resolve().parent / "workdir"))   # set KG_ROOT to your working folder
SEC_DIR = ROOT / "data/raw/sec"
TEXT_DIR = ROOT / "data/interim/sec_text"
TEXT_VERSION = "v2"   # bump when html_to_text changes, so stale cache is ignored
SECTION_DIR = ROOT / "data/interim/sec_sections"

# Which items to keep. 10-K: Item 1 Business, 1A Risk Factors.
# 20-F is numbered differently: Item 3 Key Information (has Risk Factors),
# Item 4 Information on the Company (the business description).
TARGETS = {"10-K": ["1", "1A"], "20-F": ["3", "4"]}
MIN_PLAUSIBLE_CHARS = 20_000   # a real Business/Risk section is never this short -> flag it

# "Item 1A", "ITEM 4." and the thin-space variant "ITEM\u20094" all match (\s covers Unicode spaces)
HEADING = re.compile(r"(?im)^[ \t]*item\s+(\d{1,2}[A-C]?)\b\.?")


def html_to_text(path):
    """HTML -> text where inline fragments stay joined and only block elements break lines.

    (The earlier version used get_text("\n"), which split sentences like "70.0%" into
    three lines and turned inline cross-references such as "Item 5" into fake headings.)
    """
    from bs4 import BeautifulSoup, NavigableString
    soup = BeautifulSoup(Path(path).read_bytes(), "html.parser")
    for tag in soup.find_all("ix:header"):          # hidden XBRL metadata, not real text
        tag.decompose()
    for s in soup.find_all(string=True):            # hard-wrapped source text -> single spaces
        if type(s) is NavigableString and s.parent.name not in ("script", "style"):
            s.replace_with(NavigableString(re.sub(r"\s+", " ", str(s))))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(["td", "th"]):         # keep table cells apart
        tag.append(" ")
    for tag in soup.find_all(["p", "div", "tr", "li", "table", "h1", "h2", "h3", "h4", "h5", "h6"]):
        tag.append("\n")                            # block elements end a line
    text = soup.get_text("")
    lines = [re.sub(r"[ \t\u00a0\u2009]+", " ", l).strip() for l in text.split("\n")]
    return "\n".join(l for l in lines if l)


def split_items(text):
    """Return {item_number: (start, end)} keeping the longest span per item."""
    hits = [(m.start(), m.group(1).upper()) for m in HEADING.finditer(text)]
    best = {}
    for i, (start, item) in enumerate(hits):
        end = hits[i + 1][0] if i + 1 < len(hits) else len(text)
        if item not in best or (end - start) > (best[item][1] - best[item][0]):
            best[item] = (start, end)
    return best


def main():
    TEXT_DIR.mkdir(parents=True, exist_ok=True)
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    recs = [json.loads(l) for l in (SEC_DIR / "sec_documents.jsonl").read_text(encoding="utf-8").splitlines()]
    for r in recs:
        cache = TEXT_DIR / f"{r['document_id']}.{TEXT_VERSION}.txt"
        if cache.exists():
            text = cache.read_text(encoding="utf-8")
        else:
            text = html_to_text(r["local_path"])
            cache.write_text(text, encoding="utf-8")
        items = split_items(text)
        print(f"\n{r['document_id']} {r['ticker']} {r['filing_type']} {r['filing_date']}  (total {len(text):,} chars)")
        for item in TARGETS.get(r["filing_type"], []):
            if item not in items:
                print(f"   Item {item}: NOT FOUND")
                continue
            s, e = items[item]
            body = text[s:e]
            flag = "" if len(body) >= MIN_PLAUSIBLE_CHARS else "   <-- SUSPICIOUSLY SHORT, check by hand"
            print(f"   Item {item}: {len(body):>9,} chars  starts at {s:,}{flag}")
            print(f"      begins: {body[:120].strip()!r}")
            (SECTION_DIR / f"{r['document_id']}_item{item}.txt").write_text(body, encoding="utf-8")


if __name__ == "__main__":
    main()
