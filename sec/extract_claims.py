"""
Step 4 of the SEC pipeline: chunks -> LLM -> validated raw claims (Contract A).

Outputs (in data/processed/):
  sec_llm_raw.jsonl            every raw LLM response, one per chunk (audit trail; lets you
                               re-validate WITHOUT calling the API again)
  sec_claims.jsonl             validated claims in the frozen Contract A format (what Person 3 reads)
  sec_claims_rejected.jsonl    claims that failed validation, with the reasons
  sec_claims_provenance.jsonl  per-claim traceability: chunk_id, source_url, retrieved_at, model, warnings

Flow:  (A) call the LLM for chunks that have no saved response yet
       (B) rebuild all claim files from the saved responses
So re-running is cheap and safe.
"""
import os
import collections
import difflib
import json
import re
import time
from pathlib import Path

ROOT = Path(os.environ.get("KG_ROOT", Path(__file__).resolve().parent / "workdir"))   # set KG_ROOT to your working folder
CHUNKS_PATH = ROOT / "data/interim/sec_chunks.jsonl"
DOCS_PATH = ROOT / "data/raw/sec/sec_documents.jsonl"
OUT_DIR = ROOT / "data/processed"

SECRET_NAME = "GeminiAPIKey"                       # name of your Colab secret (must match exactly)
MODEL = "gemini-3.5-flash-lite"                    # pinned name (not a "-latest" alias, which can change silently). Fallback: "gemini-3.1-flash-lite"
TEMPERATURE = None                                 # None = model default. Gemini 3.x docs advise against lowering it; set a number to override
ONLY_TEXT_CONTAINS = []                            # [] = every chunk. (For quick tests: phrases such as ["We utilize foundries"])
ONLY_TICKERS = {"NVDA", "TSM", "AVGO", "MU", "AMD", "ASML", "INTC", "LRCX", "AMAT", "ARM"}   # PROPOSAL = pilot set. Use set() to extract EVERY downloaded company
DELAY_SECONDS = 7                                  # stay under free-tier requests/minute
# Max claims kept per chunk for a (relation, object_type) pair. Product-catalog sentences ("Ryzen, Virtex, Kintex ...")
# otherwise flood the graph. Applied when claim files are rebuilt, so the raw model replies stay complete. {} = no cap.
CAP_PER_CHUNK = {("PROVIDES", "Product"): 5}
DROP_RELATIONS = set()                             # e.g. {"SUBSIDIARY_OF"} to leave ownership claims out of the output
# Person 1's shared code (common/ + extraction/validation.py), copied from the team repo. Flat or nested layout both work.
PERSON1_CODE_DIR = ROOT / "person1export"
TEXT_DIR = ROOT / "data/interim/sec_text"          # full filing text (.v2.txt): evidence offsets point into these files
COMPANIES_CSV = ROOT / "data/companies/companies.csv"   # optional: team registry, adds company_id to every claim
TEAM_RULES = True                                  # run Person 1's validate_claim + team rules at rebuild time (no LLM calls)
RETYPE_OFFERINGS = True                            # P5: re-type PROVIDES/USES objects as Input/Product/Technology (a few cached LLM calls)
EXTRACTOR_VERSION = "v6"
PROMPT_VERSION = "v3"

ENTITY_TYPES = {"Company", "Input", "Product", "Technology", "SupplyChainStage", "Country", "Facility"}
RELATIONS = {"PROVIDES", "USES", "SUPPLIES_TO", "CUSTOMER_OF", "PARTNERS_WITH", "SUBSIDIARY_OF",
             "LOCATED_IN", "OPERATES_IN", "BELONGS_TO_STAGE"}
REQUIRED_KEYS = ["subject_mention", "subject_type", "relation", "object_mention", "object_type", "evidence_span"]


# The evidence sentence must contain wording that actually expresses the relation; otherwise the model inferred it
# (e.g. from a company's name). Checked mechanically because the prompt rule alone was not reliable.
RELATION_CUES = {
    "SUBSIDIARY_OF": re.compile(r"(?i)subsidiar|wholly[- ]owned|majority[- ]owned|controlled by|\bparent\b"),
    "PARTNERS_WITH": re.compile(r"(?i)collaborat|partner|joint venture|alliance|cooperat|jointly|agreement with|together with"),
}
# Wording that must appear in the evidence of a supplier/customer claim. Company names are blanked out first
# (so "...Manufacturing Company" cannot count), which rejects name-only evidence such as table rows.
SUPPLY_CUE = re.compile(r"(?i)supplier|vendor|customer|purchas|\bsales\b|revenue|accounted for|represented|\bbuy|\bbought|\bsell|\bsold|suppl(?:y|ies|ied)|provid|procure|"
                        r"\bsourc|\brely|\brelies|reliance|depend|utiliz|\buse[sd]?\b|engag|manufactur|foundr|fabricat|assembl|packag|"
                        r"subcontract|distribut|licens|work(?:s|ed|ing)? with|collaborat|partner|\border|\bship|deliver|"
                        r"equipment|materials|components|memory|wafer|"
                        r"\bpower(?:s|ed|ing)?\b|\bused (?:in|by)\b|incorporat|integrat|\badopt|deploy|design[- ]?wins?|\bembedded in\b|\bfeatur(?:e|es|ed|ing)\b")
RELATION_CUES["SUPPLIES_TO"] = SUPPLY_CUE
RELATION_CUES["CUSTOMER_OF"] = SUPPLY_CUE

EDGE_RELATIONS = {"SUPPLIES_TO", "CUSTOMER_OF", "PARTNERS_WITH", "SUBSIDIARY_OF"}
COMPETITION = re.compile(r"(?i)\bcompet\w*")      # competitor lists are never supply relationships


# Customer-concentration tables ("The following customers accounted for 10% or more of net sales:" followed by rows such as
# "Samsung Electronics Co., Ltd. 12% 15%") arrive one row per line. A row naming a company plus percentages is valid
# evidence for a supplier/customer claim WHEN such a header is in the same chunk. The header is kept as evidence_context.
TABLE_ROW = re.compile(r"^\s*[^\d%\n]{3,}?(?:\s+(?:\d{1,3}(?:\.\d+)?\s?%|\*|\u2014|\u2013|-|n/?a)){1,6}\s*$", re.I)
CONCENTRATION_HEADER = re.compile(r"(?i)[^.\n]*\b(?:customers?|compan(?:y|ies)|distributors?)\b[^.\n]{0,160}?"
                                  r"\b(?:accounted for|represented|comprised)\b[^.\n]{0,160}?\b(?:revenues?|sales)\b[^.\n]*")


def table_row_header(c, chunk_text):
    """The concentration-table header that makes a bare table row valid evidence, or None."""
    if c.get("relation") not in ("SUPPLIES_TO", "CUSTOMER_OF") or not TABLE_ROW.match(c["evidence_span"].strip()):
        return None
    m = CONCENTRATION_HEADER.search(chunk_text)
    return m.group(0).strip() if m else None


ACRONYM_STOP = {"of", "the", "and", "on", "for", "de", "pte", "inc", "ltd", "llc", "gmbh", "ag", "plc", "sa", "nv", "bv", "co", "limited"}


def acronyms(m):
    """'European Semiconductor Manufacturing Company GmbH' -> {'ESMC', 'ESM'}; used when filings write the short form."""
    words = [w for w in re.findall(r"[A-Za-z]+", strip_paren(m)) if w.lower() not in ACRONYM_STOP and len(w) > 1]
    out = set()
    for drop_generic in (False, True):
        ws = [w for w in words if not (drop_generic and w.lower() in ("company", "corporation", "corp", "incorporated", "group", "holding", "holdings"))]
        acr = "".join(w[0] for w in ws).upper()
        if len(acr) >= 3:
            out.add(acr)
    return out


def surface_form(m, evidence):
    """The name as it is WRITTEN in the evidence: 'TSMC Washington, LLC' -> 'TSMC Washington', 'Japan Advanced
    Semiconductor Manufacturing, Inc.' -> 'JASM'. Returns the mention unchanged if it already appears (or nothing fits)."""
    if alnum(m) in alnum(evidence):
        return m
    core = " ".join(CORP_SUFFIX.sub(" ", strip_paren(m)).replace(",", " ").split())
    if len(core) >= 3:
        hit = re.search(r"\s*".join(re.escape(w) for w in core.split()), evidence, flags=re.I)
        if hit:
            return hit.group(0)
    for a in sorted(acronyms(m), key=len, reverse=True):
        if re.search(r"\b" + a + r"\b", evidence):
            return a
    return m


def significant_tokens(m):
    """Words of a name without 'Inc.', 'Ltd.' ...; only words of 4+ letters are checkable."""
    return [t for t in re.findall(r"\w+", strip_paren(CORP_SUFFIX.sub(" ", m)).lower()) if len(t) >= 4]


def party_in_evidence(m, evidence, filer):
    """The evidence must show both parties of a company-to-company claim (the filer is allowed to be 'we')."""
    if alnum(m) == alnum(filer):
        return True
    toks = significant_tokens(m)
    if not toks:
        return True
    ev_a = alnum(evidence)                       # parentheses kept: "(including X and Y)" names real parties
    if any(alnum(t) in ev_a for t in toks):
        return True
    return any(re.search(r"\b" + a + r"\b", evidence) for a in acronyms(m))    # "we owned 70% of ESMC"


# Legal / regulatory boilerplate never states a supply-chain relationship (trademark footers, jurisdiction language).
LEGAL_BOILERPLATE = re.compile(r"(?i)\btrademarks?\b|\bregistered (?:service )?marks?\b|\ball rights reserved\b|"
                               r"subject to the (?:laws|regulations|jurisdiction)|laws and regulations of|service of process")

# SUBSIDIARY_OF is also accepted when the evidence states an ownership share above 50% (e.g. "we owned 70.0% of ESMC").
# Minority stakes (e.g. 34.8%, 38.8%) are NOT subsidiaries; the ontology has no relation for them yet.
PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s?%")


def cue_ok(relation, evidence, mentions=()):
    cue = RELATION_CUES.get(relation)
    if cue is None:
        return True
    text = evidence
    if relation in ("SUPPLIES_TO", "CUSTOMER_OF"):
        for m in mentions:
            text = re.sub(re.escape(m), " ", text, flags=re.I)
    if cue.search(text):
        return True
    return relation == "SUBSIDIARY_OF" and any(float(p) > 50 for p in PERCENT.findall(evidence))


def canon_label(value, allowed):
    """Accept harmless capitalisation slips like 'PROVIDes' or 'company'."""
    v = value.strip()
    for a in allowed:
        if v.lower() == a.lower():
            return a
    return v


def canon_relation(value):
    """Fix model typos in relation labels: 'PROVIDes', 'PROVID_ES', 'PROVIDS' -> PROVIDES. Unknown labels stay unchanged (and get rejected)."""
    v = canon_label(value, RELATIONS)
    if v in RELATIONS:
        return v
    squashed = re.sub(r"[\s_\-]+", "", v).upper()
    for r in sorted(RELATIONS):
        if r.replace("_", "") == squashed:
            return r
    close = difflib.get_close_matches(v.upper(), sorted(RELATIONS), n=1, cutoff=0.85)
    return close[0] if close else v


def normalize_labels(c):
    if not isinstance(c, dict):
        return c
    c = dict(c)
    if isinstance(c.get("relation"), str):
        c["relation"] = canon_relation(c["relation"])
    for k in ("subject_type", "object_type"):
        if isinstance(c.get(k), str):
            c[k] = canon_label(c[k], ENTITY_TYPES)
    return c


PROMPT_TEMPLATE = """You extract supply-chain relationships from one section of a company's annual filing (SEC Form <<FORM>>).
The filing was submitted by <<FILER>> (ticker <<TICKER>>). In the text, "we", "us", "our" and "the Company" mean this filer.
Always write the filer as "<<FILER>>" in subject_mention / object_mention.
The text may contain line breaks in the middle of sentences; ignore them.

Return ONLY a JSON array. Each element has exactly these keys:
subject_mention, subject_type, relation, object_mention, object_type, evidence_span

Entity types: Company, Input, Product, Technology, SupplyChainStage, Country, Facility
  Input = materials, components, equipment or services bought to make products (wafers, memory, EUV tools, assembly services).
  Product = what a company makes and sells. Technology = a named process or technique (CoWoS, N3, EUV).

Relations (subject -> object):
  SUPPLIES_TO: Company -> Company. The subject sells to the object. Use when the sentence names SUPPLIERS ("we buy from X",
               "X is our supplier", "we use foundries such as X"): X SUPPLIES_TO filer.
  CUSTOMER_OF: Company -> Company. The subject buys from the object. Use when the sentence names CUSTOMERS
               ("our customers include X"): X CUSTOMER_OF filer.
  PROVIDES: Company -> Product or Input it makes/offers.   USES: Company -> Input, Product or Technology it relies on.
  PARTNERS_WITH: Company -> Company, a stated collaboration, joint venture or alliance.
  SUBSIDIARY_OF: Company -> Company that owns or controls it.
  LOCATED_IN: Company or Facility -> Country.   OPERATES_IN: Company -> Country where it has operations.
  BELONGS_TO_STAGE: Company or Product -> SupplyChainStage (design, foundry, memory, packaging, assembly, equipment, materials, ...).

Rules:
1. The evidence sentence ITSELF must state the relationship. Do not infer it from a company's name or from outside knowledge.
   Example: "Zed Arizona owns land in Arizona" does NOT say Zed Arizona is a subsidiary of Zed. Skip it.
   (A sentence like "Zed Arizona, our wholly-owned subsidiary, owns land" does.)
2. evidence_span: copy ONE sentence (or two consecutive sentences) EXACTLY from the text. No edits, no ellipses.
3. Companies must be NAMED. Skip anonymous ones ("a customer", "a limited number of suppliers", "Customer A").
   Generic materials or equipment ARE allowed as Input (e.g. "silicon wafers").
4. Competitors are not suppliers or partners. Competition statements produce no claims.
5. A sentence naming several companies, products, inputs or technologies produces ONE CLAIM PER ITEM.
   Every mention must be short: a company name, or a noun phrase of at most 6 words. Never copy a whole list into one mention.
6. Use names as written in the text (do not expand or translate them), except the filer.
7. If nothing qualifies, return [].
8. PROVIDES means the filer itself makes or sells the item. A list of applications, end markets or customer products
   ("technologies for smart cameras and PMICs") is NOT a list of things the filer provides. Skip those items.
9. USES means the filer itself relies on the item. If a sentence says customers have adopted a technology the filer offers,
   the claim is filer PROVIDES that technology, not USES.
10. LOCATED_IN / OPERATES_IN only when the sentence says the filer (or the named entity) has offices, fabs or operations in that place.
    Where its suppliers, customers or supply chain are located does not count.

Example 1 (supplier side). Text: "We rely on Acme Foundry Ltd. to fabricate our wafers and on Bolt Memory Inc. for memory."
Filer: Zed Chips Inc.
[{"subject_mention":"Acme Foundry Ltd.","subject_type":"Company","relation":"SUPPLIES_TO","object_mention":"Zed Chips Inc.","object_type":"Company","evidence_span":"We rely on Acme Foundry Ltd. to fabricate our wafers and on Bolt Memory Inc. for memory."},
 {"subject_mention":"Bolt Memory Inc.","subject_type":"Company","relation":"SUPPLIES_TO","object_mention":"Zed Chips Inc.","object_type":"Company","evidence_span":"We rely on Acme Foundry Ltd. to fabricate our wafers and on Bolt Memory Inc. for memory."}]

Example 2 (customer side). Text: "Our customers include Cobalt Devices Inc. and Delta Systems Corp."
Filer: Zed Chips Inc.
[{"subject_mention":"Cobalt Devices Inc.","subject_type":"Company","relation":"CUSTOMER_OF","object_mention":"Zed Chips Inc.","object_type":"Company","evidence_span":"Our customers include Cobalt Devices Inc. and Delta Systems Corp."},
 {"subject_mention":"Delta Systems Corp.","subject_type":"Company","relation":"CUSTOMER_OF","object_mention":"Zed Chips Inc.","object_type":"Company","evidence_span":"Our customers include Cobalt Devices Inc. and Delta Systems Corp."}]

Example 3 (technologies). Text: "We use Orbit packaging technology. Our 5-nanometer process has been adopted by makers of smart cameras and set-top boxes."
Filer: Zed Chips Inc.
[{"subject_mention":"Zed Chips Inc.","subject_type":"Company","relation":"USES","object_mention":"Orbit packaging technology","object_type":"Technology","evidence_span":"We use Orbit packaging technology."},
 {"subject_mention":"Zed Chips Inc.","subject_type":"Company","relation":"PROVIDES","object_mention":"5-nanometer process","object_type":"Technology","evidence_span":"Our 5-nanometer process has been adopted by makers of smart cameras and set-top boxes."}]
(Smart cameras and set-top boxes are customers' products, so they produce no claims.)

Section heading hint: <<HINT>>

TEXT:
<<TEXT>>
"""


def build_prompt(chunk):
    p = PROMPT_TEMPLATE
    for key, val in {"<<FORM>>": chunk["filing_type"], "<<FILER>>": chunk["company"], "<<TICKER>>": chunk["ticker"],
                     "<<HINT>>": chunk.get("heading_hint") or "(none)", "<<TEXT>>": chunk["text"]}.items():
        p = p.replace(key, val)
    return p


_client = None


class DailyQuotaExhausted(Exception):
    """The provider's per-day quota is used up: retrying today is pointless."""


DAILY_QUOTA = re.compile(r"(?i)retry in \d+h|per ?day")


def call_llm(prompt):
    """The ONLY provider-specific function. Swap this to change LLM provider."""
    global _client
    from google import genai
    from google.genai import types
    from google.colab import userdata
    if _client is None:
        _client = genai.Client(api_key=userdata.get(SECRET_NAME))
    for attempt in range(4):
        try:
            cfg = {"response_mime_type": "application/json"}
            if TEMPERATURE is not None:
                cfg["temperature"] = TEMPERATURE
            resp = _client.models.generate_content(
                model=MODEL, contents=prompt, config=types.GenerateContentConfig(**cfg))
            if resp.text is None:
                raise RuntimeError("empty response (possibly blocked)")
            return resp.text
        except Exception as e:                                   # rate limit -> wait and retry
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and DAILY_QUOTA.search(str(e)):
                raise DailyQuotaExhausted(str(e)[:300]) from e
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < 3:
                wait = 30 * (attempt + 1)
                print(f"   rate limited, waiting {wait}s")
                time.sleep(wait)
            else:
                raise


# ---------------- validation ----------------
def norm(s):
    """Compare text ignoring all whitespace, case and typographic quotes/dashes (robust to hard-wrapped lines)."""
    for a, b in [("\u201c", '"'), ("\u201d", '"'), ("\u2018", "'"), ("\u2019", "'"), ("\u2013", "-"), ("\u2014", "-")]:
        s = s.replace(a, b)
    return re.sub(r"\s+", "", s).lower()


CORP_SUFFIX = re.compile(r"(?i)\b(?:incorporated|inc|corporation|corp|company|co|limited|ltd|llc|gmbh|ag|n\.?v|plc|s\.?a|holdings?|group)\b\.?")


def strip_paren(s):
    """Remove (...) segments, e.g. the abbreviation in 'semiconductor (“CMOS”) logic'."""
    return re.sub(r"\([^)]*\)", " ", s)


def alnum(s):
    """Lower-case letters and digits only: ignores spaces, hyphens, quotes, brackets."""
    return re.sub(r"[\W_]+", "", s.lower())


def mention_key(m):
    """A name without 'Inc.', 'Ltd.', 'LLC' ...: 'TSMC Washington, LLC' -> 'tsmcwashington'."""
    m = strip_paren(m)
    k = alnum(CORP_SUFFIX.sub(" ", m))
    return k if len(k) >= 4 else alnum(m)


def parse_claims_json(text):
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    data = json.loads(t)
    if isinstance(data, dict) and isinstance(data.get("claims"), list):
        data = data["claims"]
    if not isinstance(data, list):
        raise ValueError("top-level JSON is not a list")
    return data


def validate_claim(c, chunk_text, filer):
    """Return (reasons_for_rejection, warnings)."""
    reasons, warnings = [], []
    if not isinstance(c, dict):
        return ["not_an_object"], warnings
    for k in REQUIRED_KEYS:
        if not isinstance(c.get(k), str) or not c[k].strip():
            reasons.append(f"missing_or_empty:{k}")
    if reasons:
        return reasons, warnings
    if c["relation"] not in RELATIONS:
        reasons.append(f"relation_not_allowed:{c['relation']}")
    for k in ("subject_type", "object_type"):
        if c[k] not in ENTITY_TYPES:
            reasons.append(f"{k}_not_allowed:{c[k]}")
    if norm(c["subject_mention"]) == norm(c["object_mention"]):
        reasons.append("subject_equals_object")
    if LEGAL_BOILERPLATE.search(c["evidence_span"]):
        reasons.append("legal_boilerplate")
    header = table_row_header(c, chunk_text)
    if header is None and not cue_ok(c["relation"], c["evidence_span"], (c["subject_mention"], c["object_mention"])):
        reasons.append(f"relation_cue_missing:{c['relation']}")
    if c["relation"] in EDGE_RELATIONS:
        if COMPETITION.search(c["evidence_span"]):
            reasons.append("evidence_is_about_competition")
        if not all(party_in_evidence(c[k], c["evidence_span"], filer) for k in ("subject_mention", "object_mention")):
            reasons.append("party_not_in_evidence")
    if c["relation"] in ("SUPPLIES_TO", "CUSTOMER_OF", "PARTNERS_WITH") and (c["subject_type"] != "Company" or c["object_type"] != "Company"):
        reasons.append("type_clash")
    if c["relation"] in ("PROVIDES", "USES") and c["object_type"] in ("Company", "Country"):
        reasons.append("type_clash")
    ev = norm(c["evidence_span"])
    if len(ev) < 20:
        reasons.append("evidence_too_short")
    elif ev not in norm(chunk_text):
        reasons.append("evidence_not_in_source")
    if not reasons:                                              # soft checks: recorded, not rejected
        ev_a, ch_a = alnum(strip_paren(c["evidence_span"])), alnum(strip_paren(chunk_text))
        for side in ("subject", "object"):
            m = c[f"{side}_mention"]
            if alnum(m) == alnum(filer):
                continue
            key = mention_key(m)
            if key not in ev_a:                                  # often harmless: the model wrote the long name, the sentence the short one
                warnings.append(f"{side}_not_in_evidence")
            if key not in ch_a:                                  # the name appears NOWHERE in the chunk: strong hallucination signal
                warnings.append(f"{side}_not_in_chunk")
        if header:
            warnings.append("table_row_evidence")
        if len(c["evidence_span"]) > 600:
            warnings.append("evidence_long")
        if len(c["subject_mention"]) > 60 or len(c["object_mention"]) > 60:
            warnings.append("mention_too_long")
    return reasons, warnings



def canonicalize_direction(c):
    """CUSTOMER_OF(A -> B) means B supplies A. Store everything as SUPPLIES_TO (supplier -> buyer).
    The LLM states the relation the way the sentence reads; this code does the inversion, which is more reliable."""
    if c["relation"] != "CUSTOMER_OF":
        return c, None
    flipped = dict(c, relation="SUPPLIES_TO",
                   subject_mention=c["object_mention"], subject_type=c["object_type"],
                   object_mention=c["subject_mention"], object_type=c["subject_type"])
    return flipped, "CUSTOMER_OF"


CUSTOMERS_PAT = re.compile(r"(?i)\bour (?:major |largest |key |principal )?customers?\b|\bcustomers? (?:include|such as)\b")
SUPPLIERS_PAT = re.compile(r"(?i)\b(?:our|principal|major|key) (?:\w+ )?suppliers?\b|\bwe (?:purchase|buy|procure|source|rely on|utilize|engage with)\b")


def direction_warnings(c, filer):
    """Cheap heuristic for reversed SUPPLIES_TO claims. Soft warning only (false positives are possible)."""
    if c["relation"] != "SUPPLIES_TO":
        return []
    ev = c["evidence_span"]
    obj_is_filer = norm(c["object_mention"]) == norm(filer)
    subj_is_filer = norm(c["subject_mention"]) == norm(filer)
    if obj_is_filer and CUSTOMERS_PAT.search(ev) and not SUPPLIERS_PAT.search(ev):
        return ["direction_suspect"]
    if subj_is_filer and SUPPLIERS_PAT.search(ev) and not CUSTOMERS_PAT.search(ev):
        return ["direction_suspect"]
    return []


# ---------------- pipeline ----------------

# ======================= team rules: Person 1's shared code (single source of truth) =======================
P1 = {}


def load_person1_code(src_dir=None):
    """Import common.* and extraction.validation from Person 1's export. The files may sit in common/ and extraction/
    subfolders or all in one folder; they are arranged into a package layout in a local cache folder first."""
    if P1:
        return P1
    import importlib
    import shutil
    import sys
    src_dir = Path(src_dir or PERSON1_CODE_DIR)
    if not src_dir.exists():
        raise SystemExit(f"Person 1's code not found at {src_dir}")
    pkg = Path("/tmp/person1_pkg")
    home = {"ontology.py": "common", "schemas.py": "common", "text.py": "common", "utils.py": "common",
            "vocab.py": "common", "validation.py": "extraction"}
    for name, sub in home.items():
        hits = sorted(src_dir.rglob(name), key=lambda p: (sub not in p.parts, len(p.parts)))
        if not hits:
            raise SystemExit(f"{name} missing from {src_dir}")
        (pkg / sub).mkdir(parents=True, exist_ok=True)
        (pkg / sub / "__init__.py").touch()
        shutil.copy(hits[0], pkg / sub / name)
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    for mod in ("common.ontology", "common.utils", "common.text", "common.vocab", "common.schemas", "extraction.validation"):
        P1[mod.split(".")[-1]] = importlib.import_module(mod)
    return P1


def one_sentence(evidence, mentions, filer):
    """Shortest verbatim slice holding every non-filer mention: one sentence (else two), using the team's sentence
    splitter; a sentence that is still very long is cut at ';'."""
    split = P1["text"].split_sentences
    need = [mention_key(m) for m in mentions if alnum(m) != alnum(filer)]

    def ok(piece):
        return all(k in alnum(strip_paren(piece)) or k in alnum(piece) for k in need)

    def best(text, spans):
        for width in (1, 2):
            for i in range(len(spans) - width + 1):
                piece = text[spans[i][0]:spans[i + width - 1][1]].strip()
                if ok(piece):
                    return piece
        return text

    piece = best(evidence, split(evidence)) if len(split(evidence)) > 1 else evidence
    if len(piece) > 400:
        clauses, start = [], 0
        for m in re.finditer(r";\s+", piece):
            clauses.append((start, m.start() + 1))
            start = m.end()
        clauses.append((start, len(piece)))
        piece = best(piece, clauses)
    return piece


def team_validate(c, chunk_text, filer, ticker):
    """Person 1's validate_claim on one claim (as the model stated it). Returns (claim, reasons, warnings)."""
    v = P1["validation"]
    if c["relation"] == "LOCATED_IN" and c["subject_type"] == "Company" and c["object_type"] == "Facility":
        c = dict(c, relation="OPERATES_IN")                   # contract: a company OPERATES_IN a facility (feedback P3)
    full = c["evidence_span"]
    tries = [full]
    if not TABLE_ROW.match(full.strip()):
        short = one_sentence(full, (c["subject_mention"], c["object_mention"]), filer)
        if short != full:
            tries.insert(0, short)                              # prefer one sentence; fall back if trimming lost the cue
    for ev in tries:
        cand = {"subject": c["subject_mention"], "subject_type": c["subject_type"], "relation": c["relation"],
                "object": c["object_mention"], "object_type": c["object_type"], "evidence": ev}
        res = v.validate_claim(cand, chunk_text, None, [filer, ticker])
        if res.ok:
            break
    if not res.ok:
        return c, res.reasons, res.warnings
    r = res.claim
    out = dict(c, subject_mention=r["subject"], subject_type=r["subject_type"], relation=r["relation"],
               object_mention=r.get("object_canonical") or r["object"], object_type=r["object_type"],
               evidence_span=r["evidence"])
    warnings = list(res.warnings)
    # LOCATED_IN of a company = headquarters only (stricter than the shared retype: no HQ wording -> OPERATES_IN)
    if out["relation"] == "LOCATED_IN" and out["subject_type"] == "Company" and not v.HQ_CUE_RE.search(out["evidence_span"]):
        out["relation"] = "OPERATES_IN"
        warnings.append("located_in_retyped_as_operates_in")
    return out, [], warnings


def contradictory_ids(claims, names_of_doc):
    """Same rule as the Wikipedia extractor: two companies that are each other's supplier (or parent) in one sentence
    -> both claims dropped, one of them is wrong."""
    v = P1["validation"]
    groups = collections.defaultdict(list)
    for c in claims:
        if c["relation"] in ("CUSTOMER_OF", "SUPPLIES_TO", "SUBSIDIARY_OF"):
            names = names_of_doc[c["document_id"]]
            key = lambda m: "__filer__" if v.is_article_company(m, names) else P1["utils"].normalize_key(m)
            groups[(c["document_id"], c["evidence_span"], frozenset((key(c["subject_mention"]), key(c["object_mention"]))))].append((c, key))
    bad = set()
    for cs in groups.values():
        suppliers, parents = set(), set()
        for c, key in cs:
            if c["relation"] == "SUPPLIES_TO":
                suppliers.add(key(c["subject_mention"]))
            elif c["relation"] == "CUSTOMER_OF":
                suppliers.add(key(c["object_mention"]))
            else:
                parents.add(key(c["object_mention"]))
        if len(suppliers) > 1:
            bad |= {c["claim_id"] for c, _ in cs if c["relation"] != "SUBSIDIARY_OF"}
        if len(parents) > 1:
            bad |= {c["claim_id"] for c, _ in cs if c["relation"] == "SUBSIDIARY_OF"}
    return bad


class DocText:
    """Locates evidence in the full filing text (.v2) to give character offsets, like the Wikipedia claims."""
    def __init__(self, text_dir):
        self.text_dir, self.cache = Path(text_dir), {}

    def locate(self, document_id, evidence):
        if document_id not in self.cache:
            p = self.text_dir / f"{document_id}.v2.txt"
            self.cache[document_id] = [p.read_text(encoding="utf-8") if p.exists() else None, None]
        entry = self.cache[document_id]
        text = entry[0]
        if text is None:
            return None
        i = text.find(evidence)
        if i >= 0:
            return i, i + len(evidence), "exact", evidence
        if entry[1] is None:                                    # normalised copy of the filing, built once per document
            entry[1] = P1["validation"]._normalized_with_map(text)
        nh, nmap = entry[1]
        nn = P1["validation"]._normalized_with_map(evidence)[0].strip().rstrip(".")
        j = nh.find(nn) if len(nn) >= 10 else -1
        if j < 0:
            return None
        s, e = nmap[j], nmap[j + len(nn) - 1] + 1
        return s, e, "normalized", text[s:e]


def company_ids(path):
    """ticker -> company_id from the team registry (optional)."""
    import csv
    if not Path(path).exists():
        return {}
    with open(path, encoding="utf-8-sig", newline="") as f:
        return {r["ticker"].strip().upper(): r["company_id"].strip() for r in csv.DictReader(f)
                if r.get("ticker") and r.get("company_id")}


OFFERING_TYPES = ("Input", "Product", "Technology")
def retype_prompt():
    d = P1["ontology"].ENTITY_TYPE_DESCRIPTIONS
    return ("Classify each item that a semiconductor company provides or uses into exactly one type, using these "
            "definitions from the project ontology:\n"
            f"Input = {d['Input']}\nProduct = {d['Product']}\nTechnology = {d['Technology']}\n"
            "Use Input for generic goods/services and chip categories, Product only for named products, Technology only "
            "for named processes/technologies.\nThe items are numbered. Return ONLY a JSON object mapping each item NUMBER "
            "(as a string) to its type, e.g. {\"1\": \"Input\", \"2\": \"Product\"}.\nItems:\n")


def retype_offerings(claims, provenance, cache_path):
    """P5: one type per distinct PROVIDES/USES object, decided by the LLM in batches and cached (re-runs cost nothing)."""
    cache = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    todo = sorted({c["object_mention"] for c in claims if c["relation"] in ("PROVIDES", "USES")
                   and c["object_type"] in OFFERING_TYPES and c["object_mention"] not in cache})
    def classify(batch):
        """Ask for one batch; on a broken reply, split the batch and retry (down to single items)."""
        listing = "\n".join(f"{n}. {item}" for n, item in enumerate(batch, 1))
        try:
            text = call_llm(retype_prompt() + listing)
            reply = json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip()))
            if not isinstance(reply, dict):
                raise ValueError("reply is not an object")
        except DailyQuotaExhausted:
            raise
        except Exception as e:
            if len(batch) == 1:
                print(f"   could not type {batch[0][:60]!r} ({str(e)[:60]}); it keeps its type")
                return
            half = len(batch) // 2
            classify(batch[:half])
            classify(batch[half:])
            return
        for n, item in enumerate(batch, 1):
            t = reply.get(str(n))
            if t in OFFERING_TYPES:
                cache[item] = t
        cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=0), encoding="utf-8")
        time.sleep(DELAY_SECONDS)

    try:
        for i in range(0, len(todo), 80):
            classify(todo[i:i + 80])
    except DailyQuotaExhausted:
        print("   retyping stopped: daily quota used up (re-run later; finished batches are cached)")
    prov = {p["claim_id"]: p for p in provenance}
    n = 0
    for c in claims:
        t = cache.get(c["object_mention"])
        if c["relation"] in ("PROVIDES", "USES") and t in OFFERING_TYPES and t != c["object_type"]:
            prov[c["claim_id"]].setdefault("team_rule_changes", []).append({"field": "object_type", "from": c["object_type"], "to": t})
            c["object_type"] = t
            n += 1
    return n


SECTION_LABEL = {"1": "Item 1. Business", "4": "Item 4. Information on the Company", "kw": "Keyword fallback (non-standard layout)"}
HQ_STRONG = re.compile(r"(?i)headquarter|principal (?:executive )?offices?|head office|corporate offices?")


def one_headquarters(claims, provenance, filer_of):
    """Each filer keeps exactly one LOCATED_IN country (the most frequent, 'headquarters' wording first); others -> OPERATES_IN."""
    prov = {p["claim_id"]: p for p in provenance}
    by_filer = collections.defaultdict(list)
    for c in claims:
        f = filer_of.get(c["document_id"])
        if c["relation"] == "LOCATED_IN" and f and alnum(c["subject_mention"]) == alnum(f):
            by_filer[f].append(c)
    n = 0
    for f, cs in by_filer.items():
        score = collections.Counter()
        for c in cs:
            score[c["object_mention"]] += 10 if HQ_STRONG.search(c["evidence_span"]) else 1
        hq = score.most_common(1)[0][0]
        for c in cs:
            if c["object_mention"] != hq:
                prov[c["claim_id"]].setdefault("team_rule_changes", []).append({"field": "relation", "from": "LOCATED_IN", "to": "OPERATES_IN"})
                c["relation"] = "OPERATES_IN"
                n += 1
    return n


def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def is_good(rec):
    """A saved answer is reusable only if the call succeeded AND the reply is valid JSON; anything else is retried."""
    if not rec or "error" in rec or "response_text" not in rec:
        return False
    try:
        parse_claims_json(rec["response_text"])
        return True
    except Exception:
        return False


def main():
    if TEAM_RULES:
        load_person1_code()
    if MODEL.startswith("PASTE"):
        raise SystemExit("Set MODEL at the top of this file to a model name from your Gemini model list.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = read_jsonl(CHUNKS_PATH)
    docs = {d["document_id"]: d for d in read_jsonl(DOCS_PATH)}
    raw_path = OUT_DIR / "sec_llm_raw.jsonl"
    raw = {(r["chunk_id"], r["prompt_version"], r["model"]): r for r in read_jsonl(raw_path)}

    todo = [c for c in chunks
            if (not ONLY_TICKERS or c["ticker"] in ONLY_TICKERS)
            and (not ONLY_TEXT_CONTAINS or any(s.lower() in c["text"].lower() for s in ONLY_TEXT_CONTAINS))]
    print(f"{len(todo)} chunks selected, {sum(is_good(raw.get((c['chunk_id'], PROMPT_VERSION, MODEL))) for c in todo)} already have a usable saved response")

    # (A) call the LLM where no saved response exists
    for c in todo:
        key = (c["chunk_id"], PROMPT_VERSION, MODEL)
        if is_good(raw.get(key)):                                 # failed or unparseable answers are retried
            continue
        print(f"calling LLM for {c['chunk_id']} ...")
        rec = {"chunk_id": c["chunk_id"], "prompt_version": PROMPT_VERSION, "model": MODEL}
        try:
            rec["response_text"] = call_llm(build_prompt(c))
        except DailyQuotaExhausted:
            print("\nDAILY QUOTA USED UP: stopping API calls. Everything finished so far is saved;"
                  "\nre-run this step after the quota resets and it continues from here.")
            break
        except Exception as e:
            rec["error"] = str(e)[:500]
            print(f"   FAILED: {rec['error']}")
        raw[key] = rec
        write_jsonl(raw_path, list(raw.values()))                 # save after every call
        time.sleep(DELAY_SECONDS)

    # (B) rebuild claim files from saved responses
    chunk_by_id = {c["chunk_id"]: c for c in chunks}
    claims, rejected, provenance = [], [], []
    dropped = capped = duplicates = 0
    seen_in_doc = set()
    for (chunk_id, pv, model), rec in raw.items():
        if pv != PROMPT_VERSION or model != MODEL or chunk_id not in chunk_by_id or "response_text" not in rec:
            continue
        ch = chunk_by_id[chunk_id]
        doc = docs.get(ch["document_id"], {})
        try:
            items = parse_claims_json(rec["response_text"])
        except Exception as e:
            rejected.append({"chunk_id": chunk_id, "claim": None, "reasons": [f"unparseable_json:{e}"]})
            continue
        seen, kept_per_pair = set(), collections.Counter()
        for idx, c in enumerate(items, 1):                        # idx = position in the model's reply: IDs never shift when rules change
            c = normalize_labels(c)
            if isinstance(c, dict) and c.get("relation") in DROP_RELATIONS:
                dropped += 1
                continue
            reasons, warnings = validate_claim(c, ch["text"], ch["company"])
            if reasons:
                rejected.append({"chunk_id": chunk_id, "claim": c, "reasons": reasons})
                continue
            rewritten = []
            for k in ("subject_mention", "object_mention"):              # copy names as written in the sentence (team contract)
                if alnum(c[k]) != alnum(ch["company"]) and c[k.replace("mention", "type")] in ("Company", "Facility"):
                    new = surface_form(c[k], c["evidence_span"])
                    if new != c[k]:
                        rewritten.append({"field": k, "from": c[k], "to": new})
                        c = dict(c, **{k: new})
            team_changes = []
            if TEAM_RULES:                                                # Person 1's shared validation (same rules as Wikipedia)
                before = dict(c)
                c, team_reasons, team_warnings = team_validate(c, ch["text"], ch["company"], ch["ticker"])
                if team_reasons:
                    rejected.append({"chunk_id": chunk_id, "claim": c, "reasons": [f"team:{r}" for r in team_reasons]})
                    continue
                warnings = warnings + [f"team:{w}" for w in team_warnings]
                team_changes = [{"field": k, "from": before[k], "to": c[k]} for k in
                                ("subject_mention", "subject_type", "relation", "object_mention", "object_type", "evidence_span")
                                if before[k] != c[k]]
            context = table_row_header(c, ch["text"])
            c, normalized_from = canonicalize_direction(c)
            warnings = warnings + direction_warnings(c, ch["company"])
            if context and c["relation"] == "SUPPLIES_TO" and alnum(c["object_mention"]) == alnum(ch["company"]):
                c = dict(c, subject_mention=c["object_mention"], subject_type=c["object_type"],    # listed companies are CUSTOMERS
                         object_mention=c["subject_mention"], object_type=c["subject_type"])
                warnings = warnings + ["direction_fixed_by_table_header"]
            sig = (norm(c["subject_mention"]), c["relation"], norm(c["object_mention"]))
            if sig in seen:                                       # exact duplicate inside one chunk
                continue
            seen.add(sig)
            doc_sig = (ch["document_id"],) + sig + (alnum(c["evidence_span"]),)
            if doc_sig in seen_in_doc:                                   # same triple + same evidence elsewhere in the filing
                duplicates += 1
                continue
            seen_in_doc.add(doc_sig)
            pair = (c["relation"], c["object_type"])
            if pair in CAP_PER_CHUNK and kept_per_pair[pair] >= CAP_PER_CHUNK[pair]:
                capped += 1
                continue
            kept_per_pair[pair] += 1
            claim_id = f"claim_{chunk_id}_{idx:02d}"              # stable and unique across sources (contains 'sec_')
            evidence = c["evidence_span"].strip()
            claims.append({
                "claim_id": claim_id,
                "subject_mention": c["subject_mention"].strip(), "subject_type": c["subject_type"],
                "relation": c["relation"],
                "object_mention": c["object_mention"].strip(), "object_type": c["object_type"],
                "evidence_span": evidence,
                "source": "SEC", "document_id": ch["document_id"], "publication_date": ch["filing_date"],
                "extractor_version": EXTRACTOR_VERSION, "prompt_version": PROMPT_VERSION,
                # optional Contract A fields, as on the Wikipedia claims
                "source_url": doc.get("source_url"), "retrieved_at": doc.get("retrieved_at"),
                "section": SECTION_LABEL.get(ch["item"], f"Item {ch['item']}"), "chunk_id": chunk_id, "llm_model": model,
            })
            provenance.append({
                "claim_id": claim_id, "chunk_id": chunk_id, "source_type": "SEC", "document_id": ch["document_id"],
                "source_url": doc.get("source_url"), "publication_date": ch["filing_date"],
                "retrieved_at": doc.get("retrieved_at"), "evidence_span": evidence,
                "extractor_version": EXTRACTOR_VERSION, "prompt_version": PROMPT_VERSION,
                "model": model, "warnings": warnings, "normalized_from": normalized_from, "evidence_context": context,
                "mentions_rewritten": rewritten, "team_rule_changes": team_changes,
            })
    retyped = hq_fixed = n_contra = 0
    if TEAM_RULES:
        filer_of = {d["document_id"]: d.get("company") for d in docs.values()}
        names_of = {d["document_id"]: [d.get("company") or "", d.get("ticker") or ""] for d in docs.values()}
        bad = contradictory_ids(claims, names_of)
        for c in [c for c in claims if c["claim_id"] in bad]:
            rejected.append({"chunk_id": c["chunk_id"], "claim": c, "reasons": ["team:contradictory_pair"]})
        claims = [c for c in claims if c["claim_id"] not in bad]
        provenance = [p for p in provenance if p["claim_id"] not in bad]
        n_contra = len(bad)
        hq_fixed = one_headquarters(claims, provenance, filer_of)
        if RETYPE_OFFERINGS:
            retyped = retype_offerings(claims, provenance, OUT_DIR / "sec_object_types.json")
        # character offsets into the full filing text, like the Wikipedia claims
        texts, located = DocText(TEXT_DIR), 0
        for c in claims:
            hit = texts.locate(c["document_id"], c["evidence_span"])
            if hit:
                c["evidence_char_start"], c["evidence_char_end"], c["evidence_match"], c["evidence_span"] = hit
                located += 1
        ids = company_ids(COMPANIES_CSV)
        tick = {d["document_id"]: (d.get("ticker") or "").upper() for d in docs.values()}
        for c in claims:
            if ids.get(tick.get(c["document_id"])):
                c["company_id"] = ids[tick[c["document_id"]]]
        invalid = [c["claim_id"] for c in claims if not _contract_ok(c)]
        print(f"team rules: {n_contra} contradictory claims dropped | {hq_fixed} extra 'headquarters' -> OPERATES_IN | "
              f"{retyped} offerings re-typed | offsets for {located}/{len(claims)} claims | contract violations: {len(invalid)}")
    write_jsonl(OUT_DIR / "sec_claims.jsonl", claims)
    write_jsonl(OUT_DIR / "sec_claims_rejected.jsonl", rejected)
    write_jsonl(OUT_DIR / "sec_claims_provenance.jsonl", provenance)
    n_bad = sum(1 for r in raw.values() if "error" in r)
    print(f"\nclaims kept: {len(claims)} | rejected: {len(rejected)} | dropped by DROP_RELATIONS: {dropped} | capped by CAP_PER_CHUNK: {capped} | duplicates removed: {duplicates} | LLM call errors: {n_bad}")
    if claims or rejected:
        print(f"rejection rate: {len(rejected) / (len(claims) + len(rejected)):.0%}")


def _contract_ok(c):
    try:
        P1["schemas"].RawClaim(**c)
        return True
    except Exception:
        return False

if __name__ == "__main__":
    main()
