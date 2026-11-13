"""Shared configuration and helpers for Person 2's KG stages (resolution, canonicalization, fusion, evaluation).

Inputs  : Person 1's wiki_claims.jsonl and Person 2's sec_claims.jsonl (both Contract A), the company registry.
Outputs : Contract B/C files for Person 1's Neo4j loader, plus evaluation reports.

Everything that costs an LLM call is cached in data/kg/cache/, so re-running any stage is free unless its inputs change.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

# ============================== configuration ==============================
ROOT = Path(os.environ.get("KG_ROOT", Path(__file__).resolve().parent / "workdir"))   # set KG_ROOT to your working folder
PERSON1_CODE_DIR = ROOT / "person1export"            # Person 1's common/*.py + extraction/validation.py
SEC_CLAIMS = ROOT / "data/processed/sec_claims.jsonl"
SEC_DOCUMENTS = ROOT / "data/raw/sec/sec_documents.jsonl"
WIKI_CLAIMS_CANDIDATES = [ROOT / "data/person1/wiki_claims.jsonl", ROOT / "outputs/claims/wiki_claims.jsonl",
                          ROOT / "person1export/wiki_claims.jsonl"]
REGISTRY_CANDIDATES = [ROOT / "data/companies/companies.jsonl", ROOT / "data/companies/company_id_registry.json",
                       ROOT / "data/companies/companies.csv"]
ETO_DIR = ROOT / "data/eto"                          # inputs.csv, providers.csv, provision.csv, sequence.csv, stages.csv
KG_DIR = ROOT / "data/kg"                            # all outputs of these stages
CACHE_DIR = KG_DIR / "cache"
EVAL_DIR = KG_DIR / "eval"

SECRET_NAME = "GeminiAPIKey"
MODEL = "gemini-3.5-flash-lite"
DELAY_SECONDS = 5                                    # pause between LLM calls (free-tier requests/minute)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"   # used when sentence-transformers is installed (Colab)

OFFERING_TYPES = ("Input", "Product", "Technology")
PROVISIONAL_COMPANY_START = 1001                     # registry IDs are company_001..company_999


def wiki_claims_path() -> Path:
    for p in WIKI_CLAIMS_CANDIDATES:
        if p.exists():
            return p
    hits = sorted(ROOT.rglob("wiki_claims.jsonl"))
    if hits:
        return hits[0]
    raise SystemExit("wiki_claims.jsonl not found: put Person 1's file at data/person1/wiki_claims.jsonl")


# ============================== IO ==============================
def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(*parts, n=16):
    return hashlib.sha1("\x1f".join(map(str, parts)).encode("utf-8")).hexdigest()[:n]


# ============================== Person 1's shared code ==============================
P1 = {}


def load_person1_code():
    """Import Person 1's ontology / schemas / vocab (single source of truth for the contracts)."""
    if P1:
        return P1
    import importlib
    import shutil
    pkg = Path("/tmp/person1_pkg")
    home = {"ontology.py": "common", "schemas.py": "common", "text.py": "common", "utils.py": "common",
            "vocab.py": "common", "validation.py": "extraction"}
    for name, sub in home.items():
        hits = sorted(PERSON1_CODE_DIR.rglob(name), key=lambda p: (sub not in p.parts, len(p.parts)))
        if not hits:
            raise SystemExit(f"{name} missing from {PERSON1_CODE_DIR}")
        (pkg / sub).mkdir(parents=True, exist_ok=True)
        (pkg / sub / "__init__.py").touch()
        shutil.copy(hits[0], pkg / sub / name)
    if str(pkg) not in sys.path:
        sys.path.insert(0, str(pkg))
    for mod in ("common.ontology", "common.vocab", "common.schemas"):
        P1[mod.split(".")[-1]] = importlib.import_module(mod)
    return P1


# ============================== LLM (Gemini) with cache ==============================
class DailyQuotaExhausted(Exception):
    """The provider's per-day quota is used up: retrying today is pointless."""


_DAILY = re.compile(r"(?i)retry in \d+h|per ?day")
_client = None
LLM_CALLS = {"made": 0, "cached": 0}


def _gemini(prompt: str) -> str:
    global _client
    from google import genai
    from google.genai import types
    if _client is None:
        try:
            from google.colab import userdata
            key = userdata.get(SECRET_NAME)
        except Exception:
            key = os.environ.get("GEMINI_API_KEY")
        _client = genai.Client(api_key=key)
    for attempt in range(4):
        try:
            resp = _client.models.generate_content(
                model=MODEL, contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"))
            if resp.text is None:
                raise RuntimeError("empty response (possibly blocked)")
            return resp.text
        except Exception as e:
            msg = str(e)
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg) and _DAILY.search(msg):
                raise DailyQuotaExhausted(msg[:300]) from e
            if ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg) and attempt < 3:
                time.sleep(30 * (attempt + 1))
            else:
                raise


LLM_BACKEND = _gemini        # tests replace this with a stub

# Second provider for judging (different model family than the extractors/canonicalizer). OpenAI-compatible API.
JUDGE = {"provider": "groq", "model": "openai/gpt-oss-120b", "base_url": "https://api.groq.com/openai/v1",
         "secret": "GroqAPIKey", "delay": 3, "json_mode": True, "max_tokens": 8000, "reasoning_effort": "low"}
# Alternative with a larger free daily token budget (same model weights; check the model id in your Cerebras console):
# JUDGE.update(provider="cerebras", model="gpt-oss-120b", base_url="https://api.cerebras.ai/v1", secret="CerebrasAPIKey")
# Groq free tier for this model: ~200k tokens/day is the binding limit (a full ETO judge pass uses most of it).
_judge_client = None


def _openai_compatible(prompt: str) -> str:
    """Judge backend (Groq free tier by default: ~1,000 requests/day, 30/min, 8k tokens/min for gpt-oss-120b)."""
    global _judge_client
    from openai import OpenAI
    if _judge_client is None:
        try:
            from google.colab import userdata
            key = userdata.get(JUDGE["secret"])
        except Exception:
            key = os.environ.get(JUDGE["secret"].upper())
        _judge_client = OpenAI(api_key=key, base_url=JUDGE["base_url"])
    for attempt in range(6):
        try:
            kwargs = dict(model=JUDGE["model"], temperature=0, messages=[{"role": "user", "content": prompt}],
                          max_completion_tokens=JUDGE.get("max_tokens", 8000))
            if JUDGE.get("json_mode", True):
                kwargs["response_format"] = {"type": "json_object"}      # forces a parseable JSON object
            if JUDGE.get("reasoning_effort"):
                kwargs["reasoning_effort"] = JUDGE["reasoning_effort"]   # gpt-oss: far fewer (billed) reasoning tokens
            for _ in range(3):
                try:
                    r = _judge_client.chat.completions.create(**kwargs)
                    break
                except Exception as e:
                    m = str(e)
                    if "400" in m and "reasoning_effort" in m and "reasoning_effort" in kwargs:
                        kwargs.pop("reasoning_effort")           # provider/model without this option
                    elif "400" in m and ("response_format" in m or "json" in m.lower()) and "response_format" in kwargs:
                        kwargs.pop("response_format")            # provider/model without JSON mode
                    else:
                        raise
            if getattr(r, "usage", None):
                LLM_CALLS["judge_tokens"] = LLM_CALLS.get("judge_tokens", 0) + (r.usage.total_tokens or 0)
            return r.choices[0].message.content
        except Exception as e:
            msg = str(e)
            if "429" in msg and re.search(r"(?i)per day|RPD|daily", msg):
                raise DailyQuotaExhausted(msg[:300]) from e
            if ("429" in msg or "rate" in msg.lower() or "503" in msg) and attempt < 5:
                time.sleep(20 * (attempt + 1))         # tokens-per-minute limit: wait for the window to reset
            else:
                raise


JUDGE_BACKEND = _openai_compatible   # tests replace this with a stub


class LLMCache:
    """Append-only JSONL cache: key -> raw reply text. Keyed by task name, prompt version, model and the prompt itself."""

    def __init__(self, name):
        self.path = CACHE_DIR / f"{name}.jsonl"
        self.data = {}
        if self.path.exists():
            for r in read_jsonl(self.path):
                self.data[r["key"]] = r["reply"]

    def call(self, task, prompt_version, prompt, judge=False, validate=None, retries=1):
        """judge=True routes the call to the judge provider (JUDGE), else to Gemini (MODEL).
        validate(reply) -> bool: invalid replies are retried and NEVER cached (so a re-run asks again instead of
        replaying a broken answer). Returns the reply, or None if every attempt was invalid."""
        model = JUDGE["model"] if judge else MODEL
        key = sha(task, prompt_version, model, prompt, n=24)
        if key in self.data and (validate is None or validate(self.data[key])):
            LLM_CALLS["cached"] += 1
            return self.data[key]
        reply = None
        for _ in range(retries + 1):
            reply = (JUDGE_BACKEND if judge else LLM_BACKEND)(prompt)
            LLM_CALLS["made"] += 1
            if validate is None or validate(reply):
                break
            LLM_CALLS["invalid"] = LLM_CALLS.get("invalid", 0) + 1
        else:
            return None
        self.data[key] = reply
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "task": task, "prompt_version": prompt_version, "model": model,
                                "reply": reply}, ensure_ascii=False) + "\n")
        if judge and JUDGE_BACKEND is _openai_compatible:
            time.sleep(JUDGE["delay"])
        elif not judge and LLM_BACKEND is _gemini:
            time.sleep(DELAY_SECONDS)
        return reply


def json_list(text, key="results"):
    """Parse a reply that should hold a list: a bare list, or an object wrapping it (JSON mode returns objects)."""
    out = parse_json(text)
    if isinstance(out, dict):
        out = out.get(key) if isinstance(out.get(key), list) else next((v for v in out.values() if isinstance(v, list)), None)
    return out if isinstance(out, list) else None


def parse_json(text):
    """Parse a JSON reply; tolerate code fences and leading prose."""
    if text is None:
        return None
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        m = re.search(r"[\[{].*[\]}]", t, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                return None
    return None


# ============================== embeddings ==============================
class Embedder:
    """sentence-transformers when available (Colab); otherwise TF-IDF character n-grams (offline fallback).
    The backend name is written into every report so results are never mixed silently."""

    def __init__(self):
        self.backend = None
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(EMBED_MODEL)
            self.backend = EMBED_MODEL
        except Exception:
            self.model = None
            self.backend = "tfidf-char-3-5 (fallback)"
        self._tfidf = None

    def fit_vocabulary(self, texts):
        """Only the TF-IDF fallback needs a shared vocabulary; call once with every text you will embed."""
        if self.model is None:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=1)
            self._tfidf.fit([t.lower() for t in texts])

    def encode(self, texts):
        import numpy as np
        texts = list(texts)
        if not texts:
            return np.zeros((0, 1))
        if self.model is not None:
            return self.model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)
        if self._tfidf is None:
            self.fit_vocabulary(texts)
        from sklearn.preprocessing import normalize
        return normalize(self._tfidf.transform([t.lower() for t in texts])).toarray()


# ============================== name normalization ==============================
LEGAL_SUFFIX = re.compile(
    r"\b(incorporated|inc|corporation|corp|company|co|limited|ltd|llc|plc|n ?v|s ?a|ag|se|gmbh|kg|bv|b ?v|"
    r"holdings?|group|the|sa de cv|co ltd|kk|pte|pty|oyj|ab|asa|spa|s ?p ?a)\b")
DE_SUFFIX = re.compile(r"\s*/\s*[a-z]{2}\s*$")         # SEC style "APPLIED MATERIALS INC /DE"


def fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    for a, b in [("“", '"'), ("”", '"'), ("‘", "'"), ("’", "'"), ("–", "-"), ("—", "-"),
                 ("®", ""), ("™", ""), ("©", "")]:
        s = s.replace(a, b)
    return s


def norm_name(s: str) -> str:
    """Lower-case, no punctuation, no legal suffixes: 'Taiwan Semiconductor Manufacturing Co., Ltd.' -> 'taiwan semiconductor manufacturing'."""
    s = fold(s).lower()
    s = DE_SUFFIX.sub("", s)
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    prev = None
    while prev != s:
        prev = s
        s = re.sub(r"\s+", " ", LEGAL_SUFFIX.sub(" ", s)).strip()
    return s


def loose_key(s: str) -> str:
    """Whitespace/punctuation-insensitive key ('Glo Bal-Foundries' -> 'globalfoundries')."""
    return re.sub(r"[^a-z0-9]", "", fold(s).lower())


PAREN_ALIAS = re.compile(r"""^(?P<name>.+?)\s*[,(]\s*(?:or\s+|the\s+)?["“'‘]?(?P<abbr>[A-Za-z][\w&.\-]{1,15})["”'’]?\s*\)?\s*$""")


def split_paren_alias(mention: str):
    """'Taiwan Semiconductor Manufacturing Company Limited ("TSMC")' -> ('Taiwan Semiconductor ...', 'TSMC').
    Only splits when the bracketed token looks like an abbreviation of the name (initials or a name prefix)."""
    m = PAREN_ALIAS.match(fold(mention).strip())
    if not m:
        return mention, None
    name, abbr = m.group("name").strip(), m.group("abbr").strip(".")
    words = [w for w in re.findall(r"[A-Za-z0-9]+", name) if w.lower() not in {"and", "of", "the", "co", "inc", "ltd"}]
    initials = "".join(w[0] for w in words).upper()
    a = re.sub(r"[^A-Za-z0-9]", "", abbr).upper()
    if len(a) >= 2 and (a in initials or initials.startswith(a[:2]) or name.upper().startswith(a)
                        or a == "".join(w[0] for w in words[:len(a)]).upper()):
        return name, abbr
    return mention, None


# ============================== company resolution ==============================
# Former names and short forms that the registry does not list. Small, explicit and reviewable.
EXTRA_COMPANY_ALIASES = {
    "Hynix": "SK Hynix", "Hynix Semiconductor": "SK Hynix", "SK hynix": "SK Hynix",
    "Avago": "Broadcom", "Avago Technologies": "Broadcom", "Broadcom Limited": "Broadcom", "Broadcom Corporation": "Broadcom",
    "Samsung": "Samsung Electronics", "Samsung Electronics Co": "Samsung Electronics",
    "Micron": "Micron Technology", "TSMC": "TSMC", "Taiwan Semiconductor": "TSMC",
    "GF": "GlobalFoundries", "Global Foundries": "GlobalFoundries",
    "SMIC": "Semiconductor Manufacturing International Corporation",
    "UMC": "United Microelectronics Corporation",
    "ASE": "ASE Group", "Advanced Semiconductor Engineering": "ASE Group", "ASE Technology": "ASE Group",
    "Amkor": "Amkor Technology", "Amkor Technologies": "Amkor Technology",
    "Applied": "Applied Materials", "AMAT": "Applied Materials", "Lam": "Lam Research", "KLA-Tencor": "KLA Corporation",
    "ST": "STMicroelectronics", "Cadence": "Cadence Design Systems", "Nova": "Nova Ltd.", "Arm": "Arm Holdings",
    "ARM Limited": "Arm Holdings", "Arm Ltd": "Arm Holdings", "Marvell": "Marvell Technology",
    "Texas Instruments Incorporated": "Texas Instruments", "TI": "Texas Instruments", "ADI": "Analog Devices",
    "NXP": "NXP Semiconductors", "Microchip": "Microchip Technology", "Lattice": "Lattice Semiconductor",
    "onsemi": "ON Semiconductor", "Qorvo": "Qorvo", "Skyworks": "Skyworks Solutions", "Entegris": "Entegris",
    "Veeco": "Veeco", "Onto": "Onto Innovation", "Photronics": "Photronics", "FormFactor": "FormFactor, Inc.",
    "Axcelis": "Axcelis Technologies", "Kulicke & Soffa": "Kulicke and Soffa Industries", "K&S": "Kulicke and Soffa Industries",
    "Tower": "Tower Semiconductor", "Tower Jazz": "Tower Semiconductor", "TowerJazz": "Tower Semiconductor",
    "Shin-Etsu": "Shin-Etsu Chemical", "CXMT": "ChangXin Memory Technologies", "Coherent": "Coherent Corp.",
    # former names (same company, renamed)
    "Samsung Electric Industries": "Samsung Electronics", "Hyundai Electronics": "SK Hynix", "KLA Instruments": "KLA Corporation",
    "Fairchild Semiconductor International": "Fairchild Semiconductor",
    # short forms of companies outside the registry (become one provisional entity)
    "ATI": "ATI Technologies", "Zeiss": "Carl Zeiss", "Carl Zeiss AG": "Carl Zeiss",
}


GENERIC_COMPANY = re.compile(
    r"(?i)^(our |the |its |their |certain |various |third[- ]party |other |several |some |key |major |leading |large |"
    r"external |outside )*(customers?|suppliers?|subcontractors?|vendors?|foundr(y|ies)|partners?|manufacturers?|"
    r"distributors?|oems?|odms?|hyperscalers?|competitors?|licensees?|companies|firms|governments?|clients?|"
    r"contract manufacturers?|semiconductor companies|chipmakers?|osats?)$")


class CompanyResolver:
    """Deterministic company entity resolution.

    1. registry names / aliases / legal names / tickers (exact on normalized keys)
    2. explicit former names and short forms (EXTRA_COMPANY_ALIASES)
    3. 'Full Name (ABBR)' and 'Full Name ("ABBR")' patterns: resolve the name, learn the abbreviation
    4. conservative fuzzy match to registry names only (token-sort ratio >= 95)
    5. otherwise a provisional company entity per normalized name (never dropped)
    """

    def __init__(self, registry_rows, ledger_path=None):
        self.entities = {}                 # id -> {canonical_name, aliases:set, ticker, cik, provisional}
        self.key2id = {}
        self.ambiguous = set()
        self.ledger_path = Path(ledger_path) if ledger_path else None
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path and self.ledger_path.exists() else {}
        for r in registry_rows:
            cid = r.get("entity_id") or r.get("company_id")
            name = r.get("canonical_name") or r.get("name")
            if not cid or not name:
                continue
            al = set(r.get("aliases") or []) | {name}
            for k in ("legal_name",):
                if r.get(k):
                    al.add(r[k])
            self.entities[cid] = {"canonical_name": name, "aliases": set(al), "ticker": r.get("ticker"),
                                  "cik": r.get("cik") or r.get("cik_hint"), "provisional": False}
            for a in al:
                self._index(a, cid)
            if r.get("ticker"):
                self._index(r["ticker"], cid, ticker=True)
        by_name = {e["canonical_name"]: cid for cid, e in self.entities.items()}
        self.redirect = {}                 # alias key -> canonical name for companies outside the registry
        for alias, target in EXTRA_COMPANY_ALIASES.items():
            if target in by_name:
                self._index(alias, by_name[target])
            else:
                self.redirect[loose_key(alias)] = target
        self.registry_ids = set(self.entities)
        self._fuzzy_keys = [(k, v) for k, v in self.key2id.items() if v in self.registry_ids and len(k) >= 6]

    def _index(self, name, cid, ticker=False):
        keys = {loose_key(name)} if ticker else {norm_name(name), loose_key(name)}
        for k in keys:
            if not k:
                continue
            if k in self.key2id and self.key2id[k] != cid:
                self.ambiguous.add(k)
            else:
                self.key2id[k] = cid

    def _lookup(self, name):
        for k in (norm_name(name), loose_key(name)):
            if k and k in self.key2id and k not in self.ambiguous:
                return self.key2id[k]
        return None

    def resolve(self, mention, doc_company_id=None):
        """Return (entity_id, score, method)."""
        m = fold(mention).strip()
        if GENERIC_COMPANY.match(m):
            return None, 0.0, "generic_company_mention"
        if loose_key(m) in self.redirect:
            m = self.redirect[loose_key(m)]
        if m.lower() in {"we", "us", "our company", "the company", "company", "registrant"} and doc_company_id:
            return doc_company_id, 1.0, "self_reference"
        cid = self._lookup(m)
        if cid:
            if cid in self.registry_ids:
                return cid, 1.0, "registry_exact"
            self.entities[cid]["aliases"].add(m)
            return cid, 0.7, "provisional"
        name, abbr = split_paren_alias(m)
        if not abbr:                       # 'APPLIED MATERIALS, INC. (Santa Clara, CA)' -> drop a trailing bracket
            bare = re.sub(r"\s*\([^()]*\)\s*$", "", m).strip()
            if bare and bare != m:
                cid = self._lookup(bare)
                if cid and cid in self.registry_ids:
                    return cid, 0.95, "registry_bracket_removed"
        if abbr:
            cid = self._lookup(name) or self._lookup(abbr)
            if cid:
                self.entities[cid]["aliases"].add(abbr)
                self.entities[cid]["aliases"].add(name)
                if cid in self.registry_ids:
                    return cid, 0.95, "registry_abbreviation"
                return cid, 0.7, "provisional"
        from rapidfuzz import fuzz, process
        key = norm_name(name)
        if len(key) >= 6 and self._fuzzy_keys:
            hit = process.extractOne(key, [k for k, _ in self._fuzzy_keys], scorer=fuzz.token_sort_ratio)
            if hit and hit[1] >= 95:
                return dict(self._fuzzy_keys)[hit[0]], round(hit[1] / 100, 3), "registry_fuzzy"
        return self._provisional(name, abbr), 0.7, "provisional"

    def _provisional(self, name, abbr):
        key = norm_name(name) or loose_key(name)
        if key in self.key2id:
            cid = self.key2id[key]
        else:
            cid = self.ledger.get(f"company:{key}")
            if not cid:
                used = {int(v.split("_")[1]) for v in self.ledger.values() if v.startswith("company_")}
                n = max([PROVISIONAL_COMPANY_START - 1] + list(used)) + 1
                cid = f"company_{n}"
                self.ledger[f"company:{key}"] = cid
            self.entities[cid] = {"canonical_name": name, "aliases": {name}, "ticker": None, "cik": None, "provisional": True}
            self.key2id[key] = cid
            self.key2id.setdefault(loose_key(name), cid)
        self.entities[cid]["aliases"].add(name)
        if abbr:
            self.entities[cid]["aliases"].add(abbr)
        return cid

    def separate_legal_entity(self, mention):
        """A provisional company for a specific legal entity (keyed by its full spelling, not the normalised name)."""
        key = "legal:" + loose_key(mention)
        cid = self.ledger.get(f"company:{key}")
        if not cid:
            used = {int(v.split("_")[1]) for v in self.ledger.values() if v.startswith("company_")}
            cid = f"company_{max([PROVISIONAL_COMPANY_START - 1] + list(used)) + 1}"
            self.ledger[f"company:{key}"] = cid
        if cid not in self.entities:
            self.entities[cid] = {"canonical_name": fold(mention).strip(), "aliases": {mention}, "ticker": None,
                                  "cik": None, "provisional": True}
        return cid

    def save_ledger(self):
        if self.ledger_path:
            write_json(self.ledger_path, self.ledger)


def load_registry():
    for p in REGISTRY_CANDIDATES:
        if not p.exists():
            continue
        if p.suffix == ".jsonl":
            return read_jsonl(p), p
        if p.suffix == ".json":
            obj = json.loads(p.read_text())
            return list((obj.get("companies") or {}).values()), p
        if p.suffix == ".csv":
            import csv
            rows = list(csv.DictReader(open(p, encoding="utf-8")))
            for r in rows:
                r["aliases"] = [a.strip() for a in (r.get("aliases") or "").split("|") if a.strip()]
            return rows, p
    raise SystemExit("company registry not found (data/companies/companies.jsonl)")


# ============================== small ID ledger for non-company entities ==============================
class IdLedger:
    """Stable <prefix>_NNN IDs across re-runs: an existing key keeps its ID, new keys get the next number."""

    def __init__(self, path):
        self.path = Path(path)
        self.map = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, prefix, key):
        k = f"{prefix}:{key}"
        if k not in self.map:
            used = [int(v.rsplit("_", 1)[1]) for v in self.map.values() if v.startswith(prefix + "_")]
            self.map[k] = f"{prefix}_{(max(used) + 1 if used else 1):03d}"
        return self.map[k]

    def save(self):
        write_json(self.path, self.map)


# ============================== claims loading ==============================
def load_claims():
    """Both sources as Contract A dicts, each tagged with the registry company of its source document."""
    P = load_person1_code()
    RawClaim = P["schemas"].RawClaim
    wiki = read_jsonl(wiki_claims_path())
    sec = read_jsonl(SEC_CLAIMS)
    bad = []
    for c in wiki + sec:
        try:
            RawClaim(**c)
        except Exception as e:
            bad.append((c.get("claim_id"), str(e)[:120]))
    return wiki, sec, bad


def sec_doc_company(registry_rows):
    """SEC document_id -> registry company_id via ticker / CIK."""
    docs = read_jsonl(SEC_DOCUMENTS) if SEC_DOCUMENTS.exists() else []
    by_ticker = {(r.get("ticker") or "").upper(): (r.get("entity_id") or r.get("company_id")) for r in registry_rows if r.get("ticker")}
    by_cik = {str(r.get("cik") or r.get("cik_hint") or "").lstrip("0"): (r.get("entity_id") or r.get("company_id"))
              for r in registry_rows if (r.get("cik") or r.get("cik_hint"))}
    alias = {"SKHY": "000660.KS"}
    out = {}
    for d in docs:
        t = (d.get("ticker") or "").upper()
        cid = by_ticker.get(t) or by_ticker.get(alias.get(t, "")) or by_cik.get(str(d.get("cik") or "").lstrip("0"))
        if cid:
            out[d["document_id"]] = cid
    return out, docs
