"""Stage C: canonicalize offering mentions (Input / Product / Technology) into product families.

EDC-style, adapted to entities (Zhang & Soh 2024):
  1. DEFINE + CANONICALIZE: the LLM reads each mention with its company and evidence sentence and names the generic
     product family it belongs to ("TWINSCAN NXT:2100i" -> "DUV immersion lithography systems"), picks one of a
     closed list of broad categories, and writes a one-line definition. Families already created are retrieved by
     embedding similarity and shown in the prompt, so the model reuses a fitting family instead of inventing a synonym.
  2. MERGE: near-duplicate family names inside the same category (embedding similarity) are checked by the LLM in
     small groups and merged under one name.

LEAKAGE GUARD: this stage never reads ETO files. The families are our own vocabulary; ETO is only used later, at
evaluation time, and is compared semantically (evaluate_eto.py).

Output (data/kg/): offering_canon.jsonl  one row per (company, mention)
                   offering_families.json family -> category, entity_type, definition, mentions
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

import kg_common as K

PROMPT_VERSION = "canon-v2"
REFINE_PROMPT_VERSION = "refine-v2"
VERIFY_PROMPT_VERSION = "verify-v1"
MERGE_PROMPT_VERSION = "merge-v1"
BATCH = 30
RETRIEVE_PER_ITEM = 6
MAX_CANDIDATES = 70
MERGE_SIM = {"st": 0.80, "tfidf": 0.70}    # cosine threshold for proposing a merge (checked by the LLM)
ONLY_COMPANIES: set = set()               # e.g. {"company_008", "company_011", "company_012", "company_014"} for a pilot

# Broad categories (closed list) -> entity type in the team ontology. Our own, ETO-independent vocabulary.
# Person 1's ontology: Input = goods/services feeding the chip supply chain INCLUDING chip categories (GPUs, DRAM ...);
# Product = a specific named product line; Technology = process / architecture. Families are generic, so supply-chain
# families are Inputs; downstream systems, software and services (outside the chip supply chain) are typed Product.
CATEGORIES = {
    "Lithography equipment": "Input",
    "Deposition equipment": "Input",
    "Etch and clean equipment": "Input",
    "Process control, metrology and inspection equipment": "Input",
    "Ion implantation equipment": "Input",
    "CMP and planarization equipment": "Input",
    "Thermal processing equipment": "Input",
    "Wafer and crystal production equipment": "Input",
    "Assembly and packaging equipment": "Input",
    "Test equipment and test consumables": "Input",
    "Fab automation, handling and equipment subsystems": "Input",
    "Wafers and substrates": "Input",
    "Photomasks and mask blanks": "Input",
    "Lithography materials": "Input",
    "Process chemicals and gases": "Input",
    "Packaging materials": "Input",
    "Other semiconductor materials": "Input",
    "EDA software": "Input",
    "Semiconductor IP": "Input",
    "Chip design services": "Input",
    "Foundry manufacturing services": "Input",
    "Assembly, packaging and test services": "Input",
    "Semiconductor process technologies": "Technology",
    "Packaging and integration technologies": "Technology",
    "Processors and accelerators": "Input",
    "Memory and storage chips": "Input",
    "Analog, mixed-signal and power chips": "Input",
    "RF, connectivity and networking chips": "Input",
    "Sensors and optoelectronic devices": "Input",
    "Discrete and passive components": "Input",
    "Computing systems and servers": "Product",
    "Networking and communications equipment": "Product",
    "End-user devices and vehicles": "Product",
    "Software and platforms": "Product",
    "Services and support": "Product",
    "Other technologies": "Technology",
    "Other offerings": "Product",
}

PROMPT = """You normalise product mentions for a semiconductor supply-chain knowledge graph.
For every item, name the PRODUCT FAMILY the mention belongs to: the generic kind of thing a supply-chain analyst
would list, so that equivalent offerings of different companies get the same family name.

Rules for "family":
- Generic English name, plural noun phrase. No company names, brands, product-line names, model numbers,
  generations, versions or node numbers ("TWINSCAN NXT:2100i" -> "DUV immersion lithography systems";
  "GeForce RTX 50 Series" -> "discrete GPUs"; "N3E" -> "FinFET logic process technologies").
- A family is ONE LEVEL MORE SPECIFIC than its category and must NOT simply repeat the category name. Name the
  process, function or device type: e.g. in "Deposition equipment" -> "ALD systems", "CVD systems", "PVD systems",
  "electroplating systems"; in "Memory and storage chips" -> "DRAM", "NAND flash", "HBM". Only a vague mention
  (see below) may use the category name itself.
- Keep distinctions that matter technically (EUV vs DUV lithography, DRAM vs NAND vs HBM, CPU vs GPU vs FPGA,
  deposition method, what is etched or cleaned, wet vs dry processing). Do not split by performance tier or market.
- Many mentions are synonyms or variants: "EUV", "EUV machines", "EUV systems", "High-NA EUV" -> one family
  "EUV lithography systems".
- REUSE a family from EXISTING FAMILIES (copy it exactly) whenever it denotes the same kind of thing at the same
  level of detail. Create a new family only when none fits.
- Brand or code names: identify what the product is from the evidence; if the evidence does not say, you may use
  general knowledge but then set "grounding": "knowledge" (otherwise "text").
- Vague mentions ("products", "systems", "services", "hardware", "solutions"): use the company and evidence to choose
  the broadest fitting family (often the category name itself) and set "specificity": "vague".
- If the mention is not an offering at all (a market, a business unit, an activity like "research"), set "family": null.

"category": exactly one of: {categories}
"definition": one short sentence defining the family (only when you create a NEW family; else "").

Return a JSON list, one object per item, in the same order:
[{{"id": <item id>, "family": "...", "category": "...", "definition": "...", "grounding": "text|knowledge",
   "specificity": "specific|generic|vague"}}]

EXISTING FAMILIES (name | category):
{existing}

ITEMS:
{items}
"""

REFINE_PROMPT = """You normalise product mentions for a semiconductor supply-chain knowledge graph. Each item below needs a
PRODUCT FAMILY: the generic kind of thing (plural noun phrase; no brands, model numbers, generations or node numbers),
ONE LEVEL MORE SPECIFIC than its broad category: the process, function or device type.
- Use the evidence; you may use general knowledge for brand names but then set "grounding": "knowledge".
- Reuse a family from EXISTING FAMILIES only if the mention is truly that kind of thing. A wrong reuse is worse than a
  new family: if nothing fits, create a precise new one.
- If the mention names nothing more specific than its category ("equipment", "products", "deposition"), use the
  category name itself and set "specificity": "vague".
- Manufacturing activities a company performs or buys ARE offerings (services): e.g. "wafer fabrication" -> "foundry
  wafer fabrication services", "assembly"/"packaging" -> "chip assembly and packaging services", "testing" -> "chip
  testing services". Return "family": null ONLY for things that are not a good, service or technology at all
  (markets, customer groups, business-unit or company names, patents, royalties, revenue).
"category": exactly one of: {categories}

Return one JSON object: {{"results": [{{"id": <item id>, "family": "..." or null, "category": "...", "definition": "...",
"grounding": "text|knowledge", "specificity": "specific|vague"}}]}}

EXISTING FAMILIES (name | category):
{existing}

ITEMS (id | company | mention | current category | evidence):
{items}
"""

VERIFY_PROMPT = """You check product families in a semiconductor supply-chain knowledge graph. Each FAMILY lists the mentions
(product names as written in company documents) that were assigned to it. Find mentions that do NOT belong: the
mention is not that kind of thing (e.g. "mainframe operations software" under "weather forecasting services",
"base stations" under "active electrical cables"). Instances, models, brands and synonyms of the family DO belong.
For each misfit give the correct family (plural noun phrase, generic, no brands) and its category.
"category": exactly one of: {categories}

Return one JSON object: {{"results": [{{"family_id": <id>, "misfits": [{{"mention_id": <id>, "family": "...",
"category": "..."}}]}}]}}  (use an empty misfits list when all mentions fit; list every family_id)

FAMILIES:
{families}
"""

MERGE_PROMPT = """You are cleaning the product-family vocabulary of a semiconductor supply-chain knowledge graph.
Each GROUP below lists family names that are lexically similar. Within each group, decide which names denote the SAME
kind of thing at the SAME level of detail (true synonyms or spelling variants). Do not merge a general family with a
more specific one (e.g. "lithography systems" and "EUV lithography systems" stay separate), and do not merge different
technologies (e.g. "ALD systems" and "CVD systems" stay separate).

Return JSON: [{{"group": <group id>, "merge": [["canonical name", "synonym 1", "synonym 2"], ...]}}]
List only sets of 2 or more names that should be merged; use [] when nothing merges. The canonical name must be one of
the names in the set (prefer the clearest generic one).

GROUPS:
{groups}
"""


def snippet(evidence, mention, width=220):
    ev = " ".join(evidence.split())
    i = ev.lower().find(mention.lower()[:20])
    if i < 0 or len(ev) <= width:
        return ev[:width]
    start = max(0, i - width // 3)
    return ("…" if start else "") + ev[start:start + width] + ("…" if start + width < len(ev) else "")


def collect_items(claims, resolver, doc_company):
    """One item per (company, mention key). Company = the subject company for PROVIDES/USES, else the document company."""
    items = {}
    for c in claims:
        dc = c.get("company_id") or doc_company.get(c["document_id"])
        for side in ("object", "subject"):
            if c[f"{side}_type"] not in K.OFFERING_TYPES:
                continue
            if side == "object" and c["subject_type"] == "Company":
                comp = resolver.resolve(c["subject_mention"], dc)[0] or dc or "unknown"
            else:
                comp = dc or "unknown"
            m = c[f"{side}_mention"].strip()
            key = (comp, K.loose_key(m))
            it = items.setdefault(key, {"company_id": comp, "mention": m, "types": Counter(), "evidence": c["evidence_span"],
                                        "n_claims": 0, "sources": set()})
            it["types"][c[f"{side}_type"]] += 1
            it["n_claims"] += 1
            it["sources"].add(c["source"])
    return items


def run(claims, resolver, doc_company, embedder=None):
    K.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = K.LLMCache("canonicalize")
    embedder = embedder or K.Embedder()
    st = not embedder.backend.startswith("tfidf")
    items = collect_items(claims, resolver, doc_company)
    keys = sorted(items, key=lambda k: (k[0], k[1]))
    if ONLY_COMPANIES:
        keys = [k for k in keys if k[0] in ONLY_COMPANIES]
    comp_name = {cid: e["canonical_name"] for cid, e in resolver.entities.items()}
    if not st:
        embedder.fit_vocabulary([items[k]["mention"] for k in keys] + list(CATEGORIES))

    families = {}                     # name -> {category, definition}
    fam_vecs = {}
    assign = {}
    cat_list = "; ".join(CATEGORIES)
    n_batches = (len(keys) + BATCH - 1) // BATCH
    for b in range(n_batches):
        batch = keys[b * BATCH:(b + 1) * BATCH]
        # retrieve existing families similar to the mentions in this batch
        cands = []
        if families:
            import numpy as np
            names = list(families)
            missing = [n for n in names if n not in fam_vecs]
            if missing:
                for n, v in zip(missing, embedder.encode([f"{n} ({families[n]['category']})" for n in missing])):
                    fam_vecs[n] = v
            F = np.vstack([fam_vecs[n] for n in names])
            Q = embedder.encode([items[k]["mention"] for k in batch])
            sims = Q @ F.T
            seen = set()
            for row in sims:
                for j in np.argsort(-row)[:RETRIEVE_PER_ITEM]:
                    if names[j] not in seen:
                        seen.add(names[j])
                        cands.append((row[j], names[j]))
            cands = [n for _, n in sorted(cands, reverse=True)[:MAX_CANDIDATES]]
        existing = "\n".join(f"- {n} | {families[n]['category']}" for n in sorted(cands)) or "(none yet)"
        lines = []
        for i, k in enumerate(batch):
            it = items[k]
            lines.append(json.dumps({"id": i, "company": comp_name.get(it["company_id"], it["company_id"]),
                                     "mention": it["mention"], "evidence": snippet(it["evidence"], it["mention"])},
                                    ensure_ascii=False))
        prompt = PROMPT.format(categories=cat_list, existing=existing, items="\n".join(lines))
        want = set(range(len(batch)))
        ok = lambda rep: bool(K.json_list(rep)) and want <= {int(x["id"]) for x in K.json_list(rep)
                                                              if isinstance(x, dict) and str(x.get("id", "")).isdigit()}
        try:
            reply = cache.call("canonicalize", PROMPT_VERSION, prompt, validate=ok)
        except K.DailyQuotaExhausted as e:
            print(f"daily quota exhausted after batch {b}/{n_batches}; re-run tomorrow (cached batches are kept). {e}")
            break
        out = K.json_list(reply)
        if not isinstance(out, list):
            print(f"  batch {b}: unparseable reply, items left unassigned")
            continue
        for row in out:
            try:
                k = batch[int(row["id"])]
            except Exception:
                continue
            fam = row.get("family")
            cat = row.get("category") if row.get("category") in CATEGORIES else "Other offerings"
            if fam:
                fam = re.sub(r"\s+", " ", fam).strip().rstrip(".")
                first = fam.split()[0]
                if not (first.isupper() or any(ch.isdigit() for ch in first)):   # keep acronyms like "EUV", "DRAM"
                    fam = fam[0].lower() + fam[1:]
                if fam not in families:
                    families[fam] = {"category": cat, "definition": (row.get("definition") or "").strip()}
            assign[k] = {"family": fam, "category": families[fam]["category"] if fam else cat,
                         "grounding": _norm(row.get("grounding"), ("text", "knowledge"), "text"),
                         "specificity": _norm(row.get("specificity"), ("specific", "generic", "vague"), "specific")}
        if (b + 1) % 10 == 0:
            print(f"  batch {b + 1}/{n_batches}: {len(families)} families so far")

    refine_category_families(items, assign, families, cache, comp_name)
    verify_families(items, assign, families, cache, comp_name)
    rename = merge_families(families, embedder, cache, st)
    for k, a in assign.items():
        while a["family"] in rename:
            a["family"] = rename[a["family"]]
    for old in list(rename):
        families.pop(old, None)

    rows, fam_mentions = [], defaultdict(Counter)
    for k in keys:
        it, a = items[k], assign.get(k)
        fam = a["family"] if a else None
        row = {"company_id": it["company_id"], "mention": it["mention"], "key": k[1],
               "family": fam, "category": a["category"] if a else None,
               "entity_type": CATEGORIES.get(a["category"], "Product") if a and fam else None,
               "grounding": a["grounding"] if a else None, "specificity": a["specificity"] if a else None,
               "n_claims": it["n_claims"], "sources": sorted(it["sources"]),
               "status": "assigned" if fam else ("not_an_offering" if a else "unassigned")}
        rows.append(row)
        if fam:
            fam_mentions[fam][it["mention"]] += it["n_claims"]
    fam_out = {f: {"category": d["category"], "entity_type": CATEGORIES.get(d["category"], "Product"),
                   "definition": d["definition"], "n_mentions": len(fam_mentions[f]),
                   "mentions": [m for m, _ in fam_mentions[f].most_common()]}
               for f, d in sorted(families.items()) if fam_mentions[f]}
    K.write_jsonl(K.KG_DIR / "offering_canon.jsonl", rows)
    K.write_json(K.KG_DIR / "offering_families.json", fam_out)
    stats = {"items": len(keys), "assigned": sum(r["status"] == "assigned" for r in rows),
             "not_an_offering": sum(r["status"] == "not_an_offering" for r in rows),
             "unassigned": sum(r["status"] == "unassigned" for r in rows), "families": len(fam_out),
             "merged_family_names": len(rename), "embedding_backend": embedder.backend,
             "llm_calls_made": K.LLM_CALLS["made"], "llm_calls_cached": K.LLM_CALLS["cached"],
             "grounding": dict(Counter(r["grounding"] for r in rows if r["grounding"])),
             "specificity": dict(Counter(r["specificity"] for r in rows if r["specificity"]))}
    K.write_json(K.KG_DIR / "offering_canon_stats.json", stats)
    return rows, fam_out, stats


def _is_category_name(family, category):
    k = lambda x: re.sub(r"[^a-z0-9]", "", (x or "").lower()).rstrip("s")
    return k(family) == k(category) or k(family) in {k(c) for c in CATEGORIES}


def refine_category_families(items, assign, families, cache, comp_name, batch=25):
    """Second pass for (a) mentions whose family only repeats a category name and (b) mentions judged 'not an
    offering' (re-checked with a narrower rule). Only families of the same categories are offered for reuse."""
    todo = [k for k, a in assign.items() if (a["family"] and a["specificity"] != "vague"
                                             and _is_category_name(a["family"], a["category"])) or not a["family"]]
    if not todo:
        return 0
    print(f"  refining {len(todo)} mentions (category-name families or 'not an offering')")
    fixed = 0
    cat_list = "; ".join(CATEGORIES)
    for b in range(0, len(todo), batch):
        part = todo[b:b + batch]
        cats = {assign[k]["category"] for k in part}
        existing = "\n".join(f"- {f} | {d['category']}" for f, d in sorted(families.items())
                             if d["category"] in cats and not _is_category_name(f, d["category"]))[:10000] or "(none)"
        lines = "\n".join(f"{i} | {comp_name.get(items[k]['company_id'], items[k]['company_id'])} | {items[k]['mention']} | "
                          f"{assign[k]['category'] or '-'} | {snippet(items[k]['evidence'], items[k]['mention'], 180)}"
                          for i, k in enumerate(part))
        want = set(range(len(part)))
        ok = lambda rep: bool(K.json_list(rep)) and want <= {int(x["id"]) for x in K.json_list(rep)
                                                              if isinstance(x, dict) and str(x.get("id", "")).isdigit()}
        try:
            reply = cache.call("refine", REFINE_PROMPT_VERSION,
                               REFINE_PROMPT.format(categories=cat_list, existing=existing, items=lines), validate=ok)
        except K.DailyQuotaExhausted as e:
            print(f"daily quota exhausted during refinement; re-run tomorrow. {e}")
            break
        for row in K.json_list(reply) or []:
            try:
                k = part[int(row["id"])]
            except Exception:
                continue
            fam = _clean_family(row.get("family"))
            cat = row.get("category") if row.get("category") in CATEGORIES else (assign[k]["category"] or "Other offerings")
            if not fam:
                assign[k].update({"family": None})
                continue
            if fam not in families:
                families[fam] = {"category": cat, "definition": (row.get("definition") or "").strip()}
            fixed += fam != assign[k]["family"]
            assign[k].update({"family": fam, "category": families[fam]["category"],
                              "grounding": _norm(row.get("grounding"), ("text", "knowledge"), assign[k]["grounding"]),
                              "specificity": _norm(row.get("specificity"), ("specific", "generic", "vague"), "specific")})
    print(f"  refined {fixed} mentions")
    return fixed


def _clean_family(fam):
    fam = re.sub(r"\s+", " ", fam or "").strip().rstrip(".")
    if not fam or fam.lower() in ("null", "none"):
        return None
    first = fam.split()[0]
    if not (first.isupper() or any(ch.isdigit() for ch in first)):
        fam = fam[0].lower() + fam[1:]
    return fam


def _norm(v, allowed, default):
    v = (v or "").strip().lower()
    return v if v in allowed else default


def verify_families(items, assign, families, cache, comp_name, per_call=8, max_mentions=40):
    """Self-check (EDC-style verification): the LLM sees each family with its mentions and moves misfits out."""
    members = defaultdict(list)
    for k, a in assign.items():
        if a["family"]:
            members[a["family"]].append(k)
    fams = sorted(f for f, ks in members.items() if len({items[k]["mention"].lower() for k in ks}) >= 2)
    chunks = []
    for f in fams:                                         # big families are checked in slices
        ks = members[f]
        for s0 in range(0, len(ks), max_mentions):
            chunks.append((f, ks[s0:s0 + max_mentions]))
    print(f"  verifying {len(fams)} families ({len(chunks)} slices)")
    moved = 0
    cat_list = "; ".join(CATEGORIES)
    for b in range(0, len(chunks), per_call):
        part = chunks[b:b + per_call]
        text = []
        for fi, (f, ks) in enumerate(part):
            lines = "\n".join(f"   {mi} | {comp_name.get(items[k]['company_id'], '')} | {items[k]['mention']} | "
                              f"{snippet(items[k]['evidence'], items[k]['mention'], 90)}" for mi, k in enumerate(ks))
            text.append(f"family_id {fi}: {f} ({families.get(f, {}).get('category', '')}) - "
                        f"{families.get(f, {}).get('definition', '')}\n{lines}")
        want = set(range(len(part)))
        ok = lambda rep: bool(K.json_list(rep)) and want <= {int(x["family_id"]) for x in K.json_list(rep)
                                                              if isinstance(x, dict) and str(x.get("family_id", "")).isdigit()}
        try:
            reply = cache.call("verify", VERIFY_PROMPT_VERSION,
                               VERIFY_PROMPT.format(categories=cat_list, families="\n\n".join(text)), validate=ok)
        except K.DailyQuotaExhausted as e:
            print(f"daily quota exhausted during verification; re-run tomorrow. {e}")
            break
        for row in K.json_list(reply) or []:
            try:
                f, ks = part[int(row["family_id"])]
            except Exception:
                continue
            for m in row.get("misfits") or []:
                try:
                    k = ks[int(m["mention_id"])]
                except Exception:
                    continue
                fam = _clean_family(m.get("family"))
                if not fam or fam == f:
                    continue
                cat = m.get("category") if m.get("category") in CATEGORIES else assign[k]["category"]
                if fam not in families:
                    families[fam] = {"category": cat, "definition": ""}
                assign[k].update({"family": fam, "category": families[fam]["category"]})
                moved += 1
    print(f"  moved {moved} misfit mentions")
    return moved


def merge_families(families, embedder, cache, st):
    """Propose merges by embedding similarity within a category; the LLM confirms. Returns {old_name: new_name}."""
    import numpy as np
    rename = {}
    # trivial variants first (case, plural s, hyphens)
    canon_key = {}
    for f in sorted(families, key=len):
        k = re.sub(r"[^a-z0-9]", "", f.lower()).rstrip("s")
        if k in canon_key and families[canon_key[k]]["category"] == families[f]["category"]:
            rename[f] = canon_key[k]
        else:
            canon_key[k] = f
    names = [f for f in families if f not in rename]
    if len(names) < 2:
        return rename
    thr = MERGE_SIM["st" if st else "tfidf"]
    V = embedder.encode(names)
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    S = V @ V.T
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if S[i, j] >= thr and families[names[i]]["category"] == families[names[j]]["category"]:
                parent[find(i)] = find(j)
    groups = defaultdict(list)
    for i in range(len(names)):
        groups[find(i)].append(names[i])
    groups = [sorted(g) for g in groups.values() if len(g) > 1]
    chunks = []
    for g in groups:                                   # keep groups small enough to judge
        for s in range(0, len(g), 12):
            if len(g[s:s + 12]) > 1:
                chunks.append(g[s:s + 12])
    for b in range(0, len(chunks), 15):
        part = chunks[b:b + 15]
        text = "\n".join(f"group {i}: " + json.dumps(g, ensure_ascii=False) for i, g in enumerate(part))
        try:
            out = K.parse_json(cache.call("merge", MERGE_PROMPT_VERSION, MERGE_PROMPT.format(groups=text)))
        except K.DailyQuotaExhausted as e:
            print(f"daily quota exhausted during merge; re-run tomorrow. {e}")
            break
        for row in out or []:
            try:
                g = part[int(row["group"])]
            except Exception:
                continue
            for s in row.get("merge") or []:
                s = [x for x in s if x in g]
                if len(s) >= 2:
                    for x in s[1:]:
                        if x != s[0]:
                            rename[x] = s[0]
    return rename


if __name__ == "__main__":
    reg, _ = K.load_registry()
    resolver = K.CompanyResolver(reg, K.KG_DIR / "company_id_ledger.json")
    wiki, sec, _ = K.load_claims()
    doc_company, _ = K.sec_doc_company(reg)
    rows, fams, stats = run(wiki + sec, resolver, doc_company)
    resolver.save_ledger()
    print(json.dumps(stats, indent=2))
