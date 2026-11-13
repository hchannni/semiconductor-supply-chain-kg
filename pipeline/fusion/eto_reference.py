"""Evaluation reference: arrange the ETO Chip Explorer data (CC BY-NC 4.0, https://eto.tech/dataset-docs/chipexplorer/)
into (company, ETO input) reference pairs for the companies our corpus covers.

Only used at EVALUATION time. Nothing here feeds the KG.

Arrangement
* Organisation providers are linked to our company entities by name/alias (company identity, not facts), with a short
  explicit override list. The mapping is written out for review.
* Scope = companies with at least one source document in our corpus (a Wikipedia article or an SEC filing) that
  ETO lists as a provider. Recall is only meaningful for those.
* provision.csv rows from all years are merged into one pair per (company, input); the latest year and the largest
  market share are kept (shares allow a share-weighted recall).
* Each input gets its ETO type, a short description, its "is a type of" ancestors (sequence.csv) and its stage.
  The stage of an input without one is inherited from an ancestor, else from what it goes into.
* ETO often lists both a category and its subtypes for the same company (ASML: "Lithography tools" and
  "EUV lithography tools"). Each pair is flagged `most_specific` = no subtype of it is listed for that company, so
  results can be reported on all pairs and on the non-overlapping most-specific set.
"""
from __future__ import annotations

import csv
import re
from collections import defaultdict

import kg_common as K

PROVIDER_OVERRIDES = {          # ETO provider name -> our company name (verified by hand)
    "Cerebras": "Cerebras Systems",
    "Hua Hong": "Hua Hong Semiconductor",
    "TongFu": "Tongfu Microelectronics",
    # "DuPont Electronic Solutions": "Qnity Electronics",   # 2025 spin-off; left out on purpose (identity changed)
}


def read_csv(name):
    with open(K.ETO_DIR / name, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def first_sentence(text, n=220):
    t = re.sub(r"\s+", " ", text or "").strip()
    m = re.match(r"(.+?[.!?])\s", t + " ")
    return (m.group(1) if m else t)[:n]


def load_eto():
    inputs = {r["input_id"]: r for r in read_csv("inputs.csv")}
    stages = {r["stage_id"]: r["stage_name"] for r in read_csv("stages.csv")}
    seq = read_csv("sequence.csv")
    parent, goes_into = {}, defaultdict(set)
    for r in seq:
        if r.get("is_type_of_id"):
            parent[r["input_id"]] = r["is_type_of_id"]
        if r.get("goes_into_id"):
            goes_into[r["input_id"]].add(r["goes_into_id"])

    def ancestors(i):
        out, seen = [], set()
        while i in parent and parent[i] not in seen:
            i = parent[i]
            seen.add(i)
            out.append(i)
        return out

    def stage_of(i):
        for j in [i] + ancestors(i):
            if inputs.get(j, {}).get("stage_id"):
                return inputs[j]["stage_id"]
        frontier, seen = [i] + ancestors(i), set()
        while frontier:
            nxt = []
            for j in frontier:
                for g in goes_into.get(j, ()):
                    if g in seen:
                        continue
                    seen.add(g)
                    if inputs.get(g, {}).get("stage_id"):
                        return inputs[g]["stage_id"]
                    nxt.append(g)
            frontier = nxt
        return None

    info = {}
    for i, r in inputs.items():
        st = stage_of(i)
        info[i] = {"input_id": i, "name": r["input_name"], "type": r["type"], "description": first_sentence(r["description"]),
                   "ancestors": ancestors(i), "ancestor_names": [inputs[a]["input_name"] for a in ancestors(i) if a in inputs],
                   "stage_id": st, "stage": stages.get(st)}
    return info, parent


def build(resolver, corpus_companies):
    info, parent = load_eto()
    providers = read_csv("providers.csv")
    provision = read_csv("provision.csv")
    by_name = {e["canonical_name"]: cid for cid, e in resolver.entities.items()}
    prov_map, map_rows = {}, []
    for p in providers:
        if p["provider_type"] != "organization":
            continue
        cid, how = None, None
        if p["provider_name"] in PROVIDER_OVERRIDES:
            cid, how = by_name.get(PROVIDER_OVERRIDES[p["provider_name"]]), "override"
        else:
            for n in [p["provider_name"]] + [a.strip() for a in (p.get("alias") or "").split(";") if a.strip()]:
                cid = resolver._lookup(n)
                if cid:
                    how = "name"
                    break
        if cid:
            prov_map[p["provider_id"]] = cid
            map_rows.append({"provider_id": p["provider_id"], "provider_name": p["provider_name"], "company_id": cid,
                             "company_name": resolver.entities[cid]["canonical_name"], "method": how,
                             "in_corpus": cid in corpus_companies})
    pairs = {}
    for r in provision:
        cid = prov_map.get(r["provider_id"])
        if not cid or cid not in corpus_companies or r["provided_id"] not in info:
            continue
        k = (cid, r["provided_id"])
        p = pairs.setdefault(k, {"company_id": cid, "company_name": resolver.entities[cid]["canonical_name"],
                                 **{f"eto_{a}": b for a, b in info[r["provided_id"]].items()},
                                 "years": set(), "max_share": None})
        p["years"].add(r["year"])
        try:
            share = float(r["share_provided"])
            p["max_share"] = share if p["max_share"] is None else max(p["max_share"], share)
        except (TypeError, ValueError):
            pass
    by_company = defaultdict(set)
    for (cid, i) in pairs:
        by_company[cid].add(i)
    rows = []
    for (cid, i), p in sorted(pairs.items()):
        subtypes_listed = [j for j in by_company[cid] if i in info[j]["ancestors"]]
        p["most_specific"] = not subtypes_listed
        p["years"] = ",".join(sorted(p["years"]))
        p["latest_year"] = p["years"].split(",")[-1]
        p["eto_ancestors"] = "|".join(p["eto_ancestors"])
        p["eto_ancestor_names"] = "|".join(p["eto_ancestor_names"])
        rows.append(p)
    summary = {"corpus_companies": len(corpus_companies),
               "eto_org_providers_linked": len(prov_map),
               "corpus_companies_in_eto": len({r["company_id"] for r in rows}),
               "reference_pairs": len(rows), "most_specific_pairs": sum(r["most_specific"] for r in rows),
               "pairs_by_company": {r: n for r, n in sorted(((r, sum(x["company_name"] == r for x in rows))
                                                             for r in {x["company_name"] for x in rows}),
                                                            key=lambda t: -t[1])},
               "pairs_by_stage": dict(sorted(_count(rows, "eto_stage").items())),
               "pairs_by_input_type": dict(sorted(_count(rows, "eto_type").items())),
               "corpus_companies_not_in_eto": sorted(resolver.entities[c]["canonical_name"]
                                                     for c in corpus_companies - {r["company_id"] for r in rows})}
    K.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    _write_csv(K.EVAL_DIR / "eto_company_map.csv", map_rows)
    _write_csv(K.EVAL_DIR / "eto_reference_pairs.csv", rows)
    K.write_json(K.EVAL_DIR / "eto_reference_summary.json", summary)
    return rows, info, summary


def _count(rows, key):
    out = defaultdict(int)
    for r in rows:
        out[str(r.get(key))] += 1
    return out


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def corpus_companies(wiki, doc_company):
    out = {c.get("company_id") for c in wiki if c.get("company_id")} | set(doc_company.values())
    out.discard(None)
    return out


if __name__ == "__main__":
    import json
    reg, _ = K.load_registry()
    resolver = K.CompanyResolver(reg)
    wiki, sec, _ = K.load_claims()
    doc_company, _ = K.sec_doc_company(reg)
    rows, info, summary = build(resolver, corpus_companies(wiki, doc_company))
    print(json.dumps(summary, indent=2))
