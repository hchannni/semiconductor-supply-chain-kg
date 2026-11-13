"""Semantic evaluation of the KG against ETO (company -> input provision pairs).

Names in our graph and in ETO never match exactly (unless we copied ETO's vocabulary, which would be test leakage),
so matching is semantic, with two independent methods:

  M1  Embedding similarity. Each ETO input (name + definition) is compared with every offering the KG lists for the
      same company; the pair counts as recovered when the best cosine similarity >= tau. tau is calibrated on the
      human-labelled sheet (eto_label_sheet.csv) once you have filled it in.
  M2  LLM judge, granularity-aware. Per company, the judge sees ETO's inputs and the KG's offerings and labels each
      input: exact | kg_more_specific | kg_more_general | related | none, lists the matching offerings and says
      whether the match needed outside knowledge (e.g. what a brand name is). Strict match = exact or
      kg_more_specific; graded recall gives kg_more_general half credit.

Variants (RQ4): wikipedia, sec, combined_after_fusion (product families), combined_before_fusion (raw mentions, company
resolution only), plus wikipedia_raw / sec_raw. A pair is recovered by a variant if a matching offering is supported
by that variant's source(s).

Validation: eto_label_sheet.csv (60 rows) -> fill `human_label` (yes / partial / no) -> re-run with CALIBRATE = True:
judge agreement (accuracy, Cohen's kappa) and the embedding threshold are then computed from your labels.
"""
from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict

import kg_common as K
import eto_reference as ER

JUDGE_PROMPT_VERSION = "judge-v2"
USE_SEPARATE_JUDGE = True   # judge with kg_common.JUDGE (another model family) instead of Gemini: avoids self-preference
EMB_TAU_DEFAULT = {"st": 0.55, "tfidf": 0.35}   # used until calibrated on human labels
STRICT = {"exact", "kg_more_specific"}
LABEL_SHEET = "eto_label_sheet.csv"
LABELED_SHEET = "eto_label_sheet_labeled.csv"   # your filled-in copy
SEED = 13

JUDGE_PROMPT = """You evaluate a semiconductor supply-chain knowledge graph against a reference dataset.

Company: {company}

REFERENCE INPUTS - what the reference dataset says this company provides (id | name | definition | broader categories):
{reference}

GRAPH OFFERINGS - what our knowledge graph says this company provides (idx | name | details):
{offerings}

For EACH reference input, decide how the graph offerings cover it. Compare the LEVEL OF DETAIL carefully:
- "exact": an offering is the same kind of thing at the same level of detail (synonyms and wording differences are
  fine: "chemical vapor deposition (CVD)" = "CVD tools").
- "kg_more_specific": an offering is a narrower type, product line or model of that input
  ("Striker single-wafer ALD" for "deposition tools"; an EUV scanner family for "lithography tools").
- "kg_more_general": the graph only has a BROADER offering that includes the input but does not name it.
  ALWAYS use this, never "exact", when the offering is a wider category: "deposition equipment" for "CVD tools",
  "etch and clean equipment" for "dry etch tools", "process control systems" for "overlay metrology tools".
- "related": related but a different kind of thing (a service for a tool; a material vs the tool processing it).
- "none": nothing relevant.
The "details" of an offering list the original wording seen in the sources ("observed as: ..."); a specific product
named there counts like an offering of that name.
List every offering idx that supports the verdict. Set "needs_outside_knowledge": true when the match is only possible
with knowledge that is not in the offering text (for example knowing what a brand or model name is).
Judge only what the graph says. Do not use your own knowledge of what the company sells to fill gaps.

Return one JSON object with a "results" list holding one object per reference input (every id above, once):
{{"results": [{{"id": "<reference id>", "verdict": "exact|kg_more_specific|kg_more_general|related|none",
   "offerings": [<idx>, ...], "needs_outside_knowledge": true|false, "why": "<= 12 words"}}]}}
"""


# ------------------------------------------------------------------ predictions
def family_offerings(edges, entities, resolved, scores=None):
    """Company -> offerings from the fused graph (PROVIDES edges), with the edge's confidence label and score."""
    scores = scores or {}
    ev = {}
    for r in resolved:
        ev.setdefault(r["claim_id"], r)
    out = defaultdict(list)
    for e in edges:
        if e["relation"] != "PROVIDES":
            continue
        ent = entities[e["object"]]
        claims = [ev[c] for c in e["claim_ids"] if c in ev]
        mentions = list(dict.fromkeys(c["object_mention"] for c in claims))
        out[e["subject"]].append({
            "offering_id": e["object"], "name": ent["canonical_name"], "entity_type": ent["entity_type"],
            "details": "; ".join(x for x in [ent.get("description", ""),
                                             ("observed as: " + ", ".join(mentions[:6])) if mentions else ""] if x),
            "evidence": claims[0]["evidence_span"][:200] if claims else "",
            "sources": set(e["source_types"]), "confidence": e["confidence"],
            "score": scores.get(e["edge_id"])})
    return out


def raw_offerings(claims, resolver, doc_company):
    """Company -> raw mentions (no canonicalisation, no fusion): the 'before fusion' graph."""
    out = defaultdict(dict)
    for c in claims:
        if c["relation"] != "PROVIDES" or c["subject_type"] != "Company":
            continue
        dc = c.get("company_id") or doc_company.get(c["document_id"])
        cid = resolver.resolve(c["subject_mention"], dc)[0]
        if not cid:
            continue
        key = K.loose_key(c["object_mention"])
        o = out[cid].setdefault(key, {"offering_id": "raw:" + key, "name": c["object_mention"].strip(), "details": "",
                                      "evidence": c["evidence_span"][:200], "sources": set()})
        o["sources"].add(c["source"])
    return {k: list(v.values()) for k, v in out.items()}


# ------------------------------------------------------------------ M1 embeddings
def embedding_scores(ref_rows, offerings_by_company, embedder):
    """For every reference pair and every source restriction: best cosine similarity and the best offering."""
    import numpy as np
    ref_text = {r["eto_input_id"]: f"{r['eto_name']}: {r['eto_description']}" for r in ref_rows}
    all_off = [o for offs in offerings_by_company.values() for o in offs]
    off_text = {id(o): f"{o['name']}. {o['details']}"[:400] for o in all_off}
    if embedder.backend.startswith("tfidf"):
        embedder.fit_vocabulary(list(ref_text.values()) + list(off_text.values()))
    ids = list(ref_text)
    R = dict(zip(ids, embedder.encode([ref_text[i] for i in ids])))
    O = dict(zip(off_text, embedder.encode(list(off_text.values())))) if off_text else {}
    out = {}
    for r in ref_rows:
        offs = offerings_by_company.get(r["company_id"], [])
        for restrict in (None, "WIKIPEDIA", "SEC"):
            cand = [o for o in offs if restrict is None or restrict in o["sources"]]
            if not cand:
                out[(r["company_id"], r["eto_input_id"], restrict)] = (0.0, None)
                continue
            sims = np.array([float(R[r["eto_input_id"]] @ O[id(o)]) for o in cand])
            j = int(sims.argmax())
            out[(r["company_id"], r["eto_input_id"], restrict)] = (round(float(sims[j]), 4), cand[j]["name"])
    return out


# ------------------------------------------------------------------ M2 judge
JUDGE_CHUNK = 70          # offerings per judge call (keeps each request under the free tier's tokens-per-minute cap)
_RANK = {"exact": 4, "kg_more_specific": 4, "kg_more_general": 2, "related": 1, "none": 0}


def judge_company(cache, company_name, refs, offs):
    """Judge one company. Offerings are judged in chunks; an unparseable or incomplete reply is retried, then the chunk
    is split in half. Inputs the judge never answered are marked 'judge_failed' and left out of the recall
    denominators (never silently counted as misses)."""
    if not offs:
        return {r["eto_input_id"]: {"verdict": "none", "offerings": [], "needs_outside_knowledge": False,
                                    "why": "no offerings in graph"} for r in refs}
    reference = "\n".join(f"{r['eto_input_id']} | {r['eto_name']} | {r['eto_description']} | "
                          f"{r['eto_ancestor_names'] or '-'}" for r in refs)
    ids = {r["eto_input_id"] for r in refs}
    res = {}

    def valid(reply):
        out = K.json_list(reply)
        return bool(out) and ids <= {str(x.get("id")) for x in out if isinstance(x, dict)}

    def run(part, depth=0):
        offerings = "\n".join(f"{i} | {o['name']} | {(o['details'] or o['evidence'])[:220]}" for i, o in part)
        prompt = JUDGE_PROMPT.format(company=company_name, reference=reference, offerings=offerings)
        reply = cache.call("eto_judge", JUDGE_PROMPT_VERSION, prompt, judge=USE_SEPARATE_JUDGE, validate=valid,
                           retries=1 if depth == 0 else 0)      # bounded: at most ~5 calls for a stubborn chunk
        if reply is None:
            if len(part) > 1 and depth < 1:
                mid = len(part) // 2
                return run(part[:mid], depth + 1) + run(part[mid:], depth + 1)
            return []
        return [(row, {i for i, _ in part}) for row in K.json_list(reply) if isinstance(row, dict)]

    rows = []
    for start in range(0, len(offs), JUDGE_CHUNK):
        rows += run(list(enumerate(offs))[start:start + JUDGE_CHUNK])
    for row, valid_idx in rows:
        rid = str(row.get("id"))
        if rid not in ids:
            continue
        v = row.get("verdict") if row.get("verdict") in _RANK else "none"
        idx = [int(i) for i in row.get("offerings") or [] if str(i).isdigit() and int(i) in valid_idx]
        cur = res.get(rid)
        if cur is None or _RANK[v] > _RANK[cur["verdict"]]:
            res[rid] = {"verdict": v, "offerings": idx, "needs_outside_knowledge": bool(row.get("needs_outside_knowledge")),
                        "why": row.get("why", "")}
        elif _RANK[v] == _RANK[cur["verdict"]] and v != "none":
            cur["offerings"] = sorted(set(cur["offerings"]) | set(idx))
            cur["needs_outside_knowledge"] = cur["needs_outside_knowledge"] and bool(row.get("needs_outside_knowledge"))
    for r in refs:
        res.setdefault(r["eto_input_id"], {"verdict": "judge_failed", "offerings": [], "needs_outside_knowledge": False,
                                           "why": "no valid judge reply"})
    return res


def variant_verdict(j, offs, sources):
    """Restrict a judge decision to offerings supported by the given sources."""
    if j["verdict"] in ("none", "judge_failed"):
        return j["verdict"], []
    idx = [i for i in j["offerings"] if offs[i]["sources"] & sources]
    return (j["verdict"], idx) if idx else ("none", [])


# ------------------------------------------------------------------ metrics
def recall_stats(rows, key, weight=None):
    """rows: dicts with 0/1 (or graded) value in rows[key]; cluster bootstrap over companies for a 95% CI."""
    if not rows:
        return {"n": 0}
    val = lambda r: r[key] * (r[weight] if weight else 1)
    den = lambda r: (r[weight] if weight else 1)
    est = sum(map(val, rows)) / max(1e-9, sum(map(den, rows)))
    by_c = defaultdict(list)
    for r in rows:
        by_c[r["company_id"]].append(r)
    comps = list(by_c)
    rng = random.Random(SEED)
    boots = []
    for _ in range(1000):
        sample = [r for c in (rng.choice(comps) for _ in comps) for r in by_c[c]]
        d = sum(map(den, sample))
        if d:
            boots.append(sum(map(val, sample)) / d)
    boots.sort()
    return {"n": len(rows), "recall": round(est, 4),
            "ci95": [round(boots[int(0.025 * len(boots))], 4), round(boots[int(0.975 * len(boots)) - 1], 4)]}


def main(calibrate=False):
    reg, _ = K.load_registry()
    resolver = K.CompanyResolver(reg, K.KG_DIR / "company_id_ledger.json")
    wiki, sec, _ = K.load_claims()
    doc_company, _ = K.sec_doc_company(reg)
    ref_rows, info, ref_summary = ER.build(resolver, ER.corpus_companies(wiki, doc_company))
    for r in ref_rows:
        r["most_specific"] = r["most_specific"] in (True, "True")
    entities = {e["entity_id"]: e for e in K.read_jsonl(K.KG_DIR / "canonical_entities.jsonl")}
    edges = K.read_jsonl(K.KG_DIR / "fused_edges.jsonl")
    resolved = K.read_jsonl(K.KG_DIR / "resolved_claims.jsonl")
    score_path = K.KG_DIR / "edge_scores.jsonl"
    scores = {r["edge_id"]: r["confidence_score"] for r in K.read_jsonl(score_path)} if score_path.exists() else {}
    fam = family_offerings(edges, entities, resolved, scores)
    raw = raw_offerings(wiki + sec, resolver, doc_company)

    embedder = K.Embedder()
    st = not embedder.backend.startswith("tfidf")
    emb_f = embedding_scores(ref_rows, fam, embedder)
    emb_r = embedding_scores(ref_rows, raw, embedder)

    cache = K.LLMCache("eto_judge")
    refs_by_c = defaultdict(list)
    for r in ref_rows:
        refs_by_c[r["company_id"]].append(r)
    judged_f, judged_r = {}, {}
    judge_model_used = K.JUDGE["model"] if USE_SEPARATE_JUDGE else K.MODEL
    try:
        for cid, refs in sorted(refs_by_c.items()):
            name = refs[0]["company_name"]
            judged_f[cid] = judge_company(cache, name, refs, fam.get(cid, []))
            judged_r[cid] = judge_company(cache, name, refs, raw.get(cid, []))
    except K.DailyQuotaExhausted as e:
        print(f"daily quota exhausted during judging ({len(judged_f)} companies done); re-run later. {e}")

    variants = {"wikipedia": (fam, judged_f, {"WIKIPEDIA"}, "WIKIPEDIA", emb_f),
                "sec": (fam, judged_f, {"SEC"}, "SEC", emb_f),
                "combined_after_fusion": (fam, judged_f, {"WIKIPEDIA", "SEC"}, None, emb_f),
                "combined_before_fusion": (raw, judged_r, {"WIKIPEDIA", "SEC"}, None, emb_r),
                "wikipedia_raw": (raw, judged_r, {"WIKIPEDIA"}, "WIKIPEDIA", emb_r),
                "sec_raw": (raw, judged_r, {"SEC"}, "SEC", emb_r)}

    tau = EMB_TAU_DEFAULT["st" if st else "tfidf"]
    calibration = None
    if calibrate and (K.EVAL_DIR / LABELED_SHEET).exists():
        calibration = calibrate_from_labels(K.EVAL_DIR / LABELED_SHEET)
        tau = calibration.get("embedding_tau", tau)

    pair_rows = []
    for vname, (offs_by_c, judged, sources, restrict, emb) in variants.items():
        for r in ref_rows:
            cid, iid = r["company_id"], r["eto_input_id"]
            offs = offs_by_c.get(cid, [])
            j = judged.get(cid, {}).get(iid)
            if j is None:
                verdict, idx, ext = "not_judged", [], False
            else:
                verdict, idx = variant_verdict(j, offs, sources)
                ext = j["needs_outside_knowledge"]
            score, best = emb[(cid, iid, restrict)]
            pair_rows.append({
                "variant": vname, "company_id": cid, "company_name": r["company_name"], "eto_input_id": iid,
                "eto_name": r["eto_name"], "eto_type": r["eto_type"], "eto_stage": r["eto_stage"],
                "most_specific": r["most_specific"],
                "max_share": r["max_share"], "judge_verdict": verdict,
                "judge_offerings": " | ".join(offs[i]["name"] for i in idx),
                "needs_outside_knowledge": ext, "judge_why": (j or {}).get("why", ""),
                "judged": int(verdict not in ("not_judged", "judge_failed")),
                "strict": int(verdict in STRICT), "graded": 1.0 if verdict in STRICT else (0.5 if verdict == "kg_more_general" else 0.0),
                "strict_grounded": int(verdict in STRICT and not ext),
                "emb_score": score, "emb_best": best, "emb_match": int(score >= tau),
                "matched_confidence": "|".join(sorted({offs[i].get("confidence") or "" for i in idx}))
                if verdict in STRICT else "",
                "matched_best_score": max([offs[i].get("score") or 0 for i in idx], default=None)
                if verdict in STRICT else None})

    summary = {"embedding_backend": embedder.backend, "embedding_tau": tau,
               "embedding_tau_calibrated": bool(calibration), "judge_model": judge_model_used,
               "judge_prompt_version": JUDGE_PROMPT_VERSION, "reference": ref_summary, "variants": {},
               "llm_calls_made": K.LLM_CALLS["made"], "llm_calls_cached": K.LLM_CALLS["cached"],
               "judge_tokens_used": K.LLM_CALLS.get("judge_tokens", 0)}
    for vname in variants:
        all_rows = [p for p in pair_rows if p["variant"] == vname]
        rows = [p for p in all_rows if p["judged"]]          # judge metrics only over pairs the judge answered
        ms = [p for p in rows if p["most_specific"]]
        shared = [dict(p, w=p["max_share"]) for p in rows if p["max_share"] not in (None, "")]
        stage_rows = stage_level(rows)
        summary["variants"][vname] = {
            "judge_strict": recall_stats(rows, "strict"),
            "judge_strict_most_specific": recall_stats(ms, "strict"),
            "judge_graded": recall_stats(rows, "graded"),
            "judge_strict_text_grounded": recall_stats(rows, "strict_grounded"),
            "judge_strict_share_weighted": recall_stats(shared, "strict", weight="w") if shared else {"n": 0},
            "stage_level": recall_stats(stage_rows, "strict"),
            "embedding": recall_stats(all_rows, "emb_match"),
            "judge_coverage": f"{len(rows)}/{len(all_rows)}",
            "verdicts": dict(Counter(p["judge_verdict"] for p in all_rows))}
    summary["groups"] = group_breakdown(pair_rows)
    summary["recall_by_edge_confidence"] = recall_by_confidence(pair_rows)
    summary["per_company"] = per_company(pair_rows)
    summary["precision_side"] = precision_side(fam, judged_f, refs_by_c)
    if calibration:
        summary["calibration"] = calibration

    _write_csv(K.EVAL_DIR / "eto_eval_pairs.csv", pair_rows)
    K.write_json(K.EVAL_DIR / "eto_eval_summary.json", summary)
    if not (K.EVAL_DIR / LABELED_SHEET).exists():
        write_label_sheet(pair_rows, fam)
    md = report_md(summary)
    (K.EVAL_DIR / "eto_eval_report.md").write_text(md, encoding="utf-8")
    print(md)
    return summary


MAIN_VARIANTS = ("wikipedia", "sec", "combined_before_fusion", "combined_after_fusion")


def group_breakdown(pair_rows):
    """Strict judge recall per group of the reference: ETO input type, stage, most-specific flag."""
    out = {}
    for g in ("eto_type", "eto_stage", "most_specific"):
        out[g] = {}
        for v in MAIN_VARIANTS:
            by = defaultdict(list)
            for p in pair_rows:
                if p["variant"] == v and p["judged"]:
                    by[str(p[g])].append(p)
            for k, rows in by.items():
                out[g].setdefault(k, {})[v] = recall_stats(rows, "strict")
    return out


def recall_by_confidence(pair_rows):
    """How much ETO recall survives if the graph is filtered by edge confidence (after fusion)."""
    rows = [p for p in pair_rows if p["variant"] == "combined_after_fusion" and p["judged"]]
    n = len(rows) or 1
    out = {}
    for name, allowed in (("HIGH only", {"HIGH"}), ("HIGH+MEDIUM", {"HIGH", "MEDIUM"}), ("all", {"HIGH", "MEDIUM", "LOW"})):
        hit = sum(1 for p in rows if p["strict"] and set(p["matched_confidence"].split("|")) & allowed)
        out[name] = round(hit / n, 4)
    for t in (0.5, 0.7, 0.8, 0.9, 0.95):
        hit = sum(1 for p in rows if p["strict"] and (p["matched_best_score"] or 0) >= t)
        out[f"score >= {t}"] = round(hit / n, 4)
    return out


def stage_level(rows):
    agg = {}
    for p in rows:
        k = (p["company_id"], p["eto_stage"])
        a = agg.setdefault(k, {"company_id": p["company_id"], "strict": 0})
        a["strict"] = max(a["strict"], p["strict"])
    return list(agg.values())


def per_company(pair_rows):
    out = defaultdict(dict)
    for v in ("wikipedia", "sec", "combined_before_fusion", "combined_after_fusion"):
        by = defaultdict(list)
        for p in pair_rows:
            if p["variant"] == v:
                by[p["company_name"]].append(p)
        for c, ps in by.items():
            judged = [p for p in ps if p["judged"]]
            out[c][v] = f"{sum(p['strict'] for p in judged)}/{len(judged)}" + ("" if len(judged) == len(ps) else
                                                                            f" ({len(ps) - len(judged)} unjudged)")
    return dict(sorted(out.items(), key=lambda kv: -int(kv[1]["combined_after_fusion"].split("/")[1].split()[0])))


def precision_side(fam, judged, refs_by_c):
    """Of the Input-type families the KG lists for ETO-covered companies, how many match some ETO pair?
    ETO is incomplete, so unconfirmed families are not errors; they are listed for a manual check."""
    rows, confirmed, total = [], 0, 0
    by_group = defaultdict(lambda: [0, 0])
    for cid, refs in refs_by_c.items():
        offs = fam.get(cid, [])
        hit = set()
        for j in judged.get(cid, {}).values():
            if j["verdict"] in STRICT | {"kg_more_general"}:
                hit.update(j["offerings"])
        for i, o in enumerate(offs):
            if o.get("entity_type") != "Input":       # supply-chain families only (ETO's scope)
                continue
            total += 1
            confirmed += i in hit
            src = "both" if len(o["sources"]) > 1 else next(iter(o["sources"]))
            sc = o.get("score")
            sbin = "n/a" if sc is None else ("<0.7" if sc < 0.7 else "0.7-0.9" if sc < 0.9 else ">=0.9")
            for g in (f"confidence={o.get('confidence')}", f"source={src}", f"score={sbin}"):
                by_group[g][0] += i in hit
                by_group[g][1] += 1
            if i not in hit:
                rows.append({"company": refs[0]["company_name"], "offering": o["name"], "details": o["details"][:160],
                             "sources": "|".join(sorted(o["sources"]))})
    _write_csv(K.EVAL_DIR / "eto_unconfirmed_offerings.csv", rows)
    return {"scope": "Input-type families (supply-chain scope) of companies ETO lists",
            "offerings_of_eto_companies": total, "confirmed_by_eto": confirmed,
            "share_confirmed": round(confirmed / total, 4) if total else None,
            "share_confirmed_by_group": {g: {"confirmed": a, "n": b, "share": round(a / b, 3)}
                                         for g, (a, b) in sorted(by_group.items())}}


def write_label_sheet(pair_rows, fam=None, n=60):
    rows = [p for p in pair_rows if p["variant"] == "combined_after_fusion" and p["judged"]]
    rng = random.Random(SEED)
    match = [p for p in rows if p["strict"]]
    miss_hi = sorted([p for p in rows if not p["strict"]], key=lambda p: -p["emb_score"])[:40]
    rest = [p for p in rows if p not in match and p not in miss_hi]
    pick = rng.sample(match, min(20, len(match))) + rng.sample(miss_hi, min(20, len(miss_hi)))
    pick += rng.sample(rest, min(n - len(pick), len(rest)))
    rng.shuffle(pick)
    fam = fam or {}

    def all_offerings(cid):
        return " || ".join(f"{o['name']}: {o['details'].replace('observed as: ', '')}"[:160] for o in fam.get(cid, []))[:3000]
    sheet = [{"row": i, "company": p["company_name"], "eto_input": p["eto_name"],
              "kg_all_offerings": all_offerings(p["company_id"]),
              "kg_offerings_judge": p["judge_offerings"], "kg_best_by_embedding": p["emb_best"],
              "emb_score": p["emb_score"], "judge_verdict": p["judge_verdict"],
              "human_label": "", "note": ""} for i, p in enumerate(pick)]
    _write_csv(K.EVAL_DIR / LABEL_SHEET, sheet)


def calibrate_from_labels(path):
    """Human labels: yes = graph has it (same or more specific); partial = only a broader offering; no = not there.
    Compared with the judge on three classes (match / broader / none) and on the binary match decision."""
    rows = [r for r in csv.DictReader(open(path, encoding="utf-8-sig")) if r.get("human_label", "").strip()]
    if not rows:
        return {}
    hmap = {"yes": "match", "y": "match", "partial": "broader", "p": "broader", "no": "none", "n": "none"}
    jmap = lambda v: "match" if v in STRICT else ("broader" if v == "kg_more_general" else "none")
    rows = [r for r in rows if r["human_label"].strip().lower() in hmap and r["judge_verdict"] not in ("judge_failed", "not_judged")]
    if not rows:
        return {}
    h = [hmap[r["human_label"].strip().lower()] for r in rows]
    j = [jmap(r["judge_verdict"]) for r in rows]
    acc3 = sum(a == b for a, b in zip(h, j)) / len(rows)
    y = [x == "match" for x in h]
    jb = [x == "match" for x in j]
    agree = sum(a == b for a, b in zip(y, jb)) / len(rows)
    p_yes = (sum(y) / len(y)) * (sum(jb) / len(jb)) + (1 - sum(y) / len(y)) * (1 - sum(jb) / len(jb))
    kappa = (agree - p_yes) / (1 - p_yes) if p_yes < 1 else 1.0
    s_ = [float(r["emb_score"] or 0) for r in rows]
    best = (0, None)
    for t in sorted(set(s_)):
        pred = [x >= t for x in s_]
        tp = sum(a and b for a, b in zip(pred, y))
        fp = sum(a and not b for a, b in zip(pred, y))
        fn = sum((not a) and b for a, b in zip(pred, y))
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0
        if f1 > best[0]:
            best = (f1, t)
    confusion = Counter(f"human={a} / judge={b}" for a, b in zip(h, j))
    return {"labelled_rows": len(rows), "judge_accuracy_3class": round(acc3, 3), "judge_accuracy": round(agree, 3),
            "judge_cohen_kappa": round(kappa, 3),
            "judge_precision_vs_human": round(sum(a and b for a, b in zip(jb, y)) / max(1, sum(jb)), 3),
            "judge_recall_vs_human": round(sum(a and b for a, b in zip(jb, y)) / max(1, sum(y)), 3),
            "confusion": dict(sorted(confusion.items())),
            "embedding_tau": best[1], "embedding_f1_at_tau": round(best[0], 3)}


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def report_md(s):
    ref = s["reference"]
    L = ["# ETO evaluation (semantic matching)", "",
         f"Reference: {ref['reference_pairs']} (company, input) pairs for {ref['corpus_companies_in_eto']} of our "
         f"{ref['corpus_companies']} corpus companies ({ref['most_specific_pairs']} most-specific). "
         f"Judge: {s['judge_model']} ({s['judge_prompt_version']}). Embeddings: {s['embedding_backend']}, "
         f"tau = {s['embedding_tau']} ({'calibrated on human labels' if s['embedding_tau_calibrated'] else 'default, NOT calibrated'}).",
         "", "| Variant | Judged | Judge strict | 95% CI | Most-specific | Graded | Text-grounded | Stage level | Embedding |",
         "|---|---:|---:|---|---:|---:|---:|---:|---:|"]
    for v, m in s["variants"].items():
        js = m["judge_strict"]
        f = lambda k: f"{m[k].get('recall', 0):.0%}" if m[k].get("n") else "-"
        L.append(f"| {v} | {m['judge_coverage']} | {js.get('recall', 0):.0%} | {js.get('ci95', ['', ''])[0]:.0%}–{js.get('ci95', ['', ''])[1]:.0%} | "
                 f"{f('judge_strict_most_specific')} | {f('judge_graded')} | {f('judge_strict_text_grounded')} | "
                 f"{f('stage_level')} | {f('embedding')} |" if js.get("n") else f"| {v} | - |")
    L += ["", "Strict = exact or KG more specific. Graded gives half credit when the KG only has a broader offering. "
          "Text-grounded counts only matches the judge could make without outside knowledge. CIs: company-level bootstrap. "
          "Judge metrics use only pairs the judge answered ('Judged'); failed or missing replies are excluded, not counted as misses.",
          "", "Per company (strict matches / pairs):", "",
          "| Company | Wikipedia | SEC | Before fusion | After fusion |", "|---|---:|---:|---:|---:|"]
    for c, d in s["per_company"].items():
        L.append(f"| {c} | {d.get('wikipedia', '')} | {d.get('sec', '')} | {d.get('combined_before_fusion', '')} | "
                 f"{d.get('combined_after_fusion', '')} |")
    L += ["", "Strict recall by group (reference side):", "",
          "| Group | Value | " + " | ".join(MAIN_VARIANTS) + " |", "|---|---|" + "---:|" * len(MAIN_VARIANTS)]
    for g, vals in s["groups"].items():
        for k, d in sorted(vals.items()):
            cells = [f"{d[v]['recall']:.0%} (n={d[v]['n']})" if d.get(v, {}).get("n") else "-" for v in MAIN_VARIANTS]
            L.append(f"| {g} | {k} | " + " | ".join(cells) + " |")
    L += ["", "Recall kept when the fused graph is filtered by edge confidence: " +
          ", ".join(f"{k}: {v:.0%}" for k, v in s["recall_by_edge_confidence"].items())]
    ps = s["precision_side"]
    L += ["", f"Precision side: {ps['confirmed_by_eto']} of {ps['offerings_of_eto_companies']} product families of these "
          f"companies match an ETO pair (unconfirmed ones are listed in eto_unconfirmed_offerings.csv; ETO is incomplete, "
          f"so unconfirmed is not the same as wrong).",
          "Share confirmed by group: " + "; ".join(f"{g} {d['confirmed']}/{d['n']}" for g, d in
                                                    ps.get("share_confirmed_by_group", {}).items())]
    if s.get("calibration"):
        c = s["calibration"]
        L += ["", f"Validation on {c['labelled_rows']} human-labelled pairs: judge 3-class accuracy {c['judge_accuracy_3class']}, "
              f"binary accuracy {c['judge_accuracy']}, "
              f"Cohen's kappa {c['judge_cohen_kappa']}, judge precision {c['judge_precision_vs_human']}, recall "
              f"{c['judge_recall_vs_human']}; embedding tau {c['embedding_tau']} (F1 {c['embedding_f1_at_tau']})."]
    else:
        L += ["", "Not yet validated: fill in human_label in eto_label_sheet.csv, save it as "
              "eto_label_sheet_labeled.csv and re-run with calibrate=True."]
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    import sys
    main(calibrate="--calibrate" in sys.argv)
