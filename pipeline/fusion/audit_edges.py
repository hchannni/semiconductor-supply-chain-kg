"""Precision of the fused graph by group (relation, source, entity type, confidence), and calibration of confidence.

ETO only covers (company, input) provision, so it cannot evaluate SUPPLIES_TO, SUBSIDIARY_OF, LOCATED_IN ... or tell
whether companies and products were recognised correctly. This module does that with a stratified human audit,
optionally scaled up with an LLM evidence check that is validated against the human labels.

1. make_sheet()      -> eval/edge_audit_sheet.csv: ~80 fused edges, stratified by relation x source.
                        Label each row: edge_correct (yes/no/unsure), subject_ok / object_ok (did resolution pick
                        the right entity / family?), error_type (extraction | resolution | canonicalization |
                        direction | other). Save as edge_audit_sheet_labeled.csv.
2. llm_check(n)      -> eval/edge_llm_check.csv: the judge model checks n stratified edges (+ the audit edges)
                        against their evidence sentences (about n/15 calls).
3. score()           -> eval/edge_audit_report.md/.json: precision with Wilson 95% CIs per group, resolution
                        accuracy per entity type, confidence calibration (label and numeric score vs observed
                        precision), judge-vs-human agreement, and measured per-source priors for build_kg.SOURCE_PRIOR.
"""
from __future__ import annotations

import csv
import math
import random
from collections import Counter, defaultdict

import kg_common as K

SEED = 7
PER_RELATION = 10
SOURCE_SPLIT = (("WIKIPEDIA", 4), ("SEC", 4), ("both", 2))
CHECK_PROMPT_VERSION = "edgecheck-v3"
SHEET, LABELED = "edge_audit_sheet.csv", "edge_audit_sheet_labeled.csv"

CHECK_PROMPT = """You check facts in a semiconductor supply-chain knowledge graph against their evidence.
For each EDGE you get: subject and object (canonical name, type, and the wording seen in the text), the relation and its
definition, and up to two evidence sentences. Decide:
- "supported": does at least one evidence sentence state this relation between these two entities, in this direction?
  yes = clearly stated; partly = implied or hedged; no = not stated, wrong direction, or a different relation.
- "subject_ok" / "object_ok": is the canonical entity a correct reading of the wording in the text?
  (A product family is correct if the mention is an instance or synonym of it.)
Use only the evidence; no outside knowledge. Exception: BELONGS_TO_STAGE is a type classification, not a textual
claim. Judge it by whether the offering belongs to that stage (chips and IP -> Design; wafer-processing equipment,
materials and process technologies -> Fabrication; packaging/test -> Assembly, testing, and packaging).

Return one JSON object: {{"results": [{{"id": <edge id>, "supported": "yes|partly|no", "subject_ok": true|false,
"object_ok": true|false, "why": "<= 12 words"}}]}} with every edge id once.

EDGES:
{edges}
"""


def load_graph():
    ents = {e["entity_id"]: e for e in K.read_jsonl(K.KG_DIR / "canonical_entities.jsonl")}
    edges = K.read_jsonl(K.KG_DIR / "fused_edges.jsonl")
    claims = {r["claim_id"]: r for r in K.read_jsonl(K.KG_DIR / "resolved_claims.jsonl")}
    sp = K.KG_DIR / "edge_scores.jsonl"
    scores = {r["edge_id"]: r["confidence_score"] for r in K.read_jsonl(sp)} if sp.exists() else {}
    return ents, edges, claims, scores


def src_group(e):
    return "both" if len(e["source_types"]) > 1 else e["source_types"][0]


def edge_row(e, ents, claims, scores):
    cl = [claims[c] for c in e["claim_ids"] if c in claims]
    s, o = ents[e["subject"]], ents[e["object"]]
    ev = [f"[{c['source']} {c['document_id']}] {c['evidence_span'][:400]}" for c in cl[:2]]

    def oriented(c):
        # CUSTOMER_OF claims (stored as SUPPLIES_TO) and re-ordered PARTNERS_WITH claims have subject/object reversed
        # relative to the edge; show each mention on the side of the edge it actually resolved to.
        if c.get("subject_entity_id") == e["object"] and c.get("object_entity_id") == e["subject"]:
            return c["object_mention"], c["subject_mention"]
        return c["subject_mention"], c["object_mention"]
    sm = {oriented(c)[0] for c in cl}
    om = {oriented(c)[1] for c in cl}
    return {"edge_id": e["edge_id"], "relation": e["relation"], "subject": s["canonical_name"],
            "subject_type": s["entity_type"], "subject_mentions": " | ".join(sorted(sm))[:200],
            "object": o["canonical_name"], "object_type": o["entity_type"],
            "object_mentions": " | ".join(sorted(om))[:200],
            "claim_wording": " | ".join(sorted({c["relation"] for c in cl})),
            "source_group": src_group(e), "confidence": e["confidence"], "score": scores.get(e["edge_id"]),
            "group_type": e.get("group_type"), "n_claims": len(e["claim_ids"]),
            "evidence_1": ev[0] if ev else "", "evidence_2": ev[1] if len(ev) > 1 else ""}


def stratified(edges, per_relation, rng):
    by = defaultdict(lambda: defaultdict(list))
    for e in edges:
        by[e["relation"]][src_group(e)].append(e)
    pick = []
    for rel in sorted(by):
        chosen = []
        for g, k in SOURCE_SPLIT:
            pool = by[rel][g]
            chosen += rng.sample(pool, min(k, len(pool)))
        rest = [e for g in by[rel].values() for e in g if e not in chosen]
        chosen += rng.sample(rest, min(max(0, per_relation - len(chosen)), len(rest)))
        pick += chosen
    return pick


def make_sheet(per_relation=PER_RELATION):
    ents, edges, claims, scores = load_graph()
    rng = random.Random(SEED)
    rows = [edge_row(e, ents, claims, scores) for e in stratified(edges, per_relation, rng)]
    rng.shuffle(rows)
    for r in rows:
        r.update({"edge_correct": "", "subject_ok": "", "object_ok": "", "error_type": "", "note": ""})
    _write_csv(K.EVAL_DIR / SHEET, rows)
    print(f"{len(rows)} edges -> {K.EVAL_DIR / SHEET}")
    return rows


def llm_check(n=500, batch=15, model=None):
    """model: optional judge model on the same provider (e.g. "openai/gpt-oss-20b"); Groq limits are per model.
    The audit edges go first and the rest is shuffled, so a run cut short by the quota still has the agreement set
    and every relation."""
    if model:
        K.JUDGE["model"] = model
    ents, edges, claims, scores = load_graph()
    rng = random.Random(SEED + 1)
    per_rel = max(1, n // max(1, len({e["relation"] for e in edges})))
    sample = stratified(edges, per_rel, rng)
    audit = {r["edge_id"] for r in _read_csv(K.EVAL_DIR / SHEET)} if (K.EVAL_DIR / SHEET).exists() else set()
    by_id = {e["edge_id"]: e for e in edges}
    first = [by_id[i] for i in sorted(audit) if i in by_id]
    rest = [e for e in sample if e["edge_id"] not in audit]
    rng.shuffle(rest)
    sample = first + rest
    print(f"judge model: {K.JUDGE['model']}; {len(sample)} edges ({len(first)} from the human audit first)")
    P = K.load_person1_code()
    defs = {r: d["description"] for r, d in P["ontology"].RELATION_SCHEMA.items()}
    cache = K.LLMCache("edge_check")
    out = []
    try:
        for b in range(0, len(sample), batch):
            part = [edge_row(e, ents, claims, scores) for e in sample[b:b + batch]]
            text = "\n\n".join(
                f"id: {i}\nsubject: {r['subject']} ({r['subject_type']}; text: {r['subject_mentions']})\n"
                f"relation: {r['relation']} - {defs[r['relation']]}\n"
                f"object: {r['object']} ({r['object_type']}; text: {r['object_mentions']})\n"
                f"evidence: {r['evidence_1']}\n{('evidence: ' + r['evidence_2']) if r['evidence_2'] else ''}"
                for i, r in enumerate(part))
            want = set(range(len(part)))
            ok = lambda rep: bool(K.json_list(rep)) and want <= {int(x["id"]) for x in K.json_list(rep)
                                                                  if isinstance(x, dict) and str(x.get("id", "")).isdigit()}
            reply = K.json_list(cache.call("edge_check", CHECK_PROMPT_VERSION, CHECK_PROMPT.format(edges=text),
                                           judge=True, validate=ok)) or []
            got = {int(x["id"]): x for x in reply if isinstance(x, dict) and str(x.get("id", "")).isdigit()}
            for i, r in enumerate(part):
                x = got.get(i, {})
                out.append({**r, "llm_supported": x.get("supported", ""), "llm_subject_ok": x.get("subject_ok", ""),
                            "llm_object_ok": x.get("object_ok", ""), "llm_why": x.get("why", ""),
                            "in_human_audit": r["edge_id"] in audit, "judge_model": K.JUDGE["model"]})
    except K.DailyQuotaExhausted as e:
        print(f"judge quota exhausted after {len(out)} edges; re-run later (cached). {e}")
    _write_csv(K.EVAL_DIR / "edge_llm_check.csv", out)
    print(f"{len(out)} edges checked -> edge_llm_check.csv")
    return out


# ---------------------------------------------------------------- scoring
def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [round((c - m) / d, 3), round((c + m) / d, 3)]


OK = ("yes", "true", "y", "1")      # labels may be yes/no or true/false


def prec(rows, key, ok=OK):
    lab = [r for r in rows if str(r.get(key, "")).strip().lower() not in ("", "unsure")]
    k = sum(str(r[key]).strip().lower() in ok for r in lab)
    return {"n": len(lab), "precision": round(k / len(lab), 3) if lab else None, "ci95": wilson(k, len(lab))}


def by_groups(rows, key, ok=OK):
    groups = {"relation": lambda r: r["relation"], "source": lambda r: r["source_group"],
              "confidence": lambda r: r["confidence"], "object_type": lambda r: r["object_type"],
              "score_bin": lambda r: _bin(r.get("score"))}
    out = {}
    for g, f in groups.items():
        d = defaultdict(list)
        for r in rows:
            d[f(r)].append(r)
        out[g] = {k: prec(v, key, ok) for k, v in sorted(d.items())}
    return out


def _bin(s):
    try:
        s = float(s)
    except (TypeError, ValueError):
        return "n/a"
    return "<0.7" if s < 0.7 else ("0.7-0.9" if s < 0.9 else ">=0.9")


def calibration(rows):
    """Numeric score vs human correctness: per-bin mean score and observed precision; Brier score."""
    pts = []
    for r in rows:
        lab = str(r.get("edge_correct", "")).strip().lower()
        if lab in ("yes", "no") and r.get("score") not in (None, ""):
            pts.append((float(r["score"]), 1.0 if lab == "yes" else 0.0))
    if not pts:
        return {}
    bins = defaultdict(list)
    for s, y in pts:
        bins[_bin(s)].append((s, y))
    return {"brier": round(sum((s - y) ** 2 for s, y in pts) / len(pts), 4),
            "base_rate_brier": round(sum((sum(y for _, y in pts) / len(pts) - y) ** 2 for _, y in pts) / len(pts), 4),
            "bins": {b: {"n": len(v), "mean_score": round(sum(s for s, _ in v) / len(v), 3),
                         "observed_precision": round(sum(y for _, y in v) / len(v), 3)} for b, v in sorted(bins.items())}}


def kappa(a, b):
    n = len(a)
    if not n:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return round((po - pe) / (1 - pe), 3) if pe < 1 else 1.0


def score():
    rep = {}
    hpath = K.EVAL_DIR / LABELED
    human = _read_csv(hpath) if hpath.exists() else []
    if human:
        rep["human"] = {"overall": prec(human, "edge_correct"), "groups": by_groups(human, "edge_correct"),
                        "resolution": {"subject_ok": by_groups(human, "subject_ok")["relation"],
                                       "object_ok_by_type": {t: prec([r for r in human if r["object_type"] == t], "object_ok")
                                                             for t in sorted({r["object_type"] for r in human})}},
                        "error_types": dict(Counter(r.get("error_type", "").strip() or "-" for r in human
                                                    if str(r.get("edge_correct", "")).lower() == "no")),
                        "calibration": calibration(human),
                        "measured_source_prior": {s: prec([r for r in human if r["source_group"] == s and
                                                           str(r.get("n_claims")) == "1"], "edge_correct")
                                                  for s in ("WIKIPEDIA", "SEC")}}
    lpath = K.EVAL_DIR / "edge_llm_check.csv"
    llm = _read_csv(lpath) if lpath.exists() else []
    if llm:
        for r in llm:
            r["llm_correct"] = {"yes": "yes", "partly": "no", "no": "no"}.get(str(r["llm_supported"]).lower(), "")
        rep["llm"] = {"overall": prec(llm, "llm_correct"), "groups": by_groups(llm, "llm_correct")}
        if human:
            hl = {r["edge_id"]: str(r["edge_correct"]).lower() for r in human if str(r["edge_correct"]).lower() in ("yes", "no")}
            pairs = [(hl[r["edge_id"]] == "yes", r["llm_correct"] == "yes") for r in llm if r["edge_id"] in hl and r["llm_correct"]]
            rep["llm_vs_human"] = {"n": len(pairs), "accuracy": round(sum(a == b for a, b in pairs) / len(pairs), 3) if pairs else None,
                                   "cohen_kappa": kappa([a for a, _ in pairs], [b for _, b in pairs])}
    K.write_json(K.EVAL_DIR / "edge_audit_report.json", rep)
    md = report_md(rep)
    (K.EVAL_DIR / "edge_audit_report.md").write_text(md, encoding="utf-8")
    print(md)
    return rep


def report_md(rep):
    L = ["# Edge audit: precision by group and confidence calibration", ""]
    for who in ("human", "llm"):
        if who not in rep:
            continue
        r = rep[who]
        o = r["overall"]
        L += [f"## {'Human audit' if who == 'human' else 'LLM evidence check (judge model)'}",
              f"Overall precision {o['precision']} (95% CI {o['ci95']}, n={o['n']})", ""]
        for g, d in r["groups"].items():
            L += [f"**by {g}**: " + "; ".join(f"{k} {v['precision']} (n={v['n']}, CI {v['ci95']})" for k, v in d.items()
                                              if v["n"]), ""]
    h = rep.get("human", {})
    if h.get("resolution"):
        L += ["**Entity recognition (resolution correct), object side by type**: " +
              "; ".join(f"{t} {v['precision']} (n={v['n']})" for t, v in h["resolution"]["object_ok_by_type"].items() if v["n"]), ""]
    if h.get("calibration"):
        c = h["calibration"]
        L += [f"**Confidence calibration**: Brier {c['brier']} (always-guess-base-rate: {c['base_rate_brier']}); " +
              "; ".join(f"score {b}: mean {v['mean_score']} vs observed {v['observed_precision']} (n={v['n']})"
                        for b, v in c["bins"].items()), ""]
        L += [f"**Measured per-source prior** (single-claim edges): " +
              "; ".join(f"{s} {v['precision']} (n={v['n']})" for s, v in h["measured_source_prior"].items()), ""]
    if rep.get("llm_vs_human"):
        a = rep["llm_vs_human"]
        L += [f"**LLM check vs human**: accuracy {a['accuracy']}, Cohen's kappa {a['cohen_kappa']} (n={a['n']})."]
    return "\n".join(L) + "\n"


def _read_csv(p):
    with open(p, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def _write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    import sys
    if "--sheet" in sys.argv:
        make_sheet()
    if "--llm" in sys.argv:
        llm_check()
    score()
