"""Stages B + D: entity resolution and knowledge fusion -> files for Person 1's Neo4j loader.

Reads   Contract A claims (Wikipedia from Person 1, SEC from Person 2), the company registry, and (optionally)
        data/kg/offering_canon.jsonl from canonicalize_offerings.py.
Writes  data/kg/canonical_entities.jsonl   (CanonicalEntity)
        data/kg/resolved_claims.jsonl      (Contract B; one per accepted raw claim, mentions kept)
        data/kg/fused_edges.jsonl          (Contract C)
        data/kg/documents.jsonl            (document index for provenance)
        data/kg/dropped_claims.jsonl       (every claim that did not become evidence, with the reason)
        data/kg/kg_build_report.json / .md

Design decisions
* Offerings resolve MANY-TO-ONE to product-family entities (the canonical name is the family; every observed
  mention is kept as an alias and on the Claim). Without offering_canon.jsonl, each mention is its own entity.
* CUSTOMER_OF(a, b) is fused with SUPPLIES_TO(b, a) (edge stored as SUPPLIES_TO); the claim keeps its wording.
* PARTNERS_WITH is symmetric (stored with the smaller ID as subject).
* A company has one headquarters: competing LOCATED_IN countries are ranked (filer's own SEC statement > number of
  documents > Wikipedia infobox); losers stay in the graph flagged conflict=True, confidence LOW.
* first_seen / last_seen are SEC filing dates of the supporting claims (observation dates, not validity intervals).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict

import kg_common as K

USE_FAMILIES = True
# Per-document reliability priors for the numeric edge score. SEC: our extraction audit (25/30 correct).
# Wikipedia: placeholder until audited; audit_edges.py estimates both and you can paste the measured values here.
SOURCE_PRIOR = {"SEC": 0.83, "WIKIPEDIA": 0.80}
LEGAL_FORM = __import__("re").compile(r"(?i)\b(inc|incorporated|ltd|limited|llc|s\.?a|b\.?v|n\.?v|gmbh|plc|corp|corporation|"
                                      r"k\.?k|pte|co\.)\b\.?|\(")


def main():
    P = K.load_person1_code()
    onto, vocab, schemas = P["ontology"], P["vocab"], P["schemas"]
    reg, reg_path = K.load_registry()
    resolver = K.CompanyResolver(reg, K.KG_DIR / "company_id_ledger.json")
    ledger = K.IdLedger(K.KG_DIR / "entity_id_ledger.json")
    wiki, sec, contract_bad = K.load_claims()
    doc_company, sec_docs = K.sec_doc_company(reg)
    claims = wiki + sec

    canon = {}
    families = {}
    canon_path = K.KG_DIR / "offering_canon.jsonl"
    if USE_FAMILIES and canon_path.exists():
        for r in K.read_jsonl(canon_path):
            canon[(r["company_id"], r["key"])] = r
        fam_path = K.KG_DIR / "offering_families.json"
        families = json.loads(fam_path.read_text()) if fam_path.exists() else {}
    family_mode = bool(canon)

    entities = {}                     # id -> dict
    mention_counts = defaultdict(Counter)

    def ent(eid, name, etype, **extra):
        e = entities.setdefault(eid, {"entity_id": eid, "canonical_name": name, "entity_type": etype, **extra})
        return e

    def resolve_offering(mention, otype, company_id):
        r = canon.get((company_id, K.loose_key(mention)))
        if r and r.get("status") == "not_an_offering":
            return None, 0.0, "not_an_offering"
        if r and r.get("family"):
            et = r["entity_type"]
            eid = ledger.get(onto.ENTITY_ID_PREFIX[et], "family:" + r["family"].lower())
            fam = families.get(r["family"], {})
            ent(eid, r["family"], et, description=(f"{fam.get('category') or r.get('category')}. "
                                                   f"{fam.get('definition', '')}").strip())
            score = {"text": 0.9, "knowledge": 0.75}.get(r.get("grounding"), 0.8)
            if r.get("specificity") == "vague":
                score = 0.6
            return eid, score, "family"
        eid = ledger.get(onto.ENTITY_ID_PREFIX[otype], "mention:" + K.loose_key(mention))
        ent(eid, mention.strip(), otype)
        return eid, 1.0, "mention"

    def resolve(mention, etype, doc_cid, subject_company):
        if etype == "Company":
            cid, score, how = resolver.resolve(mention, doc_cid)
            if not cid:
                return None, 0.0, how
            e = resolver.entities[cid]
            ent(cid, e["canonical_name"], "Company")
            return cid, score, how
        if etype == "Country":
            name = vocab.canonical_country(mention)
            if not name:
                return None, 0.0, "unknown_country"
            return ent(ledger.get("country", name), name, "Country")["entity_id"], 1.0, "country_vocab"
        if etype == "SupplyChainStage":
            name = vocab.canonical_stage(mention)
            if not name:
                return None, 0.0, "unknown_stage"
            return ent(ledger.get("stage", name), name, "SupplyChainStage")["entity_id"], 1.0, "stage_vocab"
        if etype == "Facility":
            key = K.norm_name(mention) or K.loose_key(mention)
            return ent(ledger.get("facility", key), mention.strip(), "Facility")["entity_id"], 1.0, "facility_name"
        return resolve_offering(mention, etype, subject_company or doc_cid)

    resolved, dropped = [], []
    res_methods = Counter()
    for c in claims:
        doc_cid = c.get("company_id") or doc_company.get(c["document_id"])
        subj_company = None
        if c["subject_type"] == "Company":
            subj_company = resolver.resolve(c["subject_mention"], doc_cid)[0]
        s_id, s_sc, s_how = resolve(c["subject_mention"], c["subject_type"], doc_cid, subj_company)
        o_id, o_sc, o_how = resolve(c["object_mention"], c["object_type"], doc_cid, subj_company)
        res_methods[s_how] += 1
        res_methods[o_how] += 1
        if (s_id and s_id == o_id and c["relation"] == "SUBSIDIARY_OF" and c["source"] == "SEC"
                and K.loose_key(c["subject_mention"]) != K.loose_key(c["object_mention"])
                and LEGAL_FORM.search(c["subject_mention"])):
            # 'Arm Limited' SUBSIDIARY_OF 'Arm Holdings plc': a filing names a distinct legal entity that normalisation
            # folded into its parent. Keep it as its own (provisional) company instead of losing the fact.
            s_id = resolver.separate_legal_entity(c["subject_mention"])
            ent(s_id, resolver.entities[s_id]["canonical_name"], "Company")
            s_how = "legal_entity_separated"
        reason = None
        if not s_id or not o_id:
            reason = s_how if not s_id else o_how
        elif s_id == o_id:
            reason = "self_loop_after_resolution"
        elif not onto.is_valid_triple_types(entities[s_id]["entity_type"], c["relation"], entities[o_id]["entity_type"]):
            reason = "domain_range_after_resolution"
        if reason:
            dropped.append({"claim_id": c["claim_id"], "reason": reason, "source": c["source"], "relation": c["relation"],
                            "subject_mention": c["subject_mention"], "object_mention": c["object_mention"],
                            "evidence_span": c["evidence_span"]})
            continue
        mention_counts[s_id][c["subject_mention"].strip()] += 1
        mention_counts[o_id][c["object_mention"].strip()] += 1
        resolved.append({"claim_id": c["claim_id"], "subject_entity_id": s_id, "relation": c["relation"],
                         "object_entity_id": o_id, "source": c["source"], "document_id": c["document_id"],
                         "resolution_score": round(min(s_sc, o_sc), 3), "evidence_span": c["evidence_span"],
                         "subject_mention": c["subject_mention"], "object_mention": c["object_mention"],
                         "publication_date": c.get("publication_date"), "extractor_version": c.get("extractor_version"),
                         "prompt_version": c.get("prompt_version"),
                         "_section": c.get("section"), "_doc_company": doc_cid})

    edges = fuse(resolved, entities, onto)
    scores = {e["edge_id"]: e.pop("_score") for e in edges}      # sidecar: FusedEdge allows only HIGH/MEDIUM/LOW

    # ---- canonical entities: aliases = observed mentions (+ registry aliases for companies)
    out_entities = []
    used = {r["subject_entity_id"] for r in resolved} | {r["object_entity_id"] for r in resolved} | resolver.registry_ids
    for eid in sorted(used):
        if eid in resolver.entities:
            e = resolver.entities[eid]
            base = {"entity_id": eid, "canonical_name": e["canonical_name"], "entity_type": "Company"}
            al = set(e["aliases"]) | set(mention_counts[eid])
            if e.get("ticker"):
                base["ticker"] = e["ticker"]
            if e.get("cik"):
                base["cik"] = str(e["cik"])
            base["description"] = "provisional (not in registry)" if e["provisional"] else None
        else:
            base = dict(entities[eid])
            al = set(mention_counts[eid])
        al.discard(base["canonical_name"])
        base["aliases"] = sorted(al, key=lambda a: (-mention_counts[eid][a], a))[:40]
        out_entities.append({k: v for k, v in base.items() if v not in (None, "")})

    # ---- validate against Person 1's contracts
    errors = Counter()
    for model, rows in ((schemas.CanonicalEntity, out_entities), (schemas.FusedEdge, edges)):
        for r in rows:
            try:
                model(**r)
            except Exception:
                errors[model.__name__] += 1
    resolved_out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in resolved]
    for r in resolved_out:
        try:
            schemas.ResolvedClaim(**r)
        except Exception:
            errors["ResolvedClaim"] += 1

    K.write_jsonl(K.KG_DIR / "edge_scores.jsonl",
                  [{"edge_id": e["edge_id"], "confidence_score": scores[e["edge_id"]], "confidence": e["confidence"],
                    "source_types": e["source_types"], "n_documents": e["source_count"], "n_claims": len(e["claim_ids"])}
                   for e in edges])
    documents = build_documents(sec_docs, doc_company, claims)
    K.write_jsonl(K.KG_DIR / "canonical_entities.jsonl", out_entities)
    K.write_jsonl(K.KG_DIR / "resolved_claims.jsonl", resolved_out)
    K.write_jsonl(K.KG_DIR / "fused_edges.jsonl", edges)
    K.write_jsonl(K.KG_DIR / "documents.jsonl", documents)
    K.write_jsonl(K.KG_DIR / "dropped_claims.jsonl", dropped)
    resolver.save_ledger()
    ledger.save()

    report = build_report(claims, wiki, sec, resolved, dropped, edges, out_entities, res_methods, errors,
                          contract_bad, family_mode, reg_path)
    report["confidence_profile"] = confidence_profile(edges, scores, {e["entity_id"]: e for e in out_entities})
    report["source_prior"] = SOURCE_PRIOR
    K.write_json(K.KG_DIR / "kg_build_report.json", report)
    (K.KG_DIR / "kg_build_report.md").write_text(report_md(report), encoding="utf-8")
    print(report_md(report))
    return report


def edge_key(r):
    s, rel, o = r["subject_entity_id"], r["relation"], r["object_entity_id"]
    if rel == "CUSTOMER_OF":
        s, rel, o = o, "SUPPLIES_TO", s
    if rel == "PARTNERS_WITH" and o < s:
        s, o = o, s
    return s, rel, o


def fuse(resolved, entities, onto):
    groups = defaultdict(list)
    for r in resolved:
        groups[edge_key(r)].append(r)
    edges = {}
    for (s, rel, o), rs in groups.items():
        docs = sorted({r["document_id"] for r in rs})
        stypes = sorted({r["source"] for r in rs})
        mentions = {(K.loose_key(r["subject_mention"]), K.loose_key(r["object_mention"])) for r in rs}
        sec_dates = sorted(r["publication_date"] for r in rs if r["source"] == "SEC" and r.get("publication_date"))
        if len(stypes) > 1:
            group = "SUPPORT"
        elif len(rs) == 1:
            group = "SINGLE_SOURCE"
        elif len(mentions) == 1:
            group = "DUPLICATE"
        else:
            group = "COMPLEMENTARY"           # same fact, different surface forms merged by resolution
        conf = "HIGH" if (len(stypes) > 1 or len(docs) >= 2) else "MEDIUM"
        core = (rel in onto.CORE_RELATIONS and entities[s]["entity_type"] in onto.CORE_ENTITY_TYPES
                and entities[o]["entity_type"] in onto.CORE_ENTITY_TYPES)
        e = {"edge_id": "edge_" + K.sha(s, rel, o), "subject": s, "relation": rel, "object": o, "sources": docs,
             "confidence": conf, "conflict": False, "source_types": stypes, "source_count": len(docs),
             "claim_ids": sorted(r["claim_id"] for r in rs), "group_type": group,
             "layer": "core" if core else "extended"}
        if sec_dates:
            e["first_seen"], e["last_seen"] = sec_dates[0], sec_dates[-1]
        e["_claims"] = rs
        edges[(s, rel, o)] = e

    # ---- one headquarters per company / one country per facility
    by_subject = defaultdict(list)
    for (s, rel, o), e in edges.items():
        if rel == "LOCATED_IN":
            by_subject[s].append(e)
    for s, es in by_subject.items():
        if len(es) < 2:
            continue

        def score(e):
            rs = e["_claims"]
            filer_says = any(r["source"] == "SEC" and r["_doc_company"] == s for r in rs)
            infobox = any(r["source"] == "WIKIPEDIA" and (r.get("_section") or "").lower() == "infobox" for r in rs)
            return 2.0 * filer_says + len({r["document_id"] for r in rs}) + 0.5 * infobox
        ranked = sorted(es, key=score, reverse=True)
        top = score(ranked[0])
        winners = [e for e in ranked if score(e) == top]
        for e in es:
            e["conflict"] = True
            e["conflict_type"] = "attribute"
            e["group_type"] = "CONFLICT"
            e["conflict_with"] = sorted(x["edge_id"] for x in es if x is not e)
            if e not in winners:
                e["confidence"] = "LOW"
            elif len(winners) > 1:
                e["confidence"] = "MEDIUM"

    # ---- ownership cycles (A subsidiary of B and B subsidiary of A)
    for (s, rel, o), e in edges.items():
        if rel == "SUBSIDIARY_OF" and (o, rel, s) in edges:
            e["conflict"] = True
            e["conflict_type"] = "relation"
            e["group_type"] = "CONFLICT"
            e["conflict_with"] = [edges[(o, rel, s)]["edge_id"]]
    out = []
    for e in edges.values():
        e["_score"] = edge_score(e)
        e.pop("_claims")
        out.append(e)
    out.sort(key=lambda e: (e["subject"], e["relation"], e["object"]))
    return out


def edge_score(e):
    """Numeric confidence in [0, 1]: noisy-OR over supporting documents. Each document contributes
    prior(source) x best resolution score of its claims; independent documents add up, repeats in one document don't.
    Conflict losers are halved, ties scaled by 0.75. A heuristic aggregate, NOT a measured probability (audit_edges.py
    checks how well it tracks human judgements)."""
    by_doc = {}
    for r in e["_claims"]:
        p = SOURCE_PRIOR.get(r["source"], 0.7) * float(r["resolution_score"])
        by_doc[r["document_id"]] = max(by_doc.get(r["document_id"], 0.0), p)
    miss = 1.0
    for p in by_doc.values():
        miss *= (1.0 - p)
    score = 1.0 - miss
    if e["conflict"] and e["confidence"] == "LOW":
        score *= 0.5
    elif e["conflict"] and e["relation"] == "LOCATED_IN" and e["confidence"] == "MEDIUM":
        score *= 0.75
    return round(score, 4)


def confidence_profile(edges, scores, entities_by_id):
    """Label shares and mean numeric score by source group, relation, object type and layer."""
    groups = defaultdict(list)
    for e in edges:
        src = "both" if len(e["source_types"]) > 1 else e["source_types"][0]
        otype = entities_by_id.get(e["object"], {}).get("entity_type", "?")
        for g in (f"source={src}", f"relation={e['relation']}", f"object_type={otype}", f"layer={e['layer']}"):
            groups[g].append(e)
    out = {}
    for g, es in sorted(groups.items()):
        c = Counter(x["confidence"] for x in es)
        out[g] = {"n": len(es), "mean_score": round(sum(scores[x["edge_id"]] for x in es) / len(es), 3),
                  "HIGH": round(c["HIGH"] / len(es), 3), "MEDIUM": round(c["MEDIUM"] / len(es), 3),
                  "LOW": round(c["LOW"] / len(es), 3)}
    return out


def build_documents(sec_docs, doc_company, claims):
    docs = {}
    for d in sec_docs:
        row = {"document_id": d["document_id"], "source": "sec", "company_id": doc_company.get(d["document_id"])}
        for k in ("source_url", "filing_date", "filing_type", "accession_number", "cik", "retrieved_at", "content_hash",
                  "ticker", "company"):
            if d.get(k):
                row[k] = d[k]
        row["publication_date"] = d.get("filing_date")
        docs[d["document_id"]] = row
    for c in claims:
        if c["document_id"] in docs:
            continue
        row = {"document_id": c["document_id"], "source": c["source"].lower(), "company_id": c.get("company_id"),
               "source_url": c.get("source_url"), "publication_date": c.get("publication_date"),
               "retrieved_at": c.get("retrieved_at")}
        if c["source"] == "WIKIPEDIA" and c.get("source_url") and "title=" in c["source_url"]:
            row["page_title"] = c["source_url"].split("title=")[1].split("&")[0].replace("_", " ")
        docs[c["document_id"]] = {k: v for k, v in row.items() if v}
    return sorted(docs.values(), key=lambda d: d["document_id"])


def degree_stats(edges, entities_by_id):
    deg = Counter()
    for e in edges:
        deg[e["subject"]] += 1
        deg[e["object"]] += 1
    out = {}
    for t in K.OFFERING_TYPES:
        ids = [i for i in deg if entities_by_id.get(i, {}).get("entity_type") == t]
        if ids:
            out[t] = {"nodes": len(ids), "degree_1": sum(deg[i] == 1 for i in ids),
                      "share_degree_1": round(sum(deg[i] == 1 for i in ids) / len(ids), 3),
                      "max_degree": max(deg[i] for i in ids)}
    return out


def build_report(claims, wiki, sec, resolved, dropped, edges, entities, res_methods, errors, contract_bad,
                 family_mode, reg_path):
    eb = {e["entity_id"]: e for e in entities}
    rel_src = defaultdict(Counter)
    for e in edges:
        key = "both" if len(e["source_types"]) > 1 else e["source_types"][0]
        rel_src[e["relation"]][key] += 1
    funnel = {}
    for name, cl in (("WIKIPEDIA", wiki), ("SEC", sec)):
        ids = {c["claim_id"] for c in cl}
        funnel[name] = {"delivered": len(cl), "resolved": sum(r["claim_id"] in ids for r in resolved),
                        "dropped": sum(d["claim_id"] in ids for d in dropped),
                        "edges_supported": sum(name in e["source_types"] for e in edges),
                        "core_provides_edges": sum(name in e["source_types"] and e["relation"] == "PROVIDES"
                                                   and e["layer"] == "core" for e in edges)}
    return {
        "family_mode": family_mode, "registry": str(reg_path),
        "contract_A_violations_in_input": len(contract_bad),
        "contract_violations_in_output": dict(errors),
        "funnel": funnel,
        "dropped_reasons": dict(Counter(d["reason"] for d in dropped)),
        "resolution_methods": dict(res_methods),
        "entities_by_type": dict(Counter(e["entity_type"] for e in entities)),
        "provisional_companies": sum(1 for e in entities if e.get("description") == "provisional (not in registry)"),
        "edges": len(edges),
        "edges_by_relation_and_source": {k: dict(v) for k, v in sorted(rel_src.items())},
        "group_types": dict(Counter(e["group_type"] for e in edges)),
        "confidence": dict(Counter(e["confidence"] for e in edges)),
        "layer": dict(Counter(e["layer"] for e in edges)),
        "conflict_edges": sum(e["conflict"] for e in edges),
        "multi_source_edges": sum(len(e["source_types"]) > 1 for e in edges),
        "edges_with_observation_dates": sum(1 for e in edges if e.get("first_seen")),
        "offering_degree": degree_stats(edges, eb),
    }


def report_md(r):
    lines = [f"# KG build report ({'product families' if r['family_mode'] else 'mention-level offerings'})", ""]
    lines.append(f"Contract violations in output: {r['contract_violations_in_output'] or 'none'}")
    lines.append("")
    lines.append("| Source | Delivered | Resolved | Dropped | Edges supported | Core PROVIDES |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for s, f in r["funnel"].items():
        lines.append(f"| {s} | {f['delivered']} | {f['resolved']} | {f['dropped']} | {f['edges_supported']} | {f['core_provides_edges']} |")
    lines += ["", f"Dropped: {r['dropped_reasons']}", f"Entities: {r['entities_by_type']} "
              f"(provisional companies: {r['provisional_companies']})", f"Edges: {r['edges']}; multi-source "
              f"{r['multi_source_edges']}; conflicts {r['conflict_edges']}; dated {r['edges_with_observation_dates']}",
              f"Group types: {r['group_types']}", f"Confidence: {r['confidence']}", f"Layer: {r['layer']}", "",
              "| Relation | Wikipedia only | SEC only | Both |", "|---|---:|---:|---:|"]
    for rel, c in r["edges_by_relation_and_source"].items():
        lines.append(f"| {rel} | {c.get('WIKIPEDIA', 0)} | {c.get('SEC', 0)} | {c.get('both', 0)} |")
    if r.get("confidence_profile"):
        lines += ["", "Edge confidence profile (label shares; mean numeric score, noisy-OR over documents, "
                      f"priors {r.get('source_prior')}):", "",
                  "| Group | n | HIGH | MEDIUM | LOW | mean score |", "|---|---:|---:|---:|---:|---:|"]
        for g, d in r["confidence_profile"].items():
            lines.append(f"| {g} | {d['n']} | {d['HIGH']:.0%} | {d['MEDIUM']:.0%} | {d['LOW']:.0%} | {d['mean_score']:.2f} |")
    lines += ["", "Offering nodes with exactly one edge:"]
    for t, d in r["offering_degree"].items():
        lines.append(f"- {t}: {d['degree_1']} of {d['nodes']} ({d['share_degree_1']:.0%}), max degree {d['max_degree']}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
