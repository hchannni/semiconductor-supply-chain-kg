"""
Builds an Excel sheet of randomly sampled claims for HUMAN checking. (Excel, not CSV: the evidence text contains quotation marks
and commas, which corrupt CSV files when they are edited and re-saved.) The 'label' column has a dropdown.

Why: the validator only proves that the quoted sentence exists in the filing. It cannot tell whether
the claim is TRUE to that sentence. You read a sample and label each claim; the share labelled
'correct' is the precision of the SEC extractor, the number your project report needs.

Sampling: 10 claims from each of 3 groups, reported separately because they answer different questions:
company_edge (SUPPLIES_TO / PARTNERS_WITH / SUBSIDIARY_OF), provides (PROVIDES, the biggest group), other (USES, LOCATED_IN, ...).
Seeded, so re-running gives the same sample.

Labels for the 'label' column (one per row):
  correct          the quoted sentence states this relation, direction and entity types are right
  wrong_direction  relation is right but subject and object are swapped
  wrong_relation   entities are right but a different relation is the correct one (e.g. USES should be PROVIDES)
  unsupported      the sentence does not actually state it (the model inferred it)
  bad_entity       a mention is not a clean entity (a list, too vague) or has the wrong type
  other            anything else - explain in 'notes'
'evidence_group': rows with the same number come from the same sentence (so labels will often match).
"""
import os
import csv
import json
import random
from pathlib import Path

ROOT = Path(os.environ.get("KG_ROOT", Path(__file__).resolve().parent / "workdir"))   # set KG_ROOT to your working folder
CLAIMS_PATH = ROOT / "data/processed/sec_claims.jsonl"
N_PER_STRATUM = 10
SEED = 21                                 # change for a fresh sample; the file name includes it
OUT_PATH = ROOT / f"data/processed/sec_audit_sheet_seed{SEED}.xlsx"
EDGE_RELATIONS = {"SUPPLIES_TO", "PARTNERS_WITH", "SUBSIDIARY_OF", "CUSTOMER_OF"}


def build_rows(claims, n_per_stratum=N_PER_STRATUM, seed=SEED):
    rng = random.Random(seed)
    groups = {}
    for c in claims:
        groups.setdefault(" ".join(c["evidence_span"].split()).lower(), len(groups) + 1)
    strata = {"company_edge": [c for c in claims if c["relation"] in EDGE_RELATIONS],
              "provides": [c for c in claims if c["relation"] == "PROVIDES"],
              "other": [c for c in claims if c["relation"] not in EDGE_RELATIONS and c["relation"] != "PROVIDES"]}
    rows = []
    for name, pool in strata.items():
        for c in rng.sample(pool, min(n_per_stratum, len(pool))):
            rows.append({
                "stratum": name, "claim_id": c["claim_id"], "evidence_group": groups[" ".join(c["evidence_span"].split()).lower()],
                "subject": c["subject_mention"], "relation": c["relation"], "object": c["object_mention"],
                "subject_type": c["subject_type"], "object_type": c["object_type"],
                "evidence_span": c["evidence_span"], "label": "", "notes": ""})
    rng.shuffle(rows)                      # mix strata so you are not primed by long runs of similar claims
    for i, r in enumerate(rows, 1):
        r["audit_id"] = i
    return rows


def main(claims_path=CLAIMS_PATH, out_path=OUT_PATH):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font
    from openpyxl.worksheet.datavalidation import DataValidation
    claims = [json.loads(l) for l in Path(claims_path).read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = build_rows(claims)
    cols = ["audit_id", "stratum", "claim_id", "evidence_group", "subject", "relation", "object",
            "subject_type", "object_type", "evidence_span", "label", "notes"]
    wb = Workbook()
    ws = wb.active
    ws.title = "audit"
    ws.append(cols)
    for r in rows:
        ws.append([r[c] for c in cols])
    widths = {"audit_id": 8, "stratum": 14, "claim_id": 30, "evidence_group": 9, "subject": 30, "relation": 16, "object": 30,
              "subject_type": 12, "object_type": 12, "evidence_span": 80, "label": 16, "notes": 50}
    for i, c in enumerate(cols, 1):
        ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = widths[c]
        ws.cell(row=1, column=i).font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    label_col = ws.cell(row=1, column=cols.index("label") + 1).column_letter
    dv = DataValidation(type="list", formula1='"correct,wrong_direction,wrong_relation,unsupported,bad_entity,other"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{label_col}2:{label_col}{len(rows) + 1}")
    wb.save(out_path)
    print(f"{len(rows)} claims written to {out_path}")
    print("Pick a value in the 'label' column (dropdown): correct / wrong_direction / wrong_relation / unsupported / bad_entity / other")


if __name__ == "__main__":
    main()
