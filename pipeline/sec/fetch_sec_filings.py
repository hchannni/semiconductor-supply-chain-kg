"""
Step 1 of the SEC pipeline (Person 2): CIK mapping + raw filing download, built to scale to 10-50+ companies.

Company list: reads data/company_metadata.csv (or company_metadata.csv next to this script's folder root).
Accepted columns (case-insensitive): Symbol or Ticker (required), Name, Rank. Only rows with Rank <= MAX_RANK are used.
Companies not on EDGAR (e.g. 005930.KS) are not an error: they are listed in sec_coverage.csv as not_in_sec_ticker_file.

How many filings per company: the N most recent 10-K / 20-F filings. FULL_HISTORY_FILINGS (default 3) for the
companies in FULL_HISTORY_TICKERS (the pilot set), DEFAULT_FILINGS (default 1) for everyone else.

Writes (in data/raw/sec/):
  sec_documents.jsonl   one line per downloaded filing (document_id, company, cik, filing_type, accession_number,
                        filing_date, source_url, retrieved_at, content_hash ...)  - append-only, safe to re-run
  sec_coverage.csv      one line per company: ok / not_in_sec_ticker_file / no_eligible_filings / error
BEFORE RUNNING: put your real name + email in USER_AGENT (SEC blocks anonymous clients). Keep your existing value.
"""
import os
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

# ---------------- configuration ----------------
USER_AGENT = os.environ.get("SEC_USER_AGENT", "Your Name your.email@example.com")   # SEC requires a real contact
ROOT = Path(os.environ.get("KG_ROOT", Path(__file__).resolve().parent / "workdir"))   # set KG_ROOT to your working folder
# Team list first (Person 1: data/companies/companies.csv, columns ticker + cik_hint), else the CompaniesMarketCap download.
COMPANY_TABLE_CANDIDATES = [ROOT / "data/companies/companies.csv", ROOT / "data/company_metadata.csv", ROOT / "company_metadata.csv"]
FALLBACK_TICKERS = ["NVDA", "TSM"]                 # used only if no company table is found
MAX_RANK = 100                                     # top-N by market cap; NOT applied to the team list (companies.csv), which is already curated
# The table gives each company's HOME-market ticker. Where a US ticker exists on EDGAR, map it here.
# SK hynix: ADSs listed on Nasdaq as SKHY in 2026, but it has not yet filed its first annual report (20-F).
SEC_TICKER_OVERRIDES = {"000660.KS": "SKHY"}
FORMS = {"10-K", "20-F"}                           # amendments (10-K/A) and 40-F deliberately excluded for now
DEFAULT_FILINGS = 1                                # latest filing only for most companies
FULL_HISTORY_FILINGS = 3                           # latest 3 for the pilot set (temporal comparison, redundancy)
# Always included, whatever the company table: team-list companies outside the CompaniesMarketCap top 100.
EXTRA_TICKERS = ["CDNS", "ACLS", "VECO", "PLAB"]
FULL_HISTORY_TICKERS = {"TSM", "NVDA", "ASML", "AMD", "INTC", "MU", "TXN", "QCOM", "AVGO", "AMAT"}   # the team pilot (Person 1)
OUT_DIR = ROOT / "data/raw/sec"
DELAY = 0.2                                        # SEC limit is 10 requests/sec; stay well under

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/{name}"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc_nodash}/{doc}"


def get(session, url):
    time.sleep(DELAY)
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp


def load_ticker_to_cik(session):
    """company_tickers.json looks like {"0": {"cik_str": 1045810, "ticker": "NVDA", "title": "..."}, ...}"""
    data = get(session, TICKER_MAP_URL).json()
    return {row["ticker"].upper(): (int(row["cik_str"]), row["title"]) for row in data.values()}


def find_company_table(candidates=COMPANY_TABLE_CANDIDATES):
    return next((Path(p) for p in candidates if Path(p).exists()), None)


def load_companies(table_path=None, max_rank=MAX_RANK, fallback=FALLBACK_TICKERS):
    """Return [{'symbol','name','rank'}] in table order, de-duplicated, filtered to rank <= max_rank."""
    table_path = table_path or find_company_table()
    if table_path is None:
        print(f"[info] no company table found, using fallback tickers {fallback}")
        return [{"symbol": t, "name": "", "rank": None} for t in fallback]
    with open(table_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        find = lambda *names: next((h for h in (reader.fieldnames or []) if h.strip().lower() in names), None)
        tcol, ncol, rcol = find("symbol", "ticker"), find("name", "company", "company_name"), find("rank")
        ccol = find("cik_hint", "cik")
        if Path(table_path).name == "companies.csv":
            max_rank = None                         # the team list is curated: take every row
        if tcol is None:
            raise SystemExit(f"{table_path} needs a 'Symbol' or 'Ticker' column; found {reader.fieldnames}")
        seen, out = set(), []
        for row in reader:
            sym = (row[tcol] or "").strip().upper()
            rank = int(float(row[rcol])) if rcol and (row[rcol] or "").strip() else None
            if not sym or sym in seen or (max_rank is not None and rank is not None and rank > max_rank):
                continue
            seen.add(sym)
            cik = (row[ccol] or "").strip() if ccol else ""
            out.append({"symbol": sym, "name": (row[ncol] or "").strip() if ncol else "", "rank": rank,
                        "cik_hint": int(float(cik)) if cik.replace(".", "", 1).isdigit() else None})
    for t in EXTRA_TICKERS:
        if t.upper() not in seen:
            seen.add(t.upper())
            out.append({"symbol": t.upper(), "name": "", "rank": None, "cik_hint": None})
    print(f"[info] company table: {table_path} ({len(out)} companies, incl. EXTRA_TICKERS)")
    return out


def select_latest(sub, fetch_json, forms, n):
    """The n most recent filings of the wanted form types. Older submission pages are fetched only if needed."""
    found = []

    def take(b):
        for acc, form, fdate, doc, rdate in zip(
            b["accessionNumber"], b["form"], b["filingDate"], b["primaryDocument"], b["reportDate"]
        ):
            if form in forms:
                found.append({"accession": acc, "form": form, "filing_date": fdate, "primary_doc": doc, "report_date": rdate})

    take(sub["filings"]["recent"])
    for extra in sub["filings"].get("files", []):
        if len(found) >= n:
            break
        take(fetch_json(extra["name"]))
    found.sort(key=lambda r: r["filing_date"], reverse=True)
    return found[:n]


def load_existing(meta_path):
    existing = {}
    if meta_path.exists():
        for line in meta_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                existing[rec["accession_number"]] = rec
    return existing


def process_company(session, company, ticker_map, existing, next_id, meta_path, out_dir):
    """Download one company's latest filings. Returns (coverage_row, next_id). Never raises."""
    symbol = company["symbol"]
    sec_ticker = SEC_TICKER_OVERRIDES.get(symbol, symbol).upper()
    n = FULL_HISTORY_FILINGS if (symbol in FULL_HISTORY_TICKERS or sec_ticker in FULL_HISTORY_TICKERS) else DEFAULT_FILINGS
    row = {"rank": company.get("rank"), "name": company.get("name", ""), "symbol": symbol, "sec_ticker": sec_ticker, "cik": "",
           "status": "", "eligible_filings": 0, "new_downloads": 0, "failed_downloads": 0}
    if sec_ticker in ticker_map:
        cik, title = ticker_map[sec_ticker]
    elif company.get("cik_hint"):
        cik, title = company["cik_hint"], company.get("name") or sec_ticker      # ticker unknown to SEC's file, CIK given by the team list
    else:
        row["status"] = "not_in_sec_ticker_file"
        return row, next_id
    row["cik"] = f"{cik:010d}"
    try:
        sub = get(session, SUBMISSIONS_URL.format(name=f"CIK{cik:010d}.json")).json()
        fetch_json = lambda name: get(session, SUBMISSIONS_URL.format(name=name)).json()
        rows = select_latest(sub, fetch_json, FORMS, n)
    except Exception as e:
        row["status"] = f"error: {str(e)[:120]}"
        return row, next_id
    row["eligible_filings"] = len(rows)
    row["status"] = "ok" if rows else "no_eligible_filings"
    for r in rows:
        if r["accession"] in existing:
            continue
        url = ARCHIVE_URL.format(cik=cik, acc_nodash=r["accession"].replace("-", ""), doc=r["primary_doc"])
        try:
            content = get(session, url).content
        except Exception as e:
            row["failed_downloads"] += 1
            print(f"  [fail] {sec_ticker} {r['form']} {r['filing_date']}: {str(e)[:100]}")
            continue
        suffix = Path(r["primary_doc"]).suffix or ".htm"
        save_path = out_dir / sec_ticker / f"{r['accession']}{suffix}"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(content)
        rec = {
            "document_id": f"sec_{next_id:05d}", "source": "SEC", "company": title, "ticker": sec_ticker,
            "cik": f"{cik:010d}", "filing_type": r["form"], "accession_number": r["accession"],
            "filing_date": r["filing_date"], "report_date": r["report_date"], "source_url": url,
            "local_path": str(save_path), "retrieved_at": datetime.now(timezone.utc).isoformat(),
            "content_hash": hashlib.sha256(content).hexdigest(),
        }
        next_id += 1
        existing[r["accession"]] = rec
        with meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        row["new_downloads"] += 1
        print(f"  saved {rec['document_id']} {sec_ticker} {r['form']} {r['filing_date']}")
    return row, next_id


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = OUT_DIR / "sec_documents.jsonl"
    existing = load_existing(meta_path)
    next_id = len(existing) + 1
    companies = load_companies()
    ticker_map = load_ticker_to_cik(session)

    coverage = []
    for c in companies:
        row, next_id = process_company(session, c, ticker_map, existing, next_id, meta_path, OUT_DIR)
        print(f"{c['symbol']}: {row['status']} ({row['eligible_filings']} filings selected, {row['new_downloads']} new)")
        coverage.append(row)

    cols = ["rank", "name", "symbol", "sec_ticker", "cik", "status", "eligible_filings", "new_downloads", "failed_downloads"]
    with open(OUT_DIR / "sec_coverage.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(coverage)
    ok = sum(r["status"] == "ok" for r in coverage)
    print(f"\nSEC coverage: {ok}/{len(coverage)} companies have eligible filings -> sec_coverage.csv")


if __name__ == "__main__":
    main()
