#!/usr/bin/env python3
"""
Build Trigger Monitor demo data from SEC 10-D filings (Exhibit 99.1).

Outputs a JSON payload shaped like the MOCK data in TriggerMonitorWebsiteDemo.jsx:
{
  "asOf": "YYYY-MM-DD",
  "portfolio": {...},
  "deals": [...],
  "alerts": [...]
}

Usage:
  python3 scripts/sec_trigger_demo_builder.py \
    --config scripts/sec_demo_deals.json \
    --months 18 \
    --out out/trigger_monitor_demo.json \
    --user-agent "Your Name your@email.com"
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional

import requests
from io import StringIO, BytesIO

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit("pandas is required. Install with: pip install pandas lxml") from exc


SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_nodash}/"
SEC_INDEX_JSON = SEC_ARCHIVES_BASE + "index.json"
SEC_DOC_URL = SEC_ARCHIVES_BASE + "{filename}"
SEC_INDEX_HTML = SEC_ARCHIVES_BASE + "index.htm"
SEC_INDEX_HTML_ALT = SEC_ARCHIVES_BASE + "index.html"
NYFED_HHDC_BACKGROUND = "https://www.newyorkfed.org/microeconomics/hhdc/background.html"
NYFED_HHDC_XLS_BASE = "https://www.newyorkfed.org/medialibrary/interactives/householdcredit/data/xls/"

MONTH_FMT = "%Y-%m-%d"

METRIC_DEFS = [
    {
        "key": "pool_balance",
        "patterns": [
            r"receivables?\s+pool\s+balance",
            r"pool\s+balance",
            r"outstanding\s+balance",
        ],
        "prefer_percent": False,
    },
    {
        "key": "total_delinquency",
        "patterns": [
            r"total\s+delinquenc(?:y|ies)\s+(?:rate|ratio|percentage|%)",
            r"delinquency\s+rate",
        ],
        "prefer_percent": True,
    },
    {
        "key": "delinquency_60_plus",
        "patterns": [
            r"60\s*[-+]\s*day.*(?:delinq|percent)",  # "60-Day Delinquency" or "60+ Day"
            r"61\s*\+\s*day",
            r"60\+",
            r"60\s+days?\s+(?:or\s+)?(?:more|greater)",
        ],
        "prefer_percent": True,
    },
    {
        "key": "cumulative_loss_ratio",
        "patterns": [
            r"cumulative\s+net\s+loss\s+ratio",
            r"cumulative\s+loss\s+ratio",
            r"cumulative\s+net\s+loss",
            r"cumulative\s+losses",
        ],
        "prefer_percent": True,
    },
]


@dataclass
class FilingDoc:
    period_end: str
    accession_no: str
    ex99_url: str


def sec_headers(user_agent: str, host: str) -> Dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": host,
    }


def fetch_json(session: requests.Session, url: str, headers: Dict[str, str], sleep: float = 0.2) -> dict:
    resp = session.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    time.sleep(sleep)
    return resp.json()


def fetch_text(session: requests.Session, url: str, headers: Dict[str, str], sleep: float = 0.2) -> str:
    resp = session.get(url, headers=headers, timeout=60)
    resp.raise_for_status()
    time.sleep(sleep)
    return resp.text


def yyyymmdd_from_period(period: str) -> str:
    return f"{period[0:4]}-{period[4:6]}-{period[6:8]}"


def normalize_period(period: Optional[str], filing_date: Optional[str]) -> Optional[str]:
    if period:
        period = period.strip()
        if re.fullmatch(r"\d{8}", period):
            return yyyymmdd_from_period(period)
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
            return period
    if filing_date:
        filing_date = filing_date.strip()
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}", filing_date):
            return filing_date
    return None


def list_recent_10d_ex99(
    session: requests.Session,
    cik: str,
    months: int,
    user_agent: str,
    debug: bool = False,
) -> List[FilingDoc]:
    submissions = fetch_json(
        session,
        SEC_SUBMISSIONS.format(cik=cik),
        headers=sec_headers(user_agent, "data.sec.gov"),
    )
    recent = submissions.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accession = recent.get("accessionNumber", [])
    periods = recent.get("periodOfReport", [])
    filing_dates = recent.get("filingDate", [])

    if debug:
        total_recent = len(forms)
        ten_d_forms = {"10-D", "10-D/A"}
        ten_d_indices = [i for i, f in enumerate(forms) if f in ten_d_forms]
        print(f"  Recent filings: {total_recent} (10-D/10-D/A: {len(ten_d_indices)})")
        for i in ten_d_indices[:5]:
            per = periods[i] if i < len(periods) else "n/a"
            acc = accession[i] if i < len(accession) else "n/a"
            print(f"    sample 10-D: {forms[i]} {acc} period {per}")

    cutoff = (dt.date.today().replace(day=1) - dt.timedelta(days=months * 31))
    docs: List[FilingDoc] = []

    count = min(len(forms), len(accession))
    for i in range(count):
        form = forms[i]
        acc = accession[i]
        per = periods[i] if i < len(periods) else None
        filing_date = filing_dates[i] if i < len(filing_dates) else None
        if form not in {"10-D", "10-D/A"}:
            continue
        per_iso = normalize_period(per, filing_date)
        if not per_iso:
            if debug:
                print(f"  - {acc} -> skipped (no period/filing date)")
            continue
        per_date = dt.date.fromisoformat(per_iso)
        if per_date < cutoff:
            if debug:
                print(f"  - {per_iso} {acc} -> skipped (before cutoff)")
            continue

        acc_nodash = acc.replace("-", "")
        cik_int = int(cik)
        idx = fetch_json(
            session,
            SEC_INDEX_JSON.format(cik_int=cik_int, acc_no_nodash=acc_nodash),
            headers=sec_headers(user_agent, "www.sec.gov"),
        )
        files = [item["name"] for item in idx.get("directory", {}).get("item", [])]
        ex99_candidates = [f for f in files if re.search(r"(ex-?99\.1|dex991|exhibit99-1|ex991)", f, re.I)]
        ex99_source = "filename"
        primary_doc = None
        if not ex99_candidates:
            ex99_candidates, primary_doc = find_docs_from_index_html(
                session,
                cik_int=cik_int,
                acc_no_nodash=acc_nodash,
                user_agent=user_agent,
                debug=debug,
            )
            ex99_source = "index_html"
        if not ex99_candidates and primary_doc:
            ex99_candidates = find_ex99_from_primary_doc(
                session,
                cik_int=cik_int,
                acc_no_nodash=acc_nodash,
                primary_doc=primary_doc,
                user_agent=user_agent,
                debug=debug,
            )
            ex99_source = "primary_doc"
        if not ex99_candidates:
            if debug:
                sample_files = ", ".join(files[:10])
                print(f"  - {per_iso} {acc} -> no Exhibit 99.1 candidates (files: {sample_files})")
            continue

        ex99 = pick_ex99_candidate(ex99_candidates)
        ex99_url = SEC_DOC_URL.format(cik_int=cik_int, acc_no_nodash=acc_nodash, filename=ex99)
        if debug:
            print(f"  - {per_iso} {acc} -> using {ex99} ({ex99_source})")
        docs.append(FilingDoc(period_end=per_iso, accession_no=acc, ex99_url=ex99_url))

    uniq = {}
    for d in docs:
        uniq[d.period_end] = d
    docs_sorted = [uniq[k] for k in sorted(uniq.keys())]
    if debug:
        print(f"  Found {len(docs_sorted)} Exhibit 99.1 docs")
        for d in docs_sorted:
            print(f"    {d.period_end} {d.accession_no} {d.ex99_url}")
    return docs_sorted


def find_docs_from_index_html(
    session: requests.Session,
    cik_int: int,
    acc_no_nodash: str,
    user_agent: str,
    debug: bool = False,
) -> tuple[List[str], Optional[str]]:
    urls = [
        SEC_INDEX_HTML.format(cik_int=cik_int, acc_no_nodash=acc_no_nodash),
        SEC_INDEX_HTML_ALT.format(cik_int=cik_int, acc_no_nodash=acc_no_nodash),
    ]
    for url in urls:
        try:
            html = fetch_text(session, url, headers=sec_headers(user_agent, "www.sec.gov"), sleep=0.1)
        except Exception:
            continue

        try:
            tables = pd.read_html(html)
        except ValueError:
            tables = []

        ex99_candidates: List[str] = []
        primary_doc: Optional[str] = None
        for df in tables:
            if df.empty:
                continue
            cols = [str(c).strip().lower() for c in df.columns]
            if "type" not in cols or "document" not in cols:
                continue
            type_col = df.columns[cols.index("type")]
            doc_col = df.columns[cols.index("document")]
            for _, row in df.iterrows():
                type_val = str(row.get(type_col, "")).upper()
                if "EX-99.1" in type_val:
                    doc_val = str(row.get(doc_col, "")).strip()
                    match = re.search(r"[\w.\-]+?\.(?:htm|html|pdf|xml|txt)", doc_val, re.I)
                    if match:
                        ex99_candidates.append(match.group(0))
                if type_val in {"10-D", "10-D/A"} and not primary_doc:
                    doc_val = str(row.get(doc_col, "")).strip()
                    match = re.search(r"[\w.\-]+?\.(?:htm|html|txt)", doc_val, re.I)
                    if match:
                        primary_doc = match.group(0)

        if ex99_candidates or primary_doc:
            if debug:
                if ex99_candidates:
                    print(f"    index.htm EX-99.1 -> {', '.join(ex99_candidates)}")
                if primary_doc:
                    print(f"    index.htm primary -> {primary_doc}")
            return ex99_candidates, primary_doc

    return [], None


def find_ex99_from_primary_doc(
    session: requests.Session,
    cik_int: int,
    acc_no_nodash: str,
    primary_doc: str,
    user_agent: str,
    debug: bool = False,
) -> List[str]:
    url = SEC_DOC_URL.format(cik_int=cik_int, acc_no_nodash=acc_no_nodash, filename=primary_doc)
    try:
        html = fetch_text(session, url, headers=sec_headers(user_agent, "www.sec.gov"), sleep=0.1)
    except Exception:
        return []

    candidates: List[str] = []
    anchor_re = re.compile(r"<a[^>]+href=[\"']?([^\"'>\s]+)[^>]*>(.*?)</a>", re.I | re.S)
    label_re = re.compile(r"(ex-?99\.1|exhibit\s*99\.1|99\.1)", re.I)
    for href, text in anchor_re.findall(html):
        if not label_re.search(text) and not label_re.search(href):
            continue
        href = href.split("#", 1)[0].split("?", 1)[0]
        if href.lower().startswith(("http://", "https://")):
            filename = href.rstrip("/").split("/")[-1]
        else:
            filename = href.split("/")[-1]
        if re.search(r"\.(?:htm|html|pdf|xml|txt)$", filename, re.I):
            candidates.append(filename)
    if debug and candidates:
        print(f"    primary doc EX-99.1 -> {', '.join(candidates)}")
    return candidates


def pick_ex99_candidate(candidates: List[str]) -> str:
    html = [c for c in candidates if c.lower().endswith((".htm", ".html"))]
    if html:
        return sorted(html, key=len)[0]
    return sorted(candidates, key=len)[0]


def normalize_number(token: str, prefer_percent: bool) -> Optional[float]:
    s = token.strip()
    if not s:
        return None
    neg = "(" in s and ")" in s
    s2 = re.sub(r"[^0-9.\-%]", "", s)
    if not s2:
        return None
    is_pct = s2.endswith("%")
    if is_pct:
        s2 = s2[:-1]
    try:
        val = float(s2)
    except ValueError:
        return None
    
    # Sanity check: if prefer_percent is True and value is implausibly large (>100 without %), reject it
    if prefer_percent and not is_pct and val > 100:
        return None
    
    if is_pct:
        val = val / 100.0
    elif prefer_percent:
        # Heuristic: many reports omit the % sign for ratios (e.g., "0.30" meaning 0.30%).
        # Treat values >= 0.1 as percent-style unless they're implausibly large.
        if 0 <= val < 0.1:
            pass
        elif 0.1 <= val <= 100:
            val = val / 100.0
        else:
            return None
    return -val if neg else val


def choose_metric_value(raw_tokens: List[str], prefer_percent: bool) -> Optional[float]:
    if not raw_tokens:
        return None
    tokens_with_pct = [t for t in raw_tokens if "%" in t]
    if prefer_percent:
        if tokens_with_pct:
            return normalize_number(tokens_with_pct[-1], prefer_percent=True)
        # prefer ratio-like tokens if present
        for t in reversed(raw_tokens):
            val = normalize_number(t, prefer_percent=True)
            if val is not None:
                return val
        return None
    # non-percent metrics (pool balance): prefer non-percent tokens
    tokens_no_pct = [t for t in raw_tokens if "%" not in t]
    for t in reversed(tokens_no_pct or raw_tokens):
        val = normalize_number(t, prefer_percent=False)
        if val is not None:
            return val
    return None


def extract_metrics_from_tables(tables: List[pd.DataFrame], debug: bool = False) -> Dict[str, float]:
    found: Dict[str, float] = {}
    for df in tables:
        if df.empty:
            continue
        df2 = df.copy()
        df2 = df2.astype(str)
        
        # Try to identify label and value columns
        # Usually first column is labels, subsequent columns are values
        for i in range(len(df2)):
            row_data = df2.iloc[i]
            
            # Check if this is a header row (contains lots of text, few numbers)
            row_text = " ".join(row_data.tolist()).lower()
            num_count = len(re.findall(r'\d+', row_text))
            text_len = len(re.sub(r'[\d\s\.\,\%\$\(\)\-]', '', row_text))
            if text_len > 50 and num_count < 3:  # Likely a header
                continue
            
            # Get label (usually first column)
            label = str(row_data.iloc[0]).lower() if len(row_data) > 0 else ""
            
            # Get numeric values from remaining columns
            value_cells = row_data.iloc[1:] if len(row_data) > 1 else row_data
            tokens = []
            for cell in value_cells:
                cell_str = str(cell)
                # Extract numbers with optional % and ()
                cell_tokens = re.findall(r"[\(\-]?\$?\s*[\d,]+(?:\.\d+)?\s*%?\s*[\)]?", cell_str)
                tokens.extend(cell_tokens)
            
            if not tokens:
                continue
                
            # Match against metric patterns
            for metric in METRIC_DEFS:
                key = metric["key"]
                if key in found:
                    continue
                if any(re.search(p, label, re.I) for p in metric["patterns"]):
                    val = choose_metric_value(tokens, metric["prefer_percent"])
                    if val is not None:
                        if debug:
                            print(f"    DEBUG: {key} matched in '{label.strip()[:60]}' -> tokens {tokens[:5]} -> value {val}")
                        found[key] = val
    return found


def stddev(values: List[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return math.sqrt(var)


def normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


SCORE_WEIGHTS = {
    "cushion": 0.6,
    "trend3m": 0.2,
    "vol6m": 0.1,
    "macro": 0.1,
}
SCORE_SCALES = {
    "cushion_full": 0.20,
    "trend3m_full": 0.01,
    "vol6m_full": 0.005,
}


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def rule_based_score(
    cushion: float,
    change3m: float,
    vol6m: float,
    direction: str,
    macro_percentile: float,
) -> tuple[float, Dict[str, float]]:
    if cushion <= 0:
        return 1.0, {
            "cushion": 1.0,
            "trend3m": 0.0,
            "vol6m": 0.0,
            "macro": clamp01((macro_percentile - 0.5) / 0.5),
        }

    cushion_risk = 1.0 - min(cushion / SCORE_SCALES["cushion_full"], 1.0)
    adverse_change = change3m if direction == "<=" else -change3m
    trend_risk = clamp01(adverse_change / SCORE_SCALES["trend3m_full"])
    vol_risk = clamp01(vol6m / SCORE_SCALES["vol6m_full"])
    macro_risk = clamp01((macro_percentile - 0.5) / 0.5)

    score = (
        SCORE_WEIGHTS["cushion"] * cushion_risk
        + SCORE_WEIGHTS["trend3m"] * trend_risk
        + SCORE_WEIGHTS["vol6m"] * vol_risk
        + SCORE_WEIGHTS["macro"] * macro_risk
    )
    return clamp01(score), {
        "cushion": clamp01(cushion_risk),
        "trend3m": trend_risk,
        "vol6m": vol_risk,
        "macro": macro_risk,
    }


def breach_probability(
    current: float,
    threshold: float,
    direction: str,
    mean_change: float,
    vol_change: float,
    months: int,
) -> float:
    if threshold == 0 or vol_change == 0:
        if direction == "<=":
            return 1.0 if current > threshold else 0.0
        return 1.0 if current < threshold else 0.0

    mu = current + mean_change * months
    sigma = vol_change * math.sqrt(months)
    if sigma == 0:
        return 0.0
    if direction == "<=":
        z = (threshold - mu) / sigma
        return max(0.0, min(1.0, 1.0 - normal_cdf(z)))
    z = (threshold - mu) / sigma
    return max(0.0, min(1.0, normal_cdf(z)))


def percent_rank(values: List[float], current: float) -> float:
    if not values:
        return 0.5
    if min(values) == max(values):
        return 0.5
    sorted_vals = sorted(values)
    rank = sum(1 for v in sorted_vals if v <= current)
    return max(0.0, min(1.0, rank / len(sorted_vals)))


def parse_quarter_token(token: str) -> Optional[dt.date]:
    if not token:
        return None
    s = str(token).strip().upper()
    m = re.search(r"(\d{4})\s*Q([1-4])", s)
    if not m:
        m = re.search(r"Q([1-4])\s*(\d{4})", s)
        if m:
            year = int(m.group(2))
            q = int(m.group(1))
        else:
            return None
    else:
        year = int(m.group(1))
        q = int(m.group(2))
    month = q * 3
    day = (dt.date(year, month, 1) + dt.timedelta(days=32)).replace(day=1) - dt.timedelta(days=1)
    return day


def parse_date_series(values: List[object]) -> List[Optional[dt.date]]:
    parsed: List[Optional[dt.date]] = []
    for v in values:
        if isinstance(v, dt.date):
            parsed.append(v)
            continue
        if isinstance(v, dt.datetime):
            parsed.append(v.date())
            continue
        s = str(v).strip()
        q = parse_quarter_token(s)
        if q:
            parsed.append(q)
            continue
        try:
            dt_val = dt.date.fromisoformat(s[0:10])
            parsed.append(dt_val)
            continue
        except Exception:
            pass
        try:
            dt_val = dt.datetime.fromisoformat(s).date()
            parsed.append(dt_val)
            continue
        except Exception:
            parsed.append(None)
    return parsed


def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        parsed = parse_date_series(df[col].tolist())
        valid = sum(1 for v in parsed if v is not None)
        if valid >= max(5, int(len(parsed) * 0.5)):
            return col
    return None


def infer_date_columns_from_headers(columns: List[object]) -> Dict[str, dt.date]:
    out: Dict[str, dt.date] = {}
    for col in columns:
        d = parse_quarter_token(str(col))
        if not d:
            try:
                d = dt.date.fromisoformat(str(col)[0:10])
            except Exception:
                d = None
        if d:
            out[str(col)] = d
    return out


def find_latest_nyfed_hhdc_url(session: requests.Session, user_agent: str) -> Optional[str]:
    try:
        html = fetch_text(session, NYFED_HHDC_BACKGROUND, headers=sec_headers(user_agent, "www.newyorkfed.org"), sleep=0.1)
    except Exception:
        return None
    matches = re.findall(
        r"https://www\\.newyorkfed\\.org/medialibrary/[^\"']*hhd_c_report_\\d{4}q[1-4]\\.xlsx\\?sc_lang=en",
        html,
        re.I,
    )
    if not matches:
        files = re.findall(r"hhd_c_report_(\\d{4})q([1-4])\\.xlsx", html, re.I)
        if not files:
            return None
        year, q = max(((int(y), int(q)) for y, q in files), default=(0, 0))
        if year == 0:
            return None
        return f"{NYFED_HHDC_XLS_BASE}hhd_c_report_{year}q{q}.xlsx?sc_lang=en"
    def key(u: str) -> tuple[int, int]:
        m = re.search(r"hhd_c_report_(\\d{4})q([1-4])", u, re.I)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
    return sorted(matches, key=key)[-1]


def find_latest_local_hhdc(path: str = "src/data") -> Optional[str]:
    root = Path(path)
    if not root.exists():
        return None
    candidates = list(root.glob("HHD_C_Report_*.xlsx"))
    if not candidates:
        candidates = list(root.glob("hhd_c_report_*.xlsx"))
    if not candidates:
        return None

    def key(p: Path) -> tuple[int, int]:
        m = re.search(r"hhd[_-]?c[_-]?report[_-]?(\d{4})q([1-4])", p.name, re.I)
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return str(sorted(candidates, key=key)[-1])


def extract_macro_series_from_df(
    df: pd.DataFrame,
    value_pattern: str,
    date_col: Optional[str] = None,
    value_col: Optional[str] = None,
) -> Optional[List[tuple[dt.date, float]]]:
    df2 = df.copy()
    df2.columns = [str(c).strip() for c in df2.columns]
    df2 = df2.dropna(how="all")

    date_col = date_col or infer_date_col(df2)
    if date_col:
        if value_col is None:
            for col in df2.columns:
                if col == date_col:
                    continue
                if re.search(value_pattern, col, re.I):
                    value_col = col
                    break
        if value_col is None:
            numeric_cols = [c for c in df2.columns if c != date_col]
            for col in numeric_cols:
                if pd.to_numeric(df2[col], errors="coerce").notna().sum() >= max(5, int(len(df2) * 0.5)):
                    value_col = col
                    break
        if value_col:
            dates = parse_date_series(df2[date_col].tolist())
            values = pd.to_numeric(df2[value_col], errors="coerce").tolist()
            series = [(d, float(v)) for d, v in zip(dates, values) if d and v == v]
            return sorted(series, key=lambda x: x[0]) if series else None

    header_dates = infer_date_columns_from_headers(df2.columns.tolist())
    if header_dates:
        category_col = df2.columns[0]
        matches = df2[category_col].astype(str).str.contains(value_pattern, case=False, na=False)
        if matches.any():
            row = df2[matches].iloc[0]
            series = []
            for col, d in header_dates.items():
                v = row.get(col)
                try:
                    val = float(v)
                except Exception:
                    continue
                series.append((d, val))
            return sorted(series, key=lambda x: x[0]) if series else None

    return None


def load_macro_series(
    session: requests.Session,
    source: Optional[str],
    user_agent: str,
    sheet: Optional[str],
    date_col: Optional[str],
    value_col: Optional[str],
    value_pattern: str,
    window_months: Optional[int],
) -> Optional[List[tuple[dt.date, float]]]:
    if not source:
        return None
    source_resolved = source
    if source.lower() == "nyfed":
        local = find_latest_local_hhdc()
        if local:
            source_resolved = local
        else:
            latest = find_latest_nyfed_hhdc_url(session, user_agent)
            if not latest:
                raise SystemExit("Failed to resolve NY Fed HHDC download URL.")
            source_resolved = latest

    if re.match(r"^https?://", source_resolved, re.I):
        resp = session.get(source_resolved, headers=sec_headers(user_agent, "www.newyorkfed.org"), timeout=60)
        resp.raise_for_status()
        content = resp.content
        if source_resolved.lower().endswith((".xls", ".xlsx")):
            try:
                xls = pd.ExcelFile(BytesIO(content))
            except Exception as exc:
                raise SystemExit("Reading Excel macro source requires openpyxl. Install: pip install openpyxl") from exc
            sheets = [sheet] if sheet else xls.sheet_names
            series = None
            for sh in sheets:
                df = xls.parse(sh)
                series = extract_macro_series_from_df(df, value_pattern, date_col=date_col, value_col=value_col)
                if series:
                    break
        else:
            df = pd.read_csv(StringIO(resp.text))
            series = extract_macro_series_from_df(df, value_pattern, date_col=date_col, value_col=value_col)
    else:
        if source_resolved.lower().endswith((".xls", ".xlsx")):
            try:
                xls = pd.ExcelFile(source_resolved)
            except Exception as exc:
                raise SystemExit("Reading Excel macro source requires openpyxl. Install: pip install openpyxl") from exc
            sheets = [sheet] if sheet else xls.sheet_names
            series = None
            for sh in sheets:
                df = xls.parse(sh)
                series = extract_macro_series_from_df(df, value_pattern, date_col=date_col, value_col=value_col)
                if series:
                    break
        else:
            df = pd.read_csv(source_resolved)
            series = extract_macro_series_from_df(df, value_pattern, date_col=date_col, value_col=value_col)

    if not series:
        return None

    if window_months:
        cutoff = dt.date.today() - dt.timedelta(days=window_months * 31)
        series = [(d, v) for d, v in series if d >= cutoff]
    return series


def macro_value_for_date(series: List[tuple[dt.date, float]], target: dt.date) -> Optional[tuple[dt.date, float]]:
    if not series:
        return None
    eligible = [(d, v) for d, v in series if d <= target]
    if not eligible:
        return series[0]
    return max(eligible, key=lambda x: x[0])


def month_label(date_str: str) -> str:
    try:
        d = dt.date.fromisoformat(date_str)
        return d.strftime("%b")
    except ValueError:
        return date_str


def load_config(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    if isinstance(cfg, dict):
        return cfg.get("deals", [])
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Deals config JSON")
    ap.add_argument("--months", type=int, default=18)
    ap.add_argument("--out", default="out/trigger_monitor_demo.json")
    ap.add_argument(
        "--user-agent",
        default="Gurman Kaur gurmankdhaliwal2@gmail.com",
        help="SEC requires a real User-Agent",
    )
    ap.add_argument("--history", type=int, default=6, help="Months of history to chart")
    ap.add_argument("--debug", action="store_true", help="Print parsing diagnostics")
    ap.add_argument(
        "--public-copy",
        default="public/data/trigger_monitor_demo.json",
        help="Optional path to also write a public JSON copy for the web UI (set to '' to disable)",
    )
    ap.add_argument("--macro-source", help="Path/URL to macro series file, or 'nyfed' to auto-fetch HHDC data")
    ap.add_argument("--macro-sheet", help="Excel sheet name for macro series (optional)")
    ap.add_argument("--macro-date-col", help="Date/Quarter column name for macro series (optional)")
    ap.add_argument("--macro-value-col", help="Value column name for macro series (optional)")
    ap.add_argument(
        "--macro-value-pattern",
        default=r"auto.*(serious|90|90\\+|delinquen)",
        help="Regex to locate macro series in the file",
    )
    ap.add_argument("--macro-window-months", type=int, default=None, help="Limit macro series to last N months")
    args = ap.parse_args()

    deals_cfg = load_config(args.config)
    session = requests.Session()
    macro_series = load_macro_series(
        session,
        source=args.macro_source,
        user_agent=args.user_agent,
        sheet=args.macro_sheet,
        date_col=args.macro_date_col,
        value_col=args.macro_value_col,
        value_pattern=args.macro_value_pattern,
        window_months=args.macro_window_months,
    )

    deals_out: List[dict] = []
    alerts: List[dict] = []
    all_as_of: List[str] = []

    for deal in deals_cfg:
        deal_id = deal.get("deal_id") or deal.get("deal") or "Unknown Deal"
        cik = str(deal.get("cik", "")).zfill(10)
        triggers_cfg = deal.get("triggers") or []

        if args.debug:
            print(f"\n== {deal_id} ({cik}) ==")
        filings = list_recent_10d_ex99(
            session,
            cik=cik,
            months=args.months,
            user_agent=args.user_agent,
            debug=args.debug,
        )
        series: List[dict] = []

        for filing in filings:
            html = fetch_text(session, filing.ex99_url, headers=sec_headers(args.user_agent, "www.sec.gov"), sleep=0.25)
            try:
                tables = pd.read_html(StringIO(html))
            except ValueError:
                tables = []
            metrics = extract_metrics_from_tables(tables, debug=args.debug)
            if args.debug:
                if metrics:
                    print(f"  - {filing.period_end}: extracted {', '.join(sorted(metrics.keys()))}")
                else:
                    print(f"  - {filing.period_end}: no metrics matched")
            if metrics:
                series.append({"period_end": filing.period_end, "metrics": metrics, "source_url": filing.ex99_url})

        if not series:
            continue

        series.sort(key=lambda r: r["period_end"])
        all_as_of.append(series[-1]["period_end"])

        # Build metric arrays
        def series_values(key: str) -> List[float]:
            return [row["metrics"][key] for row in series if key in row["metrics"]]

        delinq_60 = series_values("delinquency_60_plus")
        total_dq = series_values("total_delinquency")
        cum_loss = series_values("cumulative_loss_ratio")
        pool_bal = series_values("pool_balance")

        latest_metrics = series[-1]["metrics"]
        latest_60 = latest_metrics.get("delinquency_60_plus", 0.0)
        latest_total = latest_metrics.get("total_delinquency", 0.0)
        latest_loss = latest_metrics.get("cumulative_loss_ratio", 0.0)
        macro_percentile = percent_rank(delinq_60, latest_60) if delinq_60 else 0.5
        macro_value = None
        macro_as_of = None
        if macro_series:
            latest_date = dt.date.fromisoformat(series[-1]["period_end"])
            macro_point = macro_value_for_date(macro_series, latest_date)
            if macro_point:
                macro_as_of, macro_value = macro_point
                macro_percentile = percent_rank([v for _, v in macro_series], macro_value)

        # Changes and volatility for trigger metrics
        def change_n(values: List[float], n: int) -> float:
            if len(values) <= n:
                return 0.0
            return values[-1] - values[-n - 1]

        def recent_changes(values: List[float], window: int) -> List[float]:
            if len(values) < 2:
                return []
            changes = [values[i] - values[i - 1] for i in range(1, len(values))]
            return changes[-window:]

        deal_triggers = []
        max_score = 0.0

        for trig in triggers_cfg:
            metric_key = trig.get("metric_key", "delinquency_60_plus")
            metric_label = trig.get("metric_label", "60+ DQ %")
            direction = trig.get("direction", "<=")
            threshold = float(trig.get("threshold", 0.0))

            values = series_values(metric_key)
            current = values[-1] if values else 0.0
            change3m = change_n(values, 3)
            changes = recent_changes(values, 6)
            mean_change = sum(changes) / len(changes) if changes else 0.0
            vol_change = stddev(changes) if changes else 0.0
            vol6m = vol_change
            if threshold:
                if direction == "<=":
                    cushion = (threshold - current) / threshold
                else:
                    cushion = (current - threshold) / threshold
            else:
                cushion = 0.0
            if threshold <= 0:
                score = 0.0
                score_breakdown = {"cushion": 0.0, "trend3m": 0.0, "vol6m": 0.0, "macro": 0.0}
            else:
                score, score_breakdown = rule_based_score(cushion, change3m, vol6m, direction, macro_percentile)
            max_score = max(max_score, score)

            deal_triggers.append({
                "triggerId": trig.get("trigger_id", "TRIGGER"),
                "metric": metric_label,
                "direction": direction,
                "threshold": threshold,
                "current": current,
                "cushion": cushion,
                "change3m": change3m,
                "vol6m": vol6m,
                "score": score,
                "scoreBreakdown": score_breakdown,
            })

        # Build chart series
        history = series[-args.history:]
        cushion_series = []
        dq_series = []
        for row in history:
            period_end = row["period_end"]
            label = month_label(period_end)
            entry = {"m": label}
            for trig in triggers_cfg:
                metric_key = trig.get("metric_key", "delinquency_60_plus")
                direction = trig.get("direction", "<=")
                threshold = float(trig.get("threshold", 0.0))
                value = row["metrics"].get(metric_key)
                if value is None or threshold == 0:
                    continue
                if direction == "<=":
                    entry[trig.get("series_key", "dq")] = (threshold - value) / threshold
                else:
                    entry[trig.get("series_key", "oc")] = (value - threshold) / threshold
            if len(entry) > 1:
                cushion_series.append(entry)

            dq_val = row["metrics"].get("delinquency_60_plus")
            if dq_val is not None:
                dq_series.append({"m": label, "dq60": dq_val})

        collateral_metrics = [
            {"name": "Total DQ", "cur": latest_total, "chg": change_n(total_dq, 3)},
            {"name": "61+ DQ", "cur": latest_60, "chg": change_n(delinq_60, 3)},
            {"name": "Cum Loss", "cur": latest_loss, "chg": change_n(cum_loss, 3)},
        ]

        explanation = (
            "Derived from SEC 10-D Exhibit 99.1 tables. "
            "Scores are rule-based: 60% cushion distance, 20% 3m deterioration, "
            "10% volatility, 10% macro percentile. Breached triggers score 100."
        )

        macro_payload = {
            "theme": deal.get("macro_theme", "Collateral stress"),
            "percentile": macro_percentile,
            "series": deal.get("macro_series", "60+ delinquency"),
        }
        if macro_value is not None and macro_as_of is not None:
            macro_payload["value"] = macro_value
            macro_payload["asOf"] = macro_as_of.isoformat()
            macro_payload["source"] = "NY Fed HHDC" if (args.macro_source or "").lower() == "nyfed" else "Macro series"

        deals_out.append({
            "dealId": deal_id,
            "cusip": deal.get("cusip", "—"),
            "collateral": deal.get("collateral", "Auto ABS"),
            "geo": deal.get("geo", "US"),
            "tranche": deal.get("tranche", "Class A"),
            "macro": macro_payload,
            "triggers": deal_triggers,
            "collateralMetrics": collateral_metrics,
            "cushionSeries": cushion_series,
            "dqSeries": dq_series,
            "explanation": explanation,
        })

        if deal_triggers:
            top = max(deal_triggers, key=lambda t: t["score"])
            if top["score"] >= 0.45:
                severity = "red" if top["score"] >= 0.75 else "yellow"
                alerts.append({
                    "ts": f"{series[-1]['period_end']} 09:00",
                    "dealId": deal_id,
                    "severity": severity,
                    "title": f"{top['metric']} nearing trigger",
                    "detail": f"Cushion {top['cushion']:.2%}; risk score {top['score']:.0%}.",
                })

    if not deals_out:
        raise SystemExit("No deals built. Check CIKs, filings, or metric patterns.")

    as_of = max(all_as_of) if all_as_of else dt.date.today().strftime(MONTH_FMT)
    tranches = sum(len(d.get("triggers", [])) for d in deals_out)
    flagged = sum(1 for d in deals_out if max(t["score"] for t in d.get("triggers", [])) >= 0.45)
    red = sum(1 for d in deals_out if max(t["score"] for t in d.get("triggers", [])) >= 0.75)
    yellow = max(0, flagged - red)

    payload = {
        "asOf": as_of,
        "portfolio": {
            "deals": len(deals_out),
            "tranches": tranches,
            "flagged": flagged,
            "red": red,
            "yellow": yellow,
        },
        "deals": deals_out,
        "alerts": alerts,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Wrote {args.out}")
    if args.public_copy:
        try:
            dest = Path(args.public_copy)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"Wrote {dest}")
        except Exception as exc:  # pragma: no cover
            print(f"Warning: failed to write public copy to {args.public_copy}: {exc}")


if __name__ == "__main__":
    main()
