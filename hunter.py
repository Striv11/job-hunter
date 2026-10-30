"""Daily job hunter.

Scrapes LinkedIn, Indeed, Glassdoor, Google Jobs and Naukri for roles posted in
the last N days, scores them against the profile in config.json, skips anything
already delivered on a previous day, and writes the top 100 to Excel + HTML.

Usage:
    python hunter.py            # full daily run
    python hunter.py --quick    # small smoke-test run
"""
import argparse
import datetime as dt
import hashlib
import html
import json
import re
import sqlite3
import sys
import time
import traceback
import urllib.parse
from pathlib import Path

import pandas as pd
from jobspy import scrape_jobs

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
DB_PATH = ROOT / "data" / "seen_jobs.sqlite"
OUT_ROOT = ROOT / "output"
LOG_PATH = ROOT / "data" / "run.log"

# Who to message at the company, by track.
CONTACT_TITLES = {
    "VC / Investing": ["Partner", "Principal", "Vice President investments"],
    "Founder's Office": ["Founder", "Co-founder", "Chief of Staff"],
    "Product": ["Head of Product", "Director of Product", "VP Product"],
    "Corporate Strategy": ["Head of Strategy", "Corporate Development", "Chief Strategy Officer"],
    "Consulting": ["Partner", "Engagement Manager", "Campus recruiting"],
    "Europe Grad Scheme": ["Graduate recruitment", "Early careers", "Talent acquisition"],
}


def log(msg):
    line = f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    if sys.stdout:  # None under pythonw (scheduled, windowless runs)
        print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# ---------------------------------------------------------------- storage

def db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            key TEXT PRIMARY KEY, title TEXT, company TEXT, url TEXT,
            first_seen TEXT, delivered_on TEXT)"""
    )
    return con


def job_key(title, company, location):
    norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s or "").lower())
    return hashlib.sha1(f"{norm(title)}|{norm(company)}|{norm(location)[:20]}".encode()).hexdigest()


# ---------------------------------------------------------------- scraping

def run_query(site, term, location, country_indeed, results, hours_old, track, region):
    kwargs = dict(
        site_name=[site],
        search_term=term,
        location=location,
        results_wanted=results,
        hours_old=hours_old,
        country_indeed=country_indeed,
        verbose=0,
    )
    if site == "linkedin":
        kwargs["fetch_description"] = CONFIG["settings"]["fetch_linkedin_descriptions"]
    if site == "google":
        kwargs["google_search_term"] = f"{term} jobs in {location} since last {CONFIG['settings']['max_age_days']} days"
    # LinkedIn rate-limits (429) and returns nothing; back off and retry.
    for attempt in range(3):
        df = scrape_jobs(**kwargs)
        if (df is not None and not df.empty) or site != "linkedin":
            break
        time.sleep(30 * (attempt + 1))
    if df is None or df.empty:
        return pd.DataFrame()
    df["track"] = track
    df["region"] = region
    df["query"] = term
    return df


def scrape_all(quick=False):
    s = CONFIG["settings"]
    hours_old = s["max_age_days"] * 24
    per_query = 10 if quick else s["results_per_query_per_site"]
    # Checkpoint after every query so an interrupted run resumes instead of starting over.
    ckpt = ROOT / "data" / f"checkpoint_{dt.date.today().isoformat()}{'_quick' if quick else ''}.pkl"
    frames, done = [], set()
    if ckpt.exists():
        frames, done = pd.read_pickle(ckpt)
        log(f"resuming: {len(done)} queries already done today")
    for search in CONFIG["searches"]:
        terms = search["terms"][:1] if quick else search["terms"]
        if search["region"] == "india":
            targets = [(loc, "india") for loc in CONFIG["locations"]["india"]]
            sites = s["sites_india"]
        else:
            targets = [(e["location"], e["country_indeed"]) for e in CONFIG["locations"]["europe"]]
            sites = s["sites_europe"]
        if quick:
            targets, sites = targets[:1], sites[:2]
        for term in terms:
            for location, country in targets:
                for site in sites:
                    if site == "naukri" and search["region"] != "india":
                        continue
                    qid = (site, term, location)
                    if qid in done:
                        continue
                    try:
                        df = run_query(site, term, location, country, per_query, hours_old,
                                       search["track"], search["region"])
                        log(f"{site:9s} | {search['track']:18s} | {term[:35]:35s} | {location:22s} -> {len(df)}")
                        if not df.empty:
                            frames.append(df)
                    except Exception as e:
                        log(f"{site:9s} | {term} | {location} FAILED: {str(e)[:150]}")
                    done.add(qid)
                    pd.to_pickle((frames, done), ckpt)
                    time.sleep(s["pause_between_queries_sec"])
    if not frames:
        return pd.DataFrame()
    return pd.concat([f.dropna(axis=1, how="all") for f in frames], ignore_index=True)


# ---------------------------------------------------------------- enrichment

EXP_RE = re.compile(r"(\d{1,2})\s*(?:\+|-|–|to)?\s*(\d{1,2})?\s*\+?\s*(?:years|yrs)", re.I)


def parse_experience(text):
    """Return (min_years, max_years) from the first experience mention, else (None, None)."""
    for m in EXP_RE.finditer(text or ""):
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else None
        if lo <= 25:
            return lo, hi
    return None, None


def score_job(row):
    sc = CONFIG["scoring"]
    title = str(row.get("title") or "").lower()
    company = str(row.get("company") or "").lower()
    desc = str(row.get("description") or "").lower()
    score, reasons = 50, []

    for kw, w in sc["boost_keywords"].items():
        if kw in title:
            score += w
            reasons.append(f"+{w} title:{kw}")
        elif kw in desc:
            score += w // 3
    for kw, w in sc["penalty_keywords"].items():
        if kw in title:
            score += w
            reasons.append(f"{w} title:{kw}")
    if any(f in company for f in sc["top_firms"]):
        score += 15
        reasons.append("+15 top firm")

    lo, hi = parse_experience(desc)
    me = CONFIG["profile"]["years_experience"]
    if lo is not None:
        if lo > me + 2.5:
            score -= 25
            reasons.append(f"-25 needs {lo}+ yrs")
        elif lo <= me and (hi is None or hi >= me):
            score += 15
            reasons.append(f"+15 exp fits ({lo}-{hi or '+'})")
    if row.get("description") in (None, "") or pd.isna(row.get("description")):
        score -= 5
    return max(0, min(100, score)), "; ".join(reasons), lo, hi


def li_people_url(company, title):
    q = f'{title} "{company}"'
    return "https://www.linkedin.com/search/results/people/?keywords=" + urllib.parse.quote(q)


def google_xray_url(company, titles):
    t = " OR ".join(f'"{x}"' for x in titles)
    q = f'site:linkedin.com/in "{company}" ({t})'
    return "https://www.google.com/search?q=" + urllib.parse.quote(q)


EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


def outreach_note(row):
    """LinkedIn connection note, kept under the 300-char limit."""
    first = CONFIG["profile"]["pitch"]
    note = (f"Hi, I saw the {row['title']} role at {row['company']}. {first}, "
            f"and I'd love to learn more about the team. Happy to share my CV.")
    return note[:300]


def enrich(df):
    out = []
    for _, r in df.iterrows():
        row = r.to_dict()
        company = str(row.get("company") or "").strip()
        if not company or company.lower() == "nan":
            continue
        score, reasons, lo, hi = score_job(row)
        titles = CONTACT_TITLES.get(row["track"], ["Hiring manager"])
        desc = str(row.get("description") or "")
        emails = set(EMAIL_RE.findall(desc))
        if isinstance(row.get("emails"), str):
            emails |= set(e.strip() for e in row["emails"].split(","))
        emails = {e for e in emails if not e.lower().endswith((".png", ".jpg"))}

        sal = ""
        if pd.notna(row.get("min_amount")) or pd.notna(row.get("max_amount")):
            cur = row.get("currency") or ""
            iv = row.get("interval") or ""
            mn = row.get("min_amount")
            mx = row.get("max_amount")
            fmt = lambda v: f"{v:,.0f}" if pd.notna(v) else "?"
            sal = f"{cur} {fmt(mn)} - {fmt(mx)} / {iv}".strip()

        posted = row.get("date_posted")
        posted = str(posted)[:10] if pd.notna(posted) else ""

        out.append({
            "Score": score,
            "Track": row["track"],
            "Title": row.get("title"),
            "Company": company,
            "Location": row.get("location"),
            "Remote": "Yes" if row.get("is_remote") is True else "",
            "Posted": posted,
            "Source": row.get("site"),
            "Job Type": row.get("job_type") or "",
            "Level": row.get("job_level") or "",
            "Exp Required": (f"{lo}-{hi} yrs" if hi else f"{lo}+ yrs") if lo is not None else "",
            "Salary": sal,
            "Industry": row.get("company_industry") or "",
            "Apply Link": row.get("job_url_direct") if pd.notna(row.get("job_url_direct")) else row.get("job_url"),
            "Listing Link": row.get("job_url"),
            "Company Page": row.get("company_url") if pd.notna(row.get("company_url")) else "",
            "Emails Found": ", ".join(sorted(emails)),
            "Hiring Manager Search (LinkedIn)": li_people_url(company, titles[0]),
            "Recruiter Search (LinkedIn)": li_people_url(company, "Talent Acquisition OR Recruiter"),
            "Google X-ray (LinkedIn profiles)": google_xray_url(company, titles),
            "Who to Message": ", ".join(titles),
            "Outreach Note": outreach_note({"title": row.get("title"), "company": company}),
            "Why Scored": reasons,
            "Description": desc[:3000],
            "_key": job_key(row.get("title"), company, row.get("location")),
        })
    res = pd.DataFrame(out)
    if res.empty:
        return res
    # Same job reposted on several boards: keep the best-scored copy.
    # Same role posted in several cities: keep one row per company+title.
    res["_ct"] = (res["Company"].str.lower().str.strip() + "|" + res["Title"].astype(str).str.lower().str.strip())
    res = res.sort_values("Score", ascending=False).drop_duplicates("_key").drop_duplicates("_ct")
    return res.drop(columns=["_ct"])


def pick_new(df, con, today):
    """Drop jobs delivered on an earlier day, keep the top N, record them."""
    target = CONFIG["settings"]["daily_target"]
    cutoff = (dt.date.today() - dt.timedelta(days=CONFIG["settings"]["max_age_days"])).isoformat()
    df = df[(df["Posted"] == "") | (df["Posted"] >= cutoff)]
    delivered = {k for (k,) in con.execute(
        "SELECT key FROM jobs WHERE delivered_on IS NOT NULL AND delivered_on < ?", (today,))}
    fresh = df[~df["_key"].isin(delivered)].sort_values(["Score", "Posted"], ascending=False).head(target)
    for _, r in fresh.iterrows():
        con.execute(
            "INSERT INTO jobs(key,title,company,url,first_seen,delivered_on) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET delivered_on=excluded.delivered_on",
            (r["_key"], r["Title"], r["Company"], r["Apply Link"], today, today))
    con.commit()
    return fresh.drop(columns=["_key"])


# ---------------------------------------------------------------- output

LINK_COLS = ["Apply Link", "Listing Link", "Company Page", "Hiring Manager Search (LinkedIn)",
             "Recruiter Search (LinkedIn)", "Google X-ray (LinkedIn profiles)"]


def write_excel(df, path):
    from openpyxl.styles import Font, PatternFill, Alignment
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        df.to_excel(xw, index=False, sheet_name="Jobs")
        ws = xw.sheets["Jobs"]
        header = [c.value for c in ws[1]]
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="1F3A5F")
        for col in LINK_COLS:
            ci = header.index(col) + 1
            for row in range(2, ws.max_row + 1):
                cell = ws.cell(row=row, column=ci)
                if cell.value and str(cell.value).startswith("http"):
                    cell.hyperlink = cell.value
                    cell.value = "Open"
                    cell.font = Font(color="0563C1", underline="single")
        widths = {"Score": 7, "Track": 18, "Title": 40, "Company": 26, "Location": 22,
                  "Outreach Note": 50, "Description": 60, "Why Scored": 35}
        for i, name in enumerate(header, 1):
            ws.column_dimensions[ws.cell(1, i).column_letter].width = widths.get(name, 14)
        di = header.index("Description") + 1
        for row in range(2, ws.max_row + 1):
            ws.cell(row, di).alignment = Alignment(wrap_text=False)
        ws.freeze_panes = "D2"
        ws.auto_filter.ref = ws.dimensions
        # Application tracker columns for the user to fill in.
        for j, name in enumerate(["Status", "Messaged?", "Notes"], ws.max_column + 1):
            ws.cell(1, j, name).font = Font(bold=True, color="FFFFFF")
            ws.cell(1, j).fill = PatternFill("solid", fgColor="2E7D32")
            ws.column_dimensions[ws.cell(1, j).column_letter].width = 16


def write_html(df, path, today):
    tracks = sorted(df["Track"].unique())
    cards = []
    for _, r in df.iterrows():
        e = lambda k: html.escape(str(r.get(k) or ""))
        a = lambda k, label: f'<a href="{e(k)}" target="_blank">{label}</a>' if r.get(k) else ""
        badge = "hi" if r["Score"] >= 75 else "mid" if r["Score"] >= 55 else "lo"
        meta = " · ".join(x for x in [e("Location"), e("Posted"), e("Source"), e("Exp Required"), e("Salary")] if x)
        cards.append(f"""
<article data-track="{e('Track')}" data-text="{html.escape((str(r['Title'])+' '+str(r['Company'])+' '+str(r['Location'])).lower())}">
  <div class="top"><span class="score {badge}">{r['Score']}</span><span class="track">{e('Track')}</span></div>
  <h3>{e('Title')}</h3><div class="co">{e('Company')}</div><div class="meta">{meta}</div>
  <div class="links">{a('Apply Link','Apply')}{a('Listing Link','Listing')}{a('Hiring Manager Search (LinkedIn)','Hiring manager')}{a('Recruiter Search (LinkedIn)','Recruiter')}{a('Google X-ray (LinkedIn profiles)','X-ray')}</div>
  {'<div class="email">✉ ' + e('Emails Found') + '</div>' if r.get('Emails Found') else ''}
  <details><summary>Outreach note & details</summary>
    <p class="note">{e('Outreach Note')}</p><button onclick="navigator.clipboard.writeText(this.previousElementSibling.textContent)">Copy note</button>
    <p class="why">{e('Why Scored')}</p><p class="desc">{e('Description')[:1200]}</p></details>
</article>""")
    buttons = "".join(f'<button class="f" data-t="{html.escape(t)}">{html.escape(t)} ({(df.Track==t).sum()})</button>' for t in tracks)
    page = f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Jobs {today}</title><style>
:root{{--bg:#f6f7f9;--card:#fff;--fg:#1b1f24;--mute:#5b6470;--line:#e2e5ea;--acc:#1f5fbf}}
@media (prefers-color-scheme:dark){{:root{{--bg:#121417;--card:#1b1e23;--fg:#e8eaed;--mute:#9aa3ad;--line:#2b3038;--acc:#6ea8ff}}}}
body{{margin:0;font:14px/1.45 system-ui,sans-serif;background:var(--bg);color:var(--fg)}}
header{{padding:20px 16px;max-width:1200px;margin:auto}} h1{{margin:0 0 4px;font-size:22px}}
.bar{{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}} input{{flex:1;min-width:200px;padding:8px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg)}}
button{{padding:6px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);cursor:pointer}} button.on{{background:var(--acc);color:#fff}}
main{{display:grid;grid-template-columns:repeat(auto-fill,minmax(320px,1fr));gap:12px;padding:0 16px 40px;max-width:1200px;margin:auto}}
article{{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px}}
h3{{margin:6px 0 2px;font-size:15px}} .co{{font-weight:600}} .meta,.why{{color:var(--mute);font-size:12px}}
.top{{display:flex;justify-content:space-between}} .track{{font-size:12px;color:var(--mute)}}
.score{{font-weight:700;padding:1px 8px;border-radius:10px;color:#fff}} .hi{{background:#2e7d32}} .mid{{background:#b7791f}} .lo{{background:#8a8f98}}
.links{{display:flex;flex-wrap:wrap;gap:10px;margin:8px 0}} a{{color:var(--acc)}} .email{{font-size:12px}}
.desc{{white-space:pre-wrap;font-size:12px;max-height:200px;overflow:auto}} .note{{background:var(--bg);padding:8px;border-radius:8px}}
</style></head><body><header><h1>Daily jobs · {today}</h1><div class="meta">{len(df)} new roles posted in the last {CONFIG['settings']['max_age_days']} days, ranked by fit</div>
<div class="bar"><input id="q" placeholder="Filter by title, company, city…"><button class="f on" data-t="">All</button>{buttons}</div></header>
<main id="list">{''.join(cards)}</main>
<script>
let t="";const q=document.getElementById('q');
function apply(){{const s=q.value.toLowerCase();document.querySelectorAll('article').forEach(a=>{{a.style.display=(!t||a.dataset.track===t)&&a.dataset.text.includes(s)?'':'none'}})}}
document.querySelectorAll('.f').forEach(b=>b.onclick=()=>{{t=b.dataset.t;document.querySelectorAll('.f').forEach(x=>x.classList.toggle('on',x===b));apply()}});q.oninput=apply;
</script></body></html>"""
    path.write_text(page, encoding="utf-8")


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small smoke-test run")
    ap.add_argument("--no-open", action="store_true", help="don't open the report when done")
    args = ap.parse_args()

    today = dt.date.today().isoformat()
    log(f"=== run start ({'quick' if args.quick else 'full'}) ===")
    raw = scrape_all(quick=args.quick)
    log(f"scraped {len(raw)} raw rows")
    if raw.empty:
        log("nothing scraped; exiting")
        return 1
    jobs = enrich(raw)
    con = db()
    picked = pick_new(jobs, con, today)
    log(f"{len(jobs)} unique jobs, {len(picked)} new delivered today")

    out_dir = OUT_ROOT / today
    out_dir.mkdir(parents=True, exist_ok=True)
    write_excel(picked, out_dir / f"jobs_{today}.xlsx")
    write_html(picked, out_dir / f"jobs_{today}.html", today)
    jobs.drop(columns=["_key"]).to_csv(out_dir / f"all_scraped_{today}.csv", index=False, encoding="utf-8-sig")
    log(f"wrote reports to {out_dir}")

    # Copy the day's Excel to a Desktop folder so it's easy to find.
    import os
    import shutil
    desktop = Path(os.path.expanduser("~")) / "OneDrive" / "Desktop"
    if not desktop.exists():
        desktop = Path(os.path.expanduser("~")) / "Desktop"
    daily = desktop / "Daily Jobs"
    daily.mkdir(exist_ok=True)
    xlsx = daily / f"jobs_{today}.xlsx"
    shutil.copy2(out_dir / f"jobs_{today}.xlsx", xlsx)
    log(f"Excel copied to {xlsx}")

    if not args.no_open:
        os.startfile(xlsx)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        log("CRASH\n" + traceback.format_exc())
        sys.exit(1)
