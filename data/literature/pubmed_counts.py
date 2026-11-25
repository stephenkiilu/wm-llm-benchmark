"""PubMed counts for Figure 1a-c: brain fMRI vs brain dMRI papers, 1985-2025.

Both queries use title/abstract terms only plus the same brain filter. Totals
come from one date-range search (each paper once); annual counts from one search
per year. Every response is saved under pubmed_raw/. Calls NCBI E-utilities
(free; set NCBI_EMAIL / NCBI_API_KEY if you have them). Counts drift as PubMed
grows, so a rerun gives slightly different numbers than the archived run.

    python data/literature/pubmed_counts.py
"""
import csv
import datetime
import json
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "pubmed_raw"
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EMAIL = os.environ.get("NCBI_EMAIL")
API_KEY = os.environ.get("NCBI_API_KEY")
YEAR_START, YEAR_END = 1985, 2025

# ── QUERIES ──────────────────────────────────────────────────────────────────
# Revised: title/abstract terms only, so neither modality gets MeSH help.
FMRI = ('("fMRI"[tiab] OR "functional MRI"[tiab] OR "functional magnetic resonance imaging"[tiab] OR '
        '("BOLD"[tiab] AND ("MRI"[tiab] OR "magnetic resonance"[tiab])))')
DMRI = ('("diffusion MRI"[tiab] OR "diffusion tensor imaging"[tiab] OR "diffusion weighted imaging"[tiab] OR '
        '"tractography"[tiab] OR (("DTI"[tiab] OR "DWI"[tiab]) AND ("MRI"[tiab] OR "magnetic resonance"[tiab])))')
BRAIN = '("brain"[tiab] OR "cerebral"[tiab] OR "white matter"[tiab] OR "Brain"[Mesh])'
HUMANS = '"Humans"[Mesh]'
WHITE_MATTER = '("white matter"[tiab] OR "White Matter"[Mesh])'

RANGE = f'("{YEAR_START}"[dp] : "{YEAR_END}"[dp])'
TOTALS = {   # label: term, all restricted to 1985-2025 and counted once per paper
    "fmri_brain": f"{FMRI} AND {BRAIN}",
    "dmri_brain": f"{DMRI} AND {BRAIN}",
    "fmri_brain_humans": f"{FMRI} AND {BRAIN} AND {HUMANS}",
    "dmri_brain_humans": f"{DMRI} AND {BRAIN} AND {HUMANS}",
    "dmri_white_matter": f"{DMRI} AND {WHITE_MATTER}",
}


def count(label, term):
    """PubMed hit count for `term`; the full response is saved as pubmed_raw/<label>.json."""
    params = {"db": "pubmed", "term": term, "rettype": "count", "retmode": "json"}
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    url = f"{BASE_URL}?{urllib.parse.urlencode(params)}"
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                body = json.load(resp)
            break
        except OSError:
            if attempt == 3:
                raise
            time.sleep(2 * (attempt + 1))
    (RAW / f"{label}.json").write_text(json.dumps({"term": term, "response": body}, indent=1))
    time.sleep(0.12 if API_KEY else 0.4)      # stay under NCBI's rate limit
    return int(body["esearchresult"]["count"])


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    retrieved = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")

    totals = {label: count(f"total_{label}", f"{term} AND {RANGE}") for label, term in TOTALS.items()}

    annual = []
    for year in range(YEAR_START, YEAR_END + 1):
        fmri = count(f"year{year}_fmri_brain", f'{FMRI} AND {BRAIN} AND "{year}"[dp]')
        dmri = count(f"year{year}_dmri_brain", f'{DMRI} AND {BRAIN} AND "{year}"[dp]')
        annual.append({"year": year, "fmri_brain": fmri, "dmri_brain": dmri,
                       "ratio_fmri_per_dmri": round(fmri / dmri, 3) if dmri else ""})
        print(f"{year}: fMRI={fmri:>6}   dMRI={dmri:>6}")

    with open(HERE / "annual_counts_brain.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(annual[0]))
        w.writeheader()
        w.writerows(annual)
    with open(HERE / "totals_unique.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["measure", "unique_papers_1985_2025"])
        w.writerows(totals.items())

    json.dump({"retrieved_utc": retrieved, "database": "PubMed (NCBI E-utilities esearch)",
               "date_field": "[dp] publication date; totals use one range search, annual counts one search per year",
               "years": [YEAR_START, YEAR_END],
               "queries": {"fmri": FMRI, "dmri": DMRI, "brain_filter": BRAIN, "humans_filter": HUMANS,
                           "white_matter_filter": WHITE_MATTER},
               "totals_unique": totals},
              open(HERE / "queries.json", "w"), indent=1)
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
