"""Query PubMed for annual fMRI and dMRI publication counts.

Regenerates data/publication_counts.csv, which figure_publication_growth.py
reads. Hits the live NCBI E-utilities API.

Optional settings, read from .env or the environment:
    NCBI_EMAIL    contact address sent with each request (NCBI courtesy)
    NCBI_API_KEY  raises the rate limit from 3 to 10 requests/second

Output: data/publication_counts.csv
"""

import csv
import os
import time

import requests
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_CSV = os.path.join(BASE_DIR, "data", "publication_counts.csv")


# NCBI asks callers to identify themselves so they can make contact before
# throttling a misbehaving script. Supply your own address; neither is required
# for the queries to succeed.
EMAIL = os.environ.get("NCBI_EMAIL")
API_KEY = os.environ.get("NCBI_API_KEY")

FMRI_QUERY = (
    '("fMRI"[tiab] OR "functional MRI"[tiab] OR '
    '"functional magnetic resonance imaging"[tiab] OR '
    '("BOLD"[tiab] AND ("MRI"[tiab] OR "magnetic resonance"[tiab] OR '
    '"Magnetic Resonance Imaging"[Mesh])))'
)
DMRI_QUERY = (
    '("diffusion MRI"[tiab] OR "diffusion tensor imaging"[tiab] OR '
    '"diffusion weighted imaging"[tiab] OR "tractography"[tiab] OR '
    '"Diffusion Magnetic Resonance Imaging"[Mesh] OR '
    '"Diffusion Tensor Imaging"[Mesh] OR '
    '(("DTI"[tiab] OR "DWI"[tiab]) AND ("MRI"[tiab] OR '
    '"magnetic resonance"[tiab] OR "Magnetic Resonance Imaging"[Mesh])))'
)
 
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
YEAR_START = 1985
YEAR_END = 2026  # inclusive
 
 
def get_count(term, year):
    params = {
        "db": "pubmed",
        "term": f'{term} AND ("{year}"[dp])',
        "retmax": 0,
    }
    if EMAIL:
        params["email"] = EMAIL
    if API_KEY:
        params["api_key"] = API_KEY
    resp = requests.get(BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    text = resp.text
    start = text.find("<Count>") + len("<Count>")
    end = text.find("</Count>")
    return int(text[start:end])
 
 
def main():
    sleep_time = 0.11 if API_KEY else 0.34  # stay under NCBI's rate limit
    rows = []
    for year in range(YEAR_START, YEAR_END + 1):
        fmri_n = get_count(FMRI_QUERY, year)
        time.sleep(sleep_time)
        dmri_n = get_count(DMRI_QUERY, year)
        time.sleep(sleep_time)
        rows.append((year, fmri_n, dmri_n))
        print(f"{year}: fMRI={fmri_n:>6}   dMRI={dmri_n:>6}")
 
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, OUT_CSV)
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["year", "fmri_count", "dmri_count"])
        writer.writerows(rows)
    print(f"\nSaved counts to {out_path}")
 
 
if __name__ == "__main__":
    main()
