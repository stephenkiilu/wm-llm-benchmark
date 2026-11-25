# Publication counts (Figure 1a–c, Supplementary Table S1)

Brain fMRI and brain diffusion MRI (dMRI) papers, 1985–2025, counted the same way for both modalities:
title/abstract terms only, plus one brain filter (`"brain"`, `"cerebral"`, `"white matter"` or the Brain MeSH heading).

| File | Contents |
|---|---|
| `pubmed_counts.py` | The PubMed search (NCBI E-utilities); rerun to repeat it |
| `queries.json` | Exact queries and retrieval time of the archived run (28 Oct 2026, 16:12 UTC) |
| `annual_counts_brain.csv` | PubMed brain fMRI and dMRI papers per year, with the fMRI/dMRI ratio |
| `totals_unique.csv` | PubMed totals, each paper counted once (one 1985–2025 range search per total) |
| `pubmed_raw/` | Every PubMed response from the archived run |
| `web_of_science_counts.csv` | Web of Science totals and queries (run by hand on 2 Nov 2026) |

**PubMed totals:** fMRI 81,372; dMRI 32,404. **Web of Science totals:** fMRI 87,538; dMRI 35,127.

Web of Science was searched by hand in the Core Collection (SCI-EXPANDED, SSCI, ESCI; conference and book
indexes excluded) with the same terms on the Topic field (`TS=`), so it has no archived API response.
PubMed counts drift as records are added, so a rerun gives slightly different numbers.
