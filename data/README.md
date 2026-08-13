# Data

Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## `whitematter_dataset.csv`

622 manually annotated articles on white matter, dMRI and tractography, focused
on dementia and related disorders. 27 columns: 12 bibliographic and 15
annotation columns covering the 13 scientific variables.

| Group | Columns |
|---|---|
| Bibliographic | `PMID`, `PMCID`, `Title`, `Authors`, `Citation`, `First Author`, `Journal/Book`, `Publication Year`, `Create Date`, `NIHMS ID`, `Open Source?`, `DOI` |
| Annotated | imaging modality, study vs review, human vs non-human, dementia focus, which disease, tracts studied, results presentation, analysis software, diffusion measure, direction of WM-integrity change, group comparison, template space, coordinate reporting, notes |

The paper reports quantitative results for the six variables with bounded
answer spaces: `whitematter_tracts`, `study_type`, `DTI_study`, `Human_study`,
`Dementia_study`, `Disease_study`.

## `whitematter_lut.csv`
To describe later

## `publication_counts.csv`

Annual PubMed counts for fMRI and dMRI, used by
`evaluation/figure_publication_growth.py`. Regenerate with
`python -m evaluation.publication_counts` (hits the live NCBI API).

NCBI asks callers to identify themselves, and issues optional API keys that
raise the rate limit from 3 to 10 requests/second. Neither is required — the
script runs without them. To set them, copy `.env.example` to `.env` and fill
in:

```
NCBI_EMAIL=you@email.com
NCBI_API_KEY=          # https://account.ncbi.nlm.nih.gov/settings/
```
Plain environment variables work too.

## Not included: article full text

`WMT_FULLDATA.json`, the paper text the extraction pipeline consumes, is not
distributed — the articles are under publisher copyright.

**This does not affect reproducing the paper.** Every number, table and figure
is recomputed from the model predictions in `predictions/`, which are included.
No API key and no PDFs are needed.

Re-running extraction itself does require the corpus. To rebuild it, retrieve
the full text for the PMIDs in `whitematter_dataset.csv` through your
institution or PubMed Central, and write `data/WMT_FULLDATA.json` keyed by PMID
with a `body` field per article containing the labelled `Title:`, `Abstract:`,
`Keywords:` and `Body:` sections.
