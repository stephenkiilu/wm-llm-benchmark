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

## `Whitematter_Look_up_table.csv`

The ontology-guided lookup table: 121 canonical white-matter tracts, one per
row, each with its ontology identifiers, a definition, and the surface forms
that should normalize to it.

| Column | Contents |
|---|---|
| `preferred_label` | Canonical tract name |
| `SNOMED_ID` | SNOMED CT concept ID|
| `UBERON_ID` | Uberon term ID|
| `Definition` | One-sentence anatomical description of the tract|
| `synonyms` | Ontology synonyms, Latin variants and abbreviations |
| `Source` | Atlases and references listing the tract|
| `Possible related names` | Additional surface forms seen in the literature|

Definitions are one-sentence anatomical descriptions of each tract, written for
this table with reference to Uberon, SNOMED CT and the primary literature. 

`Source` spans ICBM-DTI-81, TractSeg, XTRACT, MGH_HCP, TRACULA, AFQ,
DSI_Studio, JHU_Tractography and and existing white matter tracts literature, so a term can be traced to the atlas it comes from.

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
