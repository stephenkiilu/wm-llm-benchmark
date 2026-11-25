# wm-llm-benchmark

Code and data for *A standardized schema for automated meta-analysis of white-matter tracts literature*
(Kiilu, Ray, Leigh & Pestilli).

622 dementia diffusion-MRI articles annotated by hand, a 122-tract vocabulary linked to Uberon and SNOMED CT,
and the predictions of five OpenAI models (GPT-4o-mini, GPT-5-mini, GPT-5.4-mini, GPT-5.4-nano, GPT-5.6-luna)
with and without the vocabulary in the prompt.

## Abstract

Coordinated communication between cortical regions relies on white matter (WM) structural connectivity, which
makes up half of human brain volume and is vital for understanding development, aging, and disease. Advances in
diffusion MRI (dMRI) and tractography allow in vivo mapping of major WM tracts across individuals. However, rapid
literature growth and a lack of standardized reporting hinder automated information extraction. To address this,
we developed a schema-based, LLM-driven WM extraction framework featuring a harmonized tract vocabulary derived
from atlases, pipelines (e.g., TractSeg), and biomedical ontologies (Uberon, SNOMED CT). Evaluated on a manually
annotated dataset of PubMed articles on dementia, our framework significantly improved tract extraction F₁ scores
by up to 18% across four of five LLMs (all q < 0.001). This demonstrates that vocabulary-guided LLM extraction
can scale WM research synthesis, advancing large-scale meta-analysis.

**Keywords:** white matter tracts, large language models, information extraction, neuroimaging, meta-analysis

## Reproduce the paper

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PY=.venv/bin/python ./reproduce.sh
```

This rebuilds every table and figure from the saved predictions in about a minute. It needs no API key and makes no network calls.

| Paper | Script | Output |
|---|---|---|
| Figure 1 | `figures/figure1.py` | `results/figures/figure1.*` |
| Table 1 (both arms; see the `arm` column) | `analysis/field_metrics.py` | `results/tables/field_metrics.csv` |
| Figure 2 | `figures/figure2.py` | `results/figures/figure2.*` |
| Figure 2a, vocabulary effect | `analysis/evaluate.py` | `tract_results.csv`, `vocabulary_effect.csv` |
| Figure 2b, cost and tokens | `analysis/efficiency.py` | `efficiency.csv` |
| Figure 2c, Tables S2–S3 | `analysis/sections.py` | `section_f1.csv`, `table_s2_sections.csv`, `table_s3_abstract_vs_full.csv` |
| Figure S1, task complexity | `analysis/complexity.py`, `figures/figure_s1_complexity.py` | `task_complexity.csv` |
| Figure S2, pipeline | `figures/figure_s2_pipeline.py` | `results/figures/figure_s2_pipeline.*` |
| Vocabulary compliance | `analysis/vocab_compliance.py` | `vocabulary_compliance.csv` |
| Tract name variants | `analysis/name_variants.py` (needs the full texts; skipped without them) | `name_variants.csv` |
| Table S1, search queries | `data/literature/` | see its README |

## Layout

```
data/         annotations.csv (reference labels), vocabulary.csv (122 tracts),
              literature/ (PubMed and Web of Science counts), brain_renders/ (Figure 1d)
prompts/      build_prompt.py and the two rendered prompts (with / without the vocabulary)
extraction/   run_extraction.py, run_all.sh: the API runs (need a key and the full texts)
predictions/  full_text/ 10 runs, sections/ 14 GPT-5.6-luna section-input runs, usage/ per-call logs
analysis/     scoring (metrics.py, vocabulary.py) and one script per table
figures/      one script per figure; style.py and page.py hold the shared style
results/      tables/ and figures/, as committed
```

Runs are named `<model>__<with|without>_vocabulary[__<sections>].csv`, one row per article (`PMID`).
WM tracts are scored as micro-F1 on vocabulary ids over articles whose reference lists at least one tract;
95% CIs are percentile bootstrap over articles (10,000 resamples); vocabulary effects are paired
bootstraps with Benjamini–Hochberg q-values.

## Rerunning the extraction

The article full texts are not redistributed. Retrieve them by PMID (`data/annotations.csv`), parse them to
`data/full_text/articles.json`, copy `.env.example` to `.env` with your `OPENAI_API_KEY`, then
`extraction/run_all.sh` (all 6,220 full-text calls cost US$34; `run_extraction.py --dry-run` estimates the cost first).
`articles.json` is a list of `{"pmid", "metadata": {"title", "abstract"}, "content": {section: text}}`; the section runs
are `run_extraction.py --model gpt-5.6-luna --arm with_vocabulary --effort medium --sections all14`.
Model outputs can differ slightly between runs, and GPT-5.6-luna has no dated snapshot.

## Licence

Code: MIT (`LICENSE`). Data, predictions and results: CC BY 4.0 (`LICENSE-DATA`). Please cite the paper (`CITATION.cff`).
