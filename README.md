# Toward automated meta-analysis of white-matter research: an ontology-guided benchmark for large language models

Code and data for the paper.

## Abstract

Understanding white-matter organization is essential for characterizing how the
brain changes during development, aging and disease. Advances in
diffusion-weighted magnetic resonance imaging (dMRI) and tractography have
produced a rapidly expanding literature, making it increasingly difficult to
synthesize findings systematically. Here, we evaluated the ability of large
language models (LLMs) to extract structured information from studies of white
matter, dMRI and tractography. We focused on the literature concerning dementia
and related disorders, in which white-matter microstructural alterations have
been extensively investigated. We first constructed a white-matter look-up table
comprising major anatomical tracts and terminology derived from biomedical
ontologies, including Uberon and SNOMED CT. We then manually annotated a corpus
of 622 articles for 13 scientific variables and used this dataset to benchmark
an information-extraction pipeline based on GPT-4o-mini and GPT-5-mini,
reporting quantitative results for the six variables with bounded answer spaces.
Both models performed well on relatively simple extraction tasks (F1 0.81–0.99),
but performance declined as the complexity of the required information and
reasoning increased. Incorporating the ontology-based look-up table increased
GPT-5-mini's F1 on white-matter tract extraction by 21 points, from 0.467 to
0.682 (a 46% relative increase; p < 0.002). Together, these resources provide a
benchmark dataset, controlled anatomical vocabulary and computational framework
for evaluating LLM-based extraction from the white-matter literature. More
broadly, the study illustrates how reasoning-capable LLMs can be combined with
curated biomedical vocabularies to support scalable and reproducible literature
synthesis.

**Keywords:** white matter tracts, large language models, information
extraction, neuroimaging, meta-analysis

## Setup

```bash
git clone https://github.com/stephenkiilu/wm-llm-benchmark.git
cd wm-llm-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

An API key is needed only to re-run extraction. For evaluation, skip it.

```bash
cp .env.example .env      # then set OPENAI_API_KEY
```

## Reproducing the paper

Run from the repository root. Metrics first — the figures read their tables.

```bash
python -m evaluation.model_comparison      # F1 + accuracy per variable, bootstrap CIs, FDR
python -m evaluation.lut_ablation          # LUT vs no-LUT: 0.467 -> 0.682
python -m evaluation.section_ablation      # F1 across the 14 section combinations
python -m evaluation.task_complexity       # gold-label complexity index

python -m evaluation.figure_section_ablation    # needs section_ablation
python -m evaluation.figure_lut_complexity      # needs lut_ablation + task_complexity
python -m evaluation.figure_publication_growth
python -m evaluation.figure_model_f1            # needs model_comparison
python -m evaluation.figure_model_accuracy      # needs model_comparison
```

| Artifact | Produced by |
|---|---|
| `results/tables/model_comparison.csv` | `model_comparison.py` |
| `results/tables/model_accuracy.csv` | `figure_model_accuracy.py` |
| `results/tables/lut_ablation.csv` | `lut_ablation.py` |
| `results/tables/section_ablation.csv` | `section_ablation.py` |
| `results/tables/task_complexity.csv` | `task_complexity.py` |
| `results/figures/publication_growth.png` | `figure_publication_growth.py` |
| `results/figures/section_ablation.png` | `figure_section_ablation.py` |
| `results/figures/lut_complexity.png` | `figure_lut_complexity.py` |

Bootstrap resampling is seeded (`RANDOM_SEED = 42`, `N_BOOT = 1000`), so these
commands reproduce the published numbers exactly rather than approximately.

## Benchmark dataset

622 manually annotated articles, 13 scientific variables, of which the paper
reports the six with bounded answer spaces. Bibliographic metadata and
annotations only — see [data/README.md](data/README.md) for the data dictionary.

## Lookup table

`data/whitematter_lut.csv` holds 313 white-matter tract terms derived from
Uberon and SNOMED CT. `prompts/with_lut.py` renders the file into the system
prompt at run time, so the published table is exactly the one the models saw.

## What is not included

Article full text is under publisher copyright and is not redistributed. This
does not limit reproduction: every published number is recomputed from the model
predictions in `predictions/`, with no API key and no PDFs. Only re-running
extraction needs the corpus; [data/README.md](data/README.md) explains how to
rebuild it from the published PMIDs.

## Re-running extraction

```bash
python -m extraction.extract_with_lut --mode 1 --key 0
python -m extraction.extract_without_lut --mode 11 --key 3 --concurrency 20 --resume
./extraction/run_all.sh          # all 14 modes in parallel, one key each
```

`--mode`: 1 Abstract · 2 Methods · 3 Results · 4 Discussion · 5 Abs+Res ·
6 Abs+Meth · 7 Abs+Disc · 8 Meth+Res · 9 Meth+Disc · 10 Res+Disc ·
11 Abs+Meth+Res · 12 Abs+Meth+Disc · 13 Abs+Res+Disc · 14 Meth+Res+Disc

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT for code; CC BY 4.0 for `data/whitematter_dataset.csv` and
`data/whitematter_lut.csv`. See [LICENSE](LICENSE).
