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
python -m evaluation.model_comparison      
python -m evaluation.lut_ablation          
python -m evaluation.section_ablation     
python -m evaluation.task_complexity      

python -m evaluation.figure_section_ablation    
python -m evaluation.figure_lut_complexity      
python -m evaluation.figure_publication_growth
python -m evaluation.figure_model_f1           
python -m evaluation.figure_model_accuracy
```

Bootstrap resampling is seeded (`RANDOM_SEED = 42`, `N_BOOT = 1000`), so these
commands reproduce the published numbers exactly rather than approximately.

## Benchmark dataset

622 manually annotated articles, 13 scientific variables, of which the paper
reports the six with bounded answer spaces. Bibliographic metadata and
annotations only — see [data/README.md](data/README.md) for the data dictionary.

## Lookup table

`data/Whitematter_Look_up_table.csv` holds 121 canonical white-matter tracts,
each with its SNOMED CT and Uberon identifiers, a short definition, its
ontology synonyms and Latin variants, and the atlases that list it
(ICBM-DTI-81, TractSeg, XTRACT, TRACULA, AFQ and others). See [data/README.md](data/README.md) for the column dictionary.

The flat list of tract strings given to the models, which the reported LUT
ablation used, is inlined in `prompts/with_lut.py` as `_LUT_TERMS` rather than
loaded from a file, so that module is exactly the prompt they saw.

## What is not included

Article full text is under publisher copyright and is not redistributed. This
does not limit reproduction: every published number is recomputed from the model
predictions in `predictions/`, with no API key and no PDFs. Only re-running
extraction needs the corpus; [data/README.md](data/README.md) explains how to
rebuild it from the published PMIDs.

## Citation

See [CITATION.cff](CITATION.cff).

## License

MIT for code, see [LICENSE](LICENSE). The files in `data/` are CC BY 4.0,
see [LICENSE-DATA](LICENSE-DATA).
