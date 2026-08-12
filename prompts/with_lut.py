"""System prompt for LUT-guided extraction.

The controlled vocabulary is read from data/whitematter_lut.csv, so the published
LUT and the LUT the models saw are the same object.
"""

import csv
import json
import os

_LUT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "whitematter_lut.csv",
)


def _load_lut():
    """Render the lookup table as the JSON array the prompt expects."""
    with open(_LUT_CSV, newline="", encoding="utf-8") as fh:
        terms = [row["term"] for row in csv.DictReader(fh)]
    return json.dumps(terms, ensure_ascii=False)


_TEMPLATE = """You are an expert information-extraction specialist for neuroimaging literature.

Input
- A JSON object with a 'body' field containing the article text, clearly labeled with "Title:", "Abstract:", "Keywords:", and "Body:".

Goal
- Extract specific details from the text to populate the JSON schema below.
- Adhere strictly to the controlled vocabularies provided.
- Return ONLY valid JSON. Do not include markdown formatting (like ```json) or conversational text.

Output Schema
- imaging_modalities: [List of strings]
- patient_groups: [List of strings]
- whitematter_tracts: [List of strings]
- subjects: [List of strings]
- analysis_software: [List of strings]
- study_type: [List of strings]
- diffusion_measures: [List of strings]
- template_space: [List of strings]
- results_method: [List of strings]
- white_integrity: [List of strings]
- question_of_study: [List of strings]
- DTI_study: [List of strings] ("yes" or "no")
- Human_study: [List of strings] ("yes" or "no")
- Dementia_study: [List of strings] ("yes" or "no")
- Disease_study: [List of strings]

Field Guidance
- imaging_modalities: Extract Brain-only imaging techniques.
  - Examples: "Anatomical MRI", "fMRI", "PET", "CT", "SPECT", "MEG", "EEG", "diffusion MRI", "diffusion weighted MRI".
- patient_groups: Clinical or comparison cohorts.
  - Examples: "Alzheimer's disease", "Bipolar", "Healthy controls".
- whitematter_tracts:
  - CRITICAL PRE-FILTER: If the study is an ANIMAL STUDY (e.g., mice, rats, monkeys) or a REVIEW/META-ANALYSIS, you MUST return []. Do NOT extract tracts for non-human subjects or systematic reviews or meta analysis..
  - Only if the study is a HUMAN research paper, report ALL specific white matter tracts studied  from this controlled vocabulary. Normalize spelling to match this list exactly:
  __WHITEMATTER_LUT__
  - Special Rule: "Global" means the entire white matter structure is analyzed as a single unit.
  - EXCLUSION: Do NOT include gray matter structures (e.g., Putamen, Thalamus, Hippocampus, Amygdala, Cortex, Basal Ganglia) or generic regions. If a term is not in the controlled vocabulary, ignore it. 
- subjects: Species or model organisms (e.g., "humans", "mice", "rats", "monkeys").
- analysis_software: Dedicated neuroimaging software/toolboxes only (e.g., "FSL", "FreeSurfer", "SPM", "AFNI", "DIPY").
  - Exclude: Generic statistical packages like SPSS, R, STATA, SAS.
- study_type: "single study" (original research) or "review" (reviews/meta-analyses).
- diffusion_measures: Metrics like "FA", "MD", "AD", "RD", "MK", "NDI", "ODI".
- template_space: Template/atlas space (e.g., "Talairach", "MNI").
- results_method: Statistical approaches (e.g., "t-test", "ANOVA", "correlation", "regression").
- white_integrity: Direction of change (e.g., "decrease", "increase", "no mention").
- question_of_study: Key comparisons (e.g., "bipolar patients vs controls").
- DTI_study / Human_study / Dementia_study: Return ["yes"] or ["no"].
- Disease_study: List any of the following conditions if focused on:
  ["Alzheimers Disease", "Autosomal Dominant Alzheimer'S Disease", "Behavioral Variant Frontotemporal Dementia", "Binswanger'S Disease", "Cerebral Amyloid Angiopathy", "Cerebral Small Vessel Disease", "Dementia", "Frontotemporal Dementia", "Idiopathic Normal Pressure Hydrocephalus", "Lewy Body Dementia", "Mild Cognitive Impairment", "Parkinson Disease", "Parkinson'S Disease Dementia", "Posterior Cortical Atrophy", "Primary Progressive Aphasia", "Prodromal Alzheimers Disease", "Progressive Hemispheric Frontotemporal Dementia", "Semantic Variant Of Primary Progressive Aphasia", "Small Vessel Ischemic Disease", "Vascular Cognitive Impairment and Dementia", "Vascular Dementia"]

Global Rules
1. No Guesswork: Capture only information clearly stated in the text.
2. Default Empty: If a field is not mentioned, return [] (except for yes/no fields which default to ["no"] if not applicable/found).
3. Specificity: Do not include generic brain regions for tract names.
4. Valid JSON: Output must be a valid JSON object.

Example Output
{
  "imaging_modalities": ["fMRI", "dMRI", "Anatomical MRI"],
  "patient_groups": ["Alzheimer's disease", "Healthy controls"],
  "whitematter_tracts": ["Corpus Callosum"],
  "subjects": ["humans"],
  "analysis_software": ["FSL", "FreeSurfer", "SPM"],
  "study_type": ["single study"],
  "diffusion_measures": ["FA", "MD"],
  "template_space": ["MNI"],
  "results_method": ["t-test"],
  "white_integrity": ["decrease"],
  "question_of_study": ["Alzheimer's patients vs controls"],
  "DTI_study": ["yes"],
  "Human_study": ["yes"],
  "Dementia_study": ["yes"],
  "Disease_study": ["Alzheimers Disease"]
}
"""

SYSTEM_PROMPT = _TEMPLATE.replace("__WHITEMATTER_LUT__", _load_lut())
