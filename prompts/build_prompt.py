"""The extraction prompt, in two arms built from one template.

The arms differ only in the white-matter-tract instructions: with_vocabulary
adds the 122 preferred tract names from data/vocabulary.csv (synonyms are left
to the scorer). Run this file to check the arms and rewrite the rendered
prompts/with_vocabulary.txt and without_vocabulary.txt.
"""

import sys
from pathlib import Path

import pandas as pd

# ── LUT ───────────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
LUT_PATH = HERE.parent / "data" / "vocabulary.csv"


# Strings that are absent values, not tract names. `str(nan)` is "nan", which is
# truthy - so a blank LUT row will silently enter the prompt as a tract called
# "nan" unless it is filtered explicitly.
_NOT_A_LABEL = {"", "nan", "none", "null", "na", "n/a", "-"}


def load_vocabulary(path=LUT_PATH) -> list:
    """Canonical tract labels, sorted, from the LUT. Blank rows are dropped."""
    table = pd.read_csv(path)
    labels, dropped = set(), 0
    for value in table["preferred_label"]:
        if pd.isna(value):
            dropped += 1
            continue
        label = str(value).strip()
        if label.lower() in _NOT_A_LABEL:
            dropped += 1
            continue
        labels.add(label)
    if not labels:
        raise ValueError(f"no usable preferred_label values in {path}")
    if dropped:
        print(f"[build_prompt] warning: dropped {dropped} blank LUT row(s) "
              f"from the prompt vocabulary", file=sys.stderr)
    return sorted(labels)


VOCABULARY = load_vocabulary()
_VOCAB_BLOCK = "[" + ", ".join(f'"{t}"' for t in VOCABULARY) + "]"


# ── TRACT FIELD GUIDANCE ──────────────────────────────────────────────────────
# The ONLY text that differs between the two arms.

_TRACTS_NOLUT = """- whitematter_tracts:
  - CRITICAL PRE-FILTER: If the study is an ANIMAL STUDY (e.g., mice, rats, monkeys)
    or a REVIEW/META-ANALYSIS, you MUST return []. Do NOT extract tracts for
    non-human subjects or systematic reviews or meta-analyses.
  - Only if the study is a HUMAN research paper, report ALL specific white matter
    tracts studied, e.g., "Corpus Callosum", "Fornix", "Uncinate Fasciculus",
    "Superior Longitudinal Fasciculus", "Corticospinal Tract", etc.
  - Special Rule - "Global": report "Global" when the study reports a white-matter
    measure averaged over the whole brain, a whole-brain skeleton, or total white
    matter (e.g. whole-brain mean FA, global TBSS skeleton). "Global" is NOT
    exclusive - if the paper also reports specific tracts, list "Global" AND those
    tracts.
  - EXCLUSION: Do NOT include gray matter structures (e.g., Putamen, Thalamus,
    Hippocampus, Amygdala, Cortex, Basal Ganglia) or generic regions."""

_TRACTS_LUT = """- whitematter_tracts:
  - CRITICAL PRE-FILTER: If the study is an ANIMAL STUDY (e.g., mice, rats, monkeys)
    or a REVIEW/META-ANALYSIS, you MUST return []. Do NOT extract tracts for
    non-human subjects or systematic reviews or meta-analyses.
  - Only if the study is a HUMAN research paper, report ALL specific white matter
    tracts studied, using this controlled vocabulary. Normalize spelling to match
    these labels exactly:
    {vocab}
  - Special Rule - "Global": report "Global" when the study reports a white-matter
    measure averaged over the whole brain, a whole-brain skeleton, or total white
    matter (e.g. whole-brain mean FA, global TBSS skeleton). "Global" is NOT
    exclusive - if the paper also reports specific tracts, list "Global" AND those
    tracts.
  - Report every tract the study examined. If a tract is clearly studied but you
    cannot find an exact match in the vocabulary, use the closest label rather
    than omitting it.
  - EXCLUSION: Do NOT include gray matter structures (e.g., Putamen, Thalamus,
    Hippocampus, Amygdala, Cortex, Basal Ganglia) or generic regions.""".format(
    vocab=_VOCAB_BLOCK
)


# ── SHARED TEMPLATE ───────────────────────────────────────────────────────────

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
- imaging_modalities: Extract brain-only imaging techniques.
  - Examples: "Anatomical MRI", "fMRI", "PET", "CT", "SPECT", "MEG", "EEG", "diffusion MRI", "diffusion weighted MRI".
- patient_groups: Clinical or comparison cohorts.
  - Examples: "Alzheimer's disease", "Bipolar", "Healthy controls".
{tracts}
- subjects: Species or model organisms (e.g., "humans", "mice", "rats", "monkeys").
- analysis_software: Dedicated neuroimaging software/toolboxes only (e.g., "FSL", "FreeSurfer", "SPM", "AFNI", "DIPY").
  - Exclude: Generic statistical packages like SPSS, R, STATA, SAS.
- study_type: "single study" (original research) or "review" (reviews/meta-analyses).
- diffusion_measures: Metrics like "FA", "MD", "AD", "RD", "MK", "NDI", "ODI".
- template_space: Template/atlas space (e.g., "Talairach", "MNI", "JHU").
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
{{
  "imaging_modalities": ["diffusion MRI"],
  "patient_groups": ["Alzheimer's disease", "Healthy controls"],
  "whitematter_tracts": ["Corpus Callosum", "fornix"],
  "subjects": ["humans"],
  "analysis_software": ["FSL"],
  "study_type": ["single study"],
  "diffusion_measures": ["FA", "MD"],
  "template_space": ["MNI"],
  "results_method": ["t-test"],
  "white_integrity": ["decrease"],
  "question_of_study": ["Alzheimer's disease vs healthy controls"],
  "DTI_study": ["yes"],
  "Human_study": ["yes"],
  "Dementia_study": ["yes"],
  "Disease_study": ["Alzheimers Disease"]
}}
"""

SYSTEM_PROMPT_NOLUT15 = _TEMPLATE.format(tracts=_TRACTS_NOLUT)
SYSTEM_PROMPT_LUT15 = _TEMPLATE.format(tracts=_TRACTS_LUT)

PROMPTS = {"without_vocabulary": SYSTEM_PROMPT_NOLUT15, "with_vocabulary": SYSTEM_PROMPT_LUT15}

EXTRACTION_FIELDS = [
    "imaging_modalities", "patient_groups", "whitematter_tracts", "subjects",
    "analysis_software", "study_type", "diffusion_measures", "template_space",
    "results_method", "white_integrity", "question_of_study",
    "DTI_study", "Human_study", "Dementia_study", "Disease_study",
]


if __name__ == "__main__":
    import difflib

    print(f"vocabulary: {len(VOCABULARY)} canonical labels from {LUT_PATH.name}")
    try:                                        # token counts are optional
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        for name, text in PROMPTS.items():
            print(f"  {name:<19} {len(enc.encode(text)):>6,} tokens")
    except ImportError:
        pass

    # the two arms must differ ONLY inside the whitematter_tracts guidance
    diff = [l for l in difflib.unified_diff(
        SYSTEM_PROMPT_NOLUT15.splitlines(), SYSTEM_PROMPT_LUT15.splitlines(), lineterm="", n=0)
        if l and l[0] in "+-" and not l.startswith(("---", "+++"))]
    print(f"\ndiffering lines: {len(diff)}")
    for l in diff:
        print(f"   {l[:100]}")

    assert SYSTEM_PROMPT_LUT15.count("Output Schema") == 1
    for f in EXTRACTION_FIELDS:
        assert f in SYSTEM_PROMPT_NOLUT15 and f in SYSTEM_PROMPT_LUT15, f
    assert "controlled vocabulary" in SYSTEM_PROMPT_LUT15
    assert "controlled vocabulary" not in _TRACTS_NOLUT
    for junk in ("nan", "none", "null"):
        assert f'"{junk}"' not in _VOCAB_BLOCK, f"{junk!r} leaked into the vocabulary"
    for name, text in PROMPTS.items():
        (HERE / f"{name}.txt").write_text(text)
    print("\nOK: both arms carry all 15 fields; vocabulary in with_vocabulary only")
