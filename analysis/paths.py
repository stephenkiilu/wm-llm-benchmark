"""Where everything lives, and the names shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
ANNOTATIONS = DATA / "annotations.csv"          # reference labels, 622 articles
VOCABULARY = DATA / "vocabulary.csv"            # 122-tract look-up table
FULL_TEXT = DATA / "full_text" / "articles.json"  # not public; needed only to rerun extraction

PREDICTIONS = ROOT / "predictions"
FULL_TEXT_RUNS = PREDICTIONS / "full_text"
SECTION_RUNS = PREDICTIONS / "sections"
USAGE = PREDICTIONS / "usage"

TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"

MODELS = ["gpt-4o-mini", "gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.6-luna"]
WITH, WITHOUT = "with_vocabulary", "without_vocabulary"
ARMS = [WITH, WITHOUT]
EFFORT = "medium"                               # reasoning effort used for every GPT-5 run
GOLD_TRACTS = "What tracts were studied?"       # reference column for WM tracts


def run_name(model, arm, sections=None):
    """File stem of one run, e.g. gpt-5.6-luna__with_vocabulary__A_M."""
    return f"{model}__{arm}" + (f"__{sections.replace('+', '_')}" if sections else "")


def run_file(model, arm, sections=None):
    folder = SECTION_RUNS if sections else FULL_TEXT_RUNS
    return folder / f"{run_name(model, arm, sections)}.csv"


def usage_file(model, arm, sections=None):
    return USAGE / f"{run_name(model, arm, sections)}.jsonl"
