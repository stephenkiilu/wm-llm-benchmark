"""Extract structured metadata from paper text using the LUT-guided prompt.

Synchronous, one section combination per run.

Output: predictions/lut_ablation/
"""

from enum import IntEnum
from typing import List, Dict, Any, Tuple
import argparse
import csv
import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

from config import API_KEYS
from prompts.with_lut import SYSTEM_PROMPT

load_dotenv()

# Client is initialized in main() from the --key argument.
client: OpenAI = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHITEMATTER_JSON_PATH = os.path.join(BASE_DIR, "data", "WMT_FULLDATA.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions", "lut_ablation")


# ── Processing modes ──────────
class ProcessingMode(IntEnum):
    ABSTRACT                    = 1
    METHODS                     = 2
    RESULTS                     = 3
    DISCUSSION                  = 4
    ABSTRACT_RESULTS            = 5
    ABSTRACT_METHODS            = 6
    ABSTRACT_DISCUSSION         = 7
    METHODS_RESULTS             = 8
    METHODS_DISCUSSION          = 9
    RESULTS_DISCUSSION          = 10
    ABSTRACT_METHODS_RESULTS    = 11
    ABSTRACT_METHODS_DISCUSSION = 12
    ABSTRACT_RESULTS_DISCUSSION = 13
    METHODS_RESULTS_DISCUSSION  = 14


# Map each mode → (include_abstract: bool, content_section_keys: list[str])
# Section keys must match the keys in paper["content"], e.g. "Methods", "Results", "Discussion"
SECTION_MAP: Dict[ProcessingMode, Tuple[bool, List[str]]] = {
    ProcessingMode.ABSTRACT:                    (True,  []),
    ProcessingMode.METHODS:                     (False, ["Methods"]),
    ProcessingMode.RESULTS:                     (False, ["Results"]),
    ProcessingMode.DISCUSSION:                  (False, ["Discussion"]),
    ProcessingMode.ABSTRACT_RESULTS:            (True,  ["Results"]),
    ProcessingMode.ABSTRACT_METHODS:            (True,  ["Methods"]),
    ProcessingMode.ABSTRACT_DISCUSSION:         (True,  ["Discussion"]),
    ProcessingMode.METHODS_RESULTS:             (False, ["Methods", "Results"]),
    ProcessingMode.METHODS_DISCUSSION:          (False, ["Methods", "Discussion"]),
    ProcessingMode.RESULTS_DISCUSSION:          (False, ["Results", "Discussion"]),
    ProcessingMode.ABSTRACT_METHODS_RESULTS:    (True,  ["Methods", "Results"]),
    ProcessingMode.ABSTRACT_METHODS_DISCUSSION: (True,  ["Methods", "Discussion"]),
    ProcessingMode.ABSTRACT_RESULTS_DISCUSSION: (True,  ["Results", "Discussion"]),
    ProcessingMode.METHODS_RESULTS_DISCUSSION:  (False, ["Methods", "Results", "Discussion"]),
}

# Convenience list for batch runs
ALL_MODES = list(ProcessingMode)


# ── Extraction fields ──────────────────────────────────────────────────────────
EXTRACTION_FIELDS = [
    "whitematter_tracts", "study_type","DTI_study","Human_study","Dementia_study","Disease_study"]

CSV_FIELDNAMES = ["PMID", "title"] + EXTRACTION_FIELDS

# ── Load data ──────────────────────────────────────────────────────────────────
def load_papers() -> Dict[str, Any]:
    """Load the paper corpus. Full text is not distributed; see data/README.md."""
    with open(WHITEMATTER_JSON_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Helper utilities ───────────────────────────────────────────────────────────

def _get_paper_field(paper: Dict[str, Any], field: str) -> str:
    """Extract and convert a paper field to string, handling None values.

    Checks metadata first, then the top-level paper dict using both the
    original field name and a lowercase version (handles 'PMID' vs 'pmid').
    """
    metadata = paper.get("metadata", {})
    if field in metadata:
        value = metadata[field]
    elif field in paper:
        value = paper[field]
    else:
        # Case-insensitive fallback (e.g. 'PMID' -> 'pmid' stored at top level)
        field_lower = field.lower()
        value = next(
            (v for k, v in paper.items() if k.lower() == field_lower),
            None,
        )
    return str(value) if value is not None else ""


def _get_section_text(paper: Dict[str, Any], section_key: str) -> str:
    """
    Extract text from a content section.
    Handles both formats:
      - string values  (section → "text...")
      - dict values    (section → {"text": "...", "subsections": {...}})
    """
    content = paper.get("content", {})
    section = content.get(section_key)

    if section is None:
        return ""
    if isinstance(section, str):
        return section
    if isinstance(section, dict):
        # Collect text from the section and its subsections recursively
        return _flatten_section_dict(section)
    return str(section)


def _flatten_section_dict(section_data: Dict[str, Any]) -> str:
    """Recursively flatten a section dict into a single text string."""
    parts = []
    if "text" in section_data and section_data["text"]:
        parts.append(section_data["text"])
    if "subsections" in section_data and isinstance(section_data["subsections"], dict):
        for sub_title, sub_data in section_data["subsections"].items():
            if isinstance(sub_data, str):
                parts.append(sub_data)
            elif isinstance(sub_data, dict):
                parts.append(_flatten_section_dict(sub_data))
    return "\n".join(parts)


# ── Payload builder ───────────────────────────────────────────────────────────

def _build_payload(paper: Dict[str, Any],
                   include_abstract: bool,
                   section_keys: List[str]) -> str:
    """
    Build a JSON payload containing the title, optionally the abstract,
    and only the specified content sections.

    Args:
        paper:            Paper dictionary with metadata and content
        include_abstract: Whether to include the abstract
        section_keys:     List of section names to extract from paper["content"]
    """
    metadata = paper.get("metadata", {})
    title    = metadata.get("title") or paper.get("title", "")

    payload: Dict[str, Any] = {"title": title}

    if include_abstract:
        abstract = metadata.get("abstract") or paper.get("abstract", "")
        payload["abstract"] = abstract

    # Pull selected sections from content
    for key in section_keys:
        text = _get_section_text(paper, key)
        if text:
            payload[key.lower()] = text

    return json.dumps(payload, ensure_ascii=False)


def prepare_payload(paper: Dict[str, Any], mode: ProcessingMode) -> str:
    """
    Return the appropriate JSON payload string for the given processing mode.

    Args:
        paper: Dictionary containing paper details
        mode:  A ProcessingMode value (1–14)

    Returns:
        JSON string ready to be sent to the OpenAI API
    """
    include_abstract, section_keys = SECTION_MAP[mode]
    return _build_payload(paper, include_abstract, section_keys)



# ── API interaction ────────────────────────────────────────────────────────────

def _process_chunk_with_api(chunk: str, model: str) -> Dict[str, Any]:
    """Send a text chunk to the OpenAI API and parse the JSON response."""
    user_payload = {"body": chunk}
    content = ""
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=1,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = resp.choices[0].message.content
        return json.loads(content)

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Raw content: {content}")
        return {}
    except Exception as e:
        print(f"API error: {e}")
        return {}


def _merge_chunk_results(all_data: Dict[str, List], chunk_result: Dict[str, Any]) -> None:
    """Merge results from a single API response into the aggregated results dict."""
    for key in all_data:
        value = chunk_result.get(key)
        if isinstance(value, list):
            all_data[key].extend(value)
        elif value:
            all_data[key].append(value)


# ── Single-paper extraction ────────────────────────────────────────────────────

def extract_one(WM_paper: Dict[str, Any],
                model: str = "gpt-5-mini",
                mode: ProcessingMode = ProcessingMode.ABSTRACT) -> Dict[str, Any]:
    """
    Extract structured data from a single paper using the OpenAI API.

    Args:
        WM_paper: Dictionary containing paper details
        model:    OpenAI model identifier (default: gpt-5-mini)
        mode:     ProcessingMode (1–14)

    Returns:
        Dictionary containing extracted fields with lists of values
    """
    data = prepare_payload(WM_paper, mode)

    all_data = {field: [] for field in EXTRACTION_FIELDS}
    chunk_result = _process_chunk_with_api(data, model)
    _merge_chunk_results(all_data, chunk_result)

    # Deduplicate
    for key in all_data:
        all_data[key] = list(set(all_data[key]))

    return all_data


# ── CSV helpers ────────────────────────────────────────────────────────────────

def _build_csv_row(paper: Dict[str, Any], extracted_data: Dict[str, List]) -> Dict[str, str]:
    """Build a CSV row from paper metadata and extracted data."""
    row = {
        "PMID":  _get_paper_field(paper, "PMID"),
        "title": _get_paper_field(paper, "title"),
    }
    for field in EXTRACTION_FIELDS:
        row[field] = ";".join(extracted_data.get(field, []))
    return row


def _write_results_to_csv(results: List[Dict[str, str]], output_path: str) -> None:
    """Write extraction results to a CSV file."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(results)


def _output_csv_path(mode: ProcessingMode, model: str = "GPT_5") -> str:
    """Generate an output CSV path based on mode and model name."""
    mode_label = mode.name  # e.g. "ABSTRACT", "METHODS_RESULTS", etc.
    return f"{OUTPUT_DIR}/GPT_section_{mode_label}_{model}.csv"


# ── Batch extraction ───────────────────────────────────────────────────────────

def extract_all(WM_papers: List[Dict[str, Any]],
                out_csv: str | None = None,
                model: str = "gpt-5-mini",
                sleep_sec: float = 0.01,
                mode: ProcessingMode = ProcessingMode.ABSTRACT) -> List[Dict[str, str]]:
    """
    Extract data from multiple papers and save results to CSV.

    Args:
        WM_papers:  List of paper dictionaries to process
        out_csv:    Output CSV file path (auto-generated from mode if None)
        model:      OpenAI model identifier
        sleep_sec:  Delay between API calls to avoid rate limits
        mode:       ProcessingMode (1–14)

    Returns:
        List of CSV row dictionaries
    """
    if out_csv is None:
        out_csv = _output_csv_path(mode)

    results      = []
    total_papers = len(WM_papers)
    mode_label   = mode.name.replace("_", " + ").title()

    print(f"Starting [{mode.name}] (mode={int(mode)}) extraction for {total_papers} papers...")
    print(f"Sections: {mode_label}")
    print(f"Output → {out_csv}\n")

    for i, paper in enumerate(WM_papers, 13):
        extracted_data = extract_one(paper, model=model, mode=mode)
        row = _build_csv_row(paper, extracted_data)
        print(row)
        results.append(row)
        print(f"Processed {i}/{total_papers}: {row.get('PMID', 'Unknown')}")
        time.sleep(sleep_sec)

    _write_results_to_csv(results, out_csv)
    print(f"\n✅ Successfully saved {len(results)} records to {out_csv}")
    return results


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract entities from papers by section.")
    parser.add_argument("--mode", type=int, required=True, choices=range(1, 15),
                        help="Processing mode (1=Abstract, 2=Methods, ... 14=Methods+Results+Discussion)")
    parser.add_argument("--key", type=int, default=0, choices=range(7),
                        help="API key index (0–6, maps to OPENAI_API_KEY, OPENAI_API_KEY1, ..., OPENAI_API_KEY6)")
    args = parser.parse_args()

    # Initialize OpenAI client with the selected API key
    selected_key = API_KEYS[args.key]
    if not selected_key:
        raise ValueError(f"API key index {args.key} is not set in .env")
    client = OpenAI(api_key=selected_key)

    mode = ProcessingMode(args.mode)
    print(f"\n🚀 Mode: {int(mode)} ({mode.name}) | API Key: {args.key}")
    print(f"   Sections: {mode.name.replace('_', ' + ').title()}\n")

    extract_all(load_papers(), mode=mode)
