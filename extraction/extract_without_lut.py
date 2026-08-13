"""Extract structured metadata from paper text without the LUT in the prompt.

Asynchronous, with a concurrency limit 

Output: predictions/no_lut/
"""

from enum import IntEnum
from typing import List, Dict, Any, Tuple, Set
import argparse
import asyncio
import csv
import json
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from config import API_KEYS
from prompts.without_lut import SYSTEM_PROMPT

load_dotenv()

# Client is initialized in main() from the --key argument.
async_client: AsyncOpenAI = None

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WHITEMATTER_JSON_PATH = os.path.join(BASE_DIR, "data", "WMT_FULLDATA.json")
OUTPUT_DIR = os.path.join(BASE_DIR, "predictions", "no_lut")


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


# ── Extraction fields ──────────────────────────────────────────────────────────
EXTRACTION_FIELDS = [
    "whitematter_tracts"]

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
    """
    metadata = paper.get("metadata", {})
    title    = metadata.get("title") or paper.get("title", "")

    payload: Dict[str, Any] = {"title": title}

    if include_abstract:
        abstract = metadata.get("abstract") or paper.get("abstract", "")
        payload["abstract"] = abstract

    for key in section_keys:
        text = _get_section_text(paper, key)
        if text:
            payload[key.lower()] = text

    return json.dumps(payload, ensure_ascii=False)


def prepare_payload(paper: Dict[str, Any], mode: ProcessingMode) -> str:
    """Return the appropriate JSON payload string for the given processing mode."""
    include_abstract, section_keys = SECTION_MAP[mode]
    return _build_payload(paper, include_abstract, section_keys)


# ── Async API interaction ─────────────────────────────────────────────────────

async def _process_chunk_with_api_async(chunk: str, model: str,
                                        max_retries: int = 5,
                                        base_delay: float = 5.0) -> Dict[str, Any]:
    """Send a text chunk to the OpenAI API asynchronously and parse the JSON response.

    Retries up to `max_retries` times with exponential backoff on rate-limit
    (429) errors or any transient API failure.
    """
    user_payload = {"body": chunk}
    content = ""

    for attempt in range(1, max_retries + 1):
        try:
            resp = await async_client.chat.completions.create(
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
            print(f"JSON parsing error (attempt {attempt}/{max_retries}): {e}")
            print(f"Raw content: {content}")
            # JSON parse errors are not retryable — bad model output
            return {}

        except Exception as e:
            err_str = str(e)
            is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower() or "rate limit" in err_str.lower()

            if attempt < max_retries:
                delay = base_delay * (2 ** (attempt - 1))  # 5s, 10s, 20s, 40s ...
                if is_rate_limit:
                    print(f"⏳ Rate limit hit — waiting {delay:.0f}s before retry {attempt}/{max_retries}...")
                else:
                    print(f"⚠️  API error (attempt {attempt}/{max_retries}): {e} — retrying in {delay:.0f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"❌ API error after {max_retries} attempts: {e}")
                return {}


def _merge_chunk_results(all_data: Dict[str, List], chunk_result: Dict[str, Any]) -> None:
    """Merge results from a single API response into the aggregated results dict."""
    for key in all_data:
        value = chunk_result.get(key)
        if isinstance(value, list):
            all_data[key].extend(value)
        elif value:
            all_data[key].append(value)


# ── Single-paper extraction (async) ───────────

async def extract_one_async(WM_paper: Dict[str, Any],
                            model: str = "gpt-5-mini",
                            mode: ProcessingMode = ProcessingMode.METHODS_RESULTS) -> Dict[str, Any]:
    """
    Extract structured data from a single paper using the OpenAI API (async).

    Args:
        WM_paper: Dictionary containing paper details
        model:    OpenAI model identifier (default: gpt-5-mini)
        mode:     ProcessingMode (1–14)

    Returns:
        Dictionary containing extracted fields with lists of values
    """
    data = prepare_payload(WM_paper, mode)

    all_data = {field: [] for field in EXTRACTION_FIELDS}
    chunk_result = await _process_chunk_with_api_async(data, model)
    _merge_chunk_results(all_data, chunk_result)

    # Deduplicate
    for key in all_data:
        all_data[key] = list(set(all_data[key]))

    return all_data


# ── CSV helpers ───────────────────────

def _build_csv_row(paper: Dict[str, Any], extracted_data: Dict[str, List]) -> Dict[str, str]:
    """Build a CSV row from paper metadata and extracted data."""
    row = {
        "PMID":  _get_paper_field(paper, "PMID"),
        "title": _get_paper_field(paper, "title"),
    }
    for field in EXTRACTION_FIELDS:
        row[field] = ";".join(extracted_data.get(field, []))
    return row


def _output_csv_path(mode: ProcessingMode, model: str = "GPT_5") -> str:
    """Generate an output CSV path based on mode and model name."""
    mode_label = mode.name
    return f"{OUTPUT_DIR}/GPT_section_{mode_label}_{model}.csv"


def _load_completed_pmids(csv_path: str) -> Set[str]:
    """Load PMIDs that have already been processed from an existing CSV."""
    completed = set()
    if not os.path.exists(csv_path):
        return completed
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pmid = row.get("PMID", "").strip()
            if pmid:
                completed.add(pmid)
    return completed


# ── Async batch extraction with progressive CSV writing ───────────────────────

async def _process_single_paper(paper: Dict[str, Any],
                                index: int,
                                total: int,
                                model: str,
                                mode: ProcessingMode,
                                semaphore: asyncio.Semaphore,
                                csv_writer,
                                csv_file,
                                lock: asyncio.Lock) -> Dict[str, str]:
    """Process a single paper with semaphore-controlled concurrency."""
    async with semaphore:
        extracted_data = await extract_one_async(paper, model=model, mode=mode)
        row = _build_csv_row(paper, extracted_data)

        # Write row to CSV immediately (progressive writing)
        async with lock:
            csv_writer.writerow(row)
            csv_file.flush()
            print(f"Processed {index}/{total}: {row.get('PMID', 'Unknown')}")

        return row


async def extract_all_async(WM_papers: List[Dict[str, Any]],
                            out_csv: str | None = None,
                            model: str = "gpt-5-mini",
                            mode: ProcessingMode = ProcessingMode.METHODS_RESULTS,
                            concurrency: int = 10,
                            resume: bool = False) -> List[Dict[str, str]]:
    """
    Extract data from multiple papers using async API calls with progressive CSV writing.

    Args:
        WM_papers:    List of paper dictionaries to process
        out_csv:      Output CSV file path (auto-generated from mode if None)
        model:        OpenAI model identifier
        mode:         ProcessingMode (1–14)
        concurrency:  Max number of concurrent API calls (default: 10)
        resume:       If True, skip papers already in the CSV and append new results

    Returns:
        List of CSV row dictionaries
    """
    if out_csv is None:
        out_csv = _output_csv_path(mode)

    # If resuming, filter out already-completed papers
    if resume:
        completed_pmids = _load_completed_pmids(out_csv)
        papers_to_process = [
            p for p in WM_papers
            if _get_paper_field(p, "PMID") not in completed_pmids
        ]
        print(f"Resuming: {len(completed_pmids)} papers already done, {len(papers_to_process)} remaining")
    else:
        papers_to_process = WM_papers

    total_papers = len(papers_to_process)
    mode_label   = mode.name.replace("_", " + ").title()
    semaphore    = asyncio.Semaphore(concurrency)
    lock         = asyncio.Lock()

    print(f"Starting [{mode.name}] (mode={int(mode)}) async extraction for {total_papers} papers...")
    print(f"Sections: {mode_label}")
    print(f"Concurrency: {concurrency} simultaneous API calls")
    print(f"Output → {out_csv}\n")

    # Open CSV: append if resuming, write fresh otherwise
    file_mode = "a" if resume else "w"
    with open(out_csv, file_mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDNAMES)
        if not resume:
            writer.writeheader()

        tasks = [
            _process_single_paper(paper, i, total_papers, model, mode,
                                  semaphore, writer, csv_file, lock)
            for i, paper in enumerate(papers_to_process, 1)
        ]

        results = await asyncio.gather(*tasks)

    print(f"\n✅ Successfully saved {len(results)} new records to {out_csv}")
    return results

# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async entity extraction from papers by section (modes 1–14).")
    parser.add_argument("--mode", type=int, required=True, choices=range(1, 15),
                        help="Processing mode (1=Abstract, 2=Methods, ... 14=Methods+Results+Discussion)")
    parser.add_argument("--key", type=int, default=0, choices=range(7),
                        help="API key index (0–6)")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls (default: 10)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing CSV, skipping already-completed papers")
    args = parser.parse_args()

    # Initialize async OpenAI client with the selected API key
    selected_key = API_KEYS[args.key]
    if not selected_key:
        raise ValueError(f"API key index {args.key} is not set in .env")
    async_client = AsyncOpenAI(api_key=selected_key)

    mode = ProcessingMode(args.mode)
    print(f"\n🚀 Mode: {int(mode)} ({mode.name}) | API Key: {args.key} | Concurrency: {args.concurrency}")
    print(f"   Sections: {mode.name.replace('_', ' + ').title()}\n")

    if args.resume:
        print("   📂 RESUME MODE: will skip already-completed papers\n")
    asyncio.run(extract_all_async(load_papers(), mode=mode, concurrency=args.concurrency, resume=args.resume))
