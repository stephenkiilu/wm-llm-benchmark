"""Run the extraction: one API call per article, writing predictions/ and usage logs.

Both arms extract the same 15 fields; with_vocabulary adds the tract vocabulary
to the prompt. Needs OPENAI_API_KEY (in .env at the repo root) and the full
texts in data/full_text/articles.json, which are not redistributed.

    python extraction/run_extraction.py --model gpt-5.6-luna --arm with_vocabulary --effort medium
    python extraction/run_extraction.py --model gpt-5.6-luna --arm with_vocabulary \
        --effort medium --sections all14 --service-tier flex      # section inputs
    ... --dry-run     # estimate tokens and cost, no API calls
    ... --resume      # continue an interrupted run
"""

import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Set

from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "analysis")]

from paths import FULL_TEXT as PAPERS_PATH, run_file, usage_file  # noqa: E402
from prompts.build_prompt import EXTRACTION_FIELDS, PROMPTS      # noqa: E402

load_dotenv(ROOT / ".env")
API_KEY = os.getenv("OPENAI_API_KEY")

# reasoning_effort is rejected by non-reasoning models
REASONING_MODELS = {"gpt-5-mini", "gpt-5.4-mini", "gpt-5.4-nano", "gpt-5.6-luna"}

CSV_FIELDNAMES = ["PMID", "title"] + EXTRACTION_FIELDS
MAX_RETRIES = 5

# Flex trades latency for price and answers 429 when capacity is short, so it
# needs to wait longer and give up later than a standard run would.
FLEX_RETRIES, FLEX_BASE_DELAY = 8, 5.0

# Article-section inputs: label -> (include abstract, content sections).
# Every input also carries the title.
SECTION_MAP: Dict[str, tuple] = {
    "A":     (True,  []),
    "M":     (False, ["Methods"]),
    "R":     (False, ["Results"]),
    "D":     (False, ["Discussion"]),
    "A+M":   (True,  ["Methods"]),
    "A+R":   (True,  ["Results"]),
    "A+D":   (True,  ["Discussion"]),
    "M+R":   (False, ["Methods", "Results"]),
    "M+D":   (False, ["Methods", "Discussion"]),
    "R+D":   (False, ["Results", "Discussion"]),
    "A+M+R": (True,  ["Methods", "Results"]),
    "A+M+D": (True,  ["Methods", "Discussion"]),
    "A+R+D": (True,  ["Results", "Discussion"]),
    "M+R+D": (False, ["Methods", "Results", "Discussion"]),
}
ALL14 = list(SECTION_MAP)


# ── PAYLOAD ───────────────────────────────────────────────────────────────────

def _flatten_section(section: Any) -> str:
    """Flatten a content section, which may be a string or nested dict."""
    if isinstance(section, str):
        return section
    if not isinstance(section, dict):
        return str(section) if section else ""
    parts = []
    if section.get("text"):
        parts.append(section["text"])
    for _, sub in (section.get("subsections") or {}).items():
        parts.append(_flatten_section(sub))
    return "\n".join(p for p in parts if p)


def build_payload(paper: Dict[str, Any], config: Optional[str] = None) -> str:
    """JSON payload for one paper.

    config=None -> full text: title, abstract and every content section.
    config set  -> title, the abstract only if the configuration includes it,
                   and only the named content sections. A section absent from
                   the source JSON is simply omitted; those
                   papers are still scored, so an empty Methods is a real part
                   of the measurement rather than a dropped row.
    """
    meta = paper.get("metadata", {})
    body = {"title": meta.get("title") or ""}
    if config is None:
        body["abstract"] = meta.get("abstract") or ""
        wanted = list((paper.get("content") or {}).keys())
    else:
        include_abstract, wanted = SECTION_MAP[config]
        if include_abstract:
            body["abstract"] = meta.get("abstract") or ""
    content = paper.get("content") or {}
    for key in wanted:
        text = _flatten_section(content.get(key))
        if text:
            body[key.lower()] = text
    return json.dumps({"body": json.dumps(body, ensure_ascii=False)}, ensure_ascii=False)


def paper_id(paper: Dict[str, Any]) -> str:
    meta = paper.get("metadata", {})
    return str(meta.get("PMID") or paper.get("pmid") or "")


def paper_title(paper: Dict[str, Any]) -> str:
    meta = paper.get("metadata", {})
    return str(meta.get("title") or paper.get("title") or "")


# ── API ───────────────────────────────────────────────────────────────────────

def request_kwargs(model: str, effort: Optional[str], seed: Optional[int],
                   service_tier: Optional[str] = None) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"model": model, "temperature": 1}
    # Scheduling and billing only: service_tier does not change the model, its
    # parameters or its output, so a flex run stays comparable with a standard
    # one. Omitted entirely when None - sending "auto" would be a no-op that
    # still shows up in the request.
    if service_tier:
        kwargs["service_tier"] = service_tier
    if effort and model in REASONING_MODELS:
        kwargs["reasoning_effort"] = effort
    # Best-effort determinism. Verified to give byte-identical output across
    # repeated calls for gpt-4o-mini, gpt-5.6-luna and gpt-5.4-mini. Only
    # gpt-4o-mini returns a system_fingerprint, so for the GPT-5 family a silent
    # backend change cannot be detected - hence logging the fingerprint anyway.
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs


async def extract_one(client: AsyncOpenAI, paper: Dict[str, Any], system_prompt: str,
                      model: str, arm: str, effort: Optional[str],
                      seed: Optional[int], semaphore: asyncio.Semaphore,
                      config: Optional[str] = None,
                      service_tier: Optional[str] = None,
                      max_retries: int = MAX_RETRIES,
                      base_delay: float = 1.0) -> Dict[str, Any]:
    """One paper -> one row plus a usage record. Never raises."""
    payload = build_payload(paper, config)
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": payload}]

    async with semaphore:
        for attempt in range(1, max_retries + 1):
            started = time.perf_counter()
            try:
                resp = await client.chat.completions.create(
                    **request_kwargs(model, effort, seed, service_tier),
                    messages=messages)
                latency = time.perf_counter() - started
                usage = resp.usage
                out_details = getattr(usage, "completion_tokens_details", None)
                in_details = getattr(usage, "prompt_tokens_details", None)
                content = resp.choices[0].message.content or ""

                try:
                    parsed = json.loads(content)
                    parse_ok = isinstance(parsed, dict)
                except json.JSONDecodeError:
                    parsed, parse_ok = {}, False

                row = {"PMID": paper_id(paper), "title": paper_title(paper)}
                for field in EXTRACTION_FIELDS:
                    value = parsed.get(field, []) if parse_ok else []
                    if not isinstance(value, list):
                        value = [value]
                    row[field] = ";".join(str(v).strip() for v in value if str(v).strip())

                record = {
                    "PMID": paper_id(paper), "model": model, "arm": arm,
                    "config": config or "full",
                    "service_tier_requested": service_tier or "standard",
                    "service_tier_served": getattr(resp, "service_tier", None),
                    "effort": effort or "default", "latency_s": round(latency, 3),
                    "prompt_tokens": usage.prompt_tokens,
                    "cached_tokens": getattr(in_details, "cached_tokens", 0) or 0,
                    "completion_tokens": usage.completion_tokens,
                    "reasoning_tokens": getattr(out_details, "reasoning_tokens", 0) or 0,
                    "finish_reason": resp.choices[0].finish_reason,
                    "model_served": resp.model,          # pins the exact snapshot
                    "system_fingerprint": getattr(resp, "system_fingerprint", None),
                    "seed": seed,
                    "attempts": attempt, "parse_ok": parse_ok,
                    "raw_response": content,             # so metrics stay recomputable
                }
                return {"row": row, "failed": False, "usage": record}

            except Exception as exc:                     # rate limits, timeouts, 5xx
                if attempt == max_retries:
                    # Deliberately NO csv row. Writing one would record the PMID
                    # as done, so --resume would skip it forever and the
                    # per-arm retry passes could never recover it. Absent from
                    # the csv, it is retried on the next pass; the error is
                    # still logged, and evaluate.py excludes unrun papers.
                    return {"row": None, "failed": True, "usage": {
                        "PMID": paper_id(paper), "model": model, "arm": arm,
                        "config": config or "full",
                        "service_tier_requested": service_tier or "standard",
                        "service_tier_served": None,
                        "effort": effort or "default", "latency_s": 0,
                        "prompt_tokens": 0, "cached_tokens": 0, "completion_tokens": 0,
                        "reasoning_tokens": 0, "finish_reason": "error",
                        "model_served": "", "system_fingerprint": None, "seed": seed,
                        "attempts": attempt, "parse_ok": False,
                        "raw_response": "", "error": f"{type(exc).__name__}: {exc}"[:300]}}
                delay = base_delay * 2 ** (attempt - 1) + random.random()
                print(f"  retry {attempt}/{max_retries} in {delay:.0f}s "
                      f"({type(exc).__name__})", flush=True)
                await asyncio.sleep(delay)


# ── ORCHESTRATION ─────────────────────────────────────────────────────────────

def output_paths(model: str, arm: str, config: Optional[str] = None) -> tuple:
    return str(run_file(model, arm, config)), str(usage_file(model, arm, config))


def already_done(path: str) -> Set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["PMID"] for r in csv.DictReader(fh) if r.get("PMID")}


async def run(model: str, arm: str, effort: Optional[str], concurrency: int,
              resume: bool, limit: Optional[int],
              seed: Optional[int], config: Optional[str] = None,
              service_tier: Optional[str] = None) -> None:
    with open(PAPERS_PATH, encoding="utf-8") as fh:
        papers = json.load(fh)
    if limit:
        papers = papers[:limit]

    csv_path, usage_path = output_paths(model, arm, config)
    max_retries = FLEX_RETRIES if service_tier == "flex" else MAX_RETRIES
    base_delay = FLEX_BASE_DELAY if service_tier == "flex" else 1.0
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(usage_path), exist_ok=True)

    if not resume and os.path.exists(csv_path):
        existing = len(already_done(csv_path))
        print(f"REFUSING to overwrite {os.path.basename(csv_path)} "
              f"({existing} rows already present).\n"
              f"Pass --resume to continue it, or delete the file to start over.")
        return

    done = already_done(csv_path) if resume else set()
    todo = [p for p in papers if paper_id(p) not in done]
    if not todo:
        print(f"nothing to do: {len(done)} rows already in {os.path.basename(csv_path)}")
        return

    print(f"model={model}  arm={arm}  input={config or 'full text'}  "
          f"effort={effort or 'default'}  tier={service_tier or 'standard'}  "
          f"seed={seed}  concurrency={concurrency}")
    print(f"papers: {len(todo)} to run"
          f"{f' ({len(done)} already done)' if done else ''}  ->  {os.path.basename(csv_path)}")

    client = AsyncOpenAI(api_key=API_KEY, timeout=600.0, max_retries=0)
    semaphore = asyncio.Semaphore(concurrency)
    started = time.perf_counter()

    # Write each result the moment it lands, not at the end. A 622-paper run can
    # take 20+ minutes; buffering it all in memory means one Ctrl-C, crash or
    # dropped connection loses the whole run and leaves nothing for --resume.
    append = resume and os.path.exists(csv_path)
    results: List[Dict[str, Any]] = []
    lock = asyncio.Lock()

    with open(csv_path, "a" if append else "w", newline="", encoding="utf-8") as csv_fh, \
         open(usage_path, "a" if append else "w", encoding="utf-8") as usage_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=CSV_FIELDNAMES)
        if not append:
            writer.writeheader()
            csv_fh.flush()

        async def worker(paper):
            result = await extract_one(client, paper, PROMPTS[arm], model, arm,
                                       effort, seed, semaphore, config,
                                       service_tier, max_retries, base_delay)
            async with lock:
                if result["row"] is not None:
                    writer.writerow(result["row"])
                usage_fh.write(json.dumps(result["usage"]) + "\n")
                csv_fh.flush()
                usage_fh.flush()
                os.fsync(csv_fh.fileno())
                results.append(result)
                done_n = len(results)
                if done_n % 50 == 0 or done_n == len(todo):
                    rate = done_n / (time.perf_counter() - started)
                    eta = (len(todo) - done_n) / rate / 60 if rate else 0
                    print(f"  {done_n}/{len(todo)}  ({rate * 60:.0f}/min, "
                          f"eta {eta:.1f} min)", flush=True)
            return result

        try:
            await asyncio.gather(*[worker(p) for p in todo])
        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"\ninterrupted - {len(results)} rows already written and safe.")
            print(f"re-run the same command with --resume to continue.")
            raise

    wall = time.perf_counter() - started
    failed = [r for r in results if r["failed"]]
    ok = sum(r["usage"]["parse_ok"] for r in results)
    tok_in = sum(r["usage"]["prompt_tokens"] for r in results)
    tok_out = sum(r["usage"]["completion_tokens"] for r in results)
    reasoning = sum(r["usage"]["reasoning_tokens"] for r in results)
    print(f"done in {wall / 60:.1f} min  |  parsed {ok}/{len(results)}  |  "
          f"in {tok_in:,} out {tok_out:,} (reasoning {reasoning:,})")
    if failed:
        print(f"  !! {len(failed)} paper(s) failed after {max_retries} retries and were "
              f"NOT written - re-run with --resume to retry them")
        for r in failed[:5]:
            print(f"     PMID {r['usage']['PMID']}: {r['usage'].get('error','')[:90]}")
    print(f"  {csv_path}\n  {usage_path}")


# ── DRY RUN ───────────────────────────────────────────────────────────────────

def _token_counter():
    """tiktoken when available; otherwise chars/4, which measured 6.4% high
    against the logged full-text runs, so the fallback is corrected by that."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("o200k_base")
        return (lambda t: len(enc.encode(t or ""))), "tiktoken o200k_base", 1.0
    except ImportError:
        return (lambda t: len(t or "") // 4), "chars/4 (tiktoken not installed)", 0.936


def dry_run(model: str, arm: str, effort: Optional[str], configs: List[Optional[str]],
            service_tier: Optional[str], limit: Optional[int]) -> None:
    """Price a run without calling the API.

    Input tokens are counted from the payloads that would actually be sent.
    Output tokens cannot be counted in advance, so they are carried over from
    this model/arm's completed full-text run and shown as a range: the upper
    figure assumes output does not shrink with the smaller input, the lower
    assumes it shrinks with it. Reasoning and the fixed 15-field schema
    dominate output, so expect the upper end.
    """
    from evaluate import PRICING

    tok, tok_name, cal = _token_counter()
    with open(PAPERS_PATH, encoding="utf-8") as fh:
        papers = json.load(fh)
    if limit:
        papers = papers[:limit]

    price = PRICING.get(model)
    if not price:
        print(f"no price on record for {model}; add it to analysis/evaluate.py PRICING")
        return
    pin, pcached, pout = price
    if service_tier in ("flex", "batch"):
        pin, pcached, pout = pin / 2, pcached / 2, pout / 2

    sys_tokens = tok(PROMPTS[arm])
    full_body = sum(tok(json.loads(build_payload(p))["body"]) for p in papers)

    ref_out = _reference_output_tokens(model, arm)
    if ref_out is None:
        print("no completed full-text run for this model and arm: cannot "
              "project output tokens. Run one config with --limit first.")
        return

    print(f"tokeniser: {tok_name}")
    print(f"pricing:   ${pin}/M fresh in, ${pcached}/M cached, ${pout}/M out"
          f"  ({service_tier or 'standard'} tier)")
    print(f"papers:    {len(papers)}   system prompt: {sys_tokens:,} tokens\n")
    print(f"{'input':10s} {'fresh in':>12s} {'cached':>12s} {'output':>18s} {'cost':>16s}")

    tot_lo = tot_hi = 0.0
    for cfg in configs:
        body = sum(tok(json.loads(build_payload(p, cfg))["body"]) for p in papers)
        fresh = body * cal
        cached = sys_tokens * len(papers)
        frac = body / full_body if full_body else 1.0
        out_hi = ref_out * len(papers)
        out_lo = out_hi * (0.6 + 0.4 * frac)
        lo = (fresh * pin + cached * pcached + out_lo * pout) / 1e6
        hi = (fresh * pin + cached * pcached + out_hi * pout) / 1e6
        tot_lo += lo
        tot_hi += hi
        print(f"{cfg or 'full':10s} {fresh:12,.0f} {cached:12,.0f} "
              f"{out_lo:8,.0f}-{out_hi:<9,.0f} ${lo:6.2f}-${hi:<7.2f}")
    print(f"\n{len(configs)} configuration(s), {len(configs) * len(papers):,} requests")
    print(f"ESTIMATED COST: ${tot_lo:.2f} - ${tot_hi:.2f}")


def _reference_output_tokens(model: str, arm: str) -> Optional[float]:
    """Mean completion tokens per paper from this model/arm's full-text run."""
    _, usage_path = output_paths(model, arm)
    if not os.path.exists(usage_path):
        return None
    rows = [json.loads(l) for l in open(usage_path, encoding="utf-8") if l.strip()]
    per_pmid = {r.get("PMID"): r for r in rows}
    vals = [r.get("completion_tokens", 0) or 0 for r in per_pmid.values()]
    return float(sum(vals)) / len(vals) if vals else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract 15 fields per article with an LLM.")
    parser.add_argument("--model", required=True,
                        help="e.g. gpt-4o-mini, gpt-5-mini, gpt-5.4-mini, gpt-5.4-nano, gpt-5.6-luna")
    parser.add_argument("--arm", required=True, choices=sorted(PROMPTS),
                        help="prompt with or without the tract vocabulary")
    parser.add_argument("--effort", default=None,
                        choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                        help="reasoning_effort; omitted = each model's own default")
    parser.add_argument("--concurrency", type=int, default=30)
    parser.add_argument("--resume", action="store_true",
                        help="skip PMIDs already present in the output csv")
    parser.add_argument("--limit", type=int, default=None,
                        help="first N papers only (smoke tests)")
    parser.add_argument("--seed", type=int, default=12345,
                        help="sampling seed for reproducibility; --seed -1 disables")
    parser.add_argument("--sections", default=None,
                        help="article-section inputs: 'all14', or a comma-separated "
                             f"list from {','.join(ALL14)}. Omit for full text.")
    parser.add_argument("--service-tier", dest="service_tier", default=None,
                        choices=["flex", "priority"],
                        help="flex = half price, slower, may 429 (not charged). "
                             "Omit for standard.")
    parser.add_argument("--dry-run", action="store_true",
                        help="estimate tokens and cost without calling the API")
    args = parser.parse_args()

    if args.sections is None:
        configs: List[Optional[str]] = [None]
    elif args.sections == "all14":
        configs = list(ALL14)
    else:
        configs = [c.strip() for c in args.sections.split(",") if c.strip()]
        unknown = [c for c in configs if c not in SECTION_MAP]
        if unknown:
            parser.error(f"unknown section config(s): {', '.join(unknown)}. "
                         f"Choose from: {', '.join(ALL14)}")

    seed = None if args.seed is not None and args.seed < 0 else args.seed

    if args.dry_run:
        dry_run(args.model, args.arm, args.effort, configs,
                args.service_tier, args.limit)
        return

    if not API_KEY:
        parser.error("OPENAI_API_KEY is not set (add it to .env at the repo root)")

    # One config per run() call: each writes its own csv and usage log, so an
    # interrupted sweep resumes per configuration rather than from the start.
    for i, config in enumerate(configs, 1):
        if len(configs) > 1:
            print(f"\n[{i}/{len(configs)}] {config}")
        asyncio.run(run(args.model, args.arm, args.effort, args.concurrency,
                        args.resume, args.limit, seed, config,
                        args.service_tier))


if __name__ == "__main__":
    main()
