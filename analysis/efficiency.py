"""Tokens, latency and cost per article for every full-text run (Fig. 2b).

Per-article means from the usage totals that evaluate.py reads from the logs.
Writes results/tables/efficiency.csv
"""
import pandas as pd

from paths import TABLES

if __name__ == "__main__":
    r = pd.read_csv(TABLES / "tract_results.csv")
    n = r.usage_calls
    out = pd.DataFrame({
        "model": r.model, "arm": r.arm, "n": n, "f1": r.f1,
        "inp": r.usage_prompt_tokens / n,              # input tokens per article
        "cached": r.usage_cached_tokens / n,
        "out": r.usage_completion_tokens / n,          # output tokens, reasoning included
        "reasoning": r.usage_reasoning_tokens / n,
        "rea_pct": r.usage_reasoning_pct, "cache_pct": r.usage_cache_pct,
        "p50": r.usage_median_latency_s, "p90": r.usage_p90_latency_s,
        "cost": r.usage_cost_usd,                       # US$ for all 622 articles
        "trunc": r.usage_truncated,
    })
    out.to_csv(TABLES / "efficiency.csv", index=False)
    print(out[["model", "arm", "f1", "out", "cost"]].round(3).to_string(index=False))
