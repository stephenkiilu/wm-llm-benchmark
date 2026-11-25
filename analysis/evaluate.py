"""WM tract F1 for every full-text run, and the effect of the vocabulary (Fig. 2a).

For each model x arm: micro-F1 on vocabulary ids over articles whose reference
lists at least one tract, with a bootstrap 95% CI, precision and recall, plus
token use and cost from the usage logs. The vocabulary effect is a paired
bootstrap (with - without) on the same articles; q-values are
Benjamini-Hochberg across the five models.

Writes results/tables/tract_results.csv and vocabulary_effect.csv
"""
import json

import pandas as pd

import metrics as M
from paths import (ANNOTATIONS, ARMS, EFFORT, GOLD_TRACTS, MODELS, TABLES, WITH, WITHOUT,
                   run_file, usage_file)
from vocabulary import load_vocabulary

SKIP_EMPTY = True      # score only articles whose reference lists at least one tract

# US$ per 1M tokens: input, cached input, output (OpenAI price list, Aug 2026)
PRICING = {
    "gpt-4o-mini":  (0.15, 0.075, 0.60),
    "gpt-5-mini":   (0.25, 0.025, 2.00),
    "gpt-5.4-mini": (0.75, 0.075, 4.50),
    "gpt-5.4-nano": (0.20, 0.02,  1.25),
    "gpt-5.6-luna": (0.20, 0.02,  1.20),
}


def score_run(path, gold, vocab):
    """Join predictions to the reference on PMID and count tp/fp/fn per article."""
    pred = pd.read_csv(path, dtype={"PMID": str})
    merged = gold.merge(pred[["PMID", "whitematter_tracts"]], on="PMID", how="left",
                        indicator=True)
    missing = int((merged["_merge"] == "left_only").sum())
    merged = merged[merged["_merge"] != "left_only"]        # an unrun article is not an error
    result = M.evaluate(merged["whitematter_tracts"].fillna(""), merged[GOLD_TRACTS], vocab)
    result["missing_rows"] = missing
    return result


def usage_summary(model, arm, sections=None):
    """Token use, latency and cost from one run's usage log (last record per article)."""
    records = [json.loads(line) for line in open(usage_file(model, arm, sections)) if line.strip()]
    rows = list({r.get("PMID", id(r)): r for r in records}.values())   # keep successful retry
    n = len(rows)
    tot = lambda k: sum(r.get(k, 0) or 0 for r in rows)
    lat = sorted(r.get("latency_s", 0) or 0 for r in rows)
    pin, pcached, pout = PRICING[model]
    fresh = tot("prompt_tokens") - tot("cached_tokens")
    cost = (fresh * pin + tot("cached_tokens") * pcached + tot("completion_tokens") * pout) / 1e6
    return {"calls": n, "api_records": len(records),
            "failed_attempts": sum(1 for r in records if r.get("error")),
            "prompt_tokens": tot("prompt_tokens"),
            "cached_tokens": tot("cached_tokens"),
            "completion_tokens": tot("completion_tokens"),
            "reasoning_tokens": tot("reasoning_tokens"),
            "median_latency_s": lat[n // 2],
            "parse_failures": sum(1 for r in rows if not r.get("parse_ok", True)),
            "errors": sum(1 for r in rows if r.get("error")),
            "p90_latency_s": lat[int(0.9 * n)],
            "truncated": sum(1 for r in rows if r.get("finish_reason") == "length"),
            "reasoning_pct": 100.0 * tot("reasoning_tokens") / tot("completion_tokens"),
            "cache_pct": 100.0 * tot("cached_tokens") / tot("prompt_tokens"),
            "cost_usd": cost}


def main():
    vocab = load_vocabulary()
    gold = pd.read_csv(ANNOTATIONS, dtype={"PMID": str})
    scored, rows = {}, []
    for model in MODELS:
        for arm in ARMS:
            res = score_run(run_file(model, arm), gold, vocab)
            scored[(model, arm)] = res
            f1 = M.micro_f1(res["counts"], SKIP_EMPTY)
            lo, hi = M.bootstrap_ci(res["counts"], skip_empty_gold=SKIP_EMPTY)
            prec, rec = M.precision_recall(res["counts"], SKIP_EMPTY)
            use = usage_summary(model, arm)
            rows.append({"model": model, "arm": arm, "effort": EFFORT,
                         "f1": f1, "ci_lower": lo, "ci_upper": hi,
                         "precision": prec, "recall": rec,
                         "prefilter_f1": res["prefilter"]["f1"],
                         "spurious_tracts": res["prefilter"]["spurious_tracts"],
                         "n_rows": res["n_rows"], "n_gold_empty": res["n_gold_empty"],
                         "missing_rows": res["missing_rows"],
                         **{f"usage_{k}": v for k, v in use.items()}})
            print(f"{model:<14}{arm:<20}F1 {f1:.3f} [{lo:.3f}, {hi:.3f}]  "
                  f"P {prec:.3f}  R {rec:.3f}  US${use['cost_usd']:.2f}")

    effects = []
    for model in sorted(MODELS):
        t = M.paired_test(scored[(model, WITHOUT)]["counts"], scored[(model, WITH)]["counts"],
                          skip_empty_gold=SKIP_EMPTY)
        base = M.micro_f1(scored[(model, WITHOUT)]["counts"], SKIP_EMPTY)
        effects.append({"model": model, **t, "relative_pct": t["delta"] / base * 100,
                        "p_at_floor": t["n_extreme"] == 0})
    for e, q in zip(effects, M.benjamini_hochberg([e["p_value"] for e in effects])):
        e["q_value"] = q
        print(f"{e['model']:<14}dF1 {e['delta']:+.3f} [{e['ci_lower']:+.3f}, {e['ci_upper']:+.3f}]"
              f"  q {'< 0.001' if e['p_at_floor'] else f'= {q:.3f}'}")

    TABLES.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLES / "tract_results.csv", index=False)
    pd.DataFrame(effects).to_csv(TABLES / "vocabulary_effect.csv", index=False)
    print(f"wrote {TABLES / 'tract_results.csv'} and vocabulary_effect.csv")


if __name__ == "__main__":
    main()
