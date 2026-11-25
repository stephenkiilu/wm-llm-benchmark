"""Which article sections does the model need? (Fig. 2c, Tables S2 and S3)

GPT-5.6-luna with the vocabulary, given 14 combinations of Abstract (A),
Methods (M), Results (R) and Discussion (D), or the full text. Fields are
scored as in Table 1 (field_metrics.py).

Bootstrap: 10,000 resamples of the 622 articles, the same draw for every input,
so each input is compared with the full text on paired replicates. Two-sided
add-one p-values; Benjamini-Hochberg within each field across its 14 tests.

Writes results/tables/
    section_f1.csv                  F1, counts, 95% CI and q vs full text per field x input
    section_abstract_vs_full.csv    abstract vs full text, paired 95% CI of the difference
    table_s2_sections.csv           Table S2: F1 [95% CI] with stars, every input
    table_s3_abstract_vs_full.csv   Table S3: abstract vs full text, 3 decimals
"""
import numpy as np
import pandas as pd

import metrics as M
from field_metrics import FIELDS, field_counts, load_run
from paths import ANNOTATIONS, TABLES, WITH, run_file
from vocabulary import load_vocabulary

N_BOOT, SEED = 10_000, 42
MODEL = "gpt-5.6-luna"
INPUTS = ["A", "M", "R", "D", "A+M", "A+R", "A+D", "M+R", "M+D", "R+D",
          "A+M+R", "A+M+D", "A+R+D", "M+R+D", "Full text"]
GROUP = {1: "Single sections", 2: "Pairs", 3: "Triples", 4: "Full text"}
ORDER = ["Article type", "Study population", "Use of DTI", "Dementia focus",
         "Disorders studied", "WM tracts studied"]
S2_ORDER = ["Use of DTI", "Study population", "Dementia focus", "Article type",
            "Disorders studied", "WM tracts studied"]
WORD = {"A": "Abstract", "M": "Methods", "R": "Results", "D": "Discussion"}


def path_of(label):
    return run_file(MODEL, WITH, None if label == "Full text" else label)


def pooled_f1(counts, keep):
    tp, fp, fn = counts[keep, 0].sum(), counts[keep, 1].sum(), counts[keep, 2].sum()
    den = 2 * tp + fp + fn
    return (2 * tp / den if den else np.nan), tp, fp, fn


def boot_f1(counts, keep, idx):
    """Pooled F1 per replicate; NaN where a replicate's F1 is 0/0."""
    w = keep[idx].astype(float)
    tp, fp, fn = ((counts[:, k][idx] * w).sum(axis=1) for k in range(3))
    den = 2 * tp + fp + fn
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, 2 * tp / den, np.nan)


def run():
    gold = pd.read_csv(ANNOTATIONS, dtype={"PMID": str}).sort_values("PMID").reset_index(drop=True)
    vocab = load_vocabulary()
    counts, keeps, rows = {}, {}, []
    for label in INPUTS:
        df = load_run(path_of(label), gold)
        assert (df.PMID.values == gold.PMID.values).all()
        n_sec = 4 if label == "Full text" else label.count("+") + 1
        for field in ORDER:
            c, keep = field_counts(df, field, vocab)
            f1, tp, fp, fn = pooled_f1(c, keep)
            counts[(label, field)], keeps[field] = c, keep
            rows.append({"field": field, "config": label, "n_sections": n_sec,
                         "group": GROUP[n_sec], "source_file": path_of(label).name,
                         "n_papers": len(c), "n_scored": int(keep.sum()),
                         "tp": tp, "fp": fp, "fn": fn, "f1": f1})
    f1_df = pd.DataFrame(rows)

    idx = np.random.default_rng(SEED).integers(0, len(gold), (N_BOOT, len(gold)))
    cell = {}
    for field in ORDER:
        keep = keeps[field]                     # depends on the reference only
        full = boot_f1(counts[("Full text", field)], keep, idx)
        tested = []
        for label in INPUTS:
            b = boot_f1(counts[(label, field)], keep, idx)
            lo, hi = np.percentile(b[~np.isnan(b)], [2.5, 97.5])
            rec = {"ci_lower": lo, "ci_upper": hi, "p_vs_full": np.nan,
                   "q_vs_full": np.nan, "stars": ""}
            if label != "Full text":
                d = full - b
                d = d[~np.isnan(d)]
                p = 2.0 * min((1 + int((d <= 0).sum())) / (len(d) + 1),
                              (1 + int((d >= 0).sum())) / (len(d) + 1))
                rec["p_vs_full"] = min(1.0, p)
                tested.append(label)
            cell[(label, field)] = rec
        for label, q in zip(tested, M.benjamini_hochberg([cell[(l, field)]["p_vs_full"] for l in tested])):
            cell[(label, field)]["q_vs_full"] = q
            cell[(label, field)]["stars"] = "" if M.stars(q) == "ns" else M.stars(q)
    for k in ("ci_lower", "ci_upper", "p_vs_full", "q_vs_full", "stars"):
        f1_df[k] = [cell[(r.config, r.field)][k] for r in f1_df.itertuples()]

    comp = []
    for field in ORDER:
        keep = keeps[field]
        ca, cf = counts[("A", field)], counts[("Full text", field)]
        diff = boot_f1(cf, keep, idx) - boot_f1(ca, keep, idx)
        lo, hi = np.percentile(diff[~np.isnan(diff)], [2.5, 97.5])
        f_a, f_c = pooled_f1(ca, keep)[0], pooled_f1(cf, keep)[0]
        comp.append({"field": field, "abstract_f1": f_a, "complete_f1": f_c,
                     "delta_f1": f_c - f_a, "ci_lower": lo, "ci_upper": hi,
                     "n_papers_resampled": len(gold), "n_scored": int(keep.sum()),
                     "n_boot": N_BOOT, "seed": SEED,
                     "n_undefined_boot": int(np.isnan(diff).sum())})
    return f1_df, pd.DataFrame(comp)


def table_s2(f1_df):
    out = []
    for label in INPUTS:
        row = {"Source section": label if label == "Full text"
               else "+".join(WORD[x] for x in label.split("+"))}
        for field in S2_ORDER:
            r = f1_df[(f1_df.config == label) & (f1_df.field == field)].iloc[0]
            row[field] = f"{r.f1:.3f}{r.stars} [{r.ci_lower:.3f}, {r.ci_upper:.3f}]"
        out.append(row)
    return pd.DataFrame(out)


def table_s3(comp):
    return pd.DataFrame({
        "Field": comp.field,
        "Abstract F1": comp.abstract_f1.map("{:.3f}".format),
        "Full text F1": comp.complete_f1.map("{:.3f}".format),
        "ΔF1 (Full text − A)": comp.delta_f1.map("{:+.3f}".format),
        "Paired 95% CI for ΔF1": comp.apply(lambda r: f"[{r.ci_lower:.3f}, {r.ci_upper:.3f}]", axis=1),
    })


if __name__ == "__main__":
    f1_df, comp = run()
    TABLES.mkdir(parents=True, exist_ok=True)
    f1_df.to_csv(TABLES / "section_f1.csv", index=False)
    comp.to_csv(TABLES / "section_abstract_vs_full.csv", index=False)
    # utf-8-sig so Excel shows Δ and − correctly
    table_s2(f1_df).to_csv(TABLES / "table_s2_sections.csv", index=False, encoding="utf-8-sig")
    table_s3(comp).to_csv(TABLES / "table_s3_abstract_vs_full.csv", index=False, encoding="utf-8-sig")
    print(f1_df.pivot(index="field", columns="config", values="f1").loc[ORDER, INPUTS].round(3).to_string())
    print(f"wrote section tables to {TABLES}")
