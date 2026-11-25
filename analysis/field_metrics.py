"""Table 1: F1 and Jaccard for six fields, five models, both prompt arms.

    binary fields    F1 on the "yes" class, all 622 articles
    list fields      micro-F1 over label sets, articles with reference labels only
    WM tracts        micro-F1 on vocabulary ids, articles with reference labels only

95% CIs: percentile bootstrap over articles, 10,000 resamples.
Writes results/tables/field_metrics.csv
"""
import numpy as np
import pandas as pd

import metrics as M
from paths import ANNOTATIONS, ARMS, MODELS, TABLES, run_file
from vocabulary import load_vocabulary

# (manuscript name, kind, reference column, prediction column, canonical map)
FIELDS = [
    ("Dementia focus",    "binary", "Does this study dementia, alzheimers, or related disease? \n", "Dementia_study", None),
    ("Use of DTI",        "binary", "Is this DTI?",                         "DTI_study",          None),
    ("Study population",  "binary", "Human study or not?",                  "Human_study",        None),
    ("Article type",      "list",   "Is this a single study or a review?",  "study_type",         M.CANON_TYPE),
    ("Disorders studied", "list",   "Which one?",                           "Disease_study",      M.CANON_DISEASE),
    ("WM tracts studied", "tracts", "What tracts were studied?",            "whitematter_tracts", None),
]


def field_counts(df, field, vocab):
    """Per-article counts (tp, fp, fn, gold_empty) and the rows the field scores."""
    _, kind, gcol, pcol, cmap = next(f for f in FIELDS if f[0] == field)
    if kind == "binary":
        counts = M.binary_counts(df[gcol], df[pcol])
        return counts, np.ones(len(counts), dtype=bool)        # every article
    if kind == "list":
        counts = M.multilabel_counts(df[gcol], df[pcol], cmap)
    else:
        counts = M.lut_counts(df[gcol], df[pcol].fillna(""), vocab)
    return counts, counts[:, 3] == 0                            # reference not empty


def load_run(path, gold):
    pred = pd.read_csv(path, dtype={"PMID": str})
    assert not pred.PMID.duplicated().any(), f"{path.name}: duplicate PMIDs"
    df = gold.merge(pred, on="PMID", how="inner", suffixes=("", "_pred"))
    assert len(df) == len(gold), f"{path.name}: joined {len(df)} of {len(gold)}"
    return df


def run():
    gold = pd.read_csv(ANNOTATIONS, dtype={"PMID": str})
    vocab = load_vocabulary()
    rows = []
    for model in MODELS:
        for arm in ARMS:
            df = load_run(run_file(model, arm), gold)
            for name, kind, *_ in FIELDS:
                counts, keep = field_counts(df, name, vocab)
                scored = counts[keep]
                row = {"model": model, "arm": arm, "field": name, "kind": kind,
                       "n_scored": len(scored), "n_gold_empty": int((counts[:, 3] == 1).sum())}
                for metric in ("f1", "jaccard"):
                    row[metric] = M.point(scored, metric)
                    row[f"{metric}_ci_lower"], row[f"{metric}_ci_upper"] = M.boot_ci(scored, metric)
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    res = run()
    TABLES.mkdir(parents=True, exist_ok=True)
    res.to_csv(TABLES / "field_metrics.csv", index=False)
    for arm in ARMS:
        show = res[res.arm == arm].pivot(index="field", columns="model", values="f1")
        print(f"\nF1, {arm}\n" + show.loc[[f[0] for f in FIELDS], MODELS].round(3).to_string())
    print(f"wrote {TABLES / 'field_metrics.csv'}")
