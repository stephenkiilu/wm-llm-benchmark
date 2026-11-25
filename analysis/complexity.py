"""Task complexity of each field and F1 of every run (Supplementary Fig. S1).

Complexity is computed from the reference labels only, never from model output:
the mean of four min-max-normalised features,
    label space   number of distinct reference labels
    cardinality   mean number of labels per article
    entropy       Shannon entropy (bits) of the label distribution
    answer length mean characters of the joined reference answer
List fields use articles with at least one reference label.

Writes results/tables/task_complexity.csv (one f1_<model>_<arm> column per run)
"""
from collections import Counter
from math import log2

import numpy as np
import pandas as pd

import metrics as M
from paths import ANNOTATIONS, ARMS, MODELS, TABLES, run_file
from vocabulary import load_vocabulary

WMT = "WM tracts studied"
BINARY = [("Use of DTI", "Is this DTI?", "DTI_study"),
          ("Study population", "Human study or not?", "Human_study"),
          ("Dementia focus", "Does this study dementia, alzheimers, or related disease? \n",
           "Dementia_study")]
LISTS = [("Article type", "Is this a single study or a review?", "study_type"),
         ("Disorders studied", "Which one?", "Disease_study"),
         (WMT, "What tracts were studied?", "whitematter_tracts")]
ORDER = [k for k, *_ in BINARY + LISTS]

# spelling-only collapses: "human" means yes; "single" means single study
CANON = {"Study population": {"human": "yes"}, "Article type": {"single": "single study"}}


def labels(field, values):
    cmap = CANON.get(field, {})
    return [cmap.get(v, v) for v in values]


def binary_f1(gold, pred, field):
    g = [labels(field, [M.norm(v)])[0] == "yes" for v in gold]
    p = [labels(field, [M.norm(v)])[0] == "yes" for v in pred]
    tp = sum(a and b for a, b in zip(g, p))
    fp = sum(b and not a for a, b in zip(g, p))
    fn = sum(a and not b for a, b in zip(g, p))
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0


def list_f1(gold, pred, field):
    counts = M.row_counts([set(labels(field, M.clean_split(v))) for v in pred],
                          [set(labels(field, M.clean_split(v))) for v in gold])
    return M.micro_f1(counts, skip_empty_gold=True)


def complexity_table(gold):
    rows = []
    for key, gcol, _ in BINARY + LISTS:
        if key in {k for k, *_ in BINARY}:
            sets = [["yes"] if M.norm(v) == "yes" else ["no"] for v in gold[gcol]]
        else:
            sets = [s for s in (labels(key, M.clean_split(v)) for v in gold[gcol]) if s]
        counts = Counter(lbl for s in sets for lbl in s)
        total = sum(counts.values())
        rows.append({"field": key, "label_space": len(counts),
                     "cardinality": float(np.mean([len(s) for s in sets])),
                     "entropy": -sum(n / total * log2(n / total) for n in counts.values()),
                     "answer_len": float(np.mean([len(", ".join(s)) for s in sets]))})
    comp = pd.DataFrame(rows).set_index("field")
    feats = ["label_space", "cardinality", "entropy", "answer_len"]
    scaled = (comp[feats] - comp[feats].min()) / (comp[feats].max() - comp[feats].min())
    comp["complexity"] = scaled.mean(axis=1)
    return comp


def build():
    vocab = load_vocabulary()
    gold = pd.read_csv(ANNOTATIONS, dtype={"PMID": str})
    comp = complexity_table(gold)
    for model in MODELS:
        for arm in sorted(ARMS):
            pred = pd.read_csv(run_file(model, arm), dtype={"PMID": str})
            df = gold.merge(pred[["PMID"] + [p for *_, p in BINARY + LISTS]], on="PMID")
            assert len(df) == len(gold), f"{model} {arm}: incomplete run"
            col = f"f1_{model}_{arm}"
            for key, gcol, pcol in BINARY:
                comp.loc[key, col] = binary_f1(df[gcol], df[pcol], key)
            for key, gcol, pcol in LISTS:
                comp.loc[key, col] = (M.micro_f1(M.evaluate(df[pcol], df[gcol], vocab)["counts"], True)
                                      if key == WMT else list_f1(df[gcol], df[pcol], key))
    return comp.loc[ORDER].sort_values("complexity")


if __name__ == "__main__":
    comp = build()
    tracts = pd.read_csv(TABLES / "tract_results.csv")      # same tract F1 as evaluate.py
    for r in tracts.itertuples():
        assert abs(comp.loc[WMT, f"f1_{r.model}_{r.arm}"] - r.f1) < 1e-9, (r.model, r.arm)
    comp.to_csv(TABLES / "task_complexity.csv")
    print(comp.round(3).to_string())
    print(f"wrote {TABLES / 'task_complexity.csv'}")
