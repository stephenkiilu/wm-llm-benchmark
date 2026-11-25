"""Random sample of WM tract disagreements for the manual error audit.

For each model (with the vocabulary) every disagreement with the reference is
listed, one row per tract: FP = the model listed a tract the reference lacks,
FN = the reference lists a tract the model missed. 25 of each are drawn
(seed 42) and were then adjudicated by hand; the counts are in
data/error_audit.csv.

Writes results/tables/error_samples/<model>.csv and all_models.csv
"""
import pandas as pd

from paths import ANNOTATIONS, GOLD_TRACTS, TABLES, WITH, run_file
from vocabulary import load_vocabulary, split_cell

MODELS = {"GPT-5.4-mini": "gpt-5.4-mini", "GPT-5.4-nano": "gpt-5.4-nano",
          "GPT-5.6-luna": "gpt-5.6-luna"}
N_PER_LABEL, SEED = 25, 42
OUT = TABLES / "error_samples"
COLUMNS = ["model", "pmid", "title", "label", "tract", "tract_id", "as_written",
           "lut_tier", "gold_empty", "model_output", "golden_truth"]


def resolve_terms(cell, vocab):
    """{vocabulary id: (name as written, tier)}; the first writing of an id wins."""
    out = {}
    for term in split_cell(cell):
        tract_id, tier = vocab.resolve(term)
        if tract_id and tract_id not in out:
            out[tract_id] = (term.strip(), tier)
    return out


def collect_errors(model, label, gold, vocab):
    pred = pd.read_csv(run_file(model, WITH), dtype={"PMID": str})
    merged = gold.merge(pred[["PMID", "whitematter_tracts"]], on="PMID", how="inner")
    rows = []
    for _, r in merged.iterrows():
        gold_cell, pred_cell = r[GOLD_TRACTS], r["whitematter_tracts"]
        g, p = resolve_terms(gold_cell, vocab), resolve_terms(pred_cell, vocab)
        for kind, ids, source in (("FP", set(p) - set(g), p), ("FN", set(g) - set(p), g)):
            for tract_id in sorted(ids):
                written, tier = source[tract_id]
                rows.append({"model": label, "pmid": r["PMID"], "title": r["Title"],
                             "label": kind, "tract": vocab.label(tract_id), "tract_id": tract_id,
                             "as_written": written, "lut_tier": tier, "gold_empty": not g,
                             "model_output": "" if pd.isna(pred_cell) else pred_cell,
                             "golden_truth": "" if pd.isna(gold_cell) else gold_cell})
    return pd.DataFrame(rows, columns=COLUMNS)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(ANNOTATIONS, dtype={"PMID": str})
    vocab = load_vocabulary()
    drawn = []
    for label, model in MODELS.items():
        errs = collect_errors(model, label, gold, vocab)
        fp, fn = errs[errs.label == "FP"], errs[errs.label == "FN"]
        sample = pd.concat([fp.sample(n=min(N_PER_LABEL, len(fp)), random_state=SEED),
                            fn.sample(n=min(N_PER_LABEL, len(fn)), random_state=SEED)],
                           ignore_index=True)
        sample.to_csv(OUT / f"{model}.csv", index=False)
        drawn.append(sample)
        print(f"{label}: {len(fp)} FP, {len(fn)} FN; sampled {len(sample)}")
    pd.concat(drawn, ignore_index=True).to_csv(OUT / "all_models.csv", index=False)
    print(f"wrote {OUT}")
