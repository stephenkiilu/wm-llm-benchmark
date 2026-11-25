"""How many predicted tract names are vocabulary entries, before and after the vocabulary.

Every tract name each run produced (all 622 articles) is sorted into
    canonical  the preferred name of a vocabulary entry
    synonym    another listed form of an entry (still scores)
    unusable   matches no entry (cannot score)
Writes results/tables/vocabulary_compliance.csv
"""
import pandas as pd

from paths import ARMS, MODELS, TABLES, run_file
from vocabulary import load_vocabulary, normalise, split_cell

if __name__ == "__main__":
    vocab = load_vocabulary()
    preferred = {normalise(p) for p in vocab.table.preferred_label}
    rows = []
    for model in MODELS:
        for arm in ARMS:
            pred = pd.read_csv(run_file(model, arm), dtype={"PMID": str})
            terms = [t for cell in pred.whitematter_tracts for t in split_cell(cell)]
            mapped = [vocab.resolve(t)[1] != "self" for t in terms]
            canonical = sum(m and t in preferred for t, m in zip(terms, mapped))
            synonym = sum(m and t not in preferred for t, m in zip(terms, mapped))
            unusable = len(terms) - canonical - synonym
            rows.append({"model": model, "arm": arm, "mentions": len(terms),
                         "canonical": canonical, "synonym": synonym, "unusable": unusable,
                         "pct_canonical": 100 * canonical / len(terms),
                         "pct_synonym": 100 * synonym / len(terms),
                         "pct_unusable": 100 * unusable / len(terms)})
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "vocabulary_compliance.csv", index=False)
    print(out.round(1).to_string(index=False))
