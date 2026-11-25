"""How often each vocabulary name form appears verbatim in the 622 full texts.

A form counts in an article if it occurs as a whole word: case-insensitive for
ordinary names, case-sensitive for abbreviations (any capital after the first
letter, e.g. "AF", "CCg"), so "AL" does not match "et al.".
Needs the full texts (data/full_text/articles.json), which are not redistributed.
Writes results/tables/name_variants.csv
"""
import json
import re

import pandas as pd

from paths import FULL_TEXT, TABLES
from vocabulary import load_vocabulary


def forms(table):
    """(preferred name, form) for the preferred name and every listed synonym."""
    out = []
    for _, r in table.iterrows():
        names = [r.preferred_label] + (str(r.synonyms).split(";") if pd.notna(r.synonyms) else [])
        for f in dict.fromkeys(n.strip() for n in names if n.strip()):
            out.append((r.preferred_label, f))
    return out


def pattern(form):
    abbreviation = any(c.isupper() for c in form[1:])
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(form) + r"(?![A-Za-z0-9])",
                      0 if abbreviation else re.I)


if __name__ == "__main__":
    texts = [json.dumps(a) for a in json.load(open(FULL_TEXT))]
    lowered = [t.lower() for t in texts]
    rows = []
    for canonical, form in forms(load_vocabulary().table):
        p, low = pattern(form), form.lower()
        n = sum(1 for t, l in zip(texts, lowered) if low in l and p.search(t))
        if n:
            rows.append({"canonical": canonical, "form": form, "articles": n})
    out = pd.DataFrame(rows).sort_values(["canonical", "articles"], ascending=[True, False], kind="stable")
    out.to_csv(TABLES / "name_variants.csv", index=False)
    per_tract = out.groupby("canonical").size()
    print(f"{len(out)} forms attested across {len(per_tract)} tracts; "
          f"most forms for one tract: {per_tract.max()} ({per_tract.idxmax()})")
