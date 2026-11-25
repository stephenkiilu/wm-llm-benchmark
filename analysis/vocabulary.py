"""Map tract names to canonical vocabulary entries.

Each name is resolved in three tiers, most authoritative first:
    1. exact   the normalised name is a form listed in the vocabulary
    2. folded  same words after dropping "the", "of", word order, etc.
    3. self    not in the vocabulary: an id built from its own words, so an
               unlisted name can still match the same unlisted name

Parent and child tracts are different entries: "cingulum" does not match
"hippocampal cingulum".

    vocab = load_vocabulary()
    vocab.resolve("Corpus Callosum of Splenium")   # -> ('LUT:0040', 'folded')
    vocab.resolve_cell("Fornix;Cingulum")          # -> {'LUT:0043', 'LUT:0002'}
"""

import collections
import re
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

import pandas as pd

from paths import ANNOTATIONS, GOLD_TRACTS, VOCABULARY

# ── CONFIG ────────────────────────────────────────────────────────────────────
# `Possible related names` is a curation hint, not a matching source: including
# it maps only 2 extra gold terms while introducing 6 additional collisions.
USE_POSSIBLE_RELATED_NAMES = False

# Separators inside one CSV cell: extraction CSVs use ";", the gold file ",".
CELL_SEPARATORS = ";,"

# Dropped when folding (tier 2/3 only). Never affects tier-1 exact lookup.
FOLD_STOPWORDS = {"the", "of", "and", "structure", "brain", "tract", "tracts"}

EMPTY_TOKENS = {
    "", "none", "n.a.", "na", "n a", "n/a", "null", "_", "-", "nan",
    "not reported", "unknown",
}


# ── STRING NORMALISATION ──────────────────────────────────────────────────────
# Deliberately minimal. This exists to make the dictionary lookup robust to
# case, spacing and punctuation. It makes NO semantic decisions; synonymy is
# the LUT's job.

def normalise(text: object) -> str:
    """Lower-case, strip punctuation, collapse whitespace. No splitting."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower().strip()
    s = re.sub(r"[\(\)\.\'\"]", " ", s)       # punctuation, but NOT the separators
    s = re.sub(r"[-_/]", " ", s)              # hyphens/slashes -> space
    return " ".join(s.split())


def split_cell(cell: object) -> List[str]:
    """Split one CSV cell into individual tract strings.

    Splits on the raw separators BEFORE normalising, because normalisation
    would otherwise consume the commas that separate gold entries.
    """
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return []
    raw = str(cell)
    for sep in CELL_SEPARATORS[1:]:
        raw = raw.replace(sep, CELL_SEPARATORS[0])
    out = []
    for part in raw.split(CELL_SEPARATORS[0]):
        term = normalise(part)
        if term and term not in EMPTY_TOKENS:
            out.append(term)
    return out


def fold_key(term: str) -> FrozenSet[str]:
    """Word-set with stopwords removed, order- and filler-insensitive."""
    return frozenset(w for w in normalise(term).split() if w not in FOLD_STOPWORDS)


# ── INDEX ─────────────────────────────────────────────────────────────────────

class LutIndex:
    """Surface-form -> canonical-id index built from the LUT."""

    def __init__(self, table: pd.DataFrame, use_related: bool):
        self.table = table
        self.use_related = use_related
        self.form_to_id: Dict[str, str] = {}                    # tier 1
        self.fold_to_id: Dict[FrozenSet[str], str] = {}         # tier 2
        self.id_to_label: Dict[str, str] = {}
        self.collisions: Dict[str, Set[str]] = collections.defaultdict(set)
        self.fold_collisions: Dict[FrozenSet[str], Set[str]] = collections.defaultdict(set)
        self.unmapped: collections.Counter = collections.Counter()
        self._build()

    # -- construction --
    def _surface_forms(self, row: pd.Series) -> List[str]:
        forms = [row["preferred_label"]]
        if pd.notna(row.get("synonyms")):
            forms += str(row["synonyms"]).split(";")
        if self.use_related and pd.notna(row.get("Possible related names")):
            forms += str(row["Possible related names"]).split("|")
        return forms

    def _build(self) -> None:
        for i, row in self.table.iterrows():
            tract_id = f"LUT:{i:04d}"
            self.id_to_label[tract_id] = str(row["preferred_label"]).strip()
            for form in self._surface_forms(row):
                key = normalise(form)
                if not key:
                    continue
                claimed = self.form_to_id.get(key)
                if claimed is None:
                    self.form_to_id[key] = tract_id
                elif claimed != tract_id:
                    self.collisions[key] |= {claimed, tract_id}   # first claim wins

                fkey = fold_key(form)
                if not fkey:
                    continue
                fclaimed = self.fold_to_id.get(fkey)
                if fclaimed is None:
                    self.fold_to_id[fkey] = tract_id
                elif fclaimed != tract_id:
                    self.fold_collisions[fkey] |= {fclaimed, tract_id}

    # -- lookup --
    def resolve(self, term: str) -> Tuple[str, str]:
        """Return (id, tier) where tier is 'exact' | 'folded' | 'self' | ''."""
        key = normalise(term)
        if not key or key in EMPTY_TOKENS:
            return "", ""
        tract_id = self.form_to_id.get(key)
        if tract_id is not None:
            return tract_id, "exact"
        fkey = fold_key(term)
        tract_id = self.fold_to_id.get(fkey)
        if tract_id is not None:
            return tract_id, "folded"
        self.unmapped[key] += 1
        return "RAW:" + " ".join(sorted(fkey)), "self"

    def resolve_cell(self, cell: object) -> Set[str]:
        """Resolve one CSV cell to a set of canonical ids."""
        ids = set()
        for term in split_cell(cell):
            tract_id, _ = self.resolve(term)
            if tract_id:
                ids.add(tract_id)
        return ids

    def label(self, tract_id: str) -> str:
        if tract_id.startswith("RAW:"):
            return tract_id[4:] + "  [unmapped]"
        return self.id_to_label.get(tract_id, tract_id)

    # -- diagnostics --
    def coverage(self, cells) -> Dict[str, int]:
        """Tier counts over an iterable of CSV cells. Does not mutate state."""
        saved = self.unmapped.copy()
        tally = collections.Counter()
        for cell in cells:
            for term in split_cell(cell):
                tally[self.resolve(term)[1]] += 1
        self.unmapped = saved
        return dict(tally)

    def report(self, gold: Optional[pd.Series] = None) -> None:
        print(f"LUT: {len(self.table)} canonical terms, "
              f"{len(self.form_to_id)} exact forms, {len(self.fold_to_id)} folded forms")
        print(f"     related-names column used: {self.use_related}")
        if self.collisions:
            print(f"     exact collisions:  {len(self.collisions)} (first claim wins)")
        if self.fold_collisions:
            print(f"     folded collisions: {len(self.fold_collisions)} (first claim wins)")
        if gold is not None:
            t = self.coverage(gold)
            total = sum(t.values())
            mapped = t.get("exact", 0) + t.get("folded", 0)
            print(f"     gold mentions: {total}   "
                  f"exact {t.get('exact', 0)}  folded {t.get('folded', 0)}  "
                  f"unmapped {t.get('self', 0)}")
            print(f"     gold coverage: {mapped}/{total} ({mapped / total:.1%})"
                  if total else "     gold coverage: n/a")


# ── LOADER ────────────────────────────────────────────────────────────────────

def load_vocabulary(path=VOCABULARY,
             use_related: bool = USE_POSSIBLE_RELATED_NAMES) -> LutIndex:
    """Load the vocabulary csv and build the surface-form index."""
    table = pd.read_csv(path)
    missing = {"preferred_label", "synonyms"} - set(table.columns)
    if missing:
        raise ValueError(f"LUT is missing required columns: {sorted(missing)}")
    return LutIndex(table, use_related)


if __name__ == "__main__":
    lut = load_vocabulary()
    gold = pd.read_csv(ANNOTATIONS)[GOLD_TRACTS]
    lut.report(gold)

    print("\nsanity checks:")
    checks = [
        ("Corpus Callosum of Splenium", "Corpus Callosum - Splenium", True),
        ("body of the corpus callosum", "body of corpus callosum",    True),
        ("forceps major",               "major forceps",              True),
        ("Corpus Callosum - Genu",      "Corpus Callosum - Splenium", False),
        ("Cingulum",                    "Hippocampal Cingulum",       False),
        ("Corpus Callosum",             "Corpus Callosum - Splenium", False),
    ]
    ok = True
    for a, b, should_match in checks:
        ida, ta = lut.resolve(a)
        idb, tb = lut.resolve(b)
        matched = ida == idb
        good = matched == should_match
        ok &= good
        print(f"  [{'ok ' if good else 'FAIL'}] {'MATCH' if matched else 'MISS ':<5} "
              f"(want {'MATCH' if should_match else 'MISS'})  {a!r} / {b!r}")
        print(f"         {ida} ({ta}) vs {idb} ({tb})")
    print("\nall sanity checks passed" if ok else "\nSOME CHECKS FAILED")
