# Data

**`annotations.csv`**: reference labels for 622 PubMed articles on white matter in dementia (published 1999–2025).
Columns 1–12 are PubMed bibliographic metadata (`PMID`, `Title`, `DOI`, ...); the rest are the hand
annotations, one column per variable, headed by the question annotators answered. The six variables scored
in the paper are `Is this a single study or a review?` (article type), `Human study or not?` (study
population), `Is this DTI?` (use of DTI), `Does this study dementia, alzheimers, or related disease? ` (header ends in a space and a newline;
dementia focus), `Which one?` (disorders studied) and `What tracts were studied?` (WM tracts). Multi-label
cells are comma-separated.

**`vocabulary.csv`**: the 122-tract look-up table. `preferred_label` (the name given to the model),
`SNOMED_ID`, `UBERON_ID`, `Definition`, `synonyms` (`;`-separated forms that map to the entry, used
when scoring), `Source` (atlases, pipelines and papers listing the tract), `Possible related names`
(curation notes, not used).

**`literature/`**: PubMed and Web of Science counts behind Figure 1a–c and Table S1 (see its README).

**`brain_renders/`**: TractSeg renders of SLF I–III and the arcuate fasciculus (Figure 1d), one HCP subject
per hemisphere. Only the images are included.
