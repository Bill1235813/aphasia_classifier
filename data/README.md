# Data folder

One sub-folder per task, each with the same three files:

| folder | task | narratives | groups |
|---|---|---|---|
| `cinderella/` | retelling the Cinderella story | 1,147 | Control 399 · Anomic 276 · Broca 273 · Conduction 141 · Wernicke 58 |
| `sandwich/` | describing how to make a peanut-butter-and-jelly sandwich | 1,437 | Control 936 · Anomic 235 · Broca 131 · Conduction 99 · Wernicke 36 |

Both come from [AphasiaBank](https://aphasia.talkbank.org/). The sandwich
corpus has proportionally more controls (65 % vs 35 %), which is why its
aphasia-F1 is a little lower than Cinderella's even though its accuracy is
higher.

> **`transcripts.json` is not in the git repository.** AphasiaBank's terms do
> not allow redistributing the corpus, so the two transcript files are shared
> with collaborators separately as `aphasia_classifier_transcripts.zip`.
> Unpack it in the `aphasia_classifier/` folder (main README, *Installation
> §5*). Both paths are in `.gitignore`, so `git add` / `git commit` skip them.
> `features.csv` — numbers only, no text — *is* in the repository.

## `<task>/transcripts.json` — the corpus

One JSON object per narrative. The three columns the classifier needs:

| column | meaning |
|---|---|
| `File_DB` | unique id of the recording, e.g. `ACWT01a.cha_aphasia` |
| `Group` | diagnostic group: `Control`, `Anomic`, `Broca`, `Conduction`, `Wernicke` |
| `text_response` | the participant's narrative, as plain text, one sentence per `.` |

The classifier treats every non-`Control` group as **aphasia**.

The remaining columns are AphasiaBank metadata (`Age`, `Sex`, `Duration_(sec)`, …)
and CLAN's own linguistic measures for that transcript (`MLU_Morphemes`,
`Words_Min`, `%_Nouns`, `retracing`, …). `build_features.py` copies every numeric
one of these into `features.csv` with a `gold_` prefix, so you can experiment
with them as features (see the main README).

### What the text looks like

`text_response` is the participant's tier as flattened by CLAN's `flo`
command, so it is plain text, not CHAT:

* participant utterances only — the investigator's turns are gone;
* CHAT annotation codes, filled pauses (`&-uh`), and unintelligible-speech
  markers are removed; the words as spoken are kept, including immediate
  repetitions and retracings (`I say Cinderella I I say.`), which is what
  the repetition features rely on;
* one utterance per sentence, every utterance ending in `.` (no `?` or `!`),
  so `text.split(". ")` recovers the utterances;
* commas are separate tokens (`oh man , oh , I can do it.`), so
  `text.split()` counts words without gluing punctuation to them.

### Reading the transcripts in Python

The file is a JSON list, so pandas opens it as a table with one row per
narrative and one column per field:

```python
import pandas as pd
df = pd.read_json("data/cinderella/transcripts.json")

df["Group"].value_counts()                          # how many per group
df[["File_DB", "Group", "text_response"]].head()

for text in df[df["Group"] == "Broca"]["text_response"].head(5):   # read a few
    print(text, "\n")

df.to_csv("data/cinderella/transcripts.csv", index=False)   # browse in Excel
```

(`transcripts.csv` would contain the corpus too — keep it out of git as well.)

### Designing a new feature from the transcripts

The text is all `predict.py` has for a new speaker, so a new feature must be
computable from `text_response` alone. A loop that works:

1. Write a function `compute(text) -> dict` that turns one
   transcript into one or more numbers. `src/features/narrative.py` is the
   simplest one to copy (about 60 lines, no models).
2. **Check that it separates the groups.** Apply it to every transcript and
   compare the groups:

   ```python
   df["my_feature"] = df["text_response"].map(my_compute)
   df.groupby("Group")["my_feature"].describe()
   ```

   A feature whose distribution is the same for `Control` and the aphasia
   groups will not help the classifier, however sensible it sounds.
3. **Add it.** Either paste the column into `features.csv` (keep the row
   order and `File_DB`) or wire the function into `build_features.py` and
   `predict.py` as the main README describes; then add its name to
   `config.FEATURES` and retrain.

The `gold_*` columns are CLAN's own measures for the same transcript, so a
hand-made count can be sanity-checked against CLAN's (your utterance count
against `gold_Total_Utts`, your type/token ratio against `gold_FREQ_TTR`).
Do not expect them to agree exactly — CLAN counts on the original CHAT
transcript, before flattening — but they should track each other closely:
an utterance count from `text.split(". ")` correlates at r ≈ 0.98 with
`gold_Total_Utts`. `notebooks/quickstart.ipynb`, section 1, runs this loop on
a small example.

### Using your own transcripts

Save a JSON file that is a list of objects with at least an id, a text and a
label column as `data/<my_task>/transcripts.json`, then set `TASK = "<my_task>"`
(and, if your column names differ, `ID_COLUMN`, `TEXT_COLUMN`, `LABEL_COLUMN`,
`CONTROL_LABEL`) in `src/config.py`. Nothing else changes.

```json
[
  {"id": "p001", "text": "so Cinderella she lived with ...", "group": "Control"},
  {"id": "p002", "text": "girl. uh. ball. shoe. ...",         "group": "Broca"}
]
```

## `<task>/features.csv` — the numeric feature table

Produced by `src/build_features.py`; one row per transcript, same order as
`transcripts.json`. Columns:

| columns | produced by | needs |
|---|---|---|
| `File_DB`, `Group` | copied from the transcripts | — |
| `gold_*` | copied from the transcripts' metadata | — (AphasiaBank only) |
| `Total_Utts`, `MLU_Utts`, `MLU_Words`, `FREQ_types`, `FREQ_tokens`, `FREQ_TTR` | `features/narrative.py` | nothing |
| `rep_shortest_len`, `rep_shortest_count`, `rep_max_count` | `features/repetition.py` | nothing |
| `cossim_delta_semantic`, `cossim_delta_syntactic` | `features/similarity.py` | a small sentence-embedding model |
| `mean_surp`, `std_surp`, `p90_surp`, `p95_surp`, `max_surp`, `frac_high_surp` | `features/surprisal.py` | a large language model, GPU |

You may add your own columns to this file (Excel, pandas, R — anything that
writes CSV). Keep the row order and the `File_DB` column unchanged.

## `<task>/example_new_transcripts.csv`

Three made-up narratives per task to try `predict.py` on: a fluent one, a
halting telegraphic one, and a perseverative one.
