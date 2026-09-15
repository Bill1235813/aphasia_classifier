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
