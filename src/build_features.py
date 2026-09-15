"""
build_features.py — turn the transcripts into the numeric feature table.

Reads   config.TRANSCRIPTS_FILE   (the narratives + labels)
Writes  config.FEATURES_FILE      (one row per transcript, one column per feature)

Run it once before training. Afterwards, train.py never looks at the text for
its hand-crafted features again — it reads this table — so you can also open
features.csv in Excel, add a column of your own, and list it in config.FEATURES.

    python src/build_features.py                       # every group in config
    python src/build_features.py --groups narrative,repetition   # just these

The four groups, cheapest first:
    narrative    utterance/word counts            seconds, no models
    repetition   perseveration statistics         seconds, no models
    similarity   closeness to control vs aphasia  ~1 minute, small model
    surprisal    language-model unpredictability  needs a GPU + the model in config

A partial run only overwrites the columns of the groups you asked for; every
other column in an existing features.csv is kept. So rebuilding one group,
or adding your own columns, never destroys the rest.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from features import narrative, repetition, similarity, surprisal


def gold_columns(transcripts):
    """Numeric metadata shipped WITH the transcripts (e.g. CLAN's own measures).

    These are copied into features.csv with a ``gold_`` prefix so you can try
    them as features. They exist only for this corpus — a new transcript has
    no gold values — so predict.py cannot use them.
    """
    skip = {config.ID_COLUMN, config.TEXT_COLUMN, config.LABEL_COLUMN}
    numeric = {}
    for column in transcripts[0]:
        if column in skip:
            continue
        values = []
        for row in transcripts:
            try:
                values.append(float(row.get(column)))
            except (TypeError, ValueError):
                values.append(np.nan)
        # Keep a column only if it is genuinely numeric for most rows.
        if np.isfinite(values).mean() >= 0.9:
            numeric[f"gold_{column}"] = values
    return numeric


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", default=",".join(config.FEATURE_GROUPS_TO_BUILD),
                        help="comma-separated subset of: narrative,repetition,similarity,surprisal")
    groups = [g.strip() for g in parser.parse_args().groups.split(",")]

    transcripts = json.loads(Path(config.TRANSCRIPTS_FILE).read_text())
    texts = [row[config.TEXT_COLUMN] for row in transcripts]
    ids = [row[config.ID_COLUMN] for row in transcripts]
    labels = [row[config.LABEL_COLUMN] for row in transcripts]
    print(f"{len(transcripts)} transcripts; building groups: {groups}")

    # Start from the existing table if there is one, so partial rebuilds and
    # hand-added columns survive. Otherwise start with ids, labels and gold_*.
    if Path(config.FEATURES_FILE).exists():
        table = pd.read_csv(config.FEATURES_FILE)
        assert list(table[config.ID_COLUMN]) == ids, \
            "features.csv rows do not match transcripts.json — delete features.csv and rerun"
    else:
        table = pd.DataFrame({config.ID_COLUMN: ids, config.LABEL_COLUMN: labels,
                              **gold_columns(transcripts)})

    if "narrative" in groups:
        rows = [narrative.compute(t) for t in texts]
        for column in narrative.COLUMNS:
            table[column] = [r[column] for r in rows]
        print("  narrative   done")

    if "repetition" in groups:
        rows = [repetition.compute(t) for t in texts]
        for column in repetition.COLUMNS:
            table[column] = [r[column] for r in rows]
        print("  repetition  done")

    if "similarity" in groups:
        is_control = [label == config.CONTROL_LABEL for label in labels]
        reference = similarity.SimilarityReference(texts, is_control, config.SIMILARITY_MODEL)
        rows = reference.leave_one_out_deltas()       # leave-one-out: see similarity.py
        for column in similarity.COLUMNS:
            table[column] = [r[column] for r in rows]
        print("  similarity  done")

    if "surprisal" in groups:
        scorer = surprisal.SurprisalScorer(config.SURPRISAL_MODEL,
                                           config.SURPRISAL_HIGH_THRESHOLD,
                                           config.SURPRISAL_MAX_TOKENS)
        rows = []
        for i, text in enumerate(texts):
            rows.append(scorer.compute(text))
            if (i + 1) % 100 == 0:
                print(f"    surprisal {i + 1}/{len(texts)}", flush=True)
        for column in surprisal.COLUMNS:
            values = pd.Series([r[column] for r in rows])
            # NOTE: [edge case callout] a transcript too short to score gets
            # NaN; fill it with the column mean so no row is lost.
            table[column] = values.fillna(values.mean())
        print("  surprisal   done")

    table.to_csv(config.FEATURES_FILE, index=False)
    print(f"wrote {config.FEATURES_FILE}  ({table.shape[0]} rows x {table.shape[1]} columns)")


if __name__ == "__main__":
    main()
