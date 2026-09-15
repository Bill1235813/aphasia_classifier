"""
data.py — load the transcripts and the feature table into arrays for training.

The single function ``load_dataset`` joins config.TRANSCRIPTS_FILE (text and
labels) with config.FEATURES_FILE (numbers) on the transcript id and returns
everything the trainer needs, already converted to the binary target.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

# The two classes, in the order the model's output uses them. Aphasia is class
# 0 and is the "positive" class for precision / recall / F1.
CLASSES = ["aphasia", "control"]


def load_dataset():
    """Returns a dictionary with:
        ids       list[str]     transcript ids
        texts     list[str]     the narratives
        features  ndarray       (n, len(config.FEATURES)) float32
        y         ndarray       (n,) int64 — 0 = aphasia, 1 = control
        groups    list[str]     the original labels (used to stratify splits)
    """
    transcripts = json.loads(Path(config.TRANSCRIPTS_FILE).read_text())
    table = pd.read_csv(config.FEATURES_FILE).set_index(config.ID_COLUMN)

    missing = [f for f in config.FEATURES if f not in table.columns]
    assert not missing, (f"these features are not columns of {config.FEATURES_FILE}: {missing}\n"
                         f"run build_features.py, or add the columns yourself")

    ids = [row[config.ID_COLUMN] for row in transcripts]
    texts = [row[config.TEXT_COLUMN] for row in transcripts]
    groups = [row[config.LABEL_COLUMN] for row in transcripts]
    features = table.loc[ids, config.FEATURES].to_numpy(dtype=np.float32)
    y = np.array([0 if g != config.CONTROL_LABEL else 1 for g in groups], dtype=np.int64)

    print(f"loaded {len(ids)} transcripts: {int((y == 0).sum())} aphasia, "
          f"{int((y == 1).sum())} control; {features.shape[1]} features")
    return {"ids": ids, "texts": texts, "features": features, "y": y, "groups": groups}
