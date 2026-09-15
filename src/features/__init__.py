"""
The four feature groups. Each module exposes ``COLUMNS`` (the names of the
columns it produces) and a way to compute them from transcript text:

    narrative.compute(text)      -> dict      (no models needed)
    repetition.compute(text)     -> dict      (no models needed)
    SimilarityReference(...).compute(text)    (small embedding model)
    SurprisalScorer(...).compute(text)        (large language model, GPU)
"""

from . import narrative, repetition, similarity, surprisal  # noqa: F401
