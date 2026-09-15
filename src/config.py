"""
config.py — every setting you are likely to change lives in this one file.

If you only edit ONE file in this package, it is this one. The other scripts
(build_features.py, train.py, predict.py) read their settings from here, so
you never have to touch their code to change the data, the features, the
models, or the training budget.

The file is organised top-to-bottom in the order you will usually need it:

    1. TASK/DATA — which corpus (cinderella / sandwich / yours) and which columns
    2. FEATURES  — which numeric features go into the classifier
    3. MODELS    — which pretrained language models to use
    4. TRAINING  — how long / how many times to train

Every setting has a comment saying what it does and what happens if you
change it.
"""

import os
from pathlib import Path

# The folder this package lives in. All other paths are relative to it, so
# the package keeps working if you move or rename the whole folder.
PACKAGE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PACKAGE_DIR / "data"


# =============================================================================
# 1. TASK and DATA
# =============================================================================

# Two AphasiaBank elicitation tasks are included, each with its own corpus:
#
#   "cinderella"  retelling the Cinderella story          1,147 narratives
#   "sandwich"    describing how to make a PB&J sandwich   1,437 narratives
#
# Pick one here. The data files, the feature table and the saved models all
# live in a folder named after the task (data/<TASK>/, models/<TASK>/), so
# the two never overwrite each other.
#
# You can also choose the task without editing this file, from the terminal:
#     APHASIA_TASK=sandwich python src/train.py
TASK = os.environ.get("APHASIA_TASK", "cinderella")

# The transcript file. Each entry is one narrative plus metadata. See
# data/README.md for the columns. To use YOUR OWN data, point this at a JSON
# list of dictionaries that has at least the three columns named just below
# (or add a folder data/<your_task>/transcripts.json and set TASK to its name).
TRANSCRIPTS_FILE = DATA_DIR / TASK / "transcripts.json"

# The table of numeric features, one row per transcript. It is produced by
# build_features.py and read by train.py. You can open it in Excel and add
# columns of your own — see "Adding your own feature" in the README.
FEATURES_FILE = DATA_DIR / TASK / "features.csv"

# Where train.py saves the trained models and predict.py loads them from.
MODELS_DIR = PACKAGE_DIR / "models" / TASK

# Column names inside TRANSCRIPTS_FILE.
ID_COLUMN = "File_DB"          # a unique name for each transcript
TEXT_COLUMN = "text_response"  # the transcript text itself
LABEL_COLUMN = "Group"         # the diagnostic group

# The classifier is binary: "control" versus "aphasia". Every transcript whose
# LABEL_COLUMN equals CONTROL_LABEL is a control; every other label (Broca,
# Wernicke, Anomic, Conduction, ...) is treated as aphasia.
CONTROL_LABEL = "Control"


# =============================================================================
# 2. FEATURES
# =============================================================================
#
# The classifier reads the transcript text with a neural text encoder AND
# sees a short list of hand-crafted numeric features. This is that list.
#
#   * To DROP a feature, delete its line (or put a # in front of it).
#   * To ADD a feature, add its column name here. It must exist as a column in
#     FEATURES_FILE — either because build_features.py computes it, or because
#     you added the column yourself in Excel / pandas.
#
# Features are grouped by where they come from. The 16 features below are the
# ones used in the paper.

FEATURES = [
    # --- Narrative measures (5) — computed from the text by features/narrative.py
    #     These follow CLAN's definitions (the standard AphasiaBank toolkit).
    "Total_Utts",      # number of utterances
    "MLU_Utts",        # number of utterances counted for mean length of utterance
    "FREQ_types",      # number of distinct words
    "FREQ_tokens",     # total number of words
    "FREQ_TTR",        # type-token ratio (lexical diversity)

    # --- Repetition profile (3) — computed from the text by features/repetition.py
    #     Perseveration is a hallmark of disordered speech; these count it.
    "rep_shortest_len",    # length (in words) of the shortest repeated unit
    "rep_shortest_count",  # how many times that shortest unit repeats
    "rep_max_count",       # the largest repeat count of ANY repeated unit

    # --- Similarity contrasts (2) — computed by features/similarity.py
    #     "Is this transcript closer to the control corpus or the aphasia
    #     corpus?"  Positive = more control-like, negative = more aphasia-like.
    "cossim_delta_semantic",   # using a sentence-meaning embedding
    "cossim_delta_syntactic",  # using word-frequency (TF-IDF) vectors

    # --- Surprisal statistics (6) — computed by features/surprisal.py
    #     How "surprising" a large language model finds each word. Disordered
    #     speech is less predictable, so its surprisal runs higher.
    "mean_surp",
    "std_surp",
    "p90_surp",        # 90th percentile of per-word surprisal
    "p95_surp",
    "max_surp",
    "frac_high_surp",  # fraction of words with surprisal above SURPRISAL_HIGH_THRESHOLD

    # --- Other columns available in features.csv but NOT used by default:
    #   "MLU_Words"        mean words per utterance (computed from text)
    #   "gold_..."         the original CLAN measures shipped with AphasiaBank
    #                      (gold_MLU_Morphemes, gold_Words_Min, ...). These exist
    #                      only for the AphasiaBank transcripts, so a classifier
    #                      trained on them cannot score NEW text with predict.py.
]

# Which feature groups build_features.py should (re)compute. Surprisal needs a
# GPU and a large language model; the other three run on a laptop in minutes.
FEATURE_GROUPS_TO_BUILD = ["narrative", "repetition", "similarity", "surprisal"]


# =============================================================================
# 3. MODELS
# =============================================================================

# The neural text encoder. It reads the whole transcript and produces a
# summary vector (ENCODER_DIM numbers) that is concatenated with the FEATURES
# above. Any Hugging Face encoder works; if you change it, ENCODER_DIM must
# match the model's hidden size. Measured peak GPU memory (BATCH_SIZE 4 / 8):
#
#     "BAAI/bge-small-en-v1.5"   ENCODER_DIM 384    train 0.8 / 1.4 GB   evaluate 0.2 GB
#     "BAAI/bge-base-en-v1.5"    ENCODER_DIM 768    train 1.7 / 2.9 GB   evaluate 0.5 GB
#     "BAAI/bge-large-en-v1.5"   ENCODER_DIM 1024   train 4.6 / 7.7 GB   evaluate 1.5 GB   (default)
#
# Smaller encoders train several times faster at some cost in accuracy.
TEXT_ENCODER = "BAAI/bge-large-en-v1.5"
ENCODER_DIM = 1024

# We do not fine-tune all of the encoder's weights (too many for ~1,000
# transcripts). LoRA adds a small number of trainable weights instead;
# LORA_RANK controls how many. 16 is the paper setting; 8 is a cheaper choice.
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# The language model that computes surprisal. Llama-2-7B-chat is the paper
# setting but is a gated model (you must accept Meta's licence on Hugging Face
# and log in with `huggingface-cli login`). Measured GPU memory, inference only:
#
#     "gpt2"                             0.5 GB
#     "gpt2-xl"                          3.3 GB     open, a good no-licence choice
#     "meta-llama/Llama-2-7b-chat-hf"   13.0 GB     (default; needs a 16 GB+ GPU)
#
# If you change this, rebuild the surprisal features AND retrain.
SURPRISAL_MODEL = "meta-llama/Llama-2-7b-chat-hf"
SURPRISAL_HIGH_THRESHOLD = 8.0   # nats; a word above this counts as "high surprisal"
SURPRISAL_MAX_TOKENS = 2048      # longest transcript the surprisal model reads

# The small sentence-embedding model behind cossim_delta_semantic.
SIMILARITY_MODEL = "all-MiniLM-L6-v2"


# =============================================================================
# 4. TRAINING
# =============================================================================

# The classifier is trained N_SEEDS times with different random splits, and the
# final prediction averages all of them (an "ensemble"). More seeds = more
# stable and slightly more accurate, but training time scales linearly:
# roughly 8 minutes per seed on one GPU. The paper used 20. Use 3–5 to try
# things out.
N_SEEDS = 20

# How the transcripts are divided for each seed: train / validation / test.
SPLIT_FRACTIONS = (0.70, 0.10, 0.20)

EPOCHS = 20          # maximum passes over the training data
PATIENCE = 5         # stop early if validation F1 has not improved for this many epochs
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
MAX_TOKENS = 384     # transcripts longer than this are truncated by the encoder
