# Aphasia classifier

A classifier that reads a spoken narrative and estimates the probability that
the speaker has aphasia. It combines a fine-tuned text encoder with 16
interpretable linguistic features. Two AphasiaBank elicitation tasks are
included, each with its own corpus and its own trained models:

| task | what participants do | transcripts | held-out accuracy | aphasia F1 |
|---|---|---|---|---|
| `cinderella` | retell the Cinderella story | 1,147 | 95.7 % | 0.967 |
| `sandwich` | describe making a peanut-butter-and-jelly sandwich | 1,437 | 96.8 % | 0.952 |

(20-seed ensembles trained with `src/train.py` exactly as shipped; the
per-transcript predictions behind these numbers are in
`models/<task>/ensemble_predictions.csv`.)

On hand-labelled language-model outputs the Cinderella classifier has perfect
precision (it never calls a genuinely fluent narrative aphasic).

This folder is self-contained. You can:

* **train** the classifier on the included data (or your own),
* **change** which features, models and data it uses by editing one file, and
* **score** new transcripts with the trained model.

No prior experience with PyTorch is assumed. Every script has a comment at the
top explaining what it does, and every setting in `src/config.py` says what
happens if you change it.

---

## Quick start

```bash
# 1. install (once)
conda create -n aphasia python=3.11 -y && conda activate aphasia
pip install torch --index-url https://download.pytorch.org/whl/cu126   # GPU; see "Installation" for CPU
pip install -r requirements.txt

# 2. train on the included features (about 8 minutes per seed on one GPU)
python src/train.py --n-seeds 3

# 3. score new transcripts
python src/predict.py --csv data/cinderella/example_new_transcripts.csv --text-column transcript

# the same, for the sandwich task:
APHASIA_TASK=sandwich python src/train.py --n-seeds 3
APHASIA_TASK=sandwich python src/predict.py --csv data/sandwich/example_new_transcripts.csv --text-column transcript
```

The included `data/<task>/features.csv` already holds every feature for every
transcript, so step 2 needs no language model — only the text encoder, which
downloads automatically (~1.3 GB).

---

## What is in this folder

```
aphasia_classifier/
├── README.md                 ← you are here
├── requirements.txt
├── src/
│   ├── config.py             ← EVERY setting you might change. Start here.
│   ├── build_features.py     ← transcripts → features.csv
│   ├── train.py              ← train N models, report accuracy, save to models/
│   ├── predict.py            ← score new transcripts with the saved models
│   ├── data.py               ← loads transcripts + features into arrays
│   ├── model.py              ← the network (encoder + features → head)
│   └── features/             ← one file per feature group
│       ├── narrative.py      ← utterance & word counts (CLAN-style)
│       ├── repetition.py     ← perseveration statistics
│       ├── similarity.py     ← closeness to control vs aphasia corpus
│       └── surprisal.py      ← language-model unpredictability
├── data/
│   ├── cinderella/           ← one folder per task, same layout in each:
│   │   ├── transcripts.json      the narratives + labels
│   │   ├── features.csv          the numeric feature table (precomputed)
│   │   └── example_new_transcripts.csv
│   ├── sandwich/
│   └── README.md             ← data dictionary
├── models/
│   ├── cinderella/           ← created by train.py, one folder per task
│   └── sandwich/
└── notebooks/
    └── quickstart.ipynb      ← the same workflow, step by step, in Jupyter
```

---

## Installation

### 1. Python environment

Python 3.10 or newer. We recommend a fresh conda environment:

```bash
conda create -n aphasia python=3.11 -y
conda activate aphasia
```

### 2. PyTorch

Install PyTorch first, choosing the line for your hardware
(see <https://pytorch.org/get-started/locally/> if unsure):

```bash
# NVIDIA GPU (CUDA 12.6):
pip install torch --index-url https://download.pytorch.org/whl/cu126
# CPU only (training will be slow; prediction is fine):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### 3. Everything else

```bash
pip install -r requirements.txt
```

### 4. (Only if you will compute surprisal) access to Llama-2

The surprisal features are computed with Meta's Llama-2-7B-chat, a *gated*
model. You need it only to **rebuild the surprisal features** or to
**score new transcripts** — training on the included `features.csv` does not.

1. Accept the licence at <https://huggingface.co/meta-llama/Llama-2-7b-chat-hf>.
2. Create a token at <https://huggingface.co/settings/tokens>.
3. Run `huggingface-cli login` and paste the token.

If you would rather avoid this, open `src/config.py` and set
`SURPRISAL_MODEL = "gpt2-xl"` (open, 1.5B parameters, runs on any GPU),
then rebuild the surprisal features and retrain — see *Changing the models*.

### 5. Check it works

```bash
python -c "import torch, transformers, peft, sentence_transformers; print('ok, GPU:', torch.cuda.is_available())"
```

### 6. How much GPU memory you need

Measured peak memory on an NVIDIA RTX A6000 with the default settings
(`MAX_TOKENS = 384`). Add roughly 1 GB on top of any figure for the CUDA
runtime itself.

**The classifier (train.py / the encoder part of predict.py)** — depends on
the text encoder and the batch size:

| `TEXT_ENCODER` | parameters | train, `BATCH_SIZE=4` | train, `BATCH_SIZE=8` | evaluate |
|---|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 35 M | 0.8 GB | 1.4 GB | 0.2 GB |
| `BAAI/bge-base-en-v1.5` | 112 M | 1.7 GB | 2.9 GB | 0.5 GB |
| `BAAI/bge-large-en-v1.5` (default) | 342 M | **4.6 GB** | 7.7 GB | 1.5 GB |

**The surprisal model (build_features.py / predict.py)** — loaded in half
precision, inference only:

| `SURPRISAL_MODEL` | parameters | memory |
|---|---|---|
| `gpt2` | 0.12 B | 0.5 GB |
| `gpt2-xl` | 1.6 B | 3.3 GB |
| `meta-llama/Llama-2-7b-chat-hf` (default) | 6.7 B | **13.0 GB** |

Putting the two together, for the default settings:

| what you run | needs | fits comfortably on |
|---|---|---|
| `train.py` (features already built) | ~5 GB | any 8 GB GPU |
| `build_features.py` / `predict.py` with Llama-2 | ~15 GB | a 16 GB GPU (tight) or 24 GB |
| `build_features.py` / `predict.py` with `gpt2-xl` | ~5 GB | any 8 GB GPU |

If memory is short: switch `SURPRISAL_MODEL` to `"gpt2-xl"` (the big saving),
then `TEXT_ENCODER` to `bge-base`, then lower `BATCH_SIZE`. A CPU-only machine
can run everything except a full training run in reasonable time.

---

## Running from the terminal

All commands are run from inside the `aphasia_classifier/` folder.

### Choosing the task

Every script works on one task at a time — `cinderella` by default. Switch with
an environment variable in front of the command, or by editing `TASK` in
`src/config.py`:

```bash
APHASIA_TASK=sandwich python src/train.py
```

Each task keeps its own data (`data/<task>/`) and models (`models/<task>/`).

### Step A — build the features (optional: they are already included)

```bash
python src/build_features.py                            # all four groups
python src/build_features.py --groups narrative,repetition   # only the cheap ones
```

Reads `data/<task>/transcripts.json`, writes `data/<task>/features.csv`. Rebuilding only
some groups keeps every other column intact, including any you added by hand.

| group | time | needs |
|---|---|---|
| narrative, repetition | seconds | nothing |
| similarity | ~1 min | downloads a 90 MB model |
| surprisal | ~10 min on a GPU | Llama-2 access (or `gpt2-xl`) |

### Step B — train

```bash
python src/train.py                # 20 seeds, as in the paper (~3 h on one GPU)
python src/train.py --n-seeds 3    # a quick run (~25 min)
```

Each *seed* trains one model on a different random 70/10/20 split. The
final ensemble averages all of them. You will see one line per epoch, then a
summary like:

```
=== per-seed test results (mean ± std) ===
  accuracy    0.944 ± 0.019
  aphasia_f1  0.956 ± 0.015
=== ensemble (average over seeds, every transcript scored while held out) ===
  accuracy               0.957
  aphasia_f1             0.966
```

Outputs, all in `models/<task>/`:

* `seed_00/ … seed_19/` — the trained models
* `results.json` — the metrics above, plus the exact settings used
* `ensemble_predictions.csv` — `P_aphasia` for every transcript in the corpus,
  so you can look at which ones the classifier gets wrong

### Step C — score new transcripts

```bash
# one or more transcripts on the command line
python src/predict.py --text "Once upon a time there was a girl named Cinderella ..." \
                      --text "girl. uh. ball. shoe. prince. uh. happy."

# a whole CSV file: writes <file>_scored.csv with P_aphasia and predicted_class added
python src/predict.py --csv data/cinderella/example_new_transcripts.csv --text-column transcript
```

`P(aphasia)` above 0.5 means the classifier calls the transcript aphasic.
This step recomputes all 16 features for each new text, so it loads the
surprisal model (Llama-2 or whatever `config.SURPRISAL_MODEL` says).

---

## Running from a Jupyter notebook

```bash
pip install jupyterlab
jupyter lab notebooks/quickstart.ipynb
```

The notebook walks through the same three steps with explanations, and shows
how to look at individual features for a sentence you type in. Any of the
terminal commands above can also be run from a notebook cell by putting `!`
in front, e.g. `!python src/train.py --n-seeds 2`.

---

## Customising: data, features, models, training

**Everything below is done by editing `src/config.py`.** Each setting there
has a comment; this section explains the common recipes.

### Use your own transcripts

1. Save them as a JSON list with an id, a text and a label per entry
   (format in `data/README.md`) as `data/<my_task>/transcripts.json`.
2. In `config.py`, set `TASK = "<my_task>"` (or use `APHASIA_TASK=<my_task>`), and
   if your column names differ, `ID_COLUMN`, `TEXT_COLUMN`, `LABEL_COLUMN`, and
   `CONTROL_LABEL` (the label that means "not aphasic").
3. Run `build_features.py`, then `train.py`. Your data, features and models
   stay in their own folders and never touch the included tasks.

The classifier is binary: `CONTROL_LABEL` vs everything else. Your labels can
be anything — `"healthy"` vs `"PWA"`, `"TD"` vs `"DLD"` — as long as one value
means the unimpaired group.

### Change which features are used

`config.FEATURES` is a plain list of column names. Delete a line to drop a
feature; add a line to add one. Then retrain.

* **Drop a group** — e.g. to train without surprisal (so prediction no longer
  needs a large language model), delete the six `*_surp` lines.
* **Add a precomputed column** — `MLU_Words` is already in `features.csv`; just
  add `"MLU_Words"` to the list.
* **Try CLAN's gold measures** — every numeric column of the AphasiaBank
  metadata is in `features.csv` as `gold_<name>` (e.g. `gold_MLU_Morphemes`,
  `gold_Words_Min`). Add any of them. *Caveat:* they exist only for the corpus
  transcripts, so a model trained on them can be evaluated but cannot score
  new text with `predict.py`.
* **Add your own feature from Excel/R/pandas** — open `data/features.csv`,
  add a column (keep the row order), save, add its name to `FEATURES`.
  Same caveat as gold features unless you also teach `predict.py` how to
  compute it.
* **Add a feature computed from the text** — write a function
  `compute(text) -> dict` in a new file under `src/features/`, add the
  returned column names to a `COLUMNS` list, and call it from
  `build_features.py` and `predict.py` following the pattern of
  `narrative.py` (the simplest example: ~60 lines, no models).

### Change the models

* **Text encoder** (`TEXT_ENCODER`, `ENCODER_DIM`): any Hugging Face encoder.
  `BAAI/bge-base-en-v1.5` (768-dim) is a lighter alternative. Retrain after.
* **Surprisal model** (`SURPRISAL_MODEL`): any causal language model, e.g.
  `"gpt2-xl"`, `"gpt2"`, `"meta-llama/Llama-2-13b-chat-hf"`. Rebuild the
  surprisal group and retrain after changing it:
  `python src/build_features.py --groups surprisal && python src/train.py`.
* **Similarity model** (`SIMILARITY_MODEL`): any sentence-transformers model.
  Rebuild the similarity group and retrain.
* **LoRA size** (`LORA_RANK`): 16 in the paper; 8 trains faster with a small
  accuracy cost.

### Change the training budget

`N_SEEDS`, `EPOCHS`, `PATIENCE`, `BATCH_SIZE`, `LEARNING_RATE`, `MAX_TOKENS`,
`SPLIT_FRACTIONS` — all in `config.py`, all explained there. For a first look,
`N_SEEDS = 3` is plenty; use 20 for numbers you want to report.

---

## How it works, in one paragraph

Each transcript goes through a text encoder (BGE-large), which we fine-tune
lightly with LoRA. The encoder's output — a 1,024-number summary of the text —
is joined with the 16 hand-crafted features (standardised to mean 0, std 1) and
passed to a small two-layer network that outputs P(aphasia) and P(control).
Training minimises cross-entropy on the training split, with batches balanced
between the two classes, and keeps the epoch with the best validation F1. This
is repeated for `N_SEEDS` random splits and the models' probabilities are
averaged: an ensemble is more stable than any single model, and because every
transcript is held out by some seeds, the reported accuracy is an honest
held-out number for the whole corpus.

The 16 features were chosen for interpretability and to complement the
encoder: utterance/word counts (fluency and lexical diversity), repetition
statistics (perseveration), similarity to each reference group, and how
unpredictable a language model finds the text (a proxy for well-formedness).
Their definitions are in the `src/features/` files.

---

## Troubleshooting

* **`OSError: ... gated repo`** — you need Llama-2 access (Installation §4), or
  switch `SURPRISAL_MODEL` to `"gpt2-xl"`.
* **CUDA out of memory during training** — lower `BATCH_SIZE` to 2, or
  `MAX_TOKENS` to 256, in `config.py`.
* **CUDA out of memory in predict.py / build_features.py** — the surprisal
  model is the big one (Llama-2-7B needs 13 GB); use `"gpt2-xl"` (3.3 GB).
  See *How much GPU memory you need* above for every option.
* **No GPU** — training works on CPU but takes hours per seed; set
  `N_SEEDS = 1` to try it. Prediction on CPU is fine with `gpt2-xl`.
* **`these features are not columns of features.csv`** — run
  `build_features.py`, or check the spelling in `config.FEATURES`.
* **`was trained with different features`** — `config.FEATURES` changed since
  the models in `models/<task>/` were trained. Retrain, or restore the list.
* **`no trained models in models/<task>`** — you are on a task you have not
  trained yet (check `APHASIA_TASK` / `TASK`), or `train.py` has not finished.
