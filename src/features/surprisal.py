"""
Surprisal statistics: how unpredictable a large language model finds the text.

A language model assigns each word a probability given the words before it.
Surprisal is -log(probability): a low number for an expected word ("peanut
butter and ___ jelly"), a high number for an unexpected one. Disordered speech
— omitted function words, wrong words, broken syntax — is less predictable, so
its surprisal runs higher. We summarise the per-word surprisals of a transcript
with six statistics.

NOTE: [design thought] the paper uses Llama-2-7B-chat as the surprisal model
because the same model is the subject of the later ablation experiments. For
classification alone any causal language model works; "gpt2-xl" is an open,
much smaller choice (see config.SURPRISAL_MODEL).
"""

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

COLUMNS = ["mean_surp", "std_surp", "p90_surp", "p95_surp", "max_surp", "frac_high_surp"]


class SurprisalScorer:
    """Loads the language model once; ``compute`` scores one transcript."""

    def __init__(self, model_name, high_threshold, max_tokens):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # NOTE: [design thought] half precision halves the memory of a 7B model
        # (~13 GB instead of ~27 GB). We cast back to float32 before the
        # log-softmax so tiny probabilities at the tail of the vocabulary do
        # not underflow to zero.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32)
        self.model.to(self.device).eval()
        self.high_threshold = high_threshold
        self.max_tokens = max_tokens

    def compute(self, text):
        ids = self.tokenizer(text, return_tensors="pt", truncation=True,
                             max_length=self.max_tokens)["input_ids"].to(self.device)
        if ids.shape[1] < 2:
            # NOTE: [edge case callout] a one-token text has no "next word" to
            # predict; the caller replaces NaN with the corpus mean.
            return {column: float("nan") for column in COLUMNS}

        with torch.no_grad():
            logits = self.model(ids).logits
        # NOTE: [shape] logits: (1, n_tokens, vocab). The logit at position t is
        # the prediction for token t+1, so we drop the last position and
        # compare against ids shifted by one ("teacher forcing").
        log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        target = ids[:, 1:]
        token_log_probs = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        surprisal = (-token_log_probs).cpu().numpy().flatten()

        return {
            "mean_surp": float(surprisal.mean()),
            "std_surp": float(surprisal.std()),
            "p90_surp": float(np.percentile(surprisal, 90)),
            "p95_surp": float(np.percentile(surprisal, 95)),
            "max_surp": float(surprisal.max()),
            "frac_high_surp": float((surprisal > self.high_threshold).mean()),
        }
