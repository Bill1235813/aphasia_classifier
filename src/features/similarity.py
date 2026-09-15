"""
Similarity contrasts: is a transcript closer to control speech or to aphasic
speech?

We keep two reference corpora — all the control transcripts and all the
aphasia transcripts — and for a new transcript measure

    delta = (mean similarity to controls) - (mean similarity to aphasics)

so that a positive delta means "sounds more like the controls" and a negative
delta means "sounds more like the aphasia group". Two versions:

    cossim_delta_semantic   similarity of sentence-meaning embeddings
                            (a small sentence-transformer model)
    cossim_delta_syntactic  similarity of word-frequency (TF-IDF) vectors,
                            which is mostly sensitive to wording and function
                            words rather than meaning

NOTE: [design thought] when we compute this feature for a transcript that is
ITSELF in the reference corpus (i.e. during training), its similarity to its
own group would include its similarity to itself (which is 1.0) — a leak that
would hand the classifier the label. So for the training set we compute the
group means LEAVE-ONE-OUT, excluding each transcript from its own group's
average. For new transcripts (predict.py) no exclusion is needed.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

COLUMNS = ["cossim_delta_semantic", "cossim_delta_syntactic"]


class SimilarityReference:
    """The two reference corpora, encoded once and reused for every query."""

    def __init__(self, texts, is_control, model_name):
        self.is_control = np.asarray(is_control, dtype=bool)
        self.is_aphasia = ~self.is_control

        self.embedder = SentenceTransformer(model_name)
        # NOTE: [shape] embeddings: (n_transcripts, embedding_dim), each row
        # L2-normalised so a dot product between rows IS the cosine similarity.
        self.embeddings = self.embedder.encode(texts, batch_size=128,
                                               normalize_embeddings=True,
                                               show_progress_bar=False)

        # NOTE: [pedagogical] TF-IDF turns each transcript into a vector of
        # word frequencies, down-weighting words that appear in most
        # transcripts ("the", "and"). min_df=2 ignores words seen only once.
        self.tfidf = TfidfVectorizer(lowercase=True, min_df=2)
        self.tfidf_vectors = self.tfidf.fit_transform(texts)

    def leave_one_out_deltas(self):
        """Deltas for every transcript in the reference corpus (for training)."""
        semantic = self._loo_delta(cosine_similarity(self.embeddings))
        syntactic = self._loo_delta(cosine_similarity(self.tfidf_vectors))
        return [{"cossim_delta_semantic": float(s), "cossim_delta_syntactic": float(t)}
                for s, t in zip(semantic, syntactic)]

    def _loo_delta(self, similarity):
        """Mean similarity to each group, excluding the transcript itself."""
        n = similarity.shape[0]
        not_self = ~np.eye(n, dtype=bool)
        # NOTE: [shape] similarity is (n, n); the masks broadcast the (n,) group
        # membership across rows so column j is "kept" only if transcript j is
        # in that group AND j != i.
        control_mask = self.is_control[None, :] & not_self
        aphasia_mask = self.is_aphasia[None, :] & not_self
        mean_control = (similarity * control_mask).sum(1) / control_mask.sum(1)
        mean_aphasia = (similarity * aphasia_mask).sum(1) / aphasia_mask.sum(1)
        return mean_control - mean_aphasia

    def compute(self, text):
        """Deltas for one NEW transcript that is not in the reference corpus."""
        embedding = self.embedder.encode([text], normalize_embeddings=True,
                                         show_progress_bar=False)
        semantic_sim = cosine_similarity(embedding, self.embeddings)[0]
        syntactic_sim = cosine_similarity(self.tfidf.transform([text]),
                                          self.tfidf_vectors)[0]
        return {
            "cossim_delta_semantic": float(semantic_sim[self.is_control].mean()
                                           - semantic_sim[self.is_aphasia].mean()),
            "cossim_delta_syntactic": float(syntactic_sim[self.is_control].mean()
                                            - syntactic_sim[self.is_aphasia].mean()),
        }
