from spacy.tokens import Doc, Token
from gensim.models import KeyedVectors
import numpy as np
from medacy.pipeline_components.base import BaseComponent


class EmbeddingComponent(BaseComponent):
    """A pipeline component that gathers word embeddings for use as features using gensim."""

    dependencies = []
    name = "embedding_component"

    def __init__(self, spacy_pipeline, word_embeddings):
        super().__init__(component_name="embedding_component")
        self.spacy_pipeline = spacy_pipeline
        self.model = KeyedVectors.load_word2vec_format(word_embeddings, binary=True)

    def _lookup_embedding(self, token):
        try:
            word_vector = self.model[token.text]
            norm = np.linalg.norm(word_vector)
            if norm == 0:
                return [bytes(n) for n in word_vector]
            else:
                return [bytes(n) for n in word_vector / norm]
        except KeyError:
            return []

    def __call__(self, doc: Doc):
        Token.set_extension("feature_embedding", getter=self._lookup_embedding, force=True)
        return doc
