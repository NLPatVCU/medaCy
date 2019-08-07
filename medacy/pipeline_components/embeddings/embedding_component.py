from spacy.tokens import Doc, Token
from medacy.pipeline_components.base import BaseComponent


class EmbeddingComponent(BaseComponent):
    """A pipeline component that gathers word embeddings for use as features using gensim."""

    dependencies = []
    name = "embedding_component"

    def __init__(self, spacy_pipeline, word_embeddings):
        super().__init__(component_name="embedding_component")
        self.spacy_pipeline = spacy_pipeline
        self.model = word_embeddings

    def _lookup_embedding(self, token):
        try:
            return [bytes(n) for n in self.model[token.text]]
        except KeyError:
            return []

    def __call__(self, doc: Doc):
        Token.set_extension("feature_embedding", getter=self._lookup_embedding, force=True)
        return doc
