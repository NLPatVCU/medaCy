from spacy.tokens import Doc, Token
from gensim.models import Word2Vec
from medacy.pipeline_components.base import BaseComponent

class EmbeddingComponent(BaseComponent):

    def __init__(self, spacy_pipeline):
        super().__init__(component_name="embedding_component")
        self.spacy_pipeline = spacy_pipeline
        self.model = Word2Vec.load("some/path")

    def __call__(self, doc: Doc):
        # self.model.similarity('france', 'spain')
        pass
