import spacy, sklearn_crfsuite
from .base import BasePipeline
from medacy.pipeline_components import MetaMap, SystematicReviewTokenizer
from medacy.pipeline_components.feature_extraction.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.embeddings.embedding_component import EmbeddingComponent
from medacy.pipeline_components import GoldAnnotatorComponent, MetaMapComponent


class SystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. This pipeline was designed over-top of the TAC 2018 SRIE track
    challenge.
    """

    def __init__(self, word_embeddings=None, metamap=None, entities=None, embedding_extractor=False):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        You can use Gensim word embedding themselves can be used as a feature, or the embedding extractor that uses
        the distance between tokens as a feature, or both.

        If word embedding features only or both, set word_embeddings to the path to the word vector binary.
        If using the embedding extractor only, set embedding_extractor to the path to the word vector binary.

        :param word_embeddings: the path to a binary of gensim-compatible word embeddings
        :param metamap: an instance of MetaMap
        :param embedding_extractor: set to True if you want to use the embedding feature extractor, or path to
            gensim binary if using embedding extractor without word embeddings themselves as a feature
        """

        super().__init__("systematic_review_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description="Pipeline tuned for the recognition of systematic review related entities from the TAC 2018 SRIE track",
                         creators="Andriy Mulyar (andriymulyar.com)", #append if multiple creators
                         organization="NLP@VCU")

        self.entities = entities if entities is not None else []

        self.spacy_pipeline.tokenizer = self.get_tokenizer()  # set tokenizer

        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

        if word_embeddings or embedding_extractor:
            from gensim.models import KeyedVectors
            if isinstance(word_embeddings, str):
                self.word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings, binary=True)
            elif isinstance(embedding_extractor, str):
                self.word_embeddings = KeyedVectors.load_word2vec_format(embedding_extractor, binary=True)

        if word_embeddings is not None:
            self.add_component(EmbeddingComponent, self.word_embeddings)

        if metamap is not None and isinstance(metamap, MetaMap):
            self.add_component(MetaMapComponent, metamap)

        self.use_embedding_extractor = embedding_extractor

    def get_learner(self):
        return ("CRF_l2sgd",
                sklearn_crfsuite.CRF(
                    algorithm='l2sgd',
                    c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True)
                )

    def get_tokenizer(self):
        tokenizer = SystematicReviewTokenizer(self.spacy_pipeline)
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        if self.use_embedding_extractor:
            from medacy.pipeline_components.feature_extraction.embedding_feature_extractor import EmbeddingFeatureExtractor
            return EmbeddingFeatureExtractor(
                self.word_embeddings,
                window_size=10,
                spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text']
                )
        else: return FeatureExtractor(window_size=10, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
