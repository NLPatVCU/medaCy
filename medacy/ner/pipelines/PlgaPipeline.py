import sklearn_crfsuite
import spacy
from gensim.models import KeyedVectors

from medacy.pipeline_components import GoldAnnotatorComponent, MetaMapComponent
from medacy.pipeline_components import MetaMap, SystematicReviewTokenizer
from medacy.pipeline_components.feature_extraction.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_extraction.embedding_feature_extractor import EmbeddingFeatureExtractor
from .base import BasePipeline


class SystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition in the PLGA dataset.
    """

    def __init__(self, entities, metamap=None, word_embeddings=None, use_embeddings=False, use_distance=False):
        """
        :param entities: the list of entities you want to look at
        :param word_embeddings: the path to a binary of gensim-compatible word embeddings
        :param metamap: an instance of MetaMap
        :param use_embeddings: bool for if to use word embeddings as a feature
        :param use_distance: bool for if to use distance between words as a feature
        """

        super().__init__("plga_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_md"),
                         description=self.__doc__,
                         creators="Steele Farnsworth",
                         organization="NLP@VCU")

        self.entities = entities

        self.spacy_pipeline.tokenizer = self.get_tokenizer()

        self.add_component(GoldAnnotatorComponent, entities)

        if word_embeddings and not (use_embeddings and use_distance):
            raise Exception("The parameter word_embeddings is set, but neither use_embeddings or use_distance is True")

        self.use_word_embeddings = use_embeddings
        self.use_distance = use_distance

        if word_embeddings is not None:
            self.word_embeddings = KeyedVectors.load_word2vec_format(word_embeddings, binary=True)
        else:
            self.word_embeddings = None

        if metamap is not None and isinstance(metamap, MetaMap):
            self.add_component(MetaMapComponent, metamap)

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
        spacy_features = ['pos_', 'shape_', 'prefix_', 'suffix_', 'text']
        window_size = 10

        if self.use_word_embeddings or self.use_distance:
            return EmbeddingFeatureExtractor(
                self.word_embeddings,
                window_size=window_size, spacy_features=spacy_features,
                use_distance=self.use_distance, use_embedding=self.use_word_embeddings
            )
        else: return FeatureExtractor(window_size=window_size, spacy_features=spacy_features)
