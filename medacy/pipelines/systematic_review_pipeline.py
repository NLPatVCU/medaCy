import sklearn_crfsuite
import spacy

from medacy.pipeline_components import GoldAnnotatorComponent
from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapComponent
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from .base import BasePipeline


class SystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. This pipeline was designed over-top of the TAC 2018 SRIE track
    challenge.
    """

    def __init__(self, metamap=None, entities=None, word_embeddings=None, use_embeddings=False, use_distance=False, cuba_device=-1):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param word_embeddings: the path to a binary of gensim-compatible word embeddings
        :param metamap: an instance of MetaMap
        :param use_embeddings: bool for if to use word embeddings as a feature
        :param use_distance: bool for if to use distance between words as a feature
        """

        super().__init__("systematic_review_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description="Pipeline tuned for the recognition of systematic review related entities from the TAC 2018 SRIE track",
                         creators="Andriy Mulyar (andriymulyar.com)", #append if multiple creators
                         organization="NLP@VCU",
                         cuda_device=cuba_device)

        self.entities = entities if entities is not None else []

        self.spacy_pipeline.tokenizer = self.get_tokenizer()  # set tokenizer

        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

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
        return FeatureExtractor(
            window_size=10,
            spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text']
        )
