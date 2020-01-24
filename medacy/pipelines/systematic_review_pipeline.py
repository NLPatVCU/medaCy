import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class SystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. This pipeline was designed over-top of the TAC 2018 SRIE track
    challenge.

    Created by Andriy Mulyar (andriymulyar.com) of NLP@VCU
    """


    def __init__(self, entities, metamap=None, **kwargs):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param entities: a list of entities
        :param metamap: an instance of MetaMap
        """

        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

        if metamap:
            metamap = MetaMap(metamap)
            self.add_component(MetaMapOverlayer, metamap)

    def get_learner(self):
        return "CRF_l2sgd", get_crf()

    def get_tokenizer(self):
        return SystematicReviewTokenizer(self.spacy_pipeline)

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=10, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
