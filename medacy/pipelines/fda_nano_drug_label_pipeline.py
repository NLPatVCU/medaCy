import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class FDANanoDrugLabelPipeline(BasePipeline):
    """
    A pipeline for named entity recognition of FDA nanoparticle drug labels. This pipeline was designed over-top of
    the TAC 2018 SRIE track challenge.

    Created by Andriy Mulyar (andriymulyar.com) of NLP@VCU
    """

    def __init__(self, entities, metamap=None, **kwargs):
        """
        :param entities: a list of Entities
        :param metamap: an instance of MetaMap
        """

        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

        if metamap:
            self.add_component(MetaMapOverlayer, metamap)

    def get_learner(self):
        return "CRF_l2sgd", get_crf()

    def get_tokenizer(self):
        return ClinicalTokenizer(self.spacy_pipeline)  # Best run with SystematicReviewTokenizer

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=6, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num', 'text'])
