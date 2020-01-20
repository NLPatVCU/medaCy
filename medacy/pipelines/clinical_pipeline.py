import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class ClinicalPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline. It was created for the extraction of ADE related entities
    from the 2018 N2C2 Shared Task.

    Created by Andiy Mulyar (andriymulyar.com) of NLP@VCU
    """


    def __init__(self, entities, metamap=None, **kwargs):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param entities: a list of entities to use in this pipeline.
        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """

        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

        if metamap:
            self.add_component(MetaMapOverlayer, metamap)

    def get_learner(self):
        return "CRF_l2sgd", get_crf()

    def get_tokenizer(self):
        return ClinicalTokenizer(self.spacy_pipeline)

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
