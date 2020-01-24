import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipelines.base.base_pipeline import BasePipeline


class ScispacyPipeline(BasePipeline):
    """
    A pipeline for named entity recognition using ScispaCy, see https://allenai.github.io/scispacy/

    This pipeline differs from the ClinicalPipeline in that it uses AllenAI's 'en_core_sci_md' model and
    the tokenizer is simply spaCy's tokenizer.

    Created by Steele Farnsworth of NLP@VCU

    Requirements:
    scispacy
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz
    """

    def __init__(self, entities, metamap=None, **kwargs):
        """
        :param entities: a list of entities
        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """
        super().__init__(entities, spacy_pipeline=spacy.load("en_core_sci_md"), **kwargs)

        if metamap:
            self.add_component(MetaMapOverlayer, metamap)

    def get_learner(self):
        return "CRF_l2sgd", get_crf()

    def get_tokenizer(self):
        return None

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
