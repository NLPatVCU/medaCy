import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class TestingPipeline(BasePipeline):
    """
    A pipeline for test running
    """

    def __init__(self, entities, **kwargs):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.
        """

        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

    def get_learner(self):
        return "CRF_l2sgd", get_crf()

    def get_tokenizer(self):
        return SystematicReviewTokenizer(self.spacy_pipeline)

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
