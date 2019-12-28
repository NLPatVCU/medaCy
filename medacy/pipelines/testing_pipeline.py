import sklearn_crfsuite
import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class TestingPipeline(BasePipeline):
    """
    A pipeline for test running
    """

    def __init__(self, entities):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        Created by Andiy Mulyar (andriymulyar.com) of NLP@VCU
        """

        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"))

    def get_learner(self):
        return ("CRF_l2sgd", sklearn_crfsuite.CRF(
            algorithm='l2sgd',
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        ))

    def get_tokenizer(self):
        tokenizer = ClinicalTokenizer(self.spacy_pipeline)
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        extractor = FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
        return extractor