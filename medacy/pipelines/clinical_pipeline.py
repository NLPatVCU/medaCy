import spacy, sklearn_crfsuite
from .base import BasePipeline
from ..pipeline_components import ClinicalTokenizer, CharacterTokenizer
from medacy.model.feature_extractor import FeatureExtractor

from ..pipeline_components import GoldAnnotatorComponent, MetaMapComponent, UnitComponent


class ClinicalPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition
    """

    def __init__(self, metamap, entities=[]):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap
        """
        super().__init__("clinical_pipeline", spacy.load("en_core_web_sm"))

        self.entities = entities

        self.spacy_pipeline.tokenizer = self.get_tokenizer() #set tokenizer

        self.add_component(GoldAnnotatorComponent, entities) #add overlay for GoldAnnotation
        self.add_component(MetaMapComponent, metamap)
        self.add_component(UnitComponent)
            


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
        extractor = FeatureExtractor(window_size = 2, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num'])
        return extractor







