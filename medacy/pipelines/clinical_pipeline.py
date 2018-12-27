import spacy, sklearn_crfsuite
from .base import BasePipeline
from ..pipeline_components import ClinicalTokenizer
from medacy.model.feature_extractor import FeatureExtractor

from ..pipeline_components import GoldAnnotatorComponent, MetaMapComponent, UnitComponent, MetaMap


class ClinicalPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline.
    """

    def __init__(self, metamap=None, entities=[]):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """
        description="""Pipeline tuned for the extraction of ADE related entities from the 2018 N2C2 Shared Task"""
        super().__init__("clinical_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Andriy Mulyar (andriymulyar.com)", #append if multiple creators
                         organization="NLP@VCU"
                         )

        self.entities = entities

        self.spacy_pipeline.tokenizer = self.get_tokenizer() #set tokenizer

        self.add_component(GoldAnnotatorComponent, entities) #add overlay for GoldAnnotation

        if metamap is not None and isinstance(metamap, MetaMap):
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
        extractor = FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
        return extractor







