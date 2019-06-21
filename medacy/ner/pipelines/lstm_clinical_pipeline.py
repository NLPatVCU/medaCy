import spacy, sklearn_crfsuite
from .base import BasePipeline
from spacy.tokenizer import Tokenizer
from medacy.ner.learners import BiLstmCrfLearner
from medacy.pipeline_components import ClinicalTokenizer
from medacy.pipeline_components import WordOnlyFeatureExtractor

from medacy.pipeline_components import BiluoAnnotatorComponent


class LstmClinicalPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline.
    """

    def __init__(self, entities=[]):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """
        description="""Pipeline tuned for the extraction of ADE related entities from the 2018 N2C2 Shared Task"""
        super().__init__("lstm_clinical_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Jorge Vargas", #append if multiple creators
                         organization="NLP@VCU"
                         )

        self.entities = entities

        self.spacy_pipeline.tokenizer = self.get_tokenizer() #set tokenizer

        self.add_component(BiluoAnnotatorComponent, entities) #add overlay for GoldAnnotation

    def get_learner(self):
        learner = BiLstmCrfLearner()
        return ('BiLSTM+CRF', learner)

    def get_tokenizer(self):
        tokenizer = Tokenizer(self.spacy_pipeline.vocab)
        return tokenizer

    def get_feature_extractor(self):
        extractor = WordOnlyFeatureExtractor(100)
        return extractor