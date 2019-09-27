import spacy, sklearn_crfsuite
from .base import BasePipeline
from spacy.tokenizer import Tokenizer
from medacy.pipeline_components import BiLstmCrfLearner
from medacy.pipeline_components import ClinicalTokenizer, SystematicReviewTokenizer
from medacy.pipeline_components import FeatureExtractor
from medacy.pipeline_components import GoldAnnotatorComponent


class LstmSystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline.
    """

    def __init__(self, entities=[], word_embeddings=None, cuda_device=-1):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """
        description="""Pipeline tuned for the extraction of ADE related entities from the 2018 N2C2 Shared Task"""
        super().__init__("lstm_clinical_pipeline",
                         spacy_pipeline=spacy.load('en_core_web_lg'),
                         description=description,
                         creators="Jorge Vargas",  #append if multiple creators
                         organization="NLP@VCU",
                         cuda_device=cuda_device
                         )

        self.entities = entities
        self.word_embeddings = word_embeddings
        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

    def get_learner(self):
        learner = BiLstmCrfLearner(self.word_embeddings, self.cuda_device)
        return ('BiLSTM+CRF', learner)

    def get_tokenizer(self):
        tokenizer = SystematicReviewTokenizer(self.spacy_pipeline)
        return tokenizer

    def get_feature_extractor(self):
        extractor = FeatureExtractor(
            window_size=0,
            spacy_features=['text']
        )

        return extractor
