import spacy, sklearn_crfsuite
from medacy.pipelines.base.base_pipeline import BasePipeline
from spacy.tokenizer import Tokenizer
from medacy.pipeline_components import BertLearner

from medacy.pipeline_components.feature_extracters.text_extractor import TextExtractor
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer

class BertPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline.
    """

    def __init__(self, entities=[], word_embeddings=None, cuda_device=-1, batch_size=32):
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
                         organization="NLP@VCU",
                         cuda_device=cuda_device
                         )

        self.entities = entities
        self.word_embeddings = word_embeddings
        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation
        self.batch_size = batch_size

    def get_learner(self):
        learner = BertLearner(
            self.cuda_device,
            pretrained_model='bert-large-cased',
            batch_size=self.batch_size
        )
        return ('BERT', learner)

    def get_tokenizer(self):
        tokenizer = SystematicReviewTokenizer(self.spacy_pipeline)
        return tokenizer

    def get_feature_extractor(self):
        extractor = TextExtractor()
        return extractor
