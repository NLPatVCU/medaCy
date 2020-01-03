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

    def __init__(self, entities=[], **kwargs):
        """
        Create a pipeline with the name 'bert_pipeline' utilizing
        by default spaCy's small english model.

        :param entities: Possible entities.
        :param cuda_device: Which cuda device to use. -1 for CPU.
        :param batch_size: Batch size to use during training.
        :param learning_rate: Learning rate to use during training.
        :param epochs: Number of epochs to use for training.
        """
        description="""Pipeline tuned for the extraction of ADE related entities from the 2018 N2C2 Shared Task"""
        super().__init__("bert_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Jorge Vargas", #append if multiple creators
                         organization="NLP@VCU",
                         )

        self.entities = entities
        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

        self.cuda_device = kwargs['cuda_device']
        self.batch_size = kwargs['batch_size'] if kwargs['batch_size'] else 8
        self.learning_rate = kwargs['learning_rate'] if kwargs['learning_rate'] else 1e-5
        self.epochs = kwargs['epochs'] if kwargs['epochs'] else 3
        self.pretrained_model = kwargs['pretrained_model']
        self.using_crf = kwargs['using_crf']

    def get_learner(self):
        learner = BertLearner(
            self.cuda_device,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            using_crf=self.using_crf
        )
        return ('BERT', learner)

    def get_tokenizer(self):
        tokenizer = SystematicReviewTokenizer(self.spacy_pipeline)
        return tokenizer

    def get_feature_extractor(self):
        extractor = TextExtractor()
        return extractor
