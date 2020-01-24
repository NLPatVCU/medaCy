"""
BERT Pipeline
"""
import spacy

from medacy.pipelines.base.base_pipeline import BasePipeline
from medacy.pipeline_components import BertLearner
from medacy.pipeline_components import TextExtractor
from medacy.pipeline_components import SystematicReviewTokenizer

# These default values are used here and by the CLI
LEARNING_RATE = 1e-5
BATCH_SIZE = 8
EPOCHS = 3

class BertPipeline(BasePipeline):
    """
    Pipeline tuned for the extraction of ADE related entities from the 2018', 'N2C2 Shared Task

    Created by Jorge Vargas of NLP@VCU
    """

    def __init__(self, entities, **kwargs):
        """
        Create a pipeline with the name 'bert_pipeline' utilizing
        by default spaCy's small english model.

        :param entities: Possible entities.
        :param cuda_device: Which cuda device to use. -1 for CPU.
        :param batch_size: Batch size to use during training.
        :param learning_rate: Learning rate to use during training.
        :param epochs: Number of epochs to use for training.
        """
        super().__init__(entities=entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

        self.cuda_device = kwargs['cuda_device'] if 'cuda_device' in kwargs else -1
        self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else BATCH_SIZE
        self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else LEARNING_RATE
        self.epochs = kwargs['epochs'] if 'epochs' in kwargs else EPOCHS
        self.pretrained_model = kwargs['pretrained_model'] if 'pretrained_model' in kwargs else 'bert-large-cased'
        self.using_crf = kwargs['using_crf'] if 'using_crf' in kwargs else False

    def get_learner(self):
        """Get the learner object for this pipeline.

        :return: BertLearner.
        """
        learner = BertLearner(
            self.cuda_device,
            pretrained_model=self.pretrained_model,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            epochs=self.epochs,
            using_crf=self.using_crf
        )
        return 'BERT', learner

    def get_tokenizer(self):
        """Get tokenizer for this pipeline.

        :return: Systematic review tokenizer.
        """
        return SystematicReviewTokenizer(self.spacy_pipeline)

    def get_feature_extractor(self):
        """Get feature extractor for this pipeline.

        :return: Text only extractor.
        """
        return TextExtractor()
