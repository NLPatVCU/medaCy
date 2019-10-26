import spacy

from medacy.pipeline_components.feature_extracters.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapComponent
from medacy.pipeline_components.learners.bilstm_crf_learner import BiLstmCrfLearner
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline
from medacy.tools.get_metamap import get_metamap


class LstmSystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. A special tokenizer that breaks down a clinical document
    to character level tokens defines this pipeline.
    """

    def __init__(self, entities, word_embeddings, cuda_device=-1, use_metamap=False):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param entities: a list of entities to be used by this pipeline
        :param word_embeddings: path to a word embeddings file
        :param cuda_device: int for which GPU to use, defaults to using the CPU
        :param use_metamap: bool for if to use MetaMap
        """
        description="""Pipeline tuned for the extraction of ADE related entities from the 2018 N2C2 Shared Task"""
        super().__init__("lstm_clinical_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Jorge Vargas",  # append if multiple creators
                         organization="NLP@VCU",
                         cuda_device=cuda_device
                         )

        if use_metamap:
            metamap = MetaMap(get_metamap())
            self.add_component(MetaMapComponent, metamap)

        self.entities = entities
        self.word_embeddings = word_embeddings
        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

    def get_learner(self):
        learner = BiLstmCrfLearner(self.word_embeddings, self.cuda_device)
        return 'BiLSTM+CRF', learner

    def get_tokenizer(self):
        return SystematicReviewTokenizer(self.spacy_pipeline)

    def get_feature_extractor(self):
        return FeatureExtractor(
            window_size=0,
            spacy_features=['text', 'pos', 'shape', 'prefix', 'suffix']
        )
