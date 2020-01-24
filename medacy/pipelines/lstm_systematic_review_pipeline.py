import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.learners.bilstm_crf_learner import BiLstmCrfLearner
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class LstmSystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition using the BiLSTM+CRF learner.

    Created by Jorge Vargas of NLP@VCU
    """

    def __init__(self, entities, metamap=None, **kwargs):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param entities: a list of entities to be used by this pipeline
        :param metamap: MetaMap object (optional, absence of which will not use MetaMap)
        :param word_embeddings: path to a word embeddings file
        :param cuda_device: int for which GPU to use, defaults to using the CPU
        """

        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

        if metamap:
            metamap = MetaMap(metamap)
            self.add_component(MetaMapOverlayer, metamap)

        if not kwargs['word_embeddings']:
            raise ValueError('This pipeline requires word embeddings.')

        self.word_embeddings = kwargs['word_embeddings']
        self.cuda_device = kwargs['cuda_device']

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
