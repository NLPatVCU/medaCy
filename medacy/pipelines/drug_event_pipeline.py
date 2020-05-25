import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.lexicon_component import LexiconOverlayer
from medacy.pipeline_components.feature_overlayers.metamap.metamap_all_types_component import MetaMapAllTypesOverlayer
from medacy.pipeline_components.feature_overlayers.table_matcher_component import TableMatcherOverlayer
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipeline_components.tokenizers.character_tokenizer import CharacterTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class DrugEventPipeline(BasePipeline):
    """
    Pipeline for recognition of adverse drug events from the 2018/19 FDA OSE drug label challenge

    Created by Corey Sutphin of NLP@VCU
    """

    def __init__(self, entities, metamap=None, lexicon={}, **kwargs):
        """
        Init a pipeline for processing data related to identifying adverse drug events
        :param entities: a list of entities
        :param metamap: instance of MetaMap
        :param entities: entities to be identified, for this pipeline adverse drug events
        :param lexicon: Dictionary with labels and their corresponding lexicons to match on
        """
        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"), **kwargs)

        if metamap:
            self.add_component(MetaMapAllTypesOverlayer, metamap)

        if lexicon is not None:
            self.add_component(LexiconOverlayer, lexicon)

        self.add_component(TableMatcherOverlayer)

    def get_learner(self):
        return "CRF_l2sgd", get_crf()

    def get_tokenizer(self):
        return CharacterTokenizer(self.spacy_pipeline)

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num', 'text', 'head'])
