import sklearn_crfsuite
import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorOverlayer
from medacy.pipeline_components.feature_overlayers.lexicon_component import LexiconOverlayer
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.feature_overlayers.table_matcher_component import TableMatcherOverlayer
from medacy.pipeline_components.tokenizers.character_tokenizer import CharacterTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class DrugEventPipeline(BasePipeline):
    """
    Pipeline for recognition of adverse drug events from the 2018/19 FDA OSE drug label challenge

    Created by Corey Sutphin of NLP@VCU
    """

    def __init__(self, entities, metamap=None, lexicon={}):
        """
        Init a pipeline for processing data related to identifying adverse drug events
        :param entities: a list of entities
        :param metamap: instance of MetaMap
        :param entities: entities to be identified, for this pipeline adverse drug events
        :param lexicon: Dictionary with labels and their corresponding lexicons to match on
        """
        super().__init__(entities, spacy_pipeline=spacy.load("en_core_web_sm"))

        if metamap is not None:
            self.add_component(MetaMapOverlayer, metamap, semantic_type_labels=[
                                                                        'aapp',
                                                                        'acab',
                                                                        'acty',
                                                                        'aggp',
                                                                        'amas',
                                                                        'amph',
                                                                        'anab',
                                                                        'anim',
                                                                        'anst',
                                                                        'antb',
                                                                        'arch',
                                                                        'bacs',
                                                                        'bact',
                                                                        'bdsu',
                                                                        'bdsy',
                                                                        'bhvr',
                                                                        'biof',
                                                                        'bird',
                                                                        'blor',
                                                                        'bmod',
                                                                        'bodm',
                                                                        'bpoc',
                                                                        'bsoj',
                                                                        'celc',
                                                                        'celf',
                                                                        'cell',
                                                                        'cgab',
                                                                        'chem',
                                                                        'chvf',
                                                                        'chvs',
                                                                        'clas',
                                                                        'clna',
                                                                        'clnd',
                                                                        'cnce',
                                                                        'comd',
                                                                        'crbs',
                                                                        'diap',
                                                                        'dora',
                                                                        'drdd',
                                                                        'dsyn',
                                                                        'edac',
                                                                        'eehu',
                                                                        'elii',
                                                                        'emod',
                                                                        'emst',
                                                                        'enty',
                                                                        'enzy',
                                                                        'euka',
                                                                        'evnt',
                                                                        'famg',
                                                                        'ffas',
                                                                        'fish',
                                                                        'fndg',
                                                                        'fngs',
                                                                        'food',
                                                                        'ftcn',
                                                                        'genf',
                                                                        'geoa',
                                                                        'gngm',
                                                                        'gora',
                                                                        'grpa',
                                                                        'grup',
                                                                        'hcpp',
                                                                        'hcro',
                                                                        'hlca',
                                                                        'hops',
                                                                        'horm',
                                                                        'humn',
                                                                        'idcn',
                                                                        'imft',
                                                                        'inbe',
                                                                        'inch',
                                                                        'inpo',
                                                                        'inpr',
                                                                        'irda',
                                                                        'lang',
                                                                        'lbpr',
                                                                        'lbtr',
                                                                        'mamm',
                                                                        'mbrt',
                                                                        'mcha',
                                                                        'medd',
                                                                        'menp',
                                                                        'mnob',
                                                                        'mobd',
                                                                        'moft',
                                                                        'mosq',
                                                                        'neop',
                                                                        'nnon',
                                                                        'npop',
                                                                        'nusq',
                                                                        'ocac',
                                                                        'ocdi',
                                                                        'orch',
                                                                        'orga',
                                                                        'orgf',
                                                                        'orgm',
                                                                        'orgt',
                                                                        'ortf',
                                                                        'patf',
                                                                        'phob',
                                                                        'phpr',
                                                                        'phsf',
                                                                        'phsu',
                                                                        'plnt',
                                                                        'podg',
                                                                        'popg',
                                                                        'prog',
                                                                        'pros',
                                                                        'qlco',
                                                                        'qnco',
                                                                        'rcpt',
                                                                        'rept',
                                                                        'resa',
                                                                        'resd',
                                                                        'rnlw',
                                                                        'sbst',
                                                                        'shro',
                                                                        'socb',
                                                                        'sosy',
                                                                        'spco',
                                                                        'tisu',
                                                                        'tmco',
                                                                        'topp',
                                                                        'virs',
                                                                        'vita',
                                                                        'vtbt'
                                                                        ]
                               )
        if lexicon is not None:
            self.add_component(LexiconOverlayer, lexicon)

        self.add_component(TableMatcherOverlayer)

    def get_learner(self):
        return ("CRF_l2sgd",
                sklearn_crfsuite.CRF(
                    algorithm='l2sgd',
                    c2=0.1,
                    max_iterations=100,
                    all_possible_transitions=True
                )
            )

    def get_tokenizer(self):
        tokenizer = CharacterTokenizer(self.spacy_pipeline)
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        extractor = FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num', 'text', 'head'])
        return extractor
