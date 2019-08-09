import spacy, sklearn_crfsuite
from .base import BasePipeline
from medacy.pipeline_components.feature_extraction.discrete_feature_extractor import FeatureExtractor

from medacy.pipeline_components import GoldAnnotatorComponent, MetaMapComponent, CharacterTokenizer
from medacy.pipeline_components.lexicon import LexiconComponent
from medacy.pipeline_components.patterns import TableMatcherComponent

class DrugEventPipeline(BasePipeline):

    def __init__(self, metamap=None, entities=[], lexicon={}, cuda_device=-1):
        """
        Init a pipeline for processing data related to identifying adverse drug events
        :param metamap: instance of MetaMap
        :param entities: entities to be identified, for this pipeline adverse drug events
        :param lexicon: Dictionary with labels and their corresponding lexicons to match on
        """

        description= "Pipeline for recognition of adverse drug events from the 2018/19 FDA OSE drug label challenge"
        super().__init__("drug_event_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Corey Sutphin",
                         organization="NLP@VCU")
        self.entities = entities

        #self.spacy_pipeline.tokenizer = self.get_tokenizer()  # Currently using SpaCy's default tokenizer

        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation
        if metamap is not None:
            #self.add_component(MetaMapComponent, metamap, semantic_type_labels=['sosy', 'phpr', 'orga', 'npop', 'mobd', 'inpo', 'comd', 'biof', 'bdsu', 'acab'])
            self.add_component(MetaMapComponent, metamap, semantic_type_labels=['aapp',
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
'vtbt',])
        if lexicon is not None:
            self.add_component(LexiconComponent, lexicon)
        self.add_component(TableMatcherComponent)

    def get_learner(self):
        return ("CRF_l2sgd", sklearn_crfsuite.CRF(
            algorithm='l2sgd',
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        ))

    def get_tokenizer(self):
        tokenizer = CharacterTokenizer(self.spacy_pipeline)
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        extractor = FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num', 'text', 'head'])
        return extractor