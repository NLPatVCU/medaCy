import json

import spacy
import sklearn_crfsuite

from medacy.pipeline_components.feature_extracters.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapComponent
from medacy.pipeline_components.learners.bilstm_crf_learner import BiLstmCrfLearner
from medacy.pipeline_components.tokenizers.character_tokenizer import CharacterTokenizer
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline

required_keys = [
    'learner',
    'entities',
    'spacy_pipeline',
    'spacy_features',
    'window_size',
]


def json_to_pipeline(json_path):
    """
    Constructs a custom pipeline from a json file

    The json must have the following keys:

    'learner': 'CRF' or 'BiLSTM'
        if 'learner' is 'BiLSTM', two additional keys are required:
            'cuda_device': the GPU to use (or -1 for the CPU)
            'word_embeddings': path to the word embeddings file
    'entities': a list of strings
    'spacy_pipeline': the spaCy model to use
    'spacy_features': a list of features that exist as spaCy token annotations
    'window_size': the number of words +/- the target word whose features should be used along with the target word

    The following keys are optional:
    'metamap': the path to the MetaMap binary; MetaMap will only be used if this key is present
    'tokenizer': 'clinical', 'systematic_review', or 'character'; defaults to the spaCy model's tokenizer

    :param json_path: the path to the json file
    :return: a custom pipeline class
    """

    with open(json_path, 'rb') as f:
        input_json = json.load(f)

    missing_keys = [key for key in required_keys if key not in input_json.keys()]
    if missing_keys:
        raise ValueError(f"Required key(s) '{missing_keys}' was/were not found in the json file.")

    class CustomPipeline(BasePipeline):
        def __init__(self):
            super().__init__(
                "custom_pipeline",
                spacy_pipeline=spacy.load(input_json['spacy_pipeline'])
            )

            self.entities = input_json['entities']

            self.spacy_pipeline.tokenizer = self.get_tokenizer()

            self.add_component(GoldAnnotatorComponent, self.entities)

            if 'metamap' in input_json.keys():
                metamap = MetaMap(input_json['metamap'])
                self.add_component(MetaMapComponent, metamap)

        def get_tokenizer(self):
            if 'tokenizer' not in input_json.keys():
                return self.spacy_pipeline.tokenizer

            selection = input_json['tokenizer']
            options = {
                'clinical': ClinicalTokenizer,
                'systematic_review': SystematicReviewTokenizer,
                'character': CharacterTokenizer
            }

            if selection not in options:
                raise ValueError(f"Tokenizer selection '{selection}' not an option")

            Tokenizer = options[selection]
            return Tokenizer(self.spacy_pipeline).tokenizer

        def get_learner(self):
            learner_selection = input_json['learner']

            if learner_selection == 'CRF':
                return ("CRF_l2sgd",
                        sklearn_crfsuite.CRF(
                            algorithm='l2sgd',
                            c2=0.1,
                            max_iterations=100,
                            all_possible_transitions=True
                            )
                        )

            if learner_selection == 'BiLSTM':
                for k in ['word_embeddings', 'cuda_device']:
                    if k not in input_json.keys():
                        raise ValueError(f"'{k}' must be specified when the learner is BiLSTM")
                return 'BiLSTM+CRF', BiLstmCrfLearner(input_json['word_embeddings'], input_json['cuda_device'])
            else:
                raise ValueError(f"'learner' must be 'CRF' or 'BiLSTM")

        def get_feature_extractor(self):
            return FeatureExtractor(
                window_size=input_json['window_size'],
                spacy_features=input_json['spacy_features']
            )

    return CustomPipeline
