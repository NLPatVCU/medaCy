import json
import os

import spacy

from medacy.pipeline_components.feature_extractors.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_extractors.text_extractor import TextExtractor
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_all_types_component import MetaMapAllTypesOverlayer
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.pipeline_components.learners.bert_learner import BertLearner
from medacy.pipeline_components.learners.bilstm_crf_learner import BiLstmCrfLearner
from medacy.pipeline_components.learners.crf_learner import get_crf
from medacy.pipeline_components.tokenizers.character_tokenizer import CharacterTokenizer
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


required_keys = [
    'learner',
    'spacy_pipeline',
]


def json_to_pipeline(json_path):
    """
    Constructs a custom pipeline from a json file

    The json must have the following keys:

    'learner': 'CRF', 'BiLSTM', or 'BERT'
    'spacy_pipeline': the spaCy model to use

    The following keys are optional:
    'spacy_features': a list of features that exist as spaCy token annotations
    'window_size': the number of words +/- the target word whose features should be used along with the target word; defaults to 0
    'tokenizer': 'clinical', 'systematic_review', or 'character'; defaults to the spaCy model's tokenizer
    'metamap': the path to the MetaMap binary; MetaMap will only be used if this key is present
        if 'metamap' is a key, 'semantic_types' must also be a key, with value 'all', 'none', or
        a list of semantic type strings

    :param json_path: the path to the json file, or a dict of what that json would be
    :return: a custom pipeline class
    """

    if isinstance(json_path, os.PathLike):
        with open(json_path, 'rb') as f:
            input_json = json.load(f)
    elif isinstance(json_path, dict):
        input_json = json_path

    missing_keys = [key for key in required_keys if key not in input_json.keys()]
    if missing_keys:
        raise ValueError(f"Required key(s) '{missing_keys}' was/were not found in the json file.")

    class CustomPipeline(BasePipeline):
        """A custom pipeline configured from a JSON file"""

        def __init__(self, entities, **kwargs):
            super().__init__(entities, spacy_pipeline=spacy.load(input_json['spacy_pipeline']))

            if 'metamap' in input_json.keys():
                if 'semantic_types' not in input_json.keys():
                    raise ValueError("'semantic_types' must be a key when 'metamap' is a key.")

                metamap = MetaMap(input_json['metamap'])

                if input_json['semantic_types'] == 'all':
                    self.add_component(MetaMapAllTypesOverlayer, metamap)
                elif input_json['semantic_types'] == 'none':
                    self.add_component(MetaMapOverlayer, metamap, semantic_type_labels=[])
                elif isinstance(input_json['semantic_types'], list):
                    self.add_component(MetaMapOverlayer, metamap, semantic_type_labels=input_json['semantic_types'])
                else:
                    raise ValueError("'semantic_types' must be 'all', 'none', or a list of strings")

            # BERT values
            self.cuda_device = kwargs['cuda_device'] if 'cuda_device' in kwargs else -1
            self.batch_size = kwargs['batch_size'] if 'batch_size' in kwargs else 8
            self.learning_rate = kwargs['learning_rate'] if 'learning_rate' in kwargs else 1e-5
            self.epochs = kwargs['epochs'] if 'epochs' in kwargs else 3
            self.pretrained_model = kwargs['pretrained_model'] if 'pretrained_model' in kwargs else 'bert-large-cased'
            self.using_crf = kwargs['using_crf'] if 'using_crf' in kwargs else False

            # BiLSTM value
            if input_json['learner'] == 'BiLSTM':
                if 'word_embeddings' not in kwargs:
                    raise ValueError("BiLSTM learner requires word embeddings; use the parameter '--word_embeddings' "
                                     "to specify an embedding path")
            self.word_embeddings = kwargs['word_embeddings']

        def get_tokenizer(self):
            if 'tokenizer' not in input_json.keys():
                return None

            selection = input_json['tokenizer']
            options = {
                'clinical': ClinicalTokenizer,
                'systematic_review': SystematicReviewTokenizer,
                'character': CharacterTokenizer
            }

            if selection not in options:
                raise ValueError(f"Tokenizer selection '{selection}' not an option")

            Tokenizer = options[selection]
            return Tokenizer(self.spacy_pipeline)

        def get_learner(self):
            learner_selection = input_json['learner']

            if learner_selection == 'CRF':
                return "CRF_l2sgd", get_crf()
            if learner_selection == 'BiLSTM':
                return 'BiLSTM+CRF', BiLstmCrfLearner(self.word_embeddings, self.cuda_device)
            if learner_selection == 'BERT':
                learner = BertLearner(
                    self.cuda_device,
                    pretrained_model=self.pretrained_model,
                    batch_size=self.batch_size,
                    learning_rate=self.learning_rate,
                    epochs=self.epochs,
                    using_crf=self.using_crf
                )
                return 'BERT', learner
            else:
                raise ValueError(f"'learner' must be 'CRF', 'BiLSTM', or 'BERT', but is {learner_selection}")

        def get_feature_extractor(self):
            if input_json['learner'] == 'BERT':
                return TextExtractor()

            return FeatureExtractor(
                window_size=input_json['window_size'] if 'window_size' in input_json else 0,
                spacy_features=input_json['spacy_features'] if 'spacy_features' in input_json else ['text']
            )

        def get_report(self):
            report = super().get_report() + '\n'
            report += f"Pipeline configured from a JSON: {json.dumps(input_json)}\nJSON path: {json_path}"
            return report

    return CustomPipeline
