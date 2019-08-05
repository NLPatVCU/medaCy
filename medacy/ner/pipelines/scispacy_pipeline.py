import spacy, sklearn_crfsuite
from .base import BasePipeline
from spacy.tokenizer import Tokenizer
from medacy.pipeline_components.feature_extraction.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components import GoldAnnotatorComponent, MetaMapComponent, MetaMap


class ScispacyPipeline(BasePipeline):
    """
    A pipeline for named entity recognition using ScispaCy, see https://allenai.github.io/scispacy/

    This pipeline differs from the ClinicalPipeline in that it uses AllenAI's 'en_core_sci_md' model and
    the tokenizer is simply spaCy's tokenizer.

    Requirements:
    scispacy
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.0/en_core_sci_md-0.2.0.tar.gz
    """

    def __init__(self, metamap=None, entities=[], cuda_device=-1):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap if metamap should be used, defaults to None.
        """
        super().__init__("scispacy_pipeline",
                         spacy_pipeline=spacy.load("en_core_sci_md"),
                         description=self.__doc__,
                         creators="Steele W. Farnsworth",  # append if multiple creators
                         organization="NLP@VCU"
                         )

        self.entities = entities
        self.spacy_pipeline.tokenizer = Tokenizer(self.spacy_pipeline.vocab)
        self.add_component(GoldAnnotatorComponent, entities) #add overlay for GoldAnnotation

        if metamap is not None and isinstance(metamap, MetaMap):
            self.add_component(MetaMapComponent, metamap)

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
        return self.spacy_pipeline.tokenizer

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=3, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
