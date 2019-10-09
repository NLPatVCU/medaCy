import sklearn_crfsuite
import spacy

from medacy.pipeline_components.feature_extracters.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapComponent
from medacy.pipeline_components.tokenizers.systematic_review_tokenizer import SystematicReviewTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline
from medacy.tools.get_metamap import get_metamap


class SystematicReviewPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. This pipeline was designed over-top of the TAC 2018 SRIE track
    challenge.
    """

    def __init__(self, entities, cuda_device=-1):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap
        """

        super().__init__("systematic_review_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=self.__doc__,
                         creators="Andriy Mulyar (andriymulyar.com)", #append if multiple creators
                         organization="NLP@VCU")

        self.entities = entities
        self.spacy_pipeline.tokenizer = self.get_tokenizer()  # set tokenizer
        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

        metamap_path = get_metamap()
        metamap = MetaMap(metamap_path)
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
        tokenizer = SystematicReviewTokenizer(self.spacy_pipeline)
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        return FeatureExtractor(window_size=10, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'text'])
