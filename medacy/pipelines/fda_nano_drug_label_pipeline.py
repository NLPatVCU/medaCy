import sklearn_crfsuite
import spacy

from medacy.pipeline_components.feature_extracters.discrete_feature_extractor import FeatureExtractor
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapComponent
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer
from medacy.pipelines.base.base_pipeline import BasePipeline


class FDANanoDrugLabelPipeline(BasePipeline):
    """
    A pipeline for clinical named entity recognition. This pipeline was designed over-top of the TAC 2018 SRIE track
    challenge.
    """

    def __init__(self, metamap=None, entities=[], cuda_device=-1):
        """
        Create a pipeline with the name 'clinical_pipeline' utilizing
        by default spaCy's small english model.

        :param metamap: an instance of MetaMap
        """
        description="""Pipeline tuned for the recognition of entities in FDA Nanoparticle Drug Labels"""

        super().__init__("fda_nano_drug_label_pipeline",
                         spacy_pipeline=spacy.load("en_core_web_sm"),
                         description=description,
                         creators="Andriy Mulyar (andriymulyar.com)", #append if multiple creators
                         organization="NLP@VCU"
                         )


        self.entities = entities

        self.spacy_pipeline.tokenizer = self.get_tokenizer()  # set tokenizer

        self.add_component(GoldAnnotatorComponent, entities)  # add overlay for GoldAnnotation

        if metamap is not None and isinstance(metamap, MetaMap):
            self.add_component(MetaMapComponent, metamap)

        #self.add_component(UnitComponent)

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
        tokenizer = ClinicalTokenizer(self.spacy_pipeline) #Best run with SystematicReviewTokenizer
        return tokenizer.tokenizer

    def get_feature_extractor(self):
        extractor = FeatureExtractor(window_size=6, spacy_features=['pos_', 'shape_', 'prefix_', 'suffix_', 'like_num', 'text'])
        return extractor
