from unittest import TestCase
from medacy.pipelines import ClinicalPipeline
from medacy.pipeline_components import GoldAnnotatorComponent


class TestClinicalPipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.pipeline = ClinicalPipeline() # Will fail as MetaMap isn't installed



    def test_init(self):
        pipeline = self.pipeline

    def test_add(self):
        pipeline = self.pipeline
        pipeline.add_component(GoldAnnotatorComponent)
        assert pipeline.spacy_pipeline
