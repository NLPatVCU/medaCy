from unittest import TestCase

from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipelines import ClinicalPipeline


class TestClinicalPipeline(TestCase):

    @classmethod
    def setUpClass(cls):
        metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap",
                          cache_output=False)
        cls.pipeline = ClinicalPipeline(metamap) # Will fail as MetaMap isn't installed

    def test_init(self):
        pipeline = self.pipeline

    def test_contains_gold(self):
        pipeline = self.pipeline
        with self.assertRaises(AssertionError):
            pipeline.add_component(GoldAnnotatorComponent)
