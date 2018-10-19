from unittest import TestCase
from medacy.learn import FeatureExtractor

#TODO write tests for feature extractor once class is written
class TestFeatureExtractor(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.feature_extractor = FeatureExtractor()



    def test_init(self):
        extractor = self.feature_extractor
        self.assertTrue(False);