from unittest import TestCase
import os.path
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap

#TODO Cannot test metamap due to issue with processes communication during unit tests - we could place pre-metamapped
#TODO files in the data directory and utilize those simply not testing the interfacing code. But these will have to be
#TODO replaced each time the data is updated (if it is).
# class TestMetaMap(TestCase):
#
#     @classmethod
#     def setUpClass(cls):
#         cls.metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")
#
#
#
#     def test_init(self):
#         metamap = self.metamap
#         self.assertTrue(os.path.isfile(metamap.metamap_path))