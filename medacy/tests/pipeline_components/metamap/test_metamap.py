from unittest import TestCase
import os.path
from medacy.pipeline_components import MetaMap

class TestMetaMap(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.metamap = MetaMap(metamap_path="/home/share/programs/metamap/2016/public_mm/bin/metamap")



    def test_init(self):
        metamap = self.metamap
        self.assertTrue(os.path.isfile(metamap.metamap_path))