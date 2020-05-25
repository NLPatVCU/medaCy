import re
import unittest

import spacy
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapOverlayer
from medacy.tests.pipeline_components.feature_overlayers.metamap import have_metamap, reason, metamap_path
from medacy.tests.sample_data import sample_dataset

class TestMetaMapComponent(unittest.TestCase):
    """Unit tests for medacy.pipeline_components.feature_overlayers.metamap.metamap_component.MetaMapOverlayer"""

    @classmethod
    def setUpClass(cls) -> None:
        """Instantiates MetaMap and activates it"""
        if not have_metamap:
            return
        cls.metamap = MetaMap(metamap_path)
        cls.metamap.activate()

        cls.nlp = spacy.load('en_core_web_sm')

    @classmethod
    def tearDownClass(cls) -> None:
        """Deactivates MetaMap"""
        if not have_metamap:
            return
        cls.metamap.deactivate()

    @unittest.skipUnless(have_metamap, reason)
    def test_overlays_cuis(self):
        """Tests that the MetaMapOverlayer overlays CUIs correctly given a document that hasn't been metamapped"""
        sample_doc = sample_dataset.data_files[0].txt_path
        with open(sample_doc) as f:
            sample_text = f.read()

        doc = self.nlp(sample_text)

        metamap = MetaMap(metamap_path)
        metamap_component = MetaMapOverlayer(self.nlp, metamap)

        metamap_component(doc)
        self.assertTrue(Token.has_extension('feature_cui'))
        cuis = [token._.feature_cui for token in doc]

        # Test that at least one of the features are a CUI
        any_match = any(re.match(r'C\d+', c) for c in cuis)
        self.assertTrue(any_match)

        # Test that all features are a CUI or '-1'
        all_match = all(re.match(r'(C\d+)|(-1)', c) for c in cuis)
        self.assertTrue(all_match)


if __name__ == '__main__':
    unittest.main()

