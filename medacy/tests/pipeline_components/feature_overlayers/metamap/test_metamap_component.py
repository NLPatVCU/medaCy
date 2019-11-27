import re
import unittest

import spacy
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.metamap.metamap_component import MetaMapComponent
from medacy.pipeline_components.feature_overlayers.metamap.metamap import MetaMap
from medacy.tools.get_metamap import get_metamap_path

# See if MetaMap has been set for this installation
metamap_path = get_metamap_path()
have_metamap = metamap_path != 0

# Specify why MetaMap tests may be skipped
reason = "This test can only be performed if the path to MetaMap has been configured for this installation"


class TestMetaMapComponent(unittest.TestCase):
    """Unit tests for medacy.pipeline_components.feature_overlayers.metamap.metamap_component.MetaMapComponent"""

    @unittest.skipUnless(have_metamap, reason)
    def test_overlays_cuis(self):
        """Tests that the MetaMapComponent overlays CUIs correctly"""
        nlp = spacy.load('en_core_web_sm')
        doc = nlp('I took Tylenol and it gave me nausea and chest pain')

        self.assertFalse(Token.has_extension('feature_cui'))

        metamap = MetaMap(metamap_path)
        metamap_component = MetaMapComponent(nlp, metamap)

        metamap_component(doc)
        self.assertTrue(Token.has_extension('feature_cui'))
        cuis = [token._.feature_cui for token in doc]

        # Test that at least one of the features are a CUI
        any_match = any(re.match(r'C\d+', c) for c in cuis)
        self.assertTrue(any_match)

        # Test that all features are a CUI or '-1'
        all_match = all(re.match(r'(C\d+)|(-1)', c) for c in cuis)
        self.assertTrue(all_match)
