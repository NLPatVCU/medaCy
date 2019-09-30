from unittest import TestCase

import spacy
from spacy.tokens import Token

from medacy.pipeline_components.feature_overlayers.lexicon_component import LexiconComponent


class TestLexicon(TestCase):


    @classmethod
    def setUpClass(cls):
        cls.lexicon = {'ADE': ['nausea', 'chest pain'], 'DRUG': ['Tylenol', 'Alleve']}
        cls.nlp = spacy.load('en_core_web_sm')
        cls.doc = cls.nlp('I took Tylenol and it gave me nausea and chest pain')

    @classmethod
    def tearDownClass(cls):
        pass

    def test_init(self):
        """
        Tests initialization with lexicon passed in
        :return:
        """
        lexicon_component = LexiconComponent(self.nlp, self.lexicon)
        self.assertIsInstance(lexicon_component, LexiconComponent)
        self.assertIsNotNone(lexicon_component.lexicon)

    def test_call_lexicon_component(self):
        """
        Test running a doc through the lexicon component and properly overlaying features from
        the lexicon.
        :return:
        """
        lexicon_component = LexiconComponent(self.nlp, self.lexicon)
        self.assertIs(Token.has_extension('feature_is_ADE_from_lexicon'), False)
        self.assertIs(Token.has_extension('feature_is_DRUG_from_lexicon'), False)
        doc = lexicon_component(self.doc)
        self.assertIs(Token.has_extension('feature_is_ADE_from_lexicon'), True)
        self.assertIs(Token.has_extension('feature_is_DRUG_from_lexicon'), True)
