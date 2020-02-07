from unittest import TestCase

import re
import string

import torch
import unicodedata
from gensim.models import KeyedVectors

from medacy.nn.vectorizer import Vectorizer

class TestVectorizer(TestCase):
    @classmethod
    def setUpClass(cls, device=0):
        cls.device = device
        cls.word_vectors = None  # TODO: re-examine
        cls.untrained_tokens = set()
        cls.other_features = {}  # TODO: re-examine
        cls.window_size = 0  # TODO: re-examine
        cls.tag_to_index = {}  # TODO: re-examine

        cls.character_to_index = {
            character: index for index, character in enumerate(string.printable, 1)  # TODO: re-examine
        }

    @classmethod
    def tearDownClass(cls):
        pass  # TODO: Figure out if anything needs to be torn down

    def test_init(self):
        """
        Tests initialization with sample parameters.
        :return:
        """
        pass  # TODO: Write the tests described (should be easy)

    def test_vectorize_tokens(self):
        """
        Tests that a list of tokens is vectorized correctly, and that unknown tokens are handled as expected.
        """
        pass  # TODO: Write the tests described
