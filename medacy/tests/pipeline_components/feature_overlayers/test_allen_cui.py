import spacy
from unittest import TestCase

from medacy.pipeline_components.feature_overlayers.allen_cui_component import AllenCUIComponent
from medacy.pipeline_components.tokenizers.clinical_tokenizer import ClinicalTokenizer

class TestAllenCUI(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.eng = spacy.load("en_core_web_sm")
        cls.tokenizer = ClinicalTokenizer(cls.eng).tokenizer
        cls.doc = "I took 5mg of tylenol and passed out. Next, I woke up, took 5000mg more tylenol and died."

    @classmethod
    def tearDownClass(cls):
        pass

    def testInit(self):
        allen_cui_component = AllenCUIComponent(self.eng)
        self.assertIsInstance(allen_cui_component, AllenCUIComponent)

    def testTokenizer(self):
        allen_cui_component = AllenCUIComponent(self.eng)
        tokenized_doc_1 = one.sci(self.doc)
        tokenized_doc_2 = self.eng(self.doc)

        tokens_1 = [str(token) for token in tokenized_doc_1]
        tokens_2 = [str(token) for token in tokenized_doc_2]

        self.assertListEqual(tokens1, tokens2)

if __name__ == "__main__":
    TestAllenCUI.main()