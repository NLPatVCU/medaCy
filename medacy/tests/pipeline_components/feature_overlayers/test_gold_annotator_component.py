import unittest

import spacy
from spacy.tokens import Doc

from medacy.data.annotations import Annotations
from medacy.pipeline_components.feature_overlayers.gold_annotator_component import GoldAnnotatorComponent


class TestGoldAnnotatorComponent(unittest.TestCase):
    """Unit tests for medacy.pipeline_components.feature_overlayers.gold_annotator_component.GoldAnnotatorComponent"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.nlp = spacy.load('en_core_web_sm')

    def test_overlays_annotations(self):
        """
        Tests that this pipeline component adds the correct labels.
        Note that this only tests that at least one instance of each label is overlayed because the number of tokens
        that receive the label varies based on the tokenizer.
        """

        txt_file_path = 'TODO'
        ann_file_path = 'TODO'

        with open(txt_file_path) as f:
            text = f.read()
        doc: Doc = self.nlp(text)

        doc.set_extension('file_name')
        doc._.file_name = txt_file_path
        doc.set_extension('gold_annotation_file')
        doc._.gold_annotation_file = ann_file_path

        ann = Annotations(ann_file_path)
        labels = ann.get_labels()

        gold_annotator = GoldAnnotatorComponent(self.nlp, list(labels))

        doc = gold_annotator(doc)

        overlayed_labels = {t._.gold_label for t in doc}
        overlayed_labels.remove('O')

        self.assertSetEqual(overlayed_labels, labels)