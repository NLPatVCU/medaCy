import unittest

import spacy
from spacy.tokens import Doc

from medacy.pipelines.testing_pipeline import TestingPipeline
from medacy.tests.sample_data import sample_dataset


class TestPipeline(unittest.TestCase):
    """Unit tests for medacy.pipelines.testing_pipeline.TestingPipeline"""

    def test_pipeline(self):
        # Get the entities, the pipeline, and an extra spaCy pipeline
        entities = sample_dataset.get_labels(as_list=True)
        pipeline = TestingPipeline(entities)

        if "gold_annotator" in pipeline.spacy_pipeline.pipe_names:
            pipeline.spacy_pipeline.remove_pipe("gold_annotator")

        spacy_pipeline = spacy.load('en_core_web_sm')

        # Create a sample Doc
        sample_doc_path = sample_dataset.data_files[0].txt_path
        with open(sample_doc_path) as f:
            text = f.read()

        # The doc must be passed through a separate spaCy pipeline first
        doc = spacy_pipeline(text)
        doc = pipeline(doc)
        self.assertIsInstance(doc, Doc)

        # Check that the first token has all the expected spaCy features
        spacy_features = pipeline.get_feature_extractor().spacy_features
        token = doc[0]
        for feature in spacy_features:
            result = hasattr(token, feature)
            self.assertTrue(result)


if __name__ == '__main__':
    unittest.main()
