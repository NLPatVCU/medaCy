import os
import shutil
import tempfile
import unittest

import pkg_resources

from medacy.data.dataset import Dataset
from medacy.model import Model
from medacy.pipelines.lstm_systematic_review_pipeline import LstmSystematicReviewPipeline
from medacy.tests.sample_data import test_dir
from medacy.tests.pipeline_components.learners import use_cuda, cuda_device


class TestBiLstmCrf(unittest.TestCase):
    """Tests for the BiLSTM+CRF"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'))
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.prediction_directory = tempfile.mkdtemp()  # Directory to store predictions

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    @unittest.skipUnless(use_cuda, "This test only runs if a cuda device is set in the medaCy config file")
    def test_prediction_with_testing_pipeline(self):
        """Tests that a model created with the BiLSTM+CRF can be fitted and used to predict"""
        pipeline = LstmSystematicReviewPipeline(
            entities=self.entities,
            word_embeddings=os.path.join(test_dir, 'test_word_embeddings.txt'),
            cuda_device=cuda_device
        )

        model = Model(pipeline)
        model.fit(self.dataset)
        resulting_dataset = model.predict(self.dataset, prediction_directory=self.prediction_directory)
        self.assertIsInstance(resulting_dataset, Dataset)

if __name__ == '__main__':
    unittest.main()

