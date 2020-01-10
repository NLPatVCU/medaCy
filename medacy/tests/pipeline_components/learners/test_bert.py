import os
import shutil
import tempfile
import unittest

import pkg_resources

from medacy.data.dataset import Dataset
from medacy.model import Model
from medacy.pipelines.bert_pipeline import BertPipeline
from medacy.tests.sample_data import test_dir


class TestBert(unittest.TestCase):
    """
    Tests for medacy.pipeline_components.learners.bert_learner.BertLearner
    and, by extension, medacy.pipelines.bert_pipeline.BertPipeline
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'))
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.prediction_directory = tempfile.mkdtemp()  # Directory to store predictions

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    def test_prediction_with_testing_pipeline(self):
        """Tests that a model created with the BiLSTM+CRF can be fitted and used to predict"""
        pipeline = BertPipeline(
            entities=self.entities,
            cuda_device=-1
        )

        pipeline_crf = BertPipeline(
            entities=self.entities,
            cuda_device=-1,
            using_crf=True
        )

        for pipe in [pipeline, pipeline_crf]:
            model = Model(pipe)
            model.fit(self.dataset)
            resulting_dataset = model.predict(self.dataset, prediction_directory=self.prediction_directory)
            self.assertIsInstance(resulting_dataset, Dataset)


if __name__ == '__main__':
    unittest.main()