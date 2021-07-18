import os
import shutil
import tempfile
import unittest
from warnings import warn

import pkg_resources

from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.bert_pipeline import BertPipeline
from medacy.tests.sample_data import test_dir
from medacy.tests.pipeline_components.learners import use_cuda, cuda_device


class TestBert(unittest.TestCase):
    """
    Tests for medacy.pipeline_components.learners.bert_learner.BertLearner
    and, by extension, medacy.pipelines.bert_pipeline.BertPipeline
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'), data_limit=1)
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.prediction_directory = tempfile.mkdtemp()  # Directory to store predictions
        cls.batch_size = 3

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    def _bert_test(self, pipe):
        model = Model(pipe)
        model.cross_validate(self.dataset, 2)
        model.fit(self.dataset)
        resulting_dataset = model.predict(self.dataset, prediction_directory=self.prediction_directory)
        self.assertIsInstance(resulting_dataset, Dataset)
        # Test that there is at least one prediction
        if not any(resulting_dataset.generate_annotations()):
            warn("The model did not generate any predictions")

    @unittest.skipUnless(use_cuda, "This test only runs if a cuda device is set in the medaCy config file")
    def test_cross_validate_fit_predict_no_crf(self):
        """Tests that a model created with BERT can be fitted and used to predict, with and without the CRF layer"""
        self._bert_test(
                BertPipeline(
                    entities=self.entities,
                    pretrained_model='bert-base-cased',
                    batch_size=self.batch_size,
                    cuda_device=cuda_device
                )
            )

    @unittest.skipUnless(use_cuda, "This test only runs if a cuda device is set in the medaCy config file")
    def test_cross_validate_fit_predict_with_crf(self):
        """Tests that a model created with BERT can be fitted and used to predict, with and without the CRF layer"""
        self._bert_test(
            BertPipeline(
                entities=self.entities,
                pretrained_model='bert-base-cased',
                batch_size=self.batch_size,
                cuda_device=cuda_device,
                using_crf=True
            )
        )


if __name__ == '__main__':
    unittest.main()
