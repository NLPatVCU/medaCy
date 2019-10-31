import importlib
import os
import shutil
import tempfile
from unittest import TestCase

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.testing_pipeline import TestingPipeline


class TestModelTrainingAndPrediction(TestCase):
    """
    Tests model training and prediction in bulk
    """

    @classmethod
    def setUpClass(cls):

        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing. See testing instructions for details.")

        cls.train_dataset, _ = Dataset.load_external('medacy_dataset_end')
        cls.entities = list(cls.train_dataset.get_labels())
        cls.train_dataset.data_limit = 1

        cls.test_dataset, _ = Dataset.load_external('medacy_dataset_end')
        cls.test_dataset.data_limit = 2

        cls.prediction_directory = tempfile.mkdtemp() #directory to store predictions

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    def test_prediction_with_testing_pipeline(self):
        """
        Constructs a model that memorizes an entity, predicts it on same file, writes to ann
        :return:
        """

        pipeline = TestingPipeline(entities=['tradename'])

        #train on Abelcet.ann
        model = Model(pipeline, n_jobs=1)
        model.fit(self.train_dataset)

        #predict on both
        model.predict(self.test_dataset, prediction_directory=self.prediction_directory)

        second_ann_file = "%s.ann" % self.test_dataset.get_data_files()[1].file_name
        annotations = Annotations(os.path.join(self.prediction_directory, second_ann_file))
        self.assertIsInstance(annotations, Annotations)
