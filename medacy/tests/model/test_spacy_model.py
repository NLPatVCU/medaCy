"""Test file for the spaCy model class.
"""
import importlib
import os
import shutil
import tempfile
from unittest import TestCase

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model.spacy_model import SpacyModel


class TestSpacyModel(TestCase):
    """
    Tests model training and prediction in bulk
    """

    @classmethod
    def setUpClass(cls):

        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing.\
                See testing instructions for details.")

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

    def testPredict(self):
        """Test predict and fit functions."""
        model = SpacyModel()
        model.fit(
            dataset=self.train_dataset,
            iterations=1
        )

        model.predict(self.test_dataset, prediction_directory=self.prediction_directory)

        second_ann_file = "%s.ann" % self.test_dataset.all_data_files[1].file_name
        annotations = Annotations(
            os.path.join(self.prediction_directory, second_ann_file)
        )
        self.assertIsInstance(annotations, Annotations)
