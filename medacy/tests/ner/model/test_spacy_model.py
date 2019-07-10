"""Test file for the spaCy model class.
"""
import os
import importlib
import tempfile
import shutil
from unittest import TestCase
import pkg_resources
from medacy.ner import SpacyModel
from medacy.data import Dataset
from medacy.tools import Annotations


class TestSpacyModel(TestCase):
    """
    Tests model training and prediction in bulk
    """

    @classmethod
    def setUpClass(cls):

        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing.\
                See testing instructions for details.")

        cls.train_dataset, _, meta_data = Dataset.load_external('medacy_dataset_end')
        cls.entities = meta_data['entities']
        cls.train_dataset.set_data_limit(1)

        cls.test_dataset, _, _ = Dataset.load_external('medacy_dataset_end')
        cls.test_dataset.set_data_limit(2)

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
            spacy_model_name='en_core_web_sm',
            iterations=2
        )

        model.predict(self.test_dataset, prediction_directory=self.prediction_directory)

        second_ann_file = "%s.ann" % self.test_dataset.get_data_files()[1].file_name
        annotations = Annotations(
            os.path.join(self.prediction_directory, second_ann_file),
            annotation_type='ann'
        )
        self.assertIsInstance(annotations, Annotations)
