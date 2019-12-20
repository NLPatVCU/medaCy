import os
import shutil
import tempfile
from unittest import TestCase

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model.spacy_model import SpacyModel
from medacy.tests.sample_data import test_dir


class TestSpacyModel(TestCase):
    """Tests for medacy.model.spacy_model.SpacyModel"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'), data_limit=1)
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.prediction_directory = tempfile.mkdtemp() #directory to store predictions

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    def test_predict(self):
        """Test predict and fit functions."""
        model = SpacyModel()
        model.fit(dataset=self.dataset, iterations=1)

        model.predict(self.dataset, prediction_directory=self.prediction_directory)

        second_ann_file = "%s.ann" % self.dataset.all_data_files[0].file_name
        annotations = Annotations(os.path.join(self.prediction_directory, second_ann_file))
        self.assertIsInstance(annotations, Annotations)
