import os
import shutil
import tempfile
from collections import Counter
from unittest import TestCase

import pkg_resources

from medacy.data.dataset import Dataset
from medacy.data.annotations import Annotations
from medacy.tests.sample_data import test_dir


class TestDataset(TestCase):
    """Unit tests for Dataset"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'))
        cls.prediction_directory = tempfile.mkdtemp()  # Set up predict directory
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.ann_files = []

        # Fill directory of prediction files (only the text files)
        for data_file in cls.dataset:
            new_file_path = os.path.join(cls.prediction_directory, data_file.file_name + '.txt')
            shutil.copyfile(data_file.txt_path, new_file_path)

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    def test_init_training(self):
        """Tests that the sample dataset is accurately identified as being for training"""
        self.assertTrue(self.dataset.is_training_directory)

    def test_init_with_data_limit(self):
        """Tests that initializing with a data limit works"""
        dataset = Dataset(self.dataset.data_directory, data_limit=1)
        self.assertEqual(len(dataset), 1)

    def test_init_prediction(self):
        """Tests that the copy of the sample dataset with only text files is identified as being for prediction"""
        dataset = Dataset(self.prediction_directory)
        self.assertIsInstance(dataset, Dataset)
        self.assertFalse(dataset.is_training_directory)

    def test_generate_annotations(self):
        """Tests that generate_annotations() creates Annotations objects"""
        for ann in self.dataset.generate_annotations():
            self.assertIsInstance(ann, Annotations)

    def test_get_labels(self):
        """Tests that get_labels returns a set of the correct labels"""
        expected = {
            'DoseFrequency', 'SampleSize', 'TimeUnits', 'Vehicle', 'TestArticlePurity', 'Endpoint', 'TestArticle',
            'GroupName', 'DoseDurationUnits', 'GroupSize', 'TimeAtFirstDose', 'Dose', 'DoseDuration', 'Species',
            'DoseUnits', 'Sex', 'EndpointUnitOfMeasure', 'TimeEndpointAssessed', 'DoseRoute', 'CellLine', 'Strain'
        }
        actual = self.dataset.get_labels()
        self.assertSetEqual(actual, expected)

    def test_compute_counts(self):
        """Tests that compute_counts() returns a Counter containing counts for all labels"""
        counts = self.dataset.compute_counts()
        self.assertIsInstance(counts, Counter)
        for label in self.dataset.get_labels():
            self.assertIn(label, counts.keys())

    def test_getitem(self):
        """Tests that some_dataset['filename'] returns an Annotations for 'filename.ann', or raises FileNotFoundError"""
        some_file_name = self.dataset.all_data_files[0].file_name
        result = self.dataset[some_file_name]
        self.assertIsInstance(result, Annotations)

        with self.assertRaises(FileNotFoundError):
            ann = self.dataset['notafilepath']


