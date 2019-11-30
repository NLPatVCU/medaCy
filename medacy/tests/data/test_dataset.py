import importlib
import os
import shutil
import tempfile
from collections import Counter
from unittest import TestCase

import pkg_resources

from medacy.data.dataset import Dataset
from medacy.data.annotations import Annotations


class TestDataset(TestCase):
    """Unit tests for Dataset"""

    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing. See testing instructions for details.")
        cls.training_directory = tempfile.mkdtemp() #set up train directory
        cls.prediction_directory = tempfile.mkdtemp()  # set up predict directory
        training_dataset, _ = Dataset.load_external('medacy_dataset_end')
        cls.entities = list(training_dataset.get_labels())
        cls.ann_files = []

        #fill directory of training files
        for data_file in training_dataset:
            file_name, raw_text, ann_text = data_file.file_name, data_file.txt_path, data_file.ann_path
            cls.ann_files.append(file_name + '.ann')
            with open(os.path.join(cls.training_directory, "%s.txt" % file_name), 'w') as f:
                f.write(raw_text)
            with open(os.path.join(cls.training_directory, "%s.ann" % file_name), 'w') as f:
                f.write(ann_text)

            #place only text files into prediction directory.
            with open(os.path.join(cls.prediction_directory, "%s.txt" % file_name), 'w') as f:
                f.write(raw_text)

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.training_directory)
        shutil.rmtree(cls.prediction_directory)

    def test_init_training(self):
        """Tests initialization of DataManager"""
        dataset = Dataset(self.training_directory)
        self.assertIsInstance(dataset, Dataset)
        self.assertTrue(dataset.is_training_directory)

    def test_init_with_data_limit(self):
        """Tests initialization of DataManager"""
        dataset = Dataset(self.training_directory, data_limit=6)
        self.assertEqual(len(dataset), 6)

    def test_init_prediction(self):
        """Tests initialization of DataManager"""
        dataset = Dataset(self.prediction_directory)
        self.assertIsInstance(dataset, Dataset)
        self.assertFalse(dataset.is_training_directory)

    def test_generate_annotations(self):
        """Tests that generate_annotations() creates Annotations objects"""
        dataset = Dataset(self.prediction_directory)
        for ann in dataset.generate_annotations():
            self.assertIsInstance(ann, Annotations)

    def test_compute_counts(self):
        """Tests that compute_counts() returns a Counter containing counts for all labels"""
        dataset = Dataset(self.prediction_directory)
        counts = dataset.compute_counts()
        self.assertIsInstance(counts, Counter)
        for label in dataset.get_labels():
            self.assertIn(label, counts.keys())

