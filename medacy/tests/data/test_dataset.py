import importlib
import os
import pkg_resources
import shutil
import tempfile
from unittest import TestCase

from medacy.data.dataset import Dataset


class TestDatasetLocal(TestCase):
    """
    Tests working with a local datasets - imports an external dataset END and writes it locally
    Tests are run on the local versions.
    """

    @classmethod
    def setUpClass(cls):

        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing. See testing instructions for details.")
        cls.training_directory = tempfile.mkdtemp() #set up train directory
        cls.prediction_directory = tempfile.mkdtemp()  # set up predict directory
        training_dataset, _, meta_data = Dataset.load_external('medacy_dataset_end')
        cls.entities = meta_data['entities']
        cls.ann_files = []

        #fill directory of training files
        for data_file in training_dataset.get_data_files():
            file_name, raw_text, ann_text = (data_file.file_name, data_file.raw_path, data_file.ann_path)
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
        """
        Tests initialization of DataManager
        :return:
        """
        dataset = Dataset(self.training_directory)
        self.assertIsInstance( dataset , Dataset)
        self.assertTrue(dataset.is_training())

    def test_init_with_data_limit(self):
        """
        Tests initialization of DataManager
        :return:
        """
        dataset = Dataset(self.training_directory, data_limit=6)

        self.assertEqual(len(dataset.get_data_files()), 6)

    def test_init_prediction(self):
        """
        Tests initialization of DataManager
        :return:
        """
        dataset = Dataset(self.prediction_directory)

        self.assertIsInstance(dataset, Dataset)
        self.assertFalse(dataset.is_training())


class TestDatasetExternal(TestCase):
    """
    Tests working with of an external imported Dataset.
    The external dataset is loaded into a temporary cache directory by pkg_resources and
    a Dataset object passed along that directory.
    """
    @classmethod
    def setUpClass(cls):

        if importlib.util.find_spec('medacy_dataset_end') is None:
            raise ImportError("medacy_dataset_end was not automatically installed for testing. See testing instructions for details.")

        cls.dataset, _, cls.entities = Dataset.load_external('medacy_dataset_end')

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()

    def test_is_training(self):
        """Tests initialization of DataManager"""
        self.assertTrue(self.dataset.is_training())

    def test_file_count(self):
        """Tests the expected file count for our testing dataset"""
        self.assertEqual(len(self.dataset.get_data_files()), 41)

    def test_is_metamapped(self):
        """Verifies that the dataset is metamapped"""
        self.assertTrue(self.dataset.is_metamapped())

    def test_limit(self):
        """Tests limiting a file"""
        self.dataset.data_limit = 5
        self.assertEqual(len(self.dataset), 5)

    def test_compute_counts(self):
        self.assertIsInstance(self.dataset.compute_counts(), dict)

    def test_compute_confusion_matrix(self):
        self.dataset.data_limit = 3
        entities, confusion_matrix = self.dataset.compute_confusion_matrix(self.dataset)
        self.dataset.data_limit = 41
        self.assertIsInstance(confusion_matrix, list)

    def test_compute_ambiguity(self):
        self.dataset.data_limit = 3
        ambiguity = self.dataset.compute_ambiguity(self.dataset)
        self.dataset.data_limit = 41
        self.assertIsInstance(ambiguity, dict)
