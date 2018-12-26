from unittest import TestCase
from medacy.data import Dataset
import shutil, tempfile, os

class TestDataset(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.training_directory = tempfile.mkdtemp() #set up train directory
        cls.prediction_directory = tempfile.mkdtemp()  # set up predict directory
        dataset, entities = Dataset.load_external('medacy_dataset_end')
        cls.entities = entities
        cls.ann_files = []

        #fill directory of training files
        for data_file in dataset.get_files():
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

    def test_init_with_limit(self):
        """
        Tests initialization of DataManager
        :return:
        """
        dataset = Dataset(self.training_directory, limit=6)

        self.assertEqual(len(dataset.get_files()), 6)

    def test_init_prediction(self):
        """
        Tests initialization of DataManager
        :return:
        """
        dataset = Dataset(self.prediction_directory)

        self.assertIsInstance(dataset, Dataset)
        self.assertFalse(dataset.is_training())




