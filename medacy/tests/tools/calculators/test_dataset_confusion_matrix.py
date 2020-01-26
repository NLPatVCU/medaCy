import unittest

from medacy.data.dataset import Dataset
from medacy.tests.sample_data import sample_dataset_1
from medacy.tests.sample_data import sample_dataset_1_predictions
from medacy.tools.calculators.dataset_confusion_matrix import *

class TestDatasetConfusionMatrix(unittest.TestCase):

    def setUp():
        pass

    def test_calculate_dataset_confusion_matrix(self): #TODO: Revision to enforce data_limit=1
        test_ents, test_mat = dataset_confusion_matrix.calculate_dataset_confusion_matrix\
        (groundtruth_dataset.data_directory, predictions_dataset.data_directory, dl=1)
        pass
        
    def test_format_dataset_confusion_matrix(self):
        pass

    def test_format_headers(self):
        headers = ["One", "Two", "Three"]
        expected_headers = ["O\nn\ne", "T\nw\no", "T\nh\nr\ne\ne"]
        formatted_headers = dataset_confusion_matrix.format_headers(headers)
        self.assertEqual(formatted_headers, expected_headers)
    
    def tearDown():
        pass


if __name__ == '__main__':
    unittest.main()
