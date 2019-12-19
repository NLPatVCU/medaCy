import os

from medacy.data.dataset import Dataset


test_dir = os.path.dirname(__file__)
sample_dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'))
