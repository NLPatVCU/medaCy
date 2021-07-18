import os

import pytest

from medacy.data.dataset import Dataset
from medacy.tests.sample_data import test_dir


@pytest.fixture
def dataset():
    return Dataset(os.path.join(test_dir, 'sample_dataset_1'), data_limit=1)


@pytest.fixture
def prediction_directory(tmp_path):
    directory = tmp_path / 'preds'
    directory.mkdir()
    return directory
