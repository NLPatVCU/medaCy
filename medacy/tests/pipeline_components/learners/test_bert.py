import os
from warnings import warn

import pytest

from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.bert_pipeline import BertPipeline
from medacy.tests.pipeline_components.learners import cuda_device
from medacy.tests.sample_data import test_dir

BATCH_SIZE = 3


@pytest.fixture
def dataset():
    return Dataset(os.path.join(test_dir, 'sample_dataset_1'), data_limit=1)


@pytest.fixture
def prediction_directory(tmp_path):
    directory = tmp_path / 'preds'
    directory.mkdir()
    return directory


@pytest.mark.parametrize('use_crf', [True, False])
@pytest.mark.slow
def test_bert(dataset, prediction_directory, use_crf):
    pipe = BertPipeline(
        entities=dataset.get_labels(as_list=True),
        pretrained_model='bert-base-cased',
        batch_size=BATCH_SIZE,
        cuda_device=cuda_device,
        using_crf=use_crf
    )

    model = Model(pipe)
    model.cross_validate(dataset, 2)
    model.fit(dataset)
    resulting_dataset = model.predict(dataset, prediction_directory=prediction_directory)
    assert isinstance(resulting_dataset, Dataset)

    # Check that there is at least one prediction
    if not any(resulting_dataset.generate_annotations()):
        warn("The model did not generate any predictions")
