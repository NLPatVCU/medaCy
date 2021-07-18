from warnings import warn

import pytest

from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.lstm_systematic_review_pipeline import LstmSystematicReviewPipeline
from medacy.tests.pipeline_components.learners import _fixtures
from medacy.tests.pipeline_components.learners import cuda_device
from medacy.tests.pipeline_components.learners import use_cuda, word_embeddings

BATCH_SIZE = 3

prediction_directory = _fixtures.prediction_directory
dataset = _fixtures.dataset


@pytest.mark.skipif(not (use_cuda and word_embeddings), reason='This test requires a cuda device and word embeddings to be set in the medaCy config file')
@pytest.mark.slow
def test_prediction(dataset, prediction_directory):
    pipeline = LstmSystematicReviewPipeline(
        entities=dataset.get_labels(as_list=True),
        word_embeddings=word_embeddings,
        cuda_device=cuda_device
    )

    model = Model(pipeline)
    model.fit(dataset)
    resulting_dataset = model.predict(dataset, prediction_directory=prediction_directory)
    assert isinstance(resulting_dataset, Dataset)

    # Check that there is at least one prediction
    if not any(resulting_dataset.generate_annotations()):
        warn("The model did not generate any predictions")
