import os
import shutil
import tempfile
from unittest import TestCase

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.testing_pipeline import TestingPipeline
from medacy.tests.sample_data import test_dir


class TestModel(TestCase):
    """Tests for medacy.model.model.Model"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'))
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.prediction_directory = tempfile.mkdtemp()  # directory to store predictions

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)

    def test_fit_predict_dump_load(self):
        """Fits a model, tests that it predicts correctly, dumps and loads it, then tests that it still predicts"""

        pipeline = TestingPipeline(entities=self.entities)
        model = Model(pipeline)

        # Test attempting to predict before fitting
        with self.assertRaises(RuntimeError):
            model.predict('Lorem ipsum dolor sit amet.')

        model.fit(self.dataset)
        # Test X and y data are set
        self.assertTrue(model.X_data)
        self.assertTrue(model.y_data)

        # Test prediction over string
        resulting_ann = model.predict('To exclude the possibility that alterations in PSSD might be a consequence of changes in the volume of reference, we used a subset of the vibratome sections')
        self.assertIsInstance(resulting_ann, Annotations)

        # Test prediction over directory
        resulting_dataset = model.predict(self.dataset.data_directory, prediction_directory=self.prediction_directory)
        self.assertIsInstance(resulting_dataset, Dataset)
        self.assertEqual(len(self.dataset), len(resulting_dataset))

        # Test pickling a model
        pickle_path = os.path.join(self.prediction_directory, 'test.pkl')
        model.dump(pickle_path)
        new_model = Model(pipeline)
        new_model.load(pickle_path)

        resulting_ann = new_model.predict('To exclude the possibility that alterations in PSSD might be a consequence of changes in the volume of reference, we used a subset of the vibratome sections')
        self.assertIsInstance(resulting_ann, Annotations)
