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
        cls.prediction_directory_2 = tempfile.mkdtemp()
        cls.pipeline = TestingPipeline(entities=cls.entities)

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        shutil.rmtree(cls.prediction_directory)
        shutil.rmtree(cls.prediction_directory_2)

    def test_fit_predict_dump_load(self):
        """Fits a model, tests that it predicts correctly, dumps and loads it, then tests that it still predicts"""

        model = Model(self.pipeline)

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
        new_model = Model(self.pipeline)
        new_model.load(pickle_path)

        resulting_ann = new_model.predict('To exclude the possibility that alterations in PSSD might be a consequence of changes in the volume of reference, we used a subset of the vibratome sections')
        self.assertIsInstance(resulting_ann, Annotations)

    def test_predict(self):
        """
        predict() has different functionality depending on what is passed to it; therefore this test
        ensures that each type of input is handled correctly
        """

        # Init the Model
        pipe = TestingPipeline(entities=self.entities)
        sample_model_path = os.path.join(test_dir, 'sample_models', 'sample_test_pipe.pkl')
        model = Model(pipe)
        model.load(sample_model_path)

        # Test passing a Dataset
        dataset_output = model.predict(self.dataset)
        self.assertIsInstance(dataset_output, Dataset)
        self.assertEqual(len(dataset_output), len(self.dataset))

        # Test passing a directory
        directory_output = model.predict(self.dataset.data_directory)
        self.assertIsInstance(directory_output, Dataset)
        self.assertEqual(len(directory_output), len(self.dataset))

        # Test passing a string
        string_output = model.predict('This is a sample string.')
        self.assertIsInstance(string_output, Annotations)

        # Test that the predictions are written to the expected location when no path is provided
        expected_dir = os.path.join(self.dataset.data_directory, 'predictions')
        self.assertTrue(os.path.isdir(expected_dir))

        # Delete that directory
        shutil.rmtree(expected_dir)

        # Test predicting to a specific directory
        model.predict(self.dataset.data_directory, prediction_directory=self.prediction_directory_2)
        expected_files = os.listdir(self.prediction_directory_2)
        self.assertEqual(6, len(expected_files))

    def test_cross_validate(self):
        """Ensures that changes made in the package do not prevent cross_validate from running to completion"""
        model = Model(self.pipeline)
        try:
            model.cross_validate(self.dataset, 2)
        except:
            self.assertTrue(False)

    def test_run_through_pipeline(self):
        """
        Tests that this function runs a document through the pipeline by testing that it has attributes
        overlayed by the pipeline
        """
        model = Model(self.pipeline)
        sample_df = list(self.dataset)[0]
        result = model._run_through_pipeline(sample_df)

        expected = sample_df.txt_path
        actual = result._.file_name
        self.assertEqual(actual, expected)

        expected = sample_df.ann_path
        actual = result._.gold_annotation_file
        self.assertEqual(actual, expected)

