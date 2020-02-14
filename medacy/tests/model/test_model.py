import os
import shutil
import tempfile
import logging
import unittest

import pkg_resources

from medacy.data.annotations import Annotations
from medacy.data.dataset import Dataset
from medacy.model.model import Model
from medacy.pipelines.testing_pipeline import TestingPipeline
from medacy.tests.sample_data import test_dir


class TestModel(unittest.TestCase):
    """Tests for medacy.model.model.Model"""

    @classmethod
    def setUpClass(cls):
        cls.dataset = Dataset(os.path.join(test_dir, 'sample_dataset_1'))
        cls.entities = cls.dataset.get_labels(as_list=True)
        cls.prediction_directory = tempfile.mkdtemp()  # directory to store predictions
        cls.prediction_directory_2 = tempfile.mkdtemp()
        cls.prediction_directory_3 = tempfile.mkdtemp()
        cls.groundtruth_directory = tempfile.mkdtemp()
        cls.groundtruth_2_directory = tempfile.mkdtemp()
        cls.pipeline = TestingPipeline(entities=cls.entities)

    @classmethod
    def tearDownClass(cls):
        pkg_resources.cleanup_resources()
        for d in [cls.prediction_directory, cls.prediction_directory_2,
                  cls.prediction_directory_3, cls.groundtruth_directory, cls.groundtruth_2_directory]:
            shutil.rmtree(d)

    def test_fit_predict_dump_load(self):
        """Fits a model, tests that it predicts correctly, dumps and loads it, then tests that it still predicts"""

        model = Model(self.pipeline)

        # Test attempting to predict before fitting
        with self.assertRaises(RuntimeError):
            model.predict('Lorem ipsum dolor sit amet.')

        model.fit(self.dataset, groundtruth_directory=self.groundtruth_2_directory)
        # Test X and y data are set
        self.assertTrue(model.X_data)
        self.assertTrue(model.y_data)

        # Test that there is at least one prediction
        resulting_ann = model.predict('To exclude the possibility that alterations in PSSD might be a consequence of changes in the volume of reference, we used a subset of the vibratome sections')
        self.assertIsInstance(resulting_ann, Annotations)
        self.assertTrue(resulting_ann)

        # Test prediction over directory
        resulting_dataset = model.predict(self.dataset.data_directory, prediction_directory=self.prediction_directory)
        self.assertIsInstance(resulting_dataset, Dataset)
        self.assertEqual(len(self.dataset), len(resulting_dataset))

        # Test that groundtruth is written
        groundtruth_dataset = Dataset(self.groundtruth_2_directory)
        expected = [d.file_name for d in self.dataset]
        actual = [d.file_name for d in groundtruth_dataset]
        self.assertListEqual(expected, actual)

        # Test that the groundtruth ann files have content
        for ann in groundtruth_dataset.generate_annotations():
            self.assertTrue(ann)

        # Test pickling a model
        pickle_path = os.path.join(self.prediction_directory, 'test.pkl')
        model.dump(pickle_path)
        new_model = Model(self.pipeline)
        new_model.load(pickle_path)

        # Test that there is at least one prediction
        resulting_ann = new_model.predict('To exclude the possibility that alterations in PSSD might be a consequence of changes in the volume of reference, we used a subset of the vibratome sections')
        self.assertIsInstance(resulting_ann, Annotations)
        self.assertTrue(resulting_ann)

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

        # Test that invalid fold counts raise ValueError
        for num in [-1, 0, 1]:
            with self.assertRaises(ValueError):
                model.cross_validate(self.dataset, num)

        try:
            resulting_data = model.cross_validate(self.dataset, 2)
            # Checking the log can help verify that the results of cross validation are expectable
            logging.debug(resulting_data)
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

    def test_cross_validate_create_groundtruth_predictions(self):
        """
        Tests that during cross validation, the medaCy groundtruth (that is, the version of the training dataset
        used by medaCy) is written as well as the predictions that are created for each fold
        """
        model = Model(self.pipeline)
        model.cross_validate(
            self.dataset,
            num_folds=2,
            prediction_directory=self.prediction_directory_3,
            groundtruth_directory=self.groundtruth_directory
        )

        prediction_dataset = Dataset(self.prediction_directory_3)
        groundtruth_dataset = Dataset(self.groundtruth_directory)

        for d in [prediction_dataset, groundtruth_dataset]:
            self.assertIsInstance(d, Dataset)

        original_file_names = {d.file_name for d in self.dataset}
        prediction_file_names = {d.file_name for d in prediction_dataset}
        groundtruth_file_names = {d.file_name for d in groundtruth_dataset}

        for n in [prediction_file_names, groundtruth_file_names]:
            self.assertSetEqual(n, original_file_names)

        # Container for all Annotations in all files in all folds
        all_anns_all_folds_actual = Annotations([])

        # Test that fold groundtruth is written to file
        for fold_name in ["fold_1", "fold_2"]:
            fold_dataset = Dataset(groundtruth_dataset.data_directory / fold_name)
            for d in fold_dataset:
                fold_ann = Annotations(d.ann_path)
                groundtruth_ann = groundtruth_dataset[d.file_name]
                # Test that the entities in the fold groundtruth are a subset of the whole for that file
                self.assertTrue(set(fold_ann) <= set(groundtruth_ann))
                all_anns_all_folds_actual |= fold_ann

        # Container for all annotations pulled directly from the groundtruth dataset
        all_groundtruth_tuples = Annotations([])
        for ann in groundtruth_dataset.generate_annotations():
            all_groundtruth_tuples |= ann

        expected = set(all_groundtruth_tuples)
        actual = set(all_anns_all_folds_actual)
        self.assertSetEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
